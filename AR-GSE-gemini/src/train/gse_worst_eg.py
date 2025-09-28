"""
GSE Worst-group with Exponentiated-Gradient (EG) outer loop.
This implements the EG-outer algorithm for worst-group selective prediction.
"""
import torch
import numpy as np
from src.train.gse_balanced_plugin import (
    c_for_target_coverage_from_raw,
    update_alpha_fixed_point, update_alpha_fixed_point_conditional,
    worst_error_on_S
)

@torch.no_grad()
def compute_raw_margin_with_beta(eta, alpha, mu, beta, class_to_group):
    """Compute raw margin with beta weighting: (α*β)_g(y) * η̃_y - ((α*β)_g(y) - μ_g(y)) * Σ η̃_y'"""
    cg = class_to_group.to(eta.device)
    ab = (alpha * beta).to(eta.device)             # [K]
    score = (ab[cg] * eta).max(dim=1).values
    coeff = ab[cg] - mu[cg]
    thr = (coeff.unsqueeze(0) * eta).sum(dim=1)
    return score - thr

def accepted_pred_with_beta(eta, alpha, mu, beta, thr, class_to_group):
    """Accept samples and make predictions using beta-weighted margins."""
    raw = compute_raw_margin_with_beta(eta, alpha, mu, beta, class_to_group)
    accepted = (raw >= thr)
    preds = ((alpha*beta)[class_to_group] * eta).argmax(dim=1)
    return accepted, preds, raw - thr

def inner_cost_sensitive_plugin(eta_S1, y_S1, eta_S2, y_S2, class_to_group, K,
                                beta, lambda_grid, M=8, alpha_steps=4,
                                cov_target=0.58, gamma=0.25, use_conditional_alpha=False):
    """
    Inner cost-sensitive plugin optimization for given beta.
    
    Args:
        eta_S1, y_S1: S1 split data
        eta_S2, y_S2: S2 split data  
        class_to_group: class to group mapping
        K: number of groups
        beta: [K] group weights from EG outer loop
        lambda_grid: lambda values to search over
        M: number of plugin iterations
        alpha_steps: fixed-point steps for alpha
        cov_target: target coverage
        gamma: EMA factor for alpha updates
        use_conditional_alpha: use conditional acceptance for alpha updates
    
    Returns:
        best_alpha, best_mu, best_t, best_score
    """
    device = eta_S1.device
    alpha = torch.ones(K, device=device)
    best = {"score": float("inf")}
    mus = []
    for lam in lambda_grid:
        if K==2: 
            mus.append(torch.tensor([lam/2.0, -lam/2.0], device=device))
        else: 
            raise NotImplementedError("Provide mu grid for K>2")

    for _ in range(M):
        for lam, mu in zip(lambda_grid, mus):
            a_cur = alpha.clone()
            t_cur = None
            for _ in range(alpha_steps):
                raw_S1 = compute_raw_margin_with_beta(eta_S1, a_cur, mu, beta, class_to_group)
                t_cur = c_for_target_coverage_from_raw(raw_S1, cov_target)
                if use_conditional_alpha:
                    a_cur = update_alpha_fixed_point_conditional(eta_S1, y_S1, a_cur, mu, t_cur, class_to_group, K, gamma=gamma)
                else:
                    a_cur = update_alpha_fixed_point(eta_S1, y_S1, a_cur, mu, t_cur, class_to_group, K, gamma=gamma, use_conditional=False)

            w_err, gerrs = worst_error_on_S(eta_S2, y_S2, a_cur, mu, t_cur, class_to_group, K)
            if w_err < best["score"]:
                best.update(dict(score=w_err, alpha=a_cur.clone(), mu=mu.clone(), t=t_cur))
        alpha = 0.5*alpha + 0.5*best["alpha"]
    return best["alpha"], best["mu"], best["t"], best["score"]

def worst_group_eg_outer(eta_S1, y_S1, eta_S2, y_S2, class_to_group, K,
                         T=25, xi=1.0, lambda_grid=None, **inner_kwargs):
    """
    Worst-group EG-outer algorithm.
    
    Args:
        eta_S1, y_S1: S1 split data
        eta_S2, y_S2: S2 split data
        class_to_group: class to group mapping
        K: number of groups
        T: number of EG outer iterations
        xi: EG step size
        lambda_grid: lambda values for inner optimization
        **inner_kwargs: additional arguments for inner optimization
    
    Returns:
        alpha_star, mu_star, t_star, beta_star, history
    """
    device = eta_S1.device
    if lambda_grid is None:
        lambda_grid = np.linspace(-1.5, 1.5, 31).tolist()
    
    # Initialize uniform beta
    beta = torch.full((K,), 1.0/K, device=device)
    history = []

    print(f"Starting EG-outer with T={T}, xi={xi}, lambda_grid=[{lambda_grid[0]:.2f}, {lambda_grid[-1]:.2f}]")

    for t in range(T):
        print(f"EG iteration {t+1}/{T}, β={beta.detach().cpu().tolist()}")
        
        # Inner optimization with current beta
        a_t, m_t, thr_t, _ = inner_cost_sensitive_plugin(
            eta_S1, y_S1, eta_S2, y_S2, class_to_group, K, beta,
            lambda_grid=lambda_grid, **inner_kwargs
        )
        
        # Compute per-group errors on S2
        w_err, gerrs = worst_error_on_S(eta_S2, y_S2, a_t, m_t, thr_t, class_to_group, K)
        print(f"  Worst error: {w_err:.4f}, Group errors: {[f'{g:.4f}' for g in gerrs]}")
        
        # EG update: β_k ← β_k * exp(xi * error_k)
        with torch.no_grad():
            e = torch.tensor(gerrs, device=device)
            beta = beta * torch.exp(xi * e)   # EG update
            beta = beta / beta.sum()           # Normalize
            
        history.append(dict(
            iteration=t+1,
            beta=beta.detach().cpu().tolist(), 
            gerrs=[float(x) for x in gerrs],
            worst_error=float(w_err)
        ))

    print("✅ EG-outer optimization complete")
    return a_t, m_t, thr_t, beta.detach().cpu(), history