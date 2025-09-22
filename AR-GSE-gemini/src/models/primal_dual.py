# src/models/primal_dual.py
import torch

def estimate_group_acceptance(s_tau, y_true, class_to_group, num_groups):
    """Computes E_hat[s_tau * 1{y in G_k}] for each group k."""
    device = s_tau.device
    class_to_group = class_to_group.to(device)
    
    y_groups = class_to_group[y_true] # [B]
    
    acc_k = torch.zeros(num_groups, device=device)
    
    # Calculate weighted acceptance for each group present in the batch
    # This is more robust than a one-hot encoding approach for large K
    group_indices, group_counts = y_groups.unique(return_counts=True)
    
    # Using scatter_add_ to sum s_tau values for each group
    group_s_sum = torch.zeros(num_groups, device=device).scatter_add_(0, y_groups, s_tau)
    
    # Normalize by the number of samples in the entire batch that belong to group k
    # This is an empirical expectation over the batch
    batch_group_counts = torch.bincount(y_groups, minlength=num_groups)
    
    # Avoid division by zero for groups not in the batch
    valid_groups = batch_group_counts > 0
    acc_k[valid_groups] = group_s_sum[valid_groups] / batch_group_counts[valid_groups]
    
    return acc_k

def primal_dual_step(
    model, batch, optimizers, loss_fn, params
):
    """
    Performs a single primal-dual update step for AR-GSE.
    """
    # Unpack batch and params
    expert_logits, y_true = batch
    expert_logits, y_true = expert_logits.to(params['device']), y_true.to(params['device'])
    
    # Forward pass
    outputs = model(expert_logits, params['c'], params['tau'], params['class_to_group'])
    eta_mix, s_tau, w = outputs['eta_mix'], outputs['s_tau'], outputs['w']

    # --- Loss Calculation ---
    # 1. Selective classification loss
    loss_cls = loss_fn(eta_mix, y_true, s_tau, params['beta'], model.alpha, params['class_to_group'])
    
    # 2. Rejection loss
    loss_rej = params['c'] * (1 - s_tau).mean()

    # 3. Constraint violation term
    # E_hat[s * 1{y in G_k}]
    acc_k_hat = estimate_group_acceptance(s_tau.detach(), y_true, params['class_to_group'], model.num_groups)
    # The constraint is (alpha_k - K * E[...])
    cons_violation = model.alpha - model.num_groups * acc_k_hat
    
    # 4. Regularizers
    # Entropy of gating weights to encourage softer assignments
    entropy_w = -torch.sum(w * torch.log(w + 1e-8), dim=1).mean()
    loss_reg = params['lambda_ent'] * entropy_w

    # Full Lagrangian
    L = loss_cls + loss_rej + (model.Lambda * cons_violation).sum() + loss_reg
    
    # --- Primal Updates ---
    for opt in optimizers.values():
        opt.zero_grad()
    L.backward()
    for opt in optimizers.values():
        opt.step()

    # --- Post-process Primal Variables ---
    with torch.no_grad():
        model.alpha.data.clamp_(min=params['alpha_clip'])
        model.mu.data -= model.mu.data.mean()

    # --- Dual Updates ---
    with torch.no_grad():
        # Using the same batch estimate of constraint violation
        new_cons_violation = model.alpha - model.num_groups * acc_k_hat
        model.Lambda.data = (model.Lambda + params['rho'] * new_cons_violation).clamp_min(0.0)

    # Return stats for logging
    stats = {
        'loss_total': L.item(),
        'loss_cls': loss_cls.item(),
        'loss_rej': loss_rej.item(),
        'loss_reg': loss_reg.item(),
        'mean_alpha': model.alpha.mean().item(),
        'mean_mu': model.mu.mean().item(),
        'mean_lambda': model.Lambda.mean().item(),
        'mean_coverage': s_tau.mean().item(),
    }
    for k in range(model.num_groups):
        stats[f'alpha_{k}'] = model.alpha[k].item()
        stats[f'lambda_{k}'] = model.Lambda[k].item()
        stats[f'cons_viol_{k}'] = cons_violation[k].item()

    return stats