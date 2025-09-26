import torch

def estimate_group_acceptance(s_tau, y_true, class_to_group, num_groups):
    """
    Ước tính E_hat[s_tau * 1{y thuộc nhóm G_k}] cho mỗi nhóm k.
    Đây là phiên bản đã sửa lỗi, tính toán kỳ vọng một cách chính xác.
    """
    device = s_tau.device
    
    # Đảm bảo class_to_group ở đúng device
    class_to_group = class_to_group.to(device)
    
    y_groups = class_to_group[y_true]  # [B]
    
    # Tạo one-hot encoding cho nhóm của mỗi mẫu
    # Shape: [B, K]
    group_one_hot = torch.nn.functional.one_hot(y_groups, num_classes=num_groups)
    
    # Nhân s_tau với one-hot encoding
    # s_tau.unsqueeze(1) -> [B, 1]
    # group_one_hot -> [B, K]
    # Kết quả -> [B, K], mỗi cột k chứa s_tau nếu mẫu thuộc nhóm k, ngược lại là 0
    s_tau_per_group = s_tau.unsqueeze(1) * group_one_hot
    
    # Ước tính kỳ vọng bằng cách lấy trung bình trên toàn batch
    # Đây chính là (1/B) * sum(s_tau * 1_{y thuộc nhóm G_k})
    acceptance_expectation_k = torch.mean(s_tau_per_group, dim=0)  # Shape: [K]
    
    return acceptance_expectation_k

def primal_dual_step(model, batch, optimizers, loss_fn, params):
    """
    Performs one step of primal-dual optimization.
    """
    logits, labels = batch
    logits = logits.to(params['device'])
    labels = labels.to(params['device'])
    
    # Zero gradients
    for optimizer in optimizers.values():
        optimizer.zero_grad()
    
    # Forward pass
    outputs = model(logits, params['c'], params['tau'], params['class_to_group'])
    
    # Primary loss: selective classification
    loss_cls = loss_fn(
        eta_mix=outputs['eta_mix'],
        y_true=labels,
        s_tau=outputs['s_tau'],
        beta=params['beta'],
        alpha=model.alpha,
        class_to_group=params['class_to_group']
    )
    
    # Entropy regularization on GATING weights w_φ(x), NOT on class posterior!
    # This encourages sparse/hard routing or load balancing between experts
    gating_entropy = -torch.sum(outputs['w'] * torch.log(outputs['w'] + 1e-8), dim=1)  # [B]
    loss_ent = -params['lambda_ent'] * gating_entropy.mean()  # Negative for regularization
    
    # Rejection loss: L_rej = c * (1/|B|) * Σ(1 - s_τ(x_i))
    s_tau = outputs['s_tau']
    loss_rej = params['c'] * (1.0 - s_tau).mean()
    
    # Constraint loss: L_cons = Σ_k λ_k * g_k(B) - MUST be computed BEFORE backward()
    # Use soft acceptance for constraint calculation (as in paper's primal-dual formulation)
    margin = outputs['margin']
    s_tau = outputs['s_tau']  # Already computed with current tau
    
    # Group coverage constraint: K * P(r(X)=0, Y∈G_k) = α_k
    # This is equivalent to: g_k = α_k - K * (1/B) * Σ_{i∈B} s_τ(x_i) * I{y_i∈G_k}
    sample_groups = params['class_to_group'][labels]
    cons_violation = torch.zeros(model.num_groups, device=params['device'])
    
    # Get current α values - keep gradients for primal update!
    alpha = model.alpha  # [K] - keep gradients!
    B = len(labels)  # batch size
    
    for k in range(model.num_groups):
        group_mask = (sample_groups == k)  # samples from group k
        if group_mask.sum() > 0:
            # P(r=0, Y∈G_k) ≈ (1/B) * Σ_{i: y_i∈G_k} s_τ(x_i)
            joint_prob_empirical = s_tau[group_mask].mean() * (group_mask.sum().float() / B)
            
            # Constraint: K * P(r=0, Y∈G_k) - α_k = 0
            # For dual update, we use: g_k = α_k - K * P(r=0, Y∈G_k)
            target_coverage = alpha[k]  # α_k - keep gradients!
            actual_coverage_scaled = model.num_groups * joint_prob_empirical  # K * P(r=0, Y∈G_k)
            cons_violation[k] = target_coverage - actual_coverage_scaled  # g_k in paper
    
    # Constraint loss: L_cons = Σ_k λ_k * g_k(B) - BEFORE backward, with gradients!
    loss_cons = torch.sum(model.Lambda * cons_violation)
    
    # Complete total loss BEFORE backward()
    loss_total = loss_cls + loss_rej + loss_ent + loss_cons
    
    
    # Backward pass with complete loss
    loss_total.backward()
    
    # Update primal variables
    optimizers['phi'].step()
    
    # Clip alpha to prevent it from becoming too small
    with torch.no_grad():
        model.alpha.data = model.alpha.data.clamp(min=params['alpha_clip'])
    
    optimizers['alpha_mu'].step()
    
    # Post-processing for identifiability/stability (as mentioned in paper Section 5)
    with torch.no_grad():
        # (ii) μ centering: set Σ_k μ_k = 0 (remove translation invariance)  
        model.mu.data = model.mu.data - model.mu.data.mean()
        
        # (iii) Optional: α normalization: set Σ_k log(α_k) = 0
        # This removes scale invariance and helps with stability
        if params.get('normalize_alpha', True):
            log_alpha = torch.log(model.alpha.data)
            log_alpha_centered = log_alpha - log_alpha.mean()
            model.alpha.data = torch.exp(log_alpha_centered)
    
    # Update dual variables AFTER primal update (correct primal-dual sequence)
    # Calculate constraint violations with DETACHED values for dual update
    with torch.no_grad():
        cons_violation_detached = torch.zeros(model.num_groups, device=params['device'])
        alpha_detached = model.alpha.detach()  # [K]
        
        for k in range(model.num_groups):
            group_mask = (sample_groups == k)
            if group_mask.sum() > 0:
                joint_prob_empirical = s_tau[group_mask].mean() * (group_mask.sum().float() / B)
                target_coverage = alpha_detached[k]
                actual_coverage_scaled = model.num_groups * joint_prob_empirical
                cons_violation_detached[k] = target_coverage - actual_coverage_scaled
        
        # Dual update: λ ← [λ + ρ * g_k(B)]₊
        model.Lambda.data = (model.Lambda + params['rho'] * cons_violation_detached).clamp_min(0.0)
    
    # Collect statistics
    stats = {
        'loss_cls': loss_cls.item(),
        'loss_rej': loss_rej.item(),
        'loss_cons': loss_cons.item(),
        'loss_ent': loss_ent.item(), 
        'loss_total': loss_total.item(),
        'mean_coverage': s_tau.mean().item(),  # Use soft acceptance
        'mean_margin': margin.mean().item(),
    }
    
    # Add per-group constraint violations to stats
    for k in range(model.num_groups):
        stats[f'cons_viol_{k}'] = cons_violation[k].item()
        
    return stats, cons_violation