import torch

def estimate_group_acceptance(s_tau, y_true, class_to_group, num_groups):
    """
    Ước tính E_hat[s_tau * 1{y thuộc nhóm G_k}] cho mỗi nhóm k.
    Đây là phiên bản đã sửa lỗi, tính toán kỳ vọng một cách chính xác.
    """
    device = s_tau.device
    batch_size = s_tau.shape[0]
    
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
    
    # Entropy regularization
    probs_mix = torch.softmax(outputs['eta_mix'], dim=1)
    entropy = -torch.sum(probs_mix * torch.log(probs_mix + 1e-8), dim=1)
    loss_ent = -params['lambda_ent'] * entropy.mean()
    
    # Rejection loss: L_rej = c * (1/|B|) * Σ(1 - s_τ(x_i))
    s_tau = outputs['s_tau']
    loss_rej = params['c'] * (1.0 - s_tau).mean()
    
    # Constraint loss: L_cons = Σ_k λ_k * g_k(B) (computed later with current constraint violations)
    # For now, we'll compute it after constraint violations are calculated
    
    # Total loss (L_cons will be added later in the function)
    loss_total = loss_cls + loss_rej + loss_ent
    
    # Backward pass
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
    
    # Calculate constraint violations using SOFT acceptance s_τ(x) (as per paper)
    margin = outputs['margin']
    # Use soft acceptance for constraint calculation (as in paper's primal-dual formulation)
    tau = params.get('tau', 1.0)  # get current temperature
    s_tau = torch.sigmoid(tau * margin)  # soft acceptance s_τ(x) = σ(τ * m(x))
    
    # Group coverage constraint: K * P(r(X)=0, Y∈G_k) = α_k
    # This is equivalent to: α_k - K * (1/B) * Σ_{i∈B} s_τ(x_i) * I{y_i∈G_k} = 0
    sample_groups = params['class_to_group'][labels]
    cons_violation = torch.zeros(model.num_groups, device=params['device'])
    
    # Get current α values
    alpha = model.alpha.detach()  # [K]
    B = len(labels)  # batch size
    
    for k in range(model.num_groups):
        group_mask = (sample_groups == k)  # samples from group k
        if group_mask.sum() > 0:
            # P(r=0, Y∈G_k) ≈ (1/B) * Σ_{i: y_i∈G_k} s_τ(x_i)
            joint_prob_empirical = s_tau[group_mask].mean() * (group_mask.sum().float() / B)
            
            # Constraint: K * P(r=0, Y∈G_k) - α_k = 0
            # For dual update, we use: g_k = α_k - K * P(r=0, Y∈G_k)
            target_coverage = alpha[k]  # α_k
            actual_coverage_scaled = model.num_groups * joint_prob_empirical  # K * P(r=0, Y∈G_k)
            cons_violation[k] = target_coverage - actual_coverage_scaled  # g_k in paper
    
    # Constraint loss: L_cons = Σ_k λ_k * g_k(B)
    loss_cons = torch.sum(model.Lambda * cons_violation)
    
    # Add constraint loss to total (complete the total loss)
    loss_total = loss_total + loss_cons
    
    # Update dual variables IMMEDIATELY after primal update (correct primal-dual)
    with torch.no_grad():
        model.Lambda.data = (model.Lambda + params['rho'] * cons_violation).clamp_min(0.0)
    
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