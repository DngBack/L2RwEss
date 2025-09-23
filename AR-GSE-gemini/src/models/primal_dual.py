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
    
    # Total loss
    loss_total = loss_cls + loss_ent
    
    # Backward pass
    loss_total.backward()
    
    # Update primal variables
    optimizers['phi'].step()
    
    # Clip alpha to prevent it from becoming too small
    with torch.no_grad():
        model.alpha.data = model.alpha.data.clamp(min=params['alpha_clip'])
    
    optimizers['alpha_mu'].step()
    
    # Calculate constraint violations using ACTUAL acceptance decisions (not probabilities)
    margin = outputs['margin']
    actual_acceptance = (margin > 0).float()  # Hard acceptance rule
    
    # Group coverage estimation using actual acceptance
    sample_groups = params['class_to_group'][labels]
    cons_violation = torch.zeros(model.num_groups, device=params['device'])
    
    for k in range(model.num_groups):
        group_mask = (sample_groups == k)
        if group_mask.sum() > 0:
            group_coverage = actual_acceptance[group_mask].mean()
            cons_violation[k] = torch.clamp(params['c'] - group_coverage, min=0.0)
    
    # Update dual variables IMMEDIATELY after primal update (correct primal-dual)
    with torch.no_grad():
        model.Lambda.data = (model.Lambda + params['rho'] * cons_violation).clamp_min(0.0)
    
    # Collect statistics
    stats = {
        'loss_cls': loss_cls.item(),
        'loss_ent': loss_ent.item(), 
        'loss_total': loss_total.item(),
        'mean_coverage': actual_acceptance.mean().item(),
        'mean_margin': margin.mean().item(),
    }
    
    # Add per-group constraint violations to stats
    for k in range(model.num_groups):
        stats[f'cons_viol_{k}'] = cons_violation[k].item()
        
    return stats, cons_violation