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

def primal_dual_step(
    model, batch, optimizers, loss_fn, params
):
    """
    Thực hiện một bước cập nhật primal-dual cho AR-GSE.
    Phiên bản cuối cùng đã sửa lỗi và thêm điều chuẩn alpha.
    """
    # 1. Chuẩn bị dữ liệu và tham số
    expert_logits, y_true = batch
    expert_logits, y_true = expert_logits.to(params['device']), y_true.to(params['device'])
    
    class_to_group = params['class_to_group']
    beta = params['beta']
    
    # 2. Forward pass qua mô hình
    outputs = model(expert_logits, params['c'], params['tau'], class_to_group)
    eta_mix, s_tau, w = outputs['eta_mix'], outputs['s_tau'], outputs['w']

    # 3. Tính toán các thành phần của hàm loss Lagrangian
    # Lỗi phân loại chọn lọc
    loss_cls = loss_fn(eta_mix, y_true, s_tau, beta, model.alpha, class_to_group)
    
    # Lỗi từ chối
    loss_rej = params['c'] * (1 - s_tau).mean()

    # Mức độ vi phạm ràng buộc (dùng cho cả loss và cập nhật dual sau này)
    acc_k_hat = estimate_group_acceptance(s_tau.detach(), y_true, class_to_group, model.num_groups)
    cons_violation = model.alpha - model.num_groups * acc_k_hat
    
    # Điều chuẩn cho trọng số gating (entropy)
    entropy_w = -torch.sum(w * torch.log(w + 1e-8), dim=1).mean()
    loss_reg = params['lambda_ent'] * entropy_w

    # (CẢI TIẾN) Điều chuẩn để "neo giữ" alpha quanh giá trị 1.0
    alpha_reg_strength = 0.1
    loss_alpha_reg = alpha_reg_strength * ((model.alpha - 1.0)**2).sum()

    # 4. Tính toán loss Lagrangian tổng hợp
    L = loss_cls + loss_rej + (model.Lambda * cons_violation).sum() + loss_reg + loss_alpha_reg
    
    # 5. Cập nhật Primal (φ, α, μ)
    for opt in optimizers.values():
        opt.zero_grad()
    L.backward()
    for opt in optimizers.values():
        opt.step()

    # 6. Xử lý sau cho các biến Primal
    with torch.no_grad():
        model.alpha.data.clamp_(min=params['alpha_clip'])
        model.mu.data -= model.mu.data.mean()

    # (LƯU Ý) Cập nhật Dual (λ) đã được chuyển ra hàm main để sử dụng EMA

    # 7. Trả về các chỉ số và mức độ vi phạm ràng buộc
    stats = {
        'loss_total': L.item(),
        'loss_cls': loss_cls.item(),
        'loss_rej': loss_rej.item(),
        'loss_reg': (loss_reg + loss_alpha_reg).item(),
        'mean_alpha': model.alpha.mean().item(),
        'mean_mu': model.mu.mean().item(),
        'mean_lambda': model.Lambda.mean().item(),
        'mean_coverage': s_tau.mean().item(),
    }
    for k in range(model.num_groups):
        stats[f'alpha_{k}'] = model.alpha[k].item()
        stats[f'lambda_{k}'] = model.Lambda[k].item()
        stats[f'cons_viol_{k}'] = cons_violation[k].item()
        
    return stats, cons_violation