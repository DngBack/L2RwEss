# src/models/surrogate_losses.py
import torch
import torch.nn.functional as F

def selective_cls_loss(
    eta_mix: torch.Tensor,
    y_true: torch.Tensor,
    s_tau: torch.Tensor,
    beta: torch.Tensor,
    alpha: torch.Tensor,
    class_to_group: torch.LongTensor,
    kind: str = "ce"
) -> torch.Tensor:
    """
    Calculates the surrogate selective classification loss.
    (Phiên bản đã sửa lỗi device placement)
    """
    # Get group for each sample in the batch
    # All input tensors are already on the correct device.
    sample_groups = class_to_group[y_true]
    
    # Get the cost weight for each sample
    sample_beta = beta[sample_groups]
    sample_alpha = alpha[sample_groups]

    # Calculate the per-sample classification loss
    if kind == "ce":
        per_sample_loss = F.cross_entropy(eta_mix, y_true, reduction='none')
    elif kind == "one_minus_p":
        p_true = eta_mix.gather(1, y_true.unsqueeze(1)).squeeze()
        per_sample_loss = 1 - p_true
    else:
        raise ValueError(f"Unknown loss kind: {kind}")
    
    # The classification loss from paper: 
    # L_cls = Σ_k β_k * α_k * (1/|B|) * Σ_{i∈B} ℓ_cls(x_i, y_i; η̃) * s_τ(x_i) * I{y_i∈G_k}
    # Per sample: β_k * α_k * ℓ_cls * s_τ (NOT division!)
    weighted_loss = sample_beta * sample_alpha * per_sample_loss * s_tau
    
    return weighted_loss.mean()