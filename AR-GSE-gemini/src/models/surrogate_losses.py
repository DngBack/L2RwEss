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

    Args:
        eta_mix: Mixed posterior from the ensemble [B, C]
        y_true: Ground truth labels [B]
        s_tau: Soft acceptance probability [B]
        beta: Per-group cost weights [K]
        alpha: Per-group acceptance targets [K]
        class_to_group: Mapping from class to group [C]
        kind: Type of classification loss, "ce" or "one_minus_p"

    Returns:
        Scalar loss value.
    """
    B, C = eta_mix.shape
    device = eta_mix.device
    
    # Get group for each sample in the batch
    # y_true [B] -> sample_groups [B]
    sample_groups = class_to_group[y_true].to(device)
    
    # Get the cost weight for each sample
    # sample_groups [B] -> sample_beta [B]
    sample_beta = beta[sample_groups]
    # sample_groups [B] -> sample_alpha [B]
    sample_alpha = alpha[sample_groups]

    # Calculate the per-sample classification loss
    if kind == "ce":
        # We need to compute it sample-wise to apply weights
        per_sample_loss = F.cross_entropy(eta_mix, y_true, reduction='none')
    elif kind == "one_minus_p":
        # gathers the mixed posterior probability of the true class
        p_true = eta_mix.gather(1, y_true.unsqueeze(1)).squeeze()
        per_sample_loss = 1 - p_true
    else:
        raise ValueError(f"Unknown loss kind: {kind}")
    
    # The term from the paper is: (beta_k / alpha_k) * loss * s_tau
    weighted_loss = (sample_beta / (sample_alpha + 1e-8)) * per_sample_loss * s_tau
    
    return weighted_loss.mean()