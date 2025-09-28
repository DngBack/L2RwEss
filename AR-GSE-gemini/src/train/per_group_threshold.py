"""
Per-group threshold (Mondrian) implementation for GSE plugin.
Uses different thresholds t_k for each group based on predicted class group.
"""
import torch

def fit_group_thresholds_from_raw(raw_margins, pred_groups, target_cov_by_group, K, correct_mask=None):
    """
    Fit per-group thresholds t_k based on raw margins and predicted groups.
    
    Args:
        raw_margins: [N] raw margin scores
        pred_groups: [N] predicted group for each sample (based on predicted class)
        target_cov_by_group: [K] target coverage for each group
        K: number of groups
        correct_mask: [N] optional boolean mask for correct predictions only
    
    Returns:
        t_k: [K] threshold for each group
    """
    t_k = torch.zeros(K, dtype=raw_margins.dtype)
    
    for k in range(K):
        # Use only correct predictions for this group if available
        if correct_mask is not None:
            mk = (pred_groups == k) & correct_mask
            # Fallback to all predictions for this group if no correct ones
            if mk.sum() == 0:
                mk = (pred_groups == k)
                print(f"⚠️ No correct predictions for group {k}, using all predictions")
        else:
            mk = (pred_groups == k)
            
        if mk.sum() == 0:
            # No samples predicted for this group - use global threshold
            global_target_cov = float(sum(target_cov_by_group)) / K
            t_k[k] = torch.quantile(raw_margins, 1.0 - global_target_cov)
            print(f"⚠️ No samples for group {k}, using global threshold")
        else:
            # Fit threshold for this group
            t_k[k] = torch.quantile(raw_margins[mk], 1.0 - target_cov_by_group[k])
            
            # Debug info
            if correct_mask is not None:
                n_correct = ((pred_groups == k) & correct_mask).sum().item()
                n_total = (pred_groups == k).sum().item()
                print(f"Group {k}: {n_correct}/{n_total} correct predictions, threshold={t_k[k]:.4f}")
    
    return t_k

def accept_with_group_thresholds(raw_margins, pred_groups, t_k):
    """
    Accept samples based on per-group thresholds.
    
    Args:
        raw_margins: [N] raw margin scores
        pred_groups: [N] predicted group for each sample
        t_k: [K] threshold for each group
    
    Returns:
        accepted: [N] boolean mask for accepted samples
    """
    return raw_margins >= t_k[pred_groups]