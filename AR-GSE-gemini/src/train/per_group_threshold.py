"""
Per-group threshold (Mondrian) implementation for GSE plugin.
Uses different thresholds t_k for each group based on predicted class group.
"""
import torch

def fit_group_thresholds_from_raw(raw_margins, pred_groups, target_cov_by_group, K):
    """
    Fit per-group thresholds t_k based on raw margins and predicted groups.
    
    Args:
        raw_margins: [N] raw margin scores
        pred_groups: [N] predicted group for each sample (based on predicted class)
        target_cov_by_group: [K] target coverage for each group
        K: number of groups
    
    Returns:
        t_k: [K] threshold for each group
    """
    t_k = torch.zeros(K, dtype=raw_margins.dtype)
    
    for k in range(K):
        mk = (pred_groups == k)
        if mk.sum() == 0:
            # No samples predicted for this group - use global threshold
            global_target_cov = float(sum(target_cov_by_group)) / K
            t_k[k] = torch.quantile(raw_margins, 1.0 - global_target_cov)
        else:
            # Fit threshold for this group
            t_k[k] = torch.quantile(raw_margins[mk], 1.0 - target_cov_by_group[k])
    
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