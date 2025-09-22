# src/data/groups.py
from __future__ import annotations
from typing import Dict
import numpy as np
import torch

def class_to_group_paper(counts: np.ndarray, thr: int = 20) -> torch.LongTensor:
    """
    Paper setup: tail = lớp có <= thr mẫu trong TRAIN-LT, head = còn lại.
    Trả về tensor [C] với 0=head, 1=tail.
    """
    assert counts.ndim == 1
    g = (counts <= thr).astype(np.int64)  # tail=1, head=0
    return torch.from_numpy(g)

def class_to_group_by_frequency(
    counts: np.ndarray,
    K: int = 2,
    head_ratio: float = 0.5,
    mode: str = "class-quantile",
) -> torch.LongTensor:
    """
    Assign groups based on class frequency in `train`.

    Args:
        counts: np.ndarray shape [C], number of samples per class (train).
        K: number of groups (>=2).
        head_ratio: used only when K=2. Ratio of classes in the 'head' group.
        mode:
          - "class-quantile": split by *number of classes* (each group ~C/K classes).
          - "mass-quantile" : split by *total number of samples*, each group ~N/K samples.

    Returns:
        torch.LongTensor shape [C], each element ∈ {0..K-1}.
    """
    assert isinstance(counts, np.ndarray), "counts must be a numpy array"
    assert counts.ndim == 1, "counts must be 1-dimensional [C]"
    C = counts.shape[0]
    assert K >= 2 and K <= C, "2 <= K <= C"

    order = np.argsort(-counts)  # sort descending by frequency
    g = np.empty(C, dtype=np.int64)

    if K == 2:
        # number of head classes, clamp so both groups are non-empty
        H = int(round(head_ratio * C))
        H = max(1, min(H, C - 1))
        head_classes = order[:H]
        g[:] = 1  # tail by default
        g[head_classes] = 0
        return torch.from_numpy(g)

    # K > 2
    mode = mode.lower()
    if mode not in {"class-quantile", "mass-quantile"}:
        raise ValueError("mode must be 'class-quantile' or 'mass-quantile'")

    if mode == "class-quantile":
        # Split nearly evenly by number of classes, ensure non-empty
        bins = np.array_split(order, K)  # handles difference of 1 class automatically
        for k, cls_k in enumerate(bins):
            g[cls_k] = k
    else:
        # Split nearly evenly by total number of samples (mass)
        tot = counts.sum()
        # Target mass thresholds for group boundaries (loose to avoid empty groups)
        targets = [(tot * (k + 1)) / K for k in range(K - 1)]  # K-1 thresholds
        csum = counts[order].cumsum()
        boundaries = []
        start = 0
        for t in targets:
            # find first index where csum >= t
            j = int(np.searchsorted(csum, t, side="left"))
            # avoid empty group: force boundary >= start+1 and <= C-(remaining groups)
            j = max(j, start + 1)
            j = min(j, C - ((K - len(boundaries) - 1)))
            boundaries.append(j)
            start = j
        cuts = [0] + boundaries + [C]
        for k in range(K):
            cls_k = order[cuts[k]:cuts[k + 1]]
            # in extreme case with many counts=0, group may still be empty -> fallback
            if cls_k.size == 0:
                # assign one nearest remaining class
                # find unused class
                unused = np.setdiff1d(order, np.where(g >= 0)[0], assume_unique=False)
                if unused.size > 0:
                    cls_k = np.array([unused[0]])
            g[cls_k] = k

    return torch.from_numpy(g)

def summarize_groups(counts: np.ndarray, class_to_group: torch.LongTensor) -> Dict[int, dict]:
    """
    Return basic statistics for each group: number of classes, total samples, mean per class,
    and min/max for sanity-checking the imbalance.
    """
    assert isinstance(counts, np.ndarray) and counts.ndim == 1
    arr = class_to_group.detach().cpu().numpy() if torch.is_tensor(class_to_group) else np.asarray(class_to_group)
    C = counts.shape[0]
    assert arr.shape[0] == C, "class_to_group size must match number of classes C"
    K = int(arr.max()) + 1

    out = {}
    for k in range(K):
        mask = (arr == k)
        n_cls = int(mask.sum())
        if n_cls == 0:
            out[k] = {"num_classes": 0, "total_samples": 0, "mean_per_class": 0.0, "min_per_class": 0, "max_per_class": 0}
            continue
        cnts = counts[mask]
        out[k] = {
            "num_classes": n_cls,
            "total_samples": int(cnts.sum()),
            "mean_per_class": float(cnts.mean()),
            "min_per_class": int(cnts.min()),
            "max_per_class": int(cnts.max()),
        }
    return out
