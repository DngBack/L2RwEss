# src/data/datasets.py
import numpy as np
import torchvision
from typing import List, Tuple

def get_cifar100_lt_counts(imb_factor: int = 100, num_classes: int = 100) -> List[int]:
    """
    Generates class sample counts for CIFAR-100-LT.
    Follows the standard exponential decay rule.

    Args:
        imb_factor: Imbalance factor, e.g., 100 means the most frequent class
                    has 100 times more samples than the least frequent one.
        num_classes: Number of classes, 100 for CIFAR-100.

    Returns:
        A list of sample counts per class.
    """
    # Original CIFAR-100 has 500 samples per class in the training set
    img_max = 500.0
    img_min = img_max / imb_factor
    
    counts = []
    for cls_idx in range(num_classes):
        num = img_max * (img_min / img_max) ** (cls_idx / (num_classes - 1.0))
        counts.append(int(num))
        
    return counts

def generate_longtail_train_set(cifar100_train_dataset, imb_factor: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Subsamples the original CIFAR-100 training set to create a long-tailed version.

    Args:
        cifar100_train_dataset: The original torchvision CIFAR-100 training dataset.
        imb_factor: The desired imbalance factor.

    Returns:
        A tuple of (indices, targets) for the new long-tailed training set.
    """
    num_classes = 100
    targets = np.array(cifar100_train_dataset.targets)
    
    # Get target counts for LT dataset
    target_counts = get_cifar100_lt_counts(imb_factor, num_classes)
    
    # Get all indices for each class
    class_indices = [np.where(targets == i)[0] for i in range(num_classes)]
    
    # Subsample indices for each class
    lt_indices = []
    for i in range(num_classes):
        # Ensure we don't request more samples than available
        num_samples = min(target_counts[i], len(class_indices[i]))
        # Randomly sample indices
        sampled_indices = np.random.choice(class_indices[i], num_samples, replace=False)
        lt_indices.extend(sampled_indices)
        
    lt_indices = np.array(lt_indices)
    lt_targets = targets[lt_indices]
    
    return lt_indices, lt_targets

# Augmentations can be defined here later, e.g., using torchvision.transforms
def get_train_augmentations():
    # Placeholder for RandAug, Mixup, CutMix logic
    pass

def get_eval_augmentations():
    # Placeholder for normalize/center-crop
    pass