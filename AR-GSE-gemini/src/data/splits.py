# src/data/splits.py
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Dict, Any
from collections import Counter

def safe_train_test_split(X, y, test_size, random_state, min_samples_per_class=2):
    """
    Performs train_test_split with stratification, falling back to random split if 
    stratification fails due to insufficient samples per class.
    """
    try:
        # Check if we have enough samples per class for stratification
        class_counts = Counter(y)
        min_count = min(class_counts.values())
        
        if min_count < min_samples_per_class:
            print(f"Warning: Some classes have only {min_count} samples, using random split instead of stratified")
            return train_test_split(X, y, test_size=test_size, random_state=random_state)
        else:
            return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    except ValueError as e:
        if "too few" in str(e).lower():
            print(f"Warning: Stratification failed ({e}), falling back to random split")
            return train_test_split(X, y, test_size=test_size, random_state=random_state)
        else:
            raise e

def create_and_save_splits(
    lt_indices: np.ndarray,
    lt_targets: np.ndarray,
    split_ratios: Dict[str, float],
    output_dir: Path,
    seed: int = 42
):
    """
    Splits the long-tailed dataset indices into train, tuneV, val_small, calib
    and saves them to JSON files.

    Args:
        lt_indices: Indices of the full long-tailed training set.
        lt_targets: Corresponding targets for stratification.
        split_ratios: Dictionary with ratios for tuneV, val_small, calib.
                      The rest will be the 'train' set.
        output_dir: Directory to save the JSON files.
        seed: Random seed for reproducibility.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # First, split off the largest part: train
    # The rest will be a temporary set for further splitting
    remaining_ratio = sum(v for k, v in split_ratios.items() if k != 'train')
    
    train_indices, temp_indices, train_targets, temp_targets = safe_train_test_split(
        lt_indices,
        lt_targets,
        test_size=remaining_ratio,
        random_state=seed
    )
    
    # Now split the temp set into tuneV, val_small, calib
    # Ratios need to be renormalized
    tuneV_ratio = split_ratios['tuneV'] / remaining_ratio
    
    tuneV_indices, temp_indices, tuneV_targets, temp_targets = safe_train_test_split(
        temp_indices,
        temp_targets,
        test_size=1.0 - tuneV_ratio,
        random_state=seed
    )
    
    remaining_ratio_2 = split_ratios['val_small'] + split_ratios['calib']
    val_small_ratio = split_ratios['val_small'] / remaining_ratio_2
    
    val_small_indices, calib_indices, _, _ = safe_train_test_split(
        temp_indices,
        temp_targets,
        test_size=1.0 - val_small_ratio,
        random_state=seed
    )
    
    splits = {
        'train': train_indices.tolist(),
        'tuneV': tuneV_indices.tolist(),
        'val_small': val_small_indices.tolist(),
        'calib': calib_indices.tolist()
    }
    
    # Save each split to a separate file
    for name, indices in splits.items():
        filepath = output_dir / f"{name}_indices.json"
        print(f"Saving {name} split with {len(indices)} samples to {filepath}")
        with open(filepath, 'w') as f:
            json.dump(indices, f)
            
    print("\nSplits created and saved successfully!")