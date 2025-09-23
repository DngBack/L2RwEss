# src/data/splits.py
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Dict, Any
from collections import Counter

def create_longtail_val_test_splits(
    cifar100_test_dataset,
    train_class_counts: list,
    output_dir: Path,
    val_size: float = 0.2,
    seed: int = 42
):
    """
    Tạo ra tập validation và test có phân phối long-tail từ tập test cân bằng gốc.
    """
    print("\nCreating long-tail validation and test sets...")
    test_targets = np.array(cifar100_test_dataset.targets)
    num_classes = len(train_class_counts)
    
    # Số lượng mẫu tối đa cho mỗi lớp trong tập test gốc là 100
    max_samples_per_class_test = 100
    
    lt_test_pool_indices = []
    
    # Subsample tập test gốc để khớp với phân phối của tập train
    for i in range(num_classes):
        # Số lượng mẫu cần lấy cho lớp i phải nhỏ hơn hoặc bằng số lượng trong tập train
        # và không được vượt quá số lượng có sẵn trong tập test (100)
        num_samples_to_take = min(train_class_counts[i], max_samples_per_class_test)
        
        class_indices_in_test = np.where(test_targets == i)[0]
        
        # Lấy ngẫu nhiên các chỉ số từ tập test
        sampled_indices = np.random.choice(class_indices_in_test, num_samples_to_take, replace=False)
        lt_test_pool_indices.extend(sampled_indices)
        
    lt_test_pool_indices = np.array(lt_test_pool_indices)
    lt_test_pool_targets = test_targets[lt_test_pool_indices]
    
    print(f"Created a long-tail pool from test set with {len(lt_test_pool_indices)} samples.")
    
    # Chia pool này thành 20% validation và 80% test
    val_indices, test_indices = train_test_split(
        lt_test_pool_indices,
        test_size=1.0 - val_size,
        random_state=seed,
        stratify=lt_test_pool_targets
    )
    
    splits = {
        'val_lt': val_indices.tolist(),
        'test_lt': test_indices.tolist()
    }
    
    for name, indices in splits.items():
        filepath = output_dir / f"{name}_indices.json"
        print(f"Saving {name} split with {len(indices)} samples to {filepath}")
        with open(filepath, 'w') as f:
            json.dump(indices, f)

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