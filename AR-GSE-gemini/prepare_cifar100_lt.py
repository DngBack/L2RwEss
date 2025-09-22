# prepare_cifar100_lt.py
import json
from sklearn.model_selection import train_test_split
import torch
import torchvision
import numpy as np
from pathlib import Path
import collections

# Make sure src is in the python path or run as a module
from src.data.datasets import generate_longtail_train_set, get_cifar100_lt_counts
from src.data.splits import create_and_save_splits
from src.data.groups import get_class_to_group

def main():
    # --- Configuration ---
    SEED = 42
    IMB_FACTOR = 100
    DATA_ROOT = "./data"
    OUTPUT_DIR = Path(f"./data/cifar100_lt_if{IMB_FACTOR}_splits")
    
    # Ratios from your plan (train is the remainder)
    SPLIT_RATIOS = {
        'tuneV': 0.08,
        'val_small': 0.06,
        'calib': 0.03
    }
    
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # --- M1.1: Load original data and create LT train set ---
    print("Step 1: Loading original CIFAR-100 and creating long-tail training set...")
    cifar100_train = torchvision.datasets.CIFAR100(root=DATA_ROOT, train=True, download=True)
    cifar100_test = torchvision.datasets.CIFAR100(root=DATA_ROOT, train=False, download=True)

    lt_indices, lt_targets = generate_longtail_train_set(cifar100_train, IMB_FACTOR)
    
    print(f"Total samples in new long-tail train set: {len(lt_indices)}")
    
    # --- M1.2: Create and save splits from the LT train set ---
    print("\nStep 2: Splitting the LT train set and saving indices...")
    # The 'train' ratio is implied: 1.0 - 0.08 - 0.06 - 0.03 = 0.83, not 0.80.
    # We will adjust to match the plan exactly.
    # Total desired proportion: 0.8 + 0.08 + 0.06 + 0.03 = 0.97
    # We will take 97% of the data and split it according to these ratios
    
    total_prop = 0.8 + 0.08 + 0.06 + 0.03
    num_total_samples = int(len(lt_indices) * total_prop)
    
    subset_indices, _, subset_targets, _ = train_test_split(
        lt_indices, lt_targets, train_size=num_total_samples, random_state=SEED, stratify=lt_targets
    )
    
    # Renormalize ratios
    split_ratios_norm = {
        'train': 0.8 / total_prop,
        'tuneV': 0.08 / total_prop,
        'val_small': 0.06 / total_prop,
        'calib': 0.03 / total_prop
    }
    # Now we split `subset_indices` using the new logic
    create_and_save_splits(subset_indices, subset_targets, split_ratios_norm, OUTPUT_DIR, SEED)

    # --- M1.3: Define groups ---
    print("\nStep 3: Defining class groups (Head/Tail)...")
    class_counts = get_cifar100_lt_counts(IMB_FACTOR)
    class_to_group = get_class_to_group(class_counts, K=2, head_ratio=0.5)

    # --- DoD Check: Print stats ---
    print("\n--- DoD CHECK ---")
    print(f"Imbalance Factor: {IMB_FACTOR}")
    print(f"Most frequent class samples: {class_counts[0]}")
    print(f"Least frequent class samples: {class_counts[-1]}")
    
    # Verify split sizes
    print("\nSplit sizes:")
    for split_name in ['train', 'tuneV', 'val_small', 'calib']:
        with open(OUTPUT_DIR / f"{split_name}_indices.json", 'r') as f:
            indices = json.load(f)
            print(f"- {split_name}: {len(indices)} samples")
    
    # Verify test set
    print(f"- test: {len(cifar100_test)} samples (balanced)")
    
    # Verify group distribution
    head_classes = (class_to_group == 0).sum().item()
    tail_classes = (class_to_group == 1).sum().item()
    print(f"\nGroup distribution: {head_classes} head classes, {tail_classes} tail classes.")
    
    lt_target_counts = collections.Counter(lt_targets)
    head_samples = sum(lt_target_counts[i] for i in range(100) if class_to_group[i] == 0)
    tail_samples = sum(lt_target_counts[i] for i in range(100) if class_to_group[i] == 1)
    print(f"Total samples in LT set -> Head: {head_samples}, Tail: {tail_samples}")
    print("\nMilestone M1 is complete!")

if __name__ == '__main__':
    main()