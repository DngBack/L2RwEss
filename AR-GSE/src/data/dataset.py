from __future__ import annotations

import json
import os
import random 
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms


# Cifar 100 long-tailed transform
def cifar_train_transform():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
    ])

def cifar_eval_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
    ])
    
# ---------- Load CIFAR ----------
def load_cifar(root: str, name: str, train: bool):
    assert name in {"cifar10","cifar100"}
    if name == "cifar10":
        ds = datasets.CIFAR10(root, train=train, download=True,
                            transform=cifar_train_transform() if train else cifar_eval_transform())
        targets = ds.targets  # list[int]
        num_classes = 10
    else:
        ds = datasets.CIFAR100(root, train=train, download=True,
                            transform=cifar_train_transform() if train else cifar_eval_transform())
        targets = ds.targets
        num_classes = 100
    return ds, targets, num_classes

# ---------- Generate long-tailed distribution using exponential function ----------
def img_num_per_cls(base_per_cls: int, c_num: int, imb_factor: float) -> List[int]:
    """
    base_per_cls: maximum number of images per class (usually equals #images/class in original train set)
    c_num: number of classes
    ims[k] ≈ base_per_cls * (imb_factor ** (-k/(c_num-1)))  (k=0 is the head class)
    """
    if imb_factor <= 1:  # balanced
        return [base_per_cls for _ in range(c_num)]
    img_max = base_per_cls
    cls_num = []
    for k in range(c_num):
        frac = pow(imb_factor, -k / (c_num - 1))
        cls_num.append(int(round(img_max * frac)))
    # ensure >= 1
    cls_num = [max(1, n) for n in cls_num]
    return cls_num

def make_long_tailed_indices(targets: List[int], num_classes: int,
                            imb_factor: float, seed: int) -> List[int]:
    """
    Returns a list of indices forming the long-tailed train set using exponential scaling.
    Assumes the original train data is evenly distributed across classes.
    """
    rng = random.Random(seed)
    targets = np.array(targets)
    # number of images per class in the original train set
    original_counts = np.bincount(targets, minlength=num_classes)
    base_per_cls = int(original_counts.max())  # 5000 for CIFAR-10, 500 for CIFAR-100
    desired = img_num_per_cls(base_per_cls, num_classes, imb_factor)

    indices = []
    for cls in range(num_classes):
        cls_idx = np.where(targets == cls)[0].tolist()
        rng.shuffle(cls_idx)
        take = min(desired[cls], len(cls_idx))
        indices.extend(cls_idx[:take])

    rng.shuffle(indices)
    return indices

# ---------- Split indices reproducibly ----------
@dataclass
class SplitConfig:
    train: float = 0.80
    tuneV: float = 0.08
    val_small: float = 0.06
    calib: float = 0.06  # can be 0.03 if you want to reduce
    # test uses the original test set

def partition_indices(idxs: List[int], cfg: SplitConfig, seed: int) -> Dict[str, List[int]]:
    rng = random.Random(seed)
    idxs = list(idxs)
    rng.shuffle(idxs)
    n = len(idxs)
    n_train = int(n * cfg.train)
    n_tune = int(n * cfg.tuneV)
    n_val = int(n * cfg.val_small)
    n_cal = int(n * cfg.calib)
    # preserve total
    remain = n - (n_train + n_tune + n_val + n_cal)
    n_train += max(0, remain)

    splits = {
        "train": idxs[:n_train],
        "tuneV": idxs[n_train:n_train+n_tune],
        "val_small": idxs[n_train+n_tune:n_train+n_tune+n_val],
        "calib": idxs[n_train+n_tune+n_val:n_train+n_tune+n_val+n_cal],
    }
    return splits

def save_splits(path: str, splits: Dict[str, List[int]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({k: list(map(int,v)) for k,v in splits.items()}, f)

def load_splits(path: str) -> Dict[str, List[int]]:
    with open(path) as f:
        return {k: list(map(int,v)) for k,v in json.load(f).items()}

# ---------- Factory to create dataset by split ----------
def build_cifar_lt_splits(root: str, name: str, imb_factor: float, seed: int,
                        split_cfg: SplitConfig, save_path: str):
    base_train, train_targets, C = load_cifar(root, name, train=True)
    test_set, _, _ = load_cifar(root, name, train=False)

    lt_idxs = make_long_tailed_indices(train_targets, C, imb_factor, seed)
    splits = partition_indices(lt_idxs, split_cfg, seed)
    save_splits(save_path, splits)

    ds = {
        "train": Subset(base_train, splits["train"]),
        "tuneV": Subset(base_train, splits["tuneV"]),
        "val_small": Subset(base_train, splits["val_small"]),
        "calib": Subset(base_train, splits["calib"]),
        "test": test_set,  # keep original
    }
    return ds, splits, C

# ---------- Count per class ----------
def count_per_class(dataset: Dataset, num_classes: int) -> np.ndarray:
    # dataset is Subset(CIFAR), dataset.dataset.targets contains all targets
    base = dataset.dataset
    targets = np.array(base.targets)
    idxs = np.array(dataset.indices)
    return np.bincount(targets[idxs], minlength=num_classes)
