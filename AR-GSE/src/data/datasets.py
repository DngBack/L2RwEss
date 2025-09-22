# src/data/datasets.py
from __future__ import annotations
import numpy as np
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from PIL import Image

def cifar100_transforms(train: bool):
    """CIFAR-100 transforms với normalize chuẩn."""
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

class HF2Torch(Dataset):
    """Dùng khi bạn chọn nguồn HuggingFace cho CIFAR-100-LT."""
    def __init__(self, hf_split, transform):
        self.ds = hf_split
        self.t = transform
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, i):
        ex = self.ds[i]
        img = ex.get("img") or ex.get("image")
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        y = ex.get("fine_label") or ex.get("label")
        return self.t(img), int(y)

def build_cifar100lt_paper(
    root: str = "./data",
    seed: int = 42,
    source: str = "torchvision",   # "huggingface" nếu bạn dùng HF
    hf_config: str = "",           # tuỳ repo HF (nếu có nhiều IF)
    imb_factor: float = 100.0,     # Chỉ dùng khi source="torchvision"
):
    """
    Paper mode: 
      - TRAIN = train LT (CIFAR-100-LT)
      - VAL/TEST = 20% / 80% từ ORIGINAL TEST (không LT), kèm trọng số lớp để re-weight.
    Trả về: dict datasets, dict meta (indices, weights, counts, group map)
    """
    rng = np.random.RandomState(seed)

    # 1) Load TRAIN-LT
    if source == "huggingface":
        try:
            from datasets import load_dataset
            ds_hf = load_dataset("tomas-gajarsky/cifar100-lt")   # chọn config IF bạn muốn nếu repo có
            train_lt = HF2Torch(ds_hf["train"], cifar100_transforms(train=True))
            # ORIGINAL TEST chuẩn (10k)
            test_orig = HF2Torch(ds_hf["test"], cifar100_transforms(train=False))
        except ImportError:
            raise ImportError("Cần cài đặt: pip install datasets")
        except Exception as e:
            raise RuntimeError(f"Không thể load HuggingFace dataset: {e}")
    
    else:  # source == "torchvision"
        # Tự tạo CIFAR-100-LT theo exponential distribution
        from .dataset import load_cifar, make_long_tailed_indices
        
        # Load full CIFAR-100
        cifar100_train, targets, num_classes = load_cifar(root, "cifar100", train=True)
        cifar100_test, _, _ = load_cifar(root, "cifar100", train=False)
        
        # Tạo long-tailed indices
        lt_indices = make_long_tailed_indices(targets, num_classes, imb_factor, seed)
        
        # Tạo train LT dataset
        train_lt = Subset(cifar100_train, lt_indices)
        test_orig = cifar100_test

    # 2) Tính phân phối lớp train-LT
    def labels_from_torch(ds):
        ys = []
        for i in range(len(ds)):
            _, y = ds[i]
            ys.append(y)
        return np.array(ys, dtype=np.int64)

    train_labels = labels_from_torch(train_lt)
    C = int(train_labels.max() + 1)
    counts_train = np.bincount(train_labels, minlength=C)
    p_train = counts_train / counts_train.sum()

    # 3) Chia VAL/TEST = 20%/80% từ ORIGINAL TEST
    test_labels = labels_from_torch(test_orig)
    N = len(test_labels)
    idx = np.arange(N)
    rng.shuffle(idx)
    n_val = int(0.2 * N)
    val_idx, test_idx = idx[:n_val], idx[n_val:]

    val_set = Subset(test_orig, val_idx.tolist())
    test_set = Subset(test_orig, test_idx.tolist())

    # 4) Trọng số lớp cho re-weight
    counts_test0 = np.bincount(test_labels, minlength=C)
    p_test0 = counts_test0 / counts_test0.sum()
    w_class = p_train / np.clip(p_test0, 1e-12, None)   # [C]

    w_val = w_class[test_labels[val_idx]]
    w_test = w_class[test_labels[test_idx]]

    # 5) Group map theo ngưỡng 20 mẫu (paper)
    from .groups import class_to_group_paper
    class_to_group = class_to_group_paper(counts_train, thr=20)   # 0=head, 1=tail

    # 6) Gói về
    datasets_out = {"train": train_lt, "val": val_set, "test": test_set}
    meta = {
        "val_idx": val_idx.tolist(),
        "test_idx": test_idx.tolist(),
        "weights_val": w_val.tolist(),
        "weights_test": w_test.tolist(),
        "counts_train": counts_train.tolist(),
        "class_to_group": class_to_group.tolist(),
        "num_classes": C,
        "source": source,
        "seed": seed,
    }
    return datasets_out, meta

def build_cifar10lt_paper(
    root: str = "./data",
    seed: int = 42,
    imb_factor: float = 100.0,
):
    """
    Paper mode cho CIFAR-10-LT (tương tự CIFAR-100-LT).
    """
    from .dataset import load_cifar, make_long_tailed_indices
    
    rng = np.random.RandomState(seed)
    
    # Load full CIFAR-10
    cifar10_train, targets, num_classes = load_cifar(root, "cifar10", train=True)
    cifar10_test, _, _ = load_cifar(root, "cifar10", train=False)
    
    # Tạo long-tailed indices
    lt_indices = make_long_tailed_indices(targets, num_classes, imb_factor, seed)
    
    # Tạo train LT dataset
    train_lt = Subset(cifar10_train, lt_indices)
    test_orig = cifar10_test
    
    # Tính phân phối lớp train-LT
    def labels_from_torch(ds):
        ys = []
        for i in range(len(ds)):
            _, y = ds[i]
            ys.append(y)
        return np.array(ys, dtype=np.int64)

    train_labels = labels_from_torch(train_lt)
    C = int(train_labels.max() + 1)
    counts_train = np.bincount(train_labels, minlength=C)
    p_train = counts_train / counts_train.sum()

    # Chia VAL/TEST = 20%/80% từ ORIGINAL TEST
    test_labels = labels_from_torch(test_orig)
    N = len(test_labels)
    idx = np.arange(N)
    rng.shuffle(idx)
    n_val = int(0.2 * N)
    val_idx, test_idx = idx[:n_val], idx[n_val:]

    val_set = Subset(test_orig, val_idx.tolist())
    test_set = Subset(test_orig, test_idx.tolist())

    # Trọng số lớp cho re-weight
    counts_test0 = np.bincount(test_labels, minlength=C)
    p_test0 = counts_test0 / counts_test0.sum()
    w_class = p_train / np.clip(p_test0, 1e-12, None)   # [C]

    w_val = w_class[test_labels[val_idx]]
    w_test = w_class[test_labels[test_idx]]

    # Group map theo ngưỡng 20 mẫu (paper)
    from .groups import class_to_group_paper
    class_to_group = class_to_group_paper(counts_train, thr=20)   # 0=head, 1=tail

    # Gói về
    datasets_out = {"train": train_lt, "val": val_set, "test": test_set}
    meta = {
        "val_idx": val_idx.tolist(),
        "test_idx": test_idx.tolist(),
        "weights_val": w_val.tolist(),
        "weights_test": w_test.tolist(),
        "counts_train": counts_train.tolist(),
        "class_to_group": class_to_group.tolist(),
        "num_classes": C,
        "source": "torchvision",
        "seed": seed,
        "imb_factor": imb_factor,
    }
    return datasets_out, meta