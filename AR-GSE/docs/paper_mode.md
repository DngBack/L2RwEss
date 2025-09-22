# Paper Mode Dataset Implementation

Đây là implementation **paper mode** cho dataset CIFAR-10/100-LT theo yêu cầu nghiên cứu.

## ✨ Tính năng chính

### 🎯 Paper Mode Dataset Structure
- **TRAIN**: Long-tailed CIFAR-10/100 (exponential imbalance)
- **VAL**: 20% từ original test set (balanced) 
- **TEST**: 80% từ original test set (balanced)
- **Re-weighting**: Trọng số lớp để cân bằng train vs test distribution

### 📊 Class Grouping  
- **Head classes**: > 20 samples (nhiều dữ liệu)
- **Tail classes**: ≤ 20 samples (ít dữ liệu)
- Ngưỡng threshold=20 theo paper specification

### 🔧 Data Sources
- **Torchvision**: Tự tạo long-tailed từ CIFAR chuẩn
- **HuggingFace**: Load CIFAR-100-LT có sẵn (dự kiến)

## 🚀 Quick Start

### 1. Build Dataset Splits

```bash
# CIFAR-100-LT paper mode
python build_paper_cli.py --dataset cifar100 --root ./data --output splits/cifar100_paper.json

# CIFAR-10-LT paper mode  
python build_paper_cli.py --dataset cifar10 --root ./data --output splits/cifar10_paper.json

# Với HuggingFace source (CIFAR-100 only)
python build_paper_cli.py --dataset cifar100 --source huggingface --output splits/cifar100_hf_paper.json
```

### 2. Load và Sử dụng

```python
from src.data.datasets import build_cifar100lt_paper

# Build datasets
datasets_dict, meta = build_cifar100lt_paper(
    root="./data",
    seed=42,
    source="torchvision",  # or "huggingface"
    imb_factor=100.0
)

train_set = datasets_dict['train']  # Long-tailed
val_set = datasets_dict['val']      # 20% balanced test  
test_set = datasets_dict['test']    # 80% balanced test

# Lấy re-weighting factors
weights_val = meta['weights_val']   # [N_val] 
weights_test = meta['weights_test'] # [N_test]
class_to_group = meta['class_to_group']  # [C] 0=head, 1=tail
```

### 3. Demo Script

```bash
python example_paper_usage.py --metadata splits/cifar100_paper.json
```

## 📁 Files Structure

```
src/data/
├── datasets.py          # Paper mode functions
├── groups.py           # Class grouping utilities  
├── dataset.py          # Base dataset functions
└── splits.py           # Original splits script

scripts/
├── build_paper_cli.py   # CLI for building paper splits
└── example_paper_usage.py  # Demo và usage example

splits/
├── cifar100_paper.json  # CIFAR-100 paper metadata
└── cifar10_paper.json   # CIFAR-10 paper metadata
```

## 🔍 Metadata Format

```json
{
  "val_idx": [6252, 4684, ...],      // Indices cho validation split
  "test_idx": [1234, 5678, ...],     // Indices cho test split  
  "weights_val": [1.2, 0.8, ...],    // Re-weight factors cho val
  "weights_test": [1.1, 0.9, ...],   // Re-weight factors cho test
  "counts_train": [500, 450, ...],   // Số samples per class trong train
  "class_to_group": [0, 0, 1, ...],  // 0=head, 1=tail
  "num_classes": 100,
  "source": "torchvision",
  "seed": 42
}
```

## 📈 Statistics Examples

### CIFAR-100-LT (imb_factor=100)
```
Train size: 10899 (long-tailed)
Val size: 2000 (balanced)  
Test size: 8000 (balanced)

Class distribution:
  Head classes: 69 (samples: 21-500)
  Tail classes: 31 (samples: 5-20)
  
Re-weighting range: 0.046 - 4.588
```

### CIFAR-10-LT (imb_factor=100)  
```
Train size: 12408 (long-tailed)
Val size: 2000 (balanced)
Test size: 8000 (balanced)

Class distribution:
  Head classes: 10 (samples: 50-5000)  
  Tail classes: 0 (threshold=20 quá thấp)
```

## 🎛️ Configuration Options

### CLI Arguments
- `--dataset`: "cifar10" hoặc "cifar100"
- `--source`: "torchvision" hoặc "huggingface"  
- `--imb-factor`: Exponential imbalance factor (default: 100.0)
- `--seed`: Random seed (default: 42)
- `--root`: Data directory (default: "./data")
- `--output`: Output JSON path

### Function Parameters
- `source`: Data source selection
- `hf_config`: HuggingFace dataset config (if needed)
- `imb_factor`: Imbalance factor cho torchvision source
- `seed`: Reproducibility seed

## ⚖️ Re-weighting Formula

```python
# Train distribution
p_train = counts_train / counts_train.sum()

# Test distribution (balanced)  
p_test = counts_test / counts_test.sum()

# Re-weight factors
w_class = p_train / p_test  # [C]

# Sample weights
w_sample = w_class[labels]  # [N]
```

Điều này đảm bảo model được đánh giá fair trên test set balanced bằng cách re-weight theo train distribution.

## 🧪 Testing

Tất cả functions đã được test với comprehensive test suite trong `tests/data/test_dataset.py`.

## 📝 Notes

- Paper mode khác với original splits ở việc VAL/TEST từ original test set thay vì train set
- Re-weighting giúp đánh giá model performance trên balanced data nhưng theo train distribution  
- Class grouping theo threshold=20 samples phù hợp cho research comparison
- HuggingFace integration cho reproducibility với datasets có sẵn