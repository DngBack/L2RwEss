#!/usr/bin/env python3
"""
Example script để load và sử dụng paper mode dataset.
Usage: python example_paper_usage.py --metadata splits/cifar100_paper.json
"""
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

def load_paper_datasets(metadata_path, batch_size=32, num_workers=2):
    """
    Load datasets từ paper mode metadata.
    """
    # Đọc metadata
    with open(metadata_path, "r") as f:
        meta = json.load(f)
    
    print(f"Loading datasets from: {metadata_path}")
    print(f"Source: {meta['source']}")
    print(f"Seed: {meta['seed']}")
    print(f"Num classes: {meta['num_classes']}")
    
    # Build lại datasets tương ứng
    if "cifar100" in metadata_path:
        from src.data.datasets import build_cifar100lt_paper
        if meta['source'] == "torchvision":
            datasets_dict, _ = build_cifar100lt_paper(
                root="./data",
                seed=meta['seed'],
                source=meta['source'],
                imb_factor=meta.get('imb_factor', 100.0),
            )
        else:  # huggingface
            datasets_dict, _ = build_cifar100lt_paper(
                root="./data",
                seed=meta['seed'],
                source=meta['source'],
            )
    elif "cifar10" in metadata_path:
        from src.data.datasets import build_cifar10lt_paper
        datasets_dict, _ = build_cifar10lt_paper(
            root="./data",
            seed=meta['seed'],
            imb_factor=meta.get('imb_factor', 100.0),
        )
    else:
        raise ValueError(f"Không nhận diện được dataset từ: {metadata_path}")
    
    # Tạo DataLoaders
    train_loader = DataLoader(
        datasets_dict['train'], 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    val_loader = DataLoader(
        datasets_dict['val'], 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    test_loader = DataLoader(
        datasets_dict['test'], 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
    }, meta

def weighted_accuracy(outputs, targets, weights):
    """
    Tính weighted accuracy với trọng số lớp.
    """
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == targets).float()
    weighted_correct = correct * weights
    return weighted_correct.sum() / weights.sum()

def demo_training_loop(loaders, meta):
    """
    Demo một vài iteration training với re-weighting.
    """
    print("\n=== Demo Training Loop ===")
    
    # Lấy weights cho val
    weights_val = torch.tensor(meta['weights_val'], dtype=torch.float32)
    
    # Demo với vài batch đầu tiên
    train_loader = loaders['train']
    val_loader = loaders['val']
    
    print("Training batches:")
    for i, (images, labels) in enumerate(train_loader):
        if i >= 3:  # Chỉ demo 3 batch
            break
        print(f"  Batch {i+1}: {images.shape}, labels range: {labels.min()}-{labels.max()}")
    
    print("\nValidation with re-weighting:")
    for i, (images, labels) in enumerate(val_loader):
        if i >= 2:  # Chỉ demo 2 batch
            break
        
        # Lấy weights tương ứng với labels trong batch này
        batch_weights = weights_val[labels]
        
        # Fake outputs (thay bằng model thật)
        fake_outputs = torch.randn(len(labels), meta['num_classes'])
        
        # Tính weighted accuracy
        acc = weighted_accuracy(fake_outputs, labels, batch_weights)
        
        print(f"  Batch {i+1}: {images.shape}, weighted acc: {acc:.4f}")

def analyze_dataset_splits(meta):
    """
    Phân tích chi tiết các splits.
    """
    print("\n=== Dataset Analysis ===")
    
    counts_train = np.array(meta['counts_train'])
    class_to_group = np.array(meta['class_to_group'])
    
    # Head/Tail analysis
    head_classes = np.where(class_to_group == 0)[0]
    tail_classes = np.where(class_to_group == 1)[0]
    
    print(f"Head classes ({len(head_classes)}): samples range {counts_train[head_classes].min()}-{counts_train[head_classes].max()}")
    print(f"Tail classes ({len(tail_classes)}): samples range {counts_train[tail_classes].min()}-{counts_train[tail_classes].max()}")
    
    # Val/Test weights analysis
    weights_val = np.array(meta['weights_val'])
    weights_test = np.array(meta['weights_test'])
    
    print("\nReweighting factors:")
    print(f"  Val weights: min={weights_val.min():.3f}, max={weights_val.max():.3f}, mean={weights_val.mean():.3f}")
    print(f"  Test weights: min={weights_test.min():.3f}, max={weights_test.max():.3f}, mean={weights_test.mean():.3f}")

def main():
    parser = argparse.ArgumentParser(description="Demo paper mode dataset usage")
    parser.add_argument("--metadata", required=True,
                        help="Path to metadata JSON file")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for DataLoaders")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="Number of workers for DataLoaders")
    
    args = parser.parse_args()
    
    # Load datasets
    loaders, meta = load_paper_datasets(
        args.metadata, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers
    )
    
    # Analyze dataset
    analyze_dataset_splits(meta)
    
    # Demo training loop
    demo_training_loop(loaders, meta)
    
    print("\nDemo completed!")

if __name__ == "__main__":
    main()