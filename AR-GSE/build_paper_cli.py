#!/usr/bin/env python3
"""
CLI script để xây dựng và lưu datasets paper mode.
Usage: python build_paper_cli.py --dataset cifar100 --root ./data --output splits/cifar100_paper.json
"""
import argparse
import json
import os
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Build paper mode dataset splits")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], required=True,
                        help="Dataset name")
    parser.add_argument("--root", default="./data", 
                        help="Data root directory")
    parser.add_argument("--output", required=True,
                        help="Output JSON file path for metadata")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--imb-factor", type=float, default=100.0,
                        help="Imbalance factor (only for torchvision source)")
    parser.add_argument("--source", choices=["torchvision", "huggingface"], 
                        default="torchvision", help="Data source")
    parser.add_argument("--hf-config", default="",
                        help="HuggingFace dataset config (if applicable)")
    
    args = parser.parse_args()
    
    # Import phù hợp
    if args.dataset == "cifar100":
        from src.data.datasets import build_cifar100lt_paper
        
        print("Building CIFAR-100-LT paper mode...")
        print(f"  Source: {args.source}")
        print(f"  Root: {args.root}")
        print(f"  Seed: {args.seed}")
        if args.source == "torchvision":
            print(f"  Imbalance factor: {args.imb_factor}")
        
        datasets_dict, meta = build_cifar100lt_paper(
            root=args.root,
            seed=args.seed,
            source=args.source,
            hf_config=args.hf_config,
            imb_factor=args.imb_factor,
        )
        
    elif args.dataset == "cifar10":
        from src.data.datasets import build_cifar10lt_paper
        
        print("Building CIFAR-10-LT paper mode...")
        print(f"  Root: {args.root}")
        print(f"  Seed: {args.seed}")
        print(f"  Imbalance factor: {args.imb_factor}")
        
        datasets_dict, meta = build_cifar10lt_paper(
            root=args.root,
            seed=args.seed,
            imb_factor=args.imb_factor,
        )
    
    # Tạo thư mục output nếu cần
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Lưu metadata
    with open(args.output, "w") as f:
        json.dump(meta, f, indent=2)
    
    # In thống kê
    print("\n=== Dataset Statistics ===")
    print(f"Train size: {len(datasets_dict['train'])}")
    print(f"Val size: {len(datasets_dict['val'])}")
    print(f"Test size: {len(datasets_dict['test'])}")
    print(f"Number of classes: {meta['num_classes']}")
    
    # In phân phối lớp train
    counts_train = np.array(meta['counts_train'])
    print("\nTrain class distribution:")
    print(f"  Min: {counts_train.min()}")
    print(f"  Max: {counts_train.max()}")
    print(f"  Mean: {counts_train.mean():.1f}")
    print(f"  Std: {counts_train.std():.1f}")
    
    # In group mapping
    class_to_group = np.array(meta['class_to_group'])
    n_head = (class_to_group == 0).sum()
    n_tail = (class_to_group == 1).sum()
    print("\nClass grouping (threshold=20):")
    print(f"  Head classes: {n_head}")
    print(f"  Tail classes: {n_tail}")
    
    # In sample counts per group
    head_samples = counts_train[class_to_group == 0].sum()
    tail_samples = counts_train[class_to_group == 1].sum()
    print(f"  Head samples: {head_samples}")
    print(f"  Tail samples: {tail_samples}")
    
    print(f"\nMetadata saved to: {args.output}")
    print("Done!")

if __name__ == "__main__":
    main()