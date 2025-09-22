# src/data/splits.py
import argparse
import json
from .dataset import SplitConfig, build_cifar_lt_splits, count_per_class
from .groups import class_to_group_by_frequency, summarize_groups

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default="./data")
    p.add_argument("--name", type=str, choices=["cifar10","cifar100"], default="cifar10")
    p.add_argument("--imb_factor", type=float, default=100.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save", type=str, default="./splits/cifar10_lt_if100_seed42.json")
    p.add_argument("--head_ratio", type=float, default=0.5)
    args = p.parse_args()

    cfg = SplitConfig(train=0.80, tuneV=0.08, val_small=0.06, calib=0.06)
    ds, splits, C = build_cifar_lt_splits(args.root, args.name, args.imb_factor, args.seed, cfg, args.save)

    # thống kê per-class trên train
    counts = count_per_class(ds["train"], C)
    print(f"[INFO] Train size: {len(ds['train'])}, per-class counts (first 10): {counts[:10].tolist()}")

    # group mapping
    g = class_to_group_by_frequency(counts, K=2, head_ratio=args.head_ratio)
    summary = summarize_groups(counts, g)
    print("[INFO] Group summary:", json.dumps(summary, indent=2))

    # sanity: tuneV/val_small/calib
    for k in ["tuneV","val_small","calib"]:
        c = count_per_class(ds[k], C)
        print(f"[INFO] {k}: size={len(ds[k])}, mean/class={float(c.mean()):.1f}")

    print(f"[OK] Splits saved to {args.save}")

if __name__ == "__main__":
    main()
