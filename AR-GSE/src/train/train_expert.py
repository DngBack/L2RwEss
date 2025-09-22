# src/train/train_expert.py
from __future__ import annotations
import os
import json
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from src.models.experts import ExpertWrapper
from src.metrics.calibration import fit_temperature_weighted
from src.data.datasets import build_cifar100lt_paper  # M1 đã tạo
torch.backends.cudnn.benchmark = True

class WithWeights(Dataset):
    """Bọc dataset để yield (x,y,w) theo re-weight paper."""
    def __init__(self, base: Dataset, weights: np.ndarray):
        self.base = base
        self.w = torch.tensor(weights, dtype=torch.float)
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        x, y = self.base[i]
        return x, y, self.w[i]

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def gather_train_labels(ds: Dataset, num_classes: int) -> tuple[np.ndarray, np.ndarray]:
    ys = []
    for i in range(len(ds)):
        _, y = ds[i]; ys.append(y)
    ys = np.array(ys, dtype=np.int64)
    counts = np.bincount(ys, minlength=num_classes).astype(np.float32)
    priors = counts / counts.sum()
    return counts, priors

def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    loss_sum, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=True):
            logits = model(x)
            loss = model.loss(logits, y)
        scaler.scale(loss).step(optimizer)
        scaler.update()
        loss_sum += float(loss.item()) * x.size(0); n += x.size(0)
    return loss_sum / max(1, n)

@torch.no_grad()
def eval_top1_weighted(model, loader, device):
    model.eval()
    correct, total = 0.0, 0.0
    for batch in loader:
        x, y, w = batch
        x, y, w = x.to(device), y.to(device), w.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).float().mul(w).sum().item()
        total   += w.sum().item()
    return correct / max(1.0, total)

@torch.no_grad()
def export_posteriors(model, loader, device, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    probs_list, labels_list = [], []
    for batch in tqdm(loader, desc=f"Export {os.path.basename(save_path)}"):
        if len(batch) == 3:
            x, y, *_ = batch
        else:
            x, y = batch
        x = x.to(device)
        logits = model(x)
        probs = model.predict_proba(logits).cpu()
        probs_list.append(probs); labels_list.append(y)
    probs = torch.cat(probs_list, 0).numpy()
    labels= torch.cat(labels_list, 0).numpy()
    np.savez_compressed(save_path, probs=probs, labels=labels)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", choices=["ce","balsoftmax","logitadjust","decoupled"], default="ce")
    ap.add_argument("--tau_la", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=256)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="./artifacts/experts/ce")
    ap.add_argument("--source", choices=["huggingface","torchvision"], default="huggingface")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ===== DATA (paper mode M1) =====
    ds, meta = build_cifar100lt_paper(source=args.source, seed=args.seed)
    train_ds, val_ds, test_ds = ds["train"], ds["val"], ds["test"]
    w_val  = np.asarray(meta["weights_val"],  dtype=np.float32)
    w_test = np.asarray(meta["weights_test"], dtype=np.float32)

    # DataLoader
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(WithWeights(val_ds,  w_val),  batch_size=256, shuffle=False, num_workers=4)
    test_loader  = DataLoader(WithWeights(test_ds, w_test), batch_size=256, shuffle=False, num_workers=4)

    num_classes = 100
    counts, priors = gather_train_labels(train_ds, num_classes)

    # ===== MODEL & OPT =====
    model = ExpertWrapper(args.loss, num_classes, counts=counts, priors=priors, tau_la=args.tau_la).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    milestones = [96, 192, 224]
    sched = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    scaler = torch.cuda.amp.GradScaler()

    best_metric = -1.0
    log_path = os.path.join(args.out, "log.txt")
    with open(log_path, "w") as lg:
        print(json.dumps({"cfg": vars(args)}, indent=2), file=lg)

    # ===== TRAIN =====
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, scaler, device)
        sched.step()

        # weighted top-1 để theo dõi tiến trình (paper re-weight)
        val_acc = eval_top1_weighted(model, val_loader, device)
        test_acc= eval_top1_weighted(model, test_loader, device)
        msg = f"[{epoch:03d}] loss={tr_loss:.4f}  val@w-top1={val_acc:.4f}  test@w-top1={test_acc:.4f}  lr={sched.get_last_lr()[0]:.4g}"
        print(msg)
        with open(log_path, "a") as lg: print(msg, file=lg)

        if val_acc > best_metric:
            best_metric = val_acc
            torch.save(
                {"state_dict": model.state_dict(),
                "counts": counts.tolist(),
                "priors": priors.tolist(),
                "epoch": epoch},
                os.path.join(args.out, "best.pt")
            )

    # ===== CALIBRATION (weighted NLL) =====
    print("Fitting temperature (weighted NLL) on val ...")
    T_star = fit_temperature_weighted(model, val_loader, device=device)
    print(f"  Learned T = {T_star:.4f}")
    torch.save(
        {"state_dict": model.state_dict(),
        "counts": counts.tolist(),
        "priors": priors.tolist(),
        "T": T_star},
        os.path.join(args.out, "calibrated.pt")
    )

    # ===== EXPORT POSTERIORS (calibrated) =====
    export_posteriors(model, DataLoader(train_ds, batch_size=256, shuffle=False, num_workers=4),
                    device, os.path.join(args.out, "post_train.npz"))
    export_posteriors(model, val_loader,  device, os.path.join(args.out, "post_val.npz"))
    export_posteriors(model, test_loader, device, os.path.join(args.out, "post_test.npz"))

if __name__ == "__main__":
    main()
