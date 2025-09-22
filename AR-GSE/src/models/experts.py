# src/models/experts.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .backbones.resnet_cifar import ResNetCIFAR

class ExpertWrapper(nn.Module):
    """
    Expert = ResNet-32 (CIFAR) + 1 trong 4 loss long-tail.
    Sau train: fit temperature T trên validation (weighted) -> posterior đã calibrate.
    """
    def __init__(self, loss_type: str, num_classes: int,
                 counts: np.ndarray | None = None,
                 priors: np.ndarray | None = None,
                 tau_la: float = 1.0):
        super().__init__()
        assert loss_type in {"ce", "balsoftmax", "logitadjust", "decoupled"}
        self.backbone = ResNetCIFAR(depth=32, num_classes=num_classes)
        self.loss_type = loss_type

        # buffers cho counts/prior
        if counts is not None:
            counts = np.asarray(counts, dtype=np.float32)
            counts[counts < 1.0] = 1.0  # tránh log(0)
            self.register_buffer("counts", torch.tensor(counts, dtype=torch.float))
        else:
            self.register_buffer("counts", None)

        if priors is not None:
            priors = np.asarray(priors, dtype=np.float32)
            priors = priors / (priors.sum() + 1e-12)
            self.register_buffer("priors", torch.tensor(priors, dtype=torch.float))
        else:
            self.register_buffer("priors", None)

        self.tau_la = float(tau_la)
        # temperature sau calibration (buffer để không optimize trong train)
        self.register_buffer("temperature", torch.tensor(1.0, dtype=torch.float))

    def forward(self, x):
        return self.backbone(x)

    def loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "ce":
            return F.cross_entropy(logits, y)

        elif self.loss_type == "balsoftmax":
            assert self.counts is not None, "balsoftmax cần counts theo lớp"
            # Balanced Softmax: CE(z + log n_y)
            logits_adj = logits + torch.log(self.counts.to(logits.device))
            return F.cross_entropy(logits_adj, y)

        elif self.loss_type == "logitadjust":
            assert self.priors is not None, "logitadjust cần priors theo lớp"
            logits_adj = logits + self.tau_la * torch.log(self.priors.to(logits.device) + 1e-12)
            return F.cross_entropy(logits_adj, y)

        elif self.loss_type == "decoupled":
            assert self.priors is not None, "decoupled cần priors theo lớp"
            w = (1.0 / (self.priors.to(logits.device) + 1e-12))[y]
            w = w / (w.mean() + 1e-12)  # chuẩn hoá để ổn định LR
            return (F.cross_entropy(logits, y, reduction="none") * w).mean()

        else:
            raise ValueError(self.loss_type)

    @torch.no_grad()
    def predict_proba(self, logits: torch.Tensor) -> torch.Tensor:
        T = float(self.temperature.item())
        return F.softmax(logits / T, dim=-1)
