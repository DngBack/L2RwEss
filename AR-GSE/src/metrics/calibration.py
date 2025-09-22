# src/metrics/calibration.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class _TempParam(nn.Module):
    def __init__(self, init_T: float = 1.0, eps: float = 1e-6):
        super().__init__()
        # u sao cho softplus(u) ≈ init_T
        u0 = torch.log(torch.exp(torch.tensor(init_T)) - 1.0)
        self.u = nn.Parameter(u0)
        self.eps = float(eps)

    def T(self) -> torch.Tensor:
        return F.softplus(self.u) + self.eps

@torch.no_grad()
def _gather_logits_labels_weights(model, loader, device):
    logits_list, labels_list, w_list = [], [], []
    model.eval()
    for batch in loader:
        if len(batch) == 3:
            x, y, w = batch
        else:
            x, y = batch
            w = torch.ones_like(y, dtype=torch.float)
        x, y, w = x.to(device), y.to(device), w.to(device)
        logits = model(x)
        logits_list.append(logits)
        labels_list.append(y)
        w_list.append(w)
    return torch.cat(logits_list, 0), torch.cat(labels_list, 0), torch.cat(w_list, 0)

def fit_temperature_weighted(model: nn.Module, loader, device="cuda") -> float:
    """
    Tìm T>0 tối ưu NLL có trọng số trên validation (re-weight theo label).
    """
    logits, labels, w = _gather_logits_labels_weights(model, loader, device)
    w = w / (w.mean() + 1e-12)

    cal = _TempParam(init_T=1.0).to(device)
    opt = torch.optim.LBFGS(cal.parameters(), lr=0.5, max_iter=50)

    def _closure():
        opt.zero_grad()
        T = cal.T()
        loss = (F.cross_entropy(logits / T, labels, reduction="none") * w).mean()
        loss.backward()
        return loss

    opt.step(_closure)
    T_star = float(cal.T().item())
    # ghi ngược vào model
    model.temperature[...] = T_star
    return T_star
