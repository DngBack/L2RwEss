M0. Khởi tạo repo & môi trường
Cấu trúc thư mục


argse/
  configs/
    dataset/cifar10_lt.yaml
    exp/experts.yaml
    exp/argse_balanced.yaml
    exp/argse_worst.yaml
    exp/eval.yaml
  src/
    data/{datasets.py, splits.py, groups.py}
    models/
      backbones/{resnet_cifar.py, resnet_imagenet.py}
      experts.py
      gating.py
      selective.py          # margin, selective decision (hard) - chỉ dùng eval
      surrogate_losses.py   # loss chọn lọc (mask bởi s_tau)
      primal_dual.py        # cập nhật primal-dual (α, μ, φ, λ)
      argse.py              # lớp AR-GSE model wrapper
      conformal.py          # (tuỳ chọn)
    metrics/{selective_metrics.py, rc_curve.py, calibration.py, bootstrap.py}
    train/
      train_expert.py       # Stage A (huấn luyện experts + temp scaling)
      train_argse.py        # Stage B’ (AR-GSE end-to-end)
      eval_test.py
      eval_baselines.py     # U-PI, SB-PI, Chow
    utils/{seed.py, logging.py, checkpoint.py, hydra_utils.py, viz.py}
  scripts/
    run_cifar10lt_argse.sh
  README.md
requirements.txt


torch torchvision torchaudio
numpy pandas scikit-learn
hydra-core omegaconf
matplotlib tqdm rich
einops
DoD: cài đặt xong, import được src/... không lỗi.
M1. Dữ liệu, chia tách, nhóm
src/data/datasets.py

CIFAR-10/100-LT (downsample theo class để đạt imbalance factor).
Augment train: RandAug (n=2,m=10), Mixup 0.2, CutMix 0.2 (chỉ dùng cho experts).
Eval: normalize/center-crop (nếu cần).
src/data/splits.py

Sinh splits cố định seed:
train | tuneV | val_small | calib | test.

(AR-GSE dùng tuneV để huấn luyện; val_small để theo dõi/EG worst-group; calib cho conformal.)
Lưu chỉ số index .json để tái lập.
src/data/groups.py

Map class→group. Mặc định K=2 (head/tail) theo tần suất lớp (hoặc theo quantile).
API: get_class_to_group(counts, K=2, head_ratio=0.5) -> LongTensor[C].
DoD: in kích thước từng split, phân bố nhóm đúng mong đợi.
M2. Experts (Stage A)
src/models/backbones/resnet_cifar.py

ResNet-32 phiên bản CIFAR (3×3 conv, {16,32,64}, GAP+FC).
src/models/experts.py

ExpertWrapper(backbone, loss_type, priors=None) với loss_type∈{ce, balsoftmax, logitadjust, decoupled}.
Xuất logits trong train; head có thể khác nhau theo loss cho đa dạng bias/variance.
src/metrics/calibration.py

Temperature scaling mỗi expert trên val_small.
Lưu T_e và apply khi xuất posterior.
src/train/train_expert.py

Train từng expert; sau train, fit temperature.
Xuất checkpoint + (tuỳ chọn) logits/posteriors cho tuneV, val_small, calib, test (để tăng tốc AR-GSE).
Gợi ý lưu logits fp16 để nhẹ.
Command mẫu


python -m src.train.train_expert +exp=experts dataset=cifar10_lt seed=42
DoD: có ≥3–4 experts đã calibrate; có file posterior/logits cho tuneV & test.
M3. Đặc trưng gating (scalable)
src/models/gating.py

GatingNet(in_dim, hidden=[128,64], dropout=0.1, num_experts=E) → softmax weights wϕ(x)w_\phi(x)
wϕ
(x).
Builder đặc trưng (không phụ thuộc số lớp C):
entropy từng expert;
top-k mass (k=5) + residual mass;
pairwise (KL/cosine) hoặc random projection logits -> 64–128 chiều;
(tuỳ chọn) độ tập trung top-k (max−mean topk), dispersion.
API: build_gating_features({p_e or logits_e}) -> feat.
DoD: forward ok, tensor shapes nhất quán; không phụ thuộc C.
M4. Surrogate & margin
src/models/surrogate_losses.py

selective_cls_loss(eta_mix, y, s_tau, beta, class_to_group, kind="ce")
CE mask bởi s_tau, nhân trọng số nhóm beta_k/α_k (sẽ do wrapper truyền).
Hỗ trợ kind="one_minus_p" để ablation.
src/models/argse.py

Hàm tính margin & s_tau:

def selective_margin(eta_mix, alpha, mu, c, class_to_group):
    inv_a = 1.0 / alpha                      # [K]
    g = class_to_group                       # [C]
    # max_y eta/alpha[g(y)]
    score = eta_mix / inv_a[g]
    max_score, _ = score.max(dim=1)
    # threshold
    coeff = inv_a[g] - mu[g]
    thr = (eta_mix * coeff).sum(dim=1) - c
    margin = max_score - thr
    return margindef acceptance_prob(margin, tau):
    return torch.sigmoid(tau * margin)
Kiểm tra: tăng c ⇒ coverage giảm; tăng tau ⇒ s_tau sắc hơn.
DoD: unit test toy pass; không NaN, clamp hợp lý.
M5. Primal–Dual Trainer (AR-GSE core)
src/models/primal_dual.py

Trạng thái học: ϕ\phi
ϕ (gating), α∈R>0K\alpha\in\mathbb{R}^K_{>0}
α∈R>0
K
, μ∈RK\mu\in\mathbb{R}^K
μ∈RK
, λ∈R≥0K\lambda\in\mathbb{R}^K_{\ge0}
λ∈R≥0
K
, τ\tau
τ.
Cập nhật primal (SGD/Adam):
Clamp αk≥ε\alpha_k \ge \varepsilon
αk
≥ε (1e-3), EMA smoothing α\alpha
α theo epoch để ổn định.
Normalize μ\mu
μ mỗi epoch (trừ mean) để tránh drift.
Cập nhật dual:
λk←max⁡(0,λk+ρ(αk−K⋅E^[sτ1{y∈Gk}]))\lambda_k \leftarrow \max(0, \lambda_k + \rho(\alpha_k - K\cdot \widehat{\mathbb{E}}[s_\tau \mathbf{1}\{y\in G_k\}]))
λk
←max(0,λk
+ρ(αk
−K⋅E

[sτ
1{y∈Gk
}])).
Warm-up τ\tau
τ: tăng tuyến tính 2→10 trong ~30 epoch.
Regularizer: entropy H(wϕ)H(w_\phi)
H(wϕ
) (λ_ent = 1e-3); (tuỳ chọn) ℓ1\ell_1
ℓ1
 lên wϕw_\phi
wϕ
 để khuyến khích hard routing.
Pseudo-training step


def primal_dual_step(batch, state):
    x, y = batch
    p_e = get_expert_posteriors_or_logits(x)     # no grad
    feats = build_gating_features(p_e)
    w = gating(feats)                             # [B, E]
    eta_mix = mixture_posterior(w, p_e)          # [B, C]

    margin = selective_margin(eta_mix, alpha, mu, c, class_to_group)
    s = acceptance_prob(margin, tau)             # [B]

    # selective loss
    loss_cls = selective_cls_loss(eta_mix, y, s, beta, class_to_group, kind="ce")
    loss_rej = c * (1 - s).mean()

    # acceptance constraints per-group
    acc_k = estimate_group_acceptance(s, y, class_to_group, K)   # \hat E[s * 1{y∈G_k}]
    cons = alpha - K * acc_k                                    # [K]

    L = loss_cls + loss_rej + (lambda * cons).sum() + lambda_ent * entropy_w(w) + l1_w(w)

    # primal updates
    opt_phi.zero_grad(); opt_alpha.zero_grad(); opt_mu.zero_grad()
    L.backward()
    grad_clip_(...)      # optional
    opt_phi.step(); opt_alpha.step(); opt_mu.step()

    # post-process
    alpha.clamp_(min=1e-3)
    mu -= mu.mean()

    # dual updates
    with torch.no_grad():
        lambda = (lambda + rho * cons).clamp_min(0.0)

    return stats_from_batch(...)
src/models/argse.py (wrapper)

Quản lý state (φ, α, μ, λ, τ), optimizer, schedulers, EMA α; expose train_epoch, eval_epoch.
DoD: loss giảm; coverage ~ hợp lý; không diverge.
M6. Huấn luyện AR-GSE (Balanced & Worst-group)
src/train/train_argse.py

Load experts (checkpoint + T_e). Dùng logits/posteriors precomputed nếu có; nếu không, forward online (chậm hơn).
Load tuneV cho train; val_small để theo dõi; test để cuối cùng đánh giá.
Balanced: cố định β=1K1\beta=\frac1K\mathbf{1}

β=K
1
1.
Worst-group: cập nhật β\beta
β mỗi epoch bằng Exponentiated-Gradient (tham số xi), dùng ước lượng e^k\hat e_k
e
^
k
 từ val_small.
Hyper mặc định (ổn định)

Optim: Adam (φ) lr 1e-3; (α, μ) lr 5e-3; dual step ρ=1e-2.
τ: 2→10 trong 30 epoch (giữ nguyên tới hết).
λ_ent=1e-3; (optional) l1_w=5e-4.
Epochs: 80–120 (tuỳ dataset).
Batch size: 256 (CIFAR), 128 (ImageNet-LT).
Early stop theo AURC/balanced error trên val_small.
Command


# Balanced
python -m src.train.train_argse +exp=argse_balanced dataset=cifar10_lt experts="[ce,balsoftmax,logitadjust,decoupled]" seed=42# Worst-group
python -m src.train.train_argse +exp=argse_worst dataset=cifar10_lt xi=1.0 seed=42
DoD: log loss, e_k, coverage, AURC↓; lưu argse.ckpt (φ, α, μ, λ).
M7. Đánh giá, RC/AURC, Conformal (tuỳ chọn)
src/metrics/selective_metrics.py

Tính eke_k
ek
 trên accepted subset; balanced error; worst-group error; coverage; min-group coverage.
src/metrics/rc_curve.py

Sắp xếp theo margin; sweep ngưỡng (hoặc chấp nhận theo percentile) → RC-curve & AURC.
src/models/selective.py (hard decision)

Làm “hard accept” đúng theo §3.2 để report @ coverage cố định.
src/models/conformal.py (tuỳ chọn)

Từ calib, tính ngưỡng phân vị tkt_k
tk
 theo nhóm cho margin → kiểm soát lỗi hữu hạn mẫu ở test.
src/train/eval_test.py

Report:
Balanced/Worst-group error @ coverage {70,80,90}%;
AURC (±95% CI bootstrap);
Coverage per-group, min-group coverage;
ECE/NLL của η~\tilde\eta
η
~
;
(tuỳ chọn) Conformal.
Command


python -m src.train.eval_test +exp=eval dataset=cifar10_lt use_conformal=false seed=42
DoD: xuất metrics.json, rc_curve.csv, hình plots/*.png.
M8. Baselines & Ablations (đập phản biện)
src/train/eval_baselines.py

U-PI: Uniform-average + (old) plug-in fixed-point (hoặc hard rule tuyến tính–ngưỡng với α, μ tối ưu trên val_small).
SB-PI: Single-best expert + plug-in.
DE-Chow: Deep ensemble + reject theo max-prob (Chow).
AR-GSE (no-sparsity vs hard-routing): bật/tắt phạt entropy/L1.
Sensitivity: số experts {1,2,4,6}; kiến trúc gating 2-tầng vs 3-tầng; τ (2→10 vs 2→6), λ_ent (1e-4, 1e-3, 1e-2).
Seeds: ≥5 (42,43,44,45,46) → mean±std; bootstrap 95% CI.
Command


python -m src.train.eval_baselines +exp=eval dataset=cifar10_lt seeds="[42,43,44,45,46]"
DoD: bảng so sánh đầy đủ, có CI; đồ thị độ nhạy.
M9. Script tổng & Repro
scripts/run_cifar10lt_argse.sh


# 1) Train experts + temp scaling
python -m src.train.train_expert +exp=experts dataset=cifar10_lt seed=42# 2) Train AR-GSE (balanced)
python -m src.train.train_argse +exp=argse_balanced dataset=cifar10_lt seed=42# 3) (optional) Worst-group
python -m src.train.train_argse +exp=argse_worst dataset=cifar10_lt xi=1 seed=42# 4) Evaluate (test)
python -m src.train.eval_test +exp=eval dataset=cifar10_lt use_conformal=false seed=42# 5) Baselines & ablations
python -m src.train.eval_baselines +exp=eval dataset=cifar10_lt seeds="[42,43,44,45,46]"
README.md

Quickstart, mô tả configs, đường dẫn artifact, cách vẽ hình & tạo bảng.
DoD: chạy được end-to-end cho CIFAR-10-LT; sinh bảng/hình sẵn sàng chèn paper.
Cấu hình Hydra (mẫu rút gọn)
configs/dataset/cifar10_lt.yaml


name: cifar10_ltimb_factor: 100splits: { train: 0.80, tuneV: 0.08, val_small: 0.06, calib: 0.03, test: 0.03 }groups: { K: 2, head_ratio: 0.5 }
configs/exp/experts.yaml


model: { backbone: resnet32_cifar, num_classes: 10 }train: { epochs: 300, bs: 256, lr: 0.1, cosine: true, aug: { ra_n: 2, ra_m: 10, mixup: 0.2, cutmix: 0.2 } }losses: [ce, balsoftmax, logitadjust, decoupled]calibration: { temperature_scaling: true }export_posteriors: true
configs/exp/argse_balanced.yaml


argse:
  tau: { start: 2.0, end: 10.0, warmup_epochs: 30 }
  c: 0.05
  beta: balanced
  lr: { phi: 1e-3, alpha: 5e-3, mu: 5e-3 }
  rho: 1e-2
  ent_lambda: 1e-3
  l1_w: 0.0
  alpha_clip: 1e-3
  epochs: 100
  bs: 256
  early_stop_patience: 15logging: { every_n_steps: 50 }
configs/exp/argse_worst.yaml


defaults: [argse_balanced]argse:
  beta: worst
  eg_xi: 1.0
configs/exp/eval.yaml


eval:
  coverage_points: [0.7, 0.8, 0.9]
  bootstrap: { n: 1000, ci: 0.95 }
  use_conformal: false
Tiêu chí hoàn thành & bẫy cần tránh
Definition of Done

 AR-GSE train stability (loss ↓, coverage hợp lý, không NaN).
 Balanced/Worst-group: AURC và error @ coverage cải thiện hơn U-PI với CI.
 Báo cáo mean±std (5 seeds) + 95% CI bootstrap.
 Ablations: hard-routing không làm giảm mạnh AURC nhưng giảm thời gian suy luận.
 iNat/IN-LT: dùng top-k features hoặc random projection; bộ nhớ không bùng nổ.
Bẫy

Rò rỉ split: experts chỉ train; AR-GSE chỉ tuneV; EG worst dùng val_small; conformal dùng calib.
Độ dốc không ổn: tăng/giảm tau quá nhanh → dao động; khuyến nghị warm-up tuyến tính.
α chạm 0: luôn clamp>=1e-3 + EMA smoothing 0.9.
λ bùng nổ: giảm rho hoặc chuẩn hoá vi phạm theo batch size; có thể clip lambda max.
Posterior lệch: luôn temperature scaling cho từng expert.
Kế hoạch thí nghiệm tối thiểu (để ra kết quả viết paper)
CIFAR-10-LT (IF=100)
E=4 experts; AR-GSE Balanced; báo cáo AURC & balanced err @ {70,80,90}% (+CI).
Baselines: U-PI, SB-PI, DE-Chow.
Ablations: no-sparsity vs hard-routing; số experts {1,2,4,6}.
CIFAR-100-LT lặp kịch bản tốt nhất.
(Tuỳ) ImageNet-LT/iNat-2018: dùng top-k features/random projection; báo thời gian suy luận.
Worst-group (EG) trên CIFAR-LT: quỹ đạo β\beta
β + cải thiện worst-group err.
Kết quả xuất: bảng chính (AURC, errors @ coverage), hình RC-curves, histogram margin theo nhóm, quỹ đạo β\beta
β, bảng sensitivity.