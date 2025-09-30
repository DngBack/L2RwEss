# AURC Improvement Patches

## Vấn đề chính
**AURC không cải thiện vì ranking của margins không đổi.** Recalibrate threshold chỉ dịch điểm cắt, không thay đổi thứ tự.

## Các bản vá đã implement

### ✅ 1. Đồng bộ per-expert temperature (CRITICAL)
**File: `eval_gse_plugin.py`**
- ✅ `get_mixture_posteriors()` đã áp dụng temperature scaling
- ✅ `recalibrate_thresholds_on_val()` đã áp dụng temperature scaling
- ✅ Consistency: train và eval sử dụng cùng phân bố

**Impact:** Temperature scaling thay đổi ranking của margins → ảnh hưởng trực tiếp AURC

---

### ✅ 2. Fix alpha update trong inner loop (CRITICAL)
**File: `gse_worst_eg.py`**

**Vấn đề cũ:**
```python
# Placeholder - alpha không thực sự update
a_cur = 0.9 * a_cur + 0.1 * torch.ones(K, device=device)
```

**Giải pháp:**
```python
# New helper function
def update_alpha_conditional_with_beta_tgroup(eta, y, alpha, mu, beta, t_group, ...):
    """Conditional alpha update with beta weighting and per-group thresholds"""
    # 1. Compute raw margins with beta
    # 2. Get predictions with (α*β) weighting
    # 3. Use PREDICTED group for threshold (deployable)
    # 4. Compute acceptance per GT-group (training signal)
    # 5. EMA update with projection (geomean=1, clamp [0.8, 1.4])
    
# Usage in inner loop
a_cur = update_alpha_conditional_with_beta_tgroup(
    eta_S1, y_S1, a_cur, mu, beta, t_group_cur, class_to_group, K,
    gamma=gamma, a_min=0.8, a_max=1.4
)
```

**Impact:** 
- α thực sự di chuyển (không còn kẹt ở [1, 1])
- Ranking của margins thay đổi theo mục tiêu coverage/group
- AURC giảm đáng kể

---

### ✅ 3. Fit t_k theo pred-group trên ALL predictions
**File: `gse_worst_eg.py`**

**Vấn đề cũ:**
```python
# Fit trên correct-only với GT-groups
correct_mask = (preds_S1 == y_S1.cpu())
t_group_cur = fit_group_thresholds_from_raw(
    raw_S1.cpu()[correct_mask], 
    y_groups_S1[correct_mask], ...
)
```

**Giải pháp:**
```python
# Fit trên ALL predictions với PREDICTED groups (deployable)
alpha_per_class = (a_cur * beta)[class_to_group]
yhat_S1 = (alpha_per_class.unsqueeze(0) * eta_S1).argmax(dim=1).cpu()
pred_groups_S1 = class_to_group[yhat_S1]

t_group_cur = []
for k in range(K):
    mk = (pred_groups_S1 == k)
    if mk.sum() > 0:
        q = 1.0 - target_cov_by_group[k]
        t_k = float(torch.quantile(raw_S1[mk].cpu(), q))
        t_group_cur.append(t_k)
```

**Impact:**
- Thresholds đã "deployable" ngay trong training
- Giảm lệch train-eval
- Coverage ổn định hơn

---

### ✅ 4. Coverage penalty trong objective
**File: `gse_worst_eg.py`**

**Thêm vào evaluation:**
```python
# Compute coverage by predicted groups on S2
raw_S2 = compute_raw_margin_with_beta(eta_S2, a_cur, mu, beta, class_to_group)
yhat_S2 = (alpha_per_class_S2.unsqueeze(0) * eta_S2).argmax(dim=1)
pred_groups_S2 = class_to_group[yhat_S2]

# Per-group coverage
cov_by_pred_group = []
for k in range(K):
    mk = (pred_groups_S2 == k)
    cov_k = (raw_S2[mk] >= t_group_cur[k]).float().mean().item()
    cov_by_pred_group.append(cov_k)

# Coverage penalty
cov_penalty = sum((cov_by_pred_group[k] - target_cov_by_group[k])**2 for k in range(K))

# Combined objective
score = w_err + 5.0 * cov_penalty  # weight 5.0
```

**Impact:**
- EG-outer không đánh đổi worst-error bằng threshold quá khắt
- Coverage giữ ổn định gần target
- Tránh overfitting trên S2

---

### ✅ 5. Nới range α và tăng alpha_steps
**Files: `gse_balanced_plugin.py`, `gse_worst_eg.py`**

**Thay đổi:**
```python
# Old bounds
'alpha_min': 0.85, 'alpha_max': 1.15, 'alpha_steps': 5

# New bounds (more room for head/tail adjustment)
'alpha_min': 0.80, 'alpha_max': 1.40, 'alpha_steps': 7
```

**Impact:**
- α có dư địa cân bằng head/tail
- μ không phải "gánh" hết (μ hiện ±0.51 khá lớn)
- Ranking linh hoạt hơn

---

## Kỳ vọng sau patches

### Coverage metrics:
- ✅ Coverage tổng ≈ 0.5-0.6 (gần target τ_head=0.56, τ_tail=0.44)
- ✅ Per-group coverage ổn định

### AURC metrics:
- ✅ AURC (Balanced) giảm ≥ 5-10%
- ✅ AURC (Worst) giảm ≥ 10-15%
- ✅ RC curve mượt hơn, không có "sụt đột ngột"

### Alpha evolution:
- ✅ α không còn kẹt ở [1, 1]
- ✅ α di chuyển trong range [0.8, 1.4]
- ✅ Log sẽ hiển thị: `α_t = [0.92, 1.08]` (example)

### Training logs:
```
EG iteration 5/20, β=[0.4500, 0.5500]
   α_t = [0.9234, 1.0812], μ_t = [0.3200, -0.3200]
   t_group = [-0.6520, -0.5950]

✅ Best inner solution: score=0.0518 (raw_err=0.0485, cov_pen=0.0066)
```

---

## Tuỳ chọn nâng cao (chưa implement)

### Option A: Fine-tune gating với ranking-aware loss
```python
# After getting α*, μ*, t_k* from EG-outer
# Fine-tune gating_net 3-5 epochs with:

L_sel = E[s(x) · CE(q_α, y)]  # Selective CE
L_bce = BCEWithLogits(κ(m_raw - t_{g(y)}), 1{y=ŷ})  # Acceptance BCE
L_kl = KL(w(x) || prior_group)  # Group prior

L_total = L_sel + λ_bce * L_bce + λ_kl * L_kl
```

**Impact:** Đổi trực tiếp w(x) → η(x) → ranking của margins → AURC ↓↓

---

## Testing checklist

- [ ] Run training: `python run_improved_eg_outer.py`
- [ ] Check alpha evolution in logs (should move from [1, 1])
- [ ] Check coverage penalty in logs (should be < 0.01)
- [ ] Run evaluation: `python -m src.train.eval_gse_plugin`
- [ ] Compare AURC before/after (expect ≥10% reduction)
- [ ] Verify coverage hits target (0.56 head, 0.44 tail)
- [ ] Check RC curve smoothness (no sudden drops)

---

## Files modified

1. `src/train/gse_worst_eg.py`
   - Added `update_alpha_conditional_with_beta_tgroup()`
   - Updated threshold fitting (pred-group, ALL samples)
   - Added coverage penalty to objective
   - Enhanced logging for alpha tracking

2. `src/train/gse_balanced_plugin.py`
   - Updated CONFIG: alpha_min=0.80, alpha_max=1.40, alpha_steps=7
   - Updated all alpha update functions with new bounds
   - Temperature scaling already present in `cache_eta_mix()`

3. `src/train/eval_gse_plugin.py`
   - Temperature scaling already consistent
   - Recalibration uses predicted groups
   - Ready for improved AURC measurement

---

## Debug tips

If α still doesn't move:
```python
# In inner loop, add print:
print(f"  [DEBUG] alpha before: {a_cur.tolist()}")
a_cur = update_alpha_conditional_with_beta_tgroup(...)
print(f"  [DEBUG] alpha after:  {a_cur.tolist()}")
print(f"  [DEBUG] alpha_hat:    {alpha_hat.tolist()}")
```

If AURC doesn't improve:
- Check temperature is applied consistently (train & eval)
- Check margins_raw are computed with same α, μ, T
- Verify ranking changes: plot sorted margins before/after
- Compare RC curves visually

---

## Next steps

1. **Run training** and check logs for alpha movement
2. **Evaluate** and compare AURC metrics
3. **If still not satisfactory:** Consider fine-tune gating (Option A)
4. **Document results** in results/ folder
5. **Commit changes** with clear message

---

**Expected timeline:**
- Training: ~2-3 hours
- Evaluation: ~10 minutes
- Analysis: ~30 minutes
- **Total: 3-4 hours for complete cycle**

Good luck! 🚀
