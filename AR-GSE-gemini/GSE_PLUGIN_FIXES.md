# GSE Balanced Plugin Fixes - Deployable Decision Rule

## Ngày: 30/09/2025

## Tổng quan
Đã khắc phục các vấn đề lệch chuẩn quan trọng trong `gse_balanced_plugin.py` để đảm bảo luật quyết định **khả triển (deployable)** và nhất quán với paper AR-GSE.

---

## ✅ Các vấn đề đã sửa

### 1. **Ngưỡng theo nhóm của ŷ (không phải y_true)** 🎯

**Vấn đề**: 
- `worst_error_on_S_with_per_group_thresholds()` dùng `t_group[y_groups]` với `y_groups = class_to_group[y]`
- Ở test-time không biết `y_true`, không thể áp ngưỡng theo ground-truth group

**Giải pháp**:
```python
# Prediction with α
alpha_per_class = alpha[class_to_group]  # [C]
yhat = (alpha_per_class.unsqueeze(0) * eta).argmax(dim=1)  # [N]

# Threshold based on PREDICTED group (deployable!)
pred_groups = class_to_group[yhat]  # [N]
thresholds_per_sample = t_group[pred_groups]  # [N]

# Hard acceptance
accepted = (raw_margins > thresholds_per_sample)
```

**Lý do**: Luật quyết định phải dựa vào ŷ để triển khai được ở test-time (không biết y thật).

---

### 2. **Per-Expert Temperature Scaling** 🌡️

**Vấn đề**: 
- `cache_eta_mix()` tính `softmax(logits)` trực tiếp, bỏ qua temperature calibration
- Margin/threshold không ổn định nếu experts chưa calibrated

**Giải pháp**:
```python
def apply_per_expert_temperature(logits, expert_names, temp_dict):
    """Apply T_e to each expert e before softmax"""
    if not temp_dict:
        return logits
    scaled = logits.clone()
    for i, name in enumerate(expert_names):
        T = float(temp_dict.get(name, 1.0))
        if abs(T - 1.0) > 1e-6:
            scaled[:, i, :] = scaled[:, i, :] / T
    return scaled

# In cache_eta_mix:
logits = apply_per_expert_temperature(logits, expert_names, temperatures)
expert_posteriors = torch.softmax(logits, dim=-1)
```

**Lợi ích**: Mixture posterior η̃ đã calibrated → margin/threshold ổn định hơn.

---

### 3. **Alpha Update với Per-Group Thresholds** 🔄

**Vấn đề**: 
- Các hàm update α hiện dùng scalar threshold `t`
- Khi có per-group thresholds `t_k`, cần update theo luật deployable

**Giải pháp**: Thêm `update_alpha_conditional_pg()`:
```python
def update_alpha_conditional_pg(eta, y, alpha, mu, t_group, class_to_group, K, ...):
    """Conditional alpha update using per-group thresholds (deployable)"""
    
    # Prediction
    alpha_per_class = alpha[class_to_group]
    yhat = (alpha_per_class.unsqueeze(0) * eta).argmax(dim=1)
    
    # Raw margin
    raw = compute_raw_margin(eta, alpha, mu, class_to_group)
    
    # Per-sample threshold based on PREDICTED group
    t_samp = t_group[class_to_group[yhat]]
    
    # Hard acceptance
    accepted = raw > t_samp
    
    # Conditional acceptance per ground-truth group
    y_groups = class_to_group[y]
    alpha_hat = torch.ones_like(alpha)
    for k in range(K):
        mask = (y_groups == k)
        if mask.any():
            acc_rate = (accepted & mask).float().sum() / mask.float().sum()
            alpha_hat[k] = acc_rate + 1e-3
    
    # EMA + project
    new_alpha = (1 - gamma) * alpha + gamma * alpha_hat
    return project_alpha(new_alpha, alpha_min, alpha_max)
```

**Sử dụng trong inner loop**:
```python
# Blended update for stability
alpha = 0.75 * update_alpha_joint(...) + 0.25 * update_alpha_conditional_pg(...)
```

---

### 4. **Deterministic Training** 🔧

**Vấn đề**: Random initialization/ordering có thể gây khác biệt giữa các lần chạy

**Giải pháp**:
```python
torch.manual_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

**Lợi ích**: Reproducible results, dễ debug và compare.

---

### 5. **Load Temperatures từ Selective Checkpoint** 📂

**Vấn đề**: Selective training đã fit temperatures, nhưng plugin không dùng

**Giải pháp**:
```python
# When loading selective checkpoint
temperatures = sel_ckpt.get('temperatures', None)
if temperatures:
    print(f"✅ Loaded per-expert temperatures: {temperatures}")

# Apply when caching
eta_S1, y_S1 = cache_eta_mix(model, S1_loader, class_to_group,
                              expert_names=CONFIG['experts']['names'],
                              temperatures=temperatures)
```

---

## 🎯 Công thức Chuẩn (Paper-Consistent)

### Dự đoán:
```
ŷ = argmax_y α_{g(y)} * η̃_y(x)
```

### Acceptance (Per-Group Thresholds):
```
accept ⟺ m_raw(x) > t_{g(ŷ)}
```
**Quan trọng**: Ngưỡng dựa vào `g(ŷ)` (predicted group), không phải `g(y)` (true group)!

### Raw Margin:
```
m_raw(x) = max_y α_{g(y)} * η_y - Σ_y (1/α_{g(y)} - μ_{g(y)}) * η_y
```

### Conditional Acceptance Rate:
```
r̂_k = P(accept | y ∈ G_k) = #{accept ∧ y∈G_k} / #{y∈G_k}
```

---

## 📊 Các vấn đề chưa sửa (TODO)

### 🔴 Quan trọng: Bỏ β khỏi Inner Rule (EG-Outer)

**Vấn đề hiện tại** (trong EG-outer code):
```python
# ❌ WRONG: β được nhân vào rule
preds = ((alpha * beta)[class_to_group] * eta).argmax(...)
raw = compute_raw_margin_with_beta(eta, alpha, mu, beta, ...)
```

**Sửa đúng**:
```python
# ✅ CORRECT: β chỉ dùng cho objective, không can thiệp rule
preds = (alpha[class_to_group] * eta).argmax(...)  # No β
raw = compute_raw_margin(eta, alpha, mu, ...)       # No β

# β chỉ dùng để weight objective trong outer loop
weighted_errors = beta * group_errors
```

**Lý do**: β là trọng số hóa mục tiêu (outer), không phải tham số của luật dự đoán/acceptance.

---

### 🔴 Alpha Update trong Inner EG

**Vấn đề**: Placeholder update thay vì blended conditional:
```python
# ❌ Current
a_cur = 0.9 * a_cur + 0.1 * torch.ones(K)
```

**Sửa đúng**: Dùng `update_alpha_conditional_pg()` với per-group thresholds.

---

### 🟡 Self-Import

**Vấn đề**: 
```python
from src.train.gse_balanced_plugin import worst_error_on_S_with_per_group_thresholds
```
trong chính file `gse_balanced_plugin.py` → circular import risk.

**Sửa**: Gọi hàm trực tiếp, không import.

---

### 🟡 Đổi tên "c" → "t"

**Vấn đề**: Tên biến `c` gây nhầm với "rejection cost", thực tế là threshold `t`.

**Sửa**: Đổi `compute_margin(eta, alpha, mu, c, ...)` → `compute_margin(eta, alpha, mu, t, ...)`.

---

## 🧪 Testing Checklist

### ✅ Verification Tests:

1. **Prediction law**:
```python
alpha_per_class = alpha[class_to_group]
yhat_expected = (alpha_per_class.unsqueeze(0) * eta).argmax(dim=1)
assert (yhat == yhat_expected).all()
```

2. **Acceptance by pred-group**:
```python
pred_groups = class_to_group[yhat]
t_samp = t_group[pred_groups]
accepted_expected = raw_margins > t_samp
assert (accepted == accepted_expected).all()
```

3. **Temperature scaling**:
```python
# Check if temperatures applied before softmax
scaled = apply_per_expert_temperature(logits, names, temps)
assert not torch.equal(scaled, logits) if temps else torch.equal(scaled, logits)
```

4. **Determinism**:
```bash
# Run twice with same seed
python src/train/gse_balanced_plugin.py
# Results should be identical
```

---

## 📈 Expected Improvements

### Performance:
- ✅ Better tail accuracy (thresholds matched to pred-group)
- ✅ More stable training (temperature scaling)
- ✅ Reproducible results (deterministic mode)

### Consistency:
- ✅ Training/eval match test-time law
- ✅ No mismatch between inner/outer optimization
- ✅ Deployable decision rule (no access to y_true)

---

## 🔧 Usage

### Basic run:
```bash
python src/train/gse_balanced_plugin.py
```

### With selective init (recommended):
```python
CONFIG['plugin_params']['use_selective_init'] = True  # Load gating + temperatures
```

### Check temperatures loaded:
```
📂 Loaded selective gating checkpoint: ...
✅ Loaded per-expert temperatures: {'ce_baseline': 1.05, 'logitadjust_baseline': 1.12, ...}
🌡️  Applying per-expert temperature scaling: {...}
```

---

## 📚 References

- **Paper**: AR-GSE Section 2.5 (Selective Prediction)
- **Related Fix**: `SELECTIVE_TRAINING_FIXES.md` (train_gating_only.py)
- **Original Issue**: Threshold by ground-truth group → not deployable

---

## 🎓 Key Lessons

1. **Test-time Deployability**: Mọi luật quyết định phải chỉ dựa vào input x và prediction ŷ, không được dùng y_true.

2. **Consistency is King**: Training/eval/test phải dùng cùng một luật - không được mismatch giữa các giai đoạn.

3. **Temperature Matters**: Calibration trước khi ensemble → mixture posterior ổn định hơn.

4. **Determinism for Science**: Reproducibility là điều kiện cần để debug và so sánh methods.

5. **β is Outer Weight, Not Inner Rule**: Trọng số hóa objective ≠ can thiệp vào prediction/acceptance rule.

---

## 📝 Files Modified

- `src/train/gse_balanced_plugin.py`:
  - ✅ Added: `apply_per_expert_temperature()`
  - ✅ Added: `update_alpha_conditional_pg()`
  - ✅ Fixed: `worst_error_on_S_with_per_group_thresholds()` (pred-group thresholds)
  - ✅ Fixed: `cache_eta_mix()` (temperature scaling support)
  - ✅ Fixed: `main()` (deterministic mode, load temperatures)

---

## 🚀 Next Steps

1. ✅ **Immediate**: Test với selective checkpoint có temperatures
2. 🔴 **High Priority**: Fix EG-outer inner loop (bỏ β khỏi rule)
3. 🟡 **Medium**: Refactor naming (c → t)
4. 🟢 **Low**: Add diagnostic logging (per-group coverage by pred-group)
