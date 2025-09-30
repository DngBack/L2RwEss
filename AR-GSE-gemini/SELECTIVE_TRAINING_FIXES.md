# Selective Training Fixes - Consistency with Paper Law

## Ngày: 30/09/2025

## Tổng quan
Đã khắc phục các vấn đề về **mismatch giữa luật quyết định và đánh giá** trong selective training mode, đảm bảo nhất quán với paper AR-GSE.

---

## Các vấn đề đã sửa

### 1. ✅ Robust Logits Loading
**Vấn đề**: Code giả định `torch.load()` trả về tensor thẳng, dễ lỗi khi thay đổi format.

**Giải pháp**: Thêm helper functions:
```python
def _load_logits_tensor(path):
    """Handle both dict {'logits': tensor} and raw tensor formats"""
    
def _load_labels_from_file_or_base(obj, base_ds, indices):
    """Load labels from dict or fallback to base dataset"""
```

**Lợi ích**: 
- Tương thích với cả format cũ (tensor) và mới (dict)
- Tự động fallback khi cần
- Hỗ trợ float32 format cho tiết kiệm bộ nhớ

---

### 2. ✅ Prediction với α (Nhất quán với Paper)
**Vấn đề**: `evaluate_split_with_learned_thresholds()` dùng `eta.argmax()` (không áp α)

**Giải pháp**: Dự đoán theo công thức paper:
```python
# ŷ = argmax_y α_{g(y)} * η_y
alpha_per_class = alpha[class_to_group]  # [C]
yhat = (alpha_per_class.unsqueeze(0) * eta).argmax(dim=1)  # [B]
```

**Lý do**: Đảm bảo dự đoán tại evaluation khớp với test-time law trong paper (§2.5)

---

### 3. ✅ Hard Acceptance theo Group của ŷ (Không phải y_true)
**Vấn đề**: 
- Dùng soft acceptance `s = σ(κ*(m - t))` thay vì hard `1{m ≥ t}`
- Áp threshold theo group của `y_true` thay vì group của `ŷ`

**Giải pháp**: 
```python
# Get threshold based on PREDICTED class group
t_for_pred = t_param[class_to_group[yhat]]  # [B]

# Hard acceptance
accepted = m_raw > t_for_pred
```

**Lý do**: Test-time không biết y_true, chỉ dựa vào ŷ để chọn threshold

---

### 4. ✅ Update α với Per-Group Thresholds
**Vấn đề**: `update_alpha_fixed_point_conditional()` vẫn dùng scalar threshold `t`

**Giải pháp**: Chuyển sang dùng `t_param` (per-group thresholds):
```python
def update_alpha_fixed_point_conditional(eta, y, alpha, mu, t_param, ...):
    # Prediction with α
    yhat = (alpha_per_class.unsqueeze(0) * eta).argmax(dim=1)
    
    # Per-sample threshold based on predicted group
    t_samp = t_param[class_to_group[yhat]]
    
    # Hard acceptance
    accepted = raw > t_samp
    
    # Compute acceptance rate per ground-truth group
    for k in range(K):
        mask = (y_groups == k)
        grp_acc = (accepted & mask).sum() / mask.sum()
        alpha_hat[k] = grp_acc
```

**Lý do**: Conditional acceptance trong paper dùng luật:
```
r̂_k = #{accept ∧ y ∈ G_k} / #{y ∈ G_k}
```

---

### 5. ✅ Loại bỏ Code Thừa
**Vấn đề**: `compute_mixture_and_w()` có dòng `with torch.no_grad(): pass`

**Giải pháp**: Đã xóa dòng thừa

---

## Công thức Chuẩn (Paper AR-GSE)

### Dự đoán:
```
ŷ = argmax_y α_{g(y)} * η̃_y(x)
```

### Acceptance:
```
accept ⟺ m_raw(x) ≥ t_{g(ŷ)}
```

### Raw Margin:
```
m_raw(x) = max_y α_{g(y)} * η_y - Σ_y (1/α_{g(y)} - μ_{g(y)}) * η_y
```

### Conditional Acceptance Rate (cho update α):
```
r̂_k = P(accept | y ∈ G_k)
```

---

## Tác động

### ✅ Consistency
- Inner/outer optimization giờ khớp với test-time law
- Không còn mismatch giữa training và evaluation

### ✅ Correctness
- Chọn μ đúng (không bị sai lệch do soft acceptance)
- α match với acceptance thực tế
- Worst-case/balanced metrics chính xác

### ✅ Robustness
- Tương thích với nhiều format logits
- Dễ mở rộng cho K > 2 groups

---

## Testing

### Kiểm tra format compatibility:
```bash
# Test với format cũ (tensor)
python src/train/train_gating_only.py --mode selective

# Test với format mới (dict) - sau khi update export
```

### Kiểm tra metrics:
```python
# Verify prediction law
assert (yhat == (alpha[class_to_group].unsqueeze(0) * eta).argmax(1)).all()

# Verify hard acceptance
assert (accepted == (m_raw > t_param[class_to_group[yhat]])).all()
```

---

## Files Modified

1. **`src/train/train_gating_only.py`**:
   - Added: `_load_logits_tensor()`, `_load_labels_from_file_or_base()`
   - Updated: `load_data_from_logits()`, `load_two_splits_from_logits()`
   - Fixed: `compute_mixture_and_w()` (removed redundant code)
   - Fixed: `update_alpha_fixed_point_conditional()` (per-group thresholds)
   - Fixed: `evaluate_split_with_learned_thresholds()` (correct prediction & hard acceptance)
   - Updated: B2 section in `run_selective_mode()` (use t_param)

---

## Next Steps

### Optional Improvements:
1. **Export logits as dict**: Thêm labels & indices vào file export
   ```python
   torch.save({
       'logits': logits.to(torch.float16),
       'labels': labels,
       'indices': indices
   }, path)
   ```

2. **Diagnostic logging**: Log acceptance rates by group during training
   ```python
   print(f"Acceptance: head={acc_head:.2%}, tail={acc_tail:.2%}")
   ```

3. **Visualize decision boundary**: Plot m_raw distribution vs thresholds

---

## References
- Paper: AR-GSE Section 2.5 (Selective Prediction Law)
- Original issue: Mismatch between training and test-time decision rules
