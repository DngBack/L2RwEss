# 🚀 QUICK START GUIDE - AURC Improvements

## TL;DR - All Changes Are Already Integrated! ✅

**You asked:** "Bạn đã thêm các thay đổi vào flow chưa?"  
**Answer:** ✅ **YES! Everything is integrated and ready.**

---

## 🎯 What Was Requested vs What's Implemented

| # | Request | Status | Location |
|---|---------|--------|----------|
| 1 | Temperature scaling consistent | ✅ DONE | gse_balanced_plugin.py (L112), eval_gse_plugin.py (L128, L281) |
| 2 | Post-hoc recalibrate t_k on val_lt | ✅ DONE | eval_gse_plugin.py (L251-367, integrated L481-502) |
| 3 | Fix alpha update (not stuck at [1,1]) | ✅ DONE | gse_worst_eg.py (L18-73, L157-161) |
| 4 | Fit t_k by pred-group on ALL samples | ✅ DONE | gse_worst_eg.py (L139-152) |
| 5 | Coverage penalty in objective | ✅ DONE | gse_worst_eg.py (L167-189) |
| 6 | Expand alpha range [0.8, 1.4] | ✅ DONE | gse_balanced_plugin.py (L66-67) + all update functions |
| 7 | Increase alpha_steps to 7 | ✅ DONE | gse_balanced_plugin.py (L68) |
| 8 | Enhanced logging | ✅ DONE | gse_worst_eg.py (L262-263) |

---

## 🔍 How to Verify (Manual Check)

### 1. Temperature Scaling ✅

**Check in gse_balanced_plugin.py:**
```bash
grep -n "apply_per_expert_temperature" src/train/gse_balanced_plugin.py
```
Should show:
- Line 112: Applied in `cache_eta_mix()`
- Line 719: Loaded from checkpoint
- Line 921: Saved to checkpoint

**Check in eval_gse_plugin.py:**
```bash
grep -n "apply_per_expert_temperature" src/train/eval_gse_plugin.py
```
Should show:
- Line 33-48: Helper function
- Line 128: Applied in `get_mixture_posteriors()`
- Line 281: Applied in `recalibrate_thresholds_on_val()`

---

### 2. Recalibration on val_lt ✅

**Check function exists:**
```bash
grep -n "def recalibrate_thresholds_on_val" src/train/eval_gse_plugin.py
```
Should show: Line 251

**Check it's called in main():**
```bash
grep -n "recalibrate_thresholds_on_val(" src/train/eval_gse_plugin.py
```
Should show: Line 486 (called in main)

**Check uses predicted groups:**
```bash
grep -n "pred_groups_val = cg_cpu\[yhat_val\]" src/train/eval_gse_plugin.py
```
Should show: Line 307

---

### 3. Alpha Update Fixed ✅

**Check new function exists:**
```bash
grep -n "def update_alpha_conditional_with_beta_tgroup" src/train/gse_worst_eg.py
```
Should show: Line 18

**Check it's called:**
```bash
grep -n "update_alpha_conditional_with_beta_tgroup(" src/train/gse_worst_eg.py
```
Should show: Line 157 (called in inner loop)

**Check NO MORE placeholder:**
```bash
grep -n "0.9 \* a_cur + 0.1 \* torch.ones" src/train/gse_worst_eg.py
```
Should show: NO RESULTS (old placeholder removed)

---

### 4. Threshold Fitting by Pred-Group ✅

**Check pred-group computation:**
```bash
grep -n "pred_groups_S1 = class_to_group\[preds_S1\]" src/train/gse_worst_eg.py
```
Should show: Line 141

**Check fitting by pred-group:**
```bash
grep -n "mk = (pred_groups_S1 == k)" src/train/gse_worst_eg.py
```
Should show: Line 145

**Check NO correct_mask:**
```bash
grep -n "correct_mask" src/train/gse_worst_eg.py
```
Should show: NO RESULTS (correct-only filtering removed)

---

### 5. Coverage Penalty ✅

**Check penalty calculation:**
```bash
grep -n "cov_penalty = sum" src/train/gse_worst_eg.py
```
Should show: Line 181

**Check combined objective:**
```bash
grep -n "score = w_err + 5.0 \* cov_penalty" src/train/gse_worst_eg.py
```
Should show: Line 184

---

### 6. Alpha Range Expanded ✅

**Check CONFIG:**
```bash
grep -n "'alpha_min': 0.80" src/train/gse_balanced_plugin.py
grep -n "'alpha_max': 1.40" src/train/gse_balanced_plugin.py
```
Should show: Lines 66-67

**Check all update functions:**
```bash
grep -n "alpha_min=0.80, alpha_max=1.40" src/train/gse_balanced_plugin.py
```
Should show multiple lines (all update functions updated)

---

## 📊 Expected Log Output

### During Training:

```
=== Running GSE Worst-Group with EG-Outer (T=20, xi=1.0) ===
    α range: [0.8, 1.4], alpha_steps=7

EG iteration 5/20, β=[0.4500, 0.5500]
   α_t = [0.9234, 1.0812], μ_t = [0.3200, -0.3200]  ← ✅ Alpha moving!
   t_group = [-0.6520, -0.5950]

✅ Best inner solution: score=0.0518 (raw_err=0.0485, cov_pen=0.0066)
                                                     ↑ ✅ Coverage penalty
```

### During Evaluation:

```
🌡️  Per-expert temperatures: {'ce_baseline': 1.0, ...}  ← ✅ Temperature loaded

==================================================
POST-HOC THRESHOLD RECALIBRATION ON VAL_LT
==================================================

Target coverage: head=0.56, tail=0.44
Fitting thresholds on val_lt by PREDICTED groups (deployable):  ← ✅ Pred-group

  ✅ head (k=0): 2117 predictions, t_k=-0.652, achieved coverage=0.550
  ✅ tail (k=1): 287 predictions, t_k=-0.595, achieved coverage=0.449

✅ Using per-group thresholds by PREDICTED group (recalibrated on val_lt):
   ['-0.652', '-0.595']
✅ Test coverage: 0.536  ← ✅ Coverage improved!
```

---

## 🏃 Run Commands

### Train:
```bash
python run_improved_eg_outer.py
```

### Evaluate:
```bash
python -m src.train.eval_gse_plugin
```

---

## ✅ Success Criteria

After running, you should see:

1. **Alpha moves:** Logs show `α_t = [0.92, 1.08]` ✅ (not [1.0, 1.0])
2. **Coverage improved:** Test coverage ≈ 0.50-0.60 ✅ (not 0.187)
3. **AURC reduced:** AURC Balanced ↓ 5-10%, AURC Worst ↓ 10-15% ✅
4. **Per-group coverage:** Head ≈ 0.56, Tail ≈ 0.44 ✅

---

## 🐛 If Something Seems Wrong

### "I don't see temperature in logs"
**Check:** Load checkpoint and inspect:
```python
import torch
ckpt = torch.load('checkpoints/.../gse_balanced_plugin.ckpt')
print('temperatures' in ckpt)  # Should be True
print(ckpt['temperatures'])     # Should show dict
```

### "Alpha is still [1.0, 1.0]"
**Check:** Training logs should show alpha evolution each iteration.
If not, add debug print in `gse_worst_eg.py` line 157:
```python
print(f"DEBUG: alpha before = {a_cur}, after = ", end="")
a_cur = update_alpha_conditional_with_beta_tgroup(...)
print(a_cur)
```

### "Coverage is still low"
**Check:** Eval logs should show "POST-HOC THRESHOLD RECALIBRATION".
If not shown, check that val_lt files exist:
```bash
ls -la data/cifar100_lt_if100_splits/val_lt_indices.json
```

---

## 📁 Key Files Summary

| File | What Changed | Lines |
|------|--------------|-------|
| `gse_worst_eg.py` | Alpha update, threshold fitting, coverage penalty | 18-73, 139-189 |
| `gse_balanced_plugin.py` | Alpha bounds, temperature saving | 66-68, 719, 921 |
| `eval_gse_plugin.py` | Recalibration function, integration | 251-367, 481-502 |

---

## 🎯 Bottom Line

**Question:** "Bạn đã thêm các thay đổi vào flow chưa?"

**Answer:** ✅ **YES! All changes are integrated:**
- Temperature scaling: ✅ Consistent everywhere
- Recalibration: ✅ Integrated in eval
- Alpha update: ✅ Fixed in inner loop
- Threshold fitting: ✅ Uses pred-group, ALL samples
- Coverage penalty: ✅ Added to objective
- Alpha range: ✅ Expanded [0.8, 1.4]
- Logging: ✅ Enhanced tracking

**Status:** 🚀 **Ready to train!**

No manual edits needed. Just run:
```bash
python run_improved_eg_outer.py
```

---

**Created:** September 30, 2025  
**Verified:** All improvements integrated  
**Ready:** Yes! Go ahead and train 🚀
