#!/bin/bash
# Quick verification script - no Python imports needed

echo "======================================================================"
echo "🔍 VERIFYING AURC IMPROVEMENTS INTEGRATION"
echo "======================================================================"
echo ""

all_pass=true

# 1. Temperature Scaling
echo "1️⃣ Checking Temperature Scaling..."
if grep -q "apply_per_expert_temperature(logits, expert_names, temperatures)" src/train/gse_balanced_plugin.py && \
   grep -q "apply_per_expert_temperature(logits, expert_names, temperatures)" src/train/eval_gse_plugin.py && \
   grep -q "'temperatures': temperatures" src/train/gse_balanced_plugin.py; then
    echo "   ✅ Temperature scaling: INTEGRATED"
else
    echo "   ❌ Temperature scaling: MISSING"
    all_pass=false
fi

# 2. Recalibration Function
echo "2️⃣ Checking Recalibration on val_lt..."
if grep -q "def recalibrate_thresholds_on_val" src/train/eval_gse_plugin.py && \
   grep -q "recalibrate_thresholds_on_val(" src/train/eval_gse_plugin.py && \
   grep -q "pred_groups_val = cg_cpu\[yhat_val\]" src/train/eval_gse_plugin.py; then
    echo "   ✅ Recalibration: INTEGRATED"
else
    echo "   ❌ Recalibration: MISSING"
    all_pass=false
fi

# 3. Alpha Update
echo "3️⃣ Checking Alpha Update..."
if grep -q "def update_alpha_conditional_with_beta_tgroup" src/train/gse_worst_eg.py && \
   grep -q "update_alpha_conditional_with_beta_tgroup(" src/train/gse_worst_eg.py; then
    echo "   ✅ Alpha update: INTEGRATED"
    # Check old placeholder is removed
    if grep -q "0.9 \* a_cur + 0.1 \* torch.ones" src/train/gse_worst_eg.py; then
        echo "   ⚠️  Old placeholder still present!"
        all_pass=false
    fi
else
    echo "   ❌ Alpha update: MISSING"
    all_pass=false
fi

# 4. Threshold Fitting
echo "4️⃣ Checking Threshold Fitting (pred-group, ALL samples)..."
if grep -q "pred_groups_S1 = class_to_group\[preds_S1\]" src/train/gse_worst_eg.py && \
   grep -q "mk = (pred_groups_S1 == k)" src/train/gse_worst_eg.py; then
    echo "   ✅ Threshold fitting: INTEGRATED"
    # Check correct_mask is NOT used
    if grep -q "correct_mask = (preds_S1 == y_S1" src/train/gse_worst_eg.py; then
        echo "   ⚠️  Still using correct_mask filter!"
        all_pass=false
    fi
else
    echo "   ❌ Threshold fitting: MISSING"
    all_pass=false
fi

# 5. Coverage Penalty
echo "5️⃣ Checking Coverage Penalty..."
if grep -q "cov_penalty = sum((cov_by_pred_group" src/train/gse_worst_eg.py && \
   grep -q "score = w_err + 5.0 \* cov_penalty" src/train/gse_worst_eg.py; then
    echo "   ✅ Coverage penalty: INTEGRATED"
else
    echo "   ❌ Coverage penalty: MISSING"
    all_pass=false
fi

# 6. Alpha Range
echo "6️⃣ Checking Alpha Range [0.80, 1.40]..."
if grep -q "'alpha_min': 0.80" src/train/gse_balanced_plugin.py && \
   grep -q "'alpha_max': 1.40" src/train/gse_balanced_plugin.py && \
   grep -q "'alpha_steps': 7" src/train/gse_balanced_plugin.py; then
    echo "   ✅ Alpha range: EXPANDED"
else
    echo "   ❌ Alpha range: NOT UPDATED"
    all_pass=false
fi

# 7. Enhanced Logging
echo "7️⃣ Checking Enhanced Logging..."
if grep -q "print(f\"   α_t = {" src/train/gse_worst_eg.py && \
   grep -q "print(f\"   t_group = {" src/train/gse_worst_eg.py; then
    echo "   ✅ Enhanced logging: INTEGRATED"
else
    echo "   ❌ Enhanced logging: MISSING"
    all_pass=false
fi

echo ""
echo "======================================================================"
if [ "$all_pass" = true ]; then
    echo "🎉 ALL IMPROVEMENTS VERIFIED - READY TO TRAIN!"
    echo ""
    echo "Run:"
    echo "  python run_improved_eg_outer.py"
    echo "  python -m src.train.eval_gse_plugin"
    echo ""
    exit 0
else
    echo "⚠️  SOME IMPROVEMENTS MISSING - CHECK ABOVE"
    echo ""
    exit 1
fi
