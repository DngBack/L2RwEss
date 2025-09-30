#!/usr/bin/env python3
"""
Verify that all AURC improvements are properly integrated.
"""
import torch
import sys
from pathlib import Path

def check_gse_worst_eg():
    """Check gse_worst_eg.py has all improvements"""
    print("\n" + "="*60)
    print("Checking gse_worst_eg.py...")
    print("="*60)
    
    with open('src/train/gse_worst_eg.py', 'r') as f:
        content = f.read()
    
    checks = {
        "✅ Has update_alpha_conditional_with_beta_tgroup": 
            "def update_alpha_conditional_with_beta_tgroup" in content,
        
        "✅ Fit t_k by pred-group on ALL samples":
            "pred_groups_S1 = class_to_group[yhat_S1]" in content and
            "mk = (pred_groups_S1 == k)" in content,
        
        "✅ Alpha update called in inner loop":
            "a_cur = update_alpha_conditional_with_beta_tgroup(" in content,
        
        "✅ Coverage penalty in objective":
            "cov_penalty = sum((cov_by_pred_group[k] - target_cov_by_group[k])**2" in content and
            "score = w_err + 5.0 * cov_penalty" in content,
        
        "✅ Alpha logging in EG-outer":
            'print(f"   α_t = {[f\'{a:.4f}\' for a in a_t.cpu().tolist()]}"' in content,
    }
    
    all_pass = True
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check}")
        if not passed:
            all_pass = False
    
    return all_pass

def check_gse_balanced_plugin():
    """Check gse_balanced_plugin.py has all improvements"""
    print("\n" + "="*60)
    print("Checking gse_balanced_plugin.py...")
    print("="*60)
    
    with open('src/train/gse_balanced_plugin.py', 'r') as f:
        content = f.read()
    
    checks = {
        "✅ Temperature loading from selective checkpoint":
            "temperatures = sel_ckpt.get('temperatures', None)" in content,
        
        "✅ Temperature passed to cache_eta_mix":
            "temperatures=temperatures)" in content,
        
        "✅ Temperature saved in checkpoint":
            "'temperatures': temperatures" in content,
        
        "✅ Alpha bounds [0.80, 1.40]":
            "alpha_min': 0.80" in content and "alpha_max': 1.40" in content,
        
        "✅ Alpha steps = 7":
            "alpha_steps': 7" in content,
        
        "✅ Apply temperature in cache_eta_mix":
            "apply_per_expert_temperature(logits, expert_names, temperatures)" in content,
    }
    
    all_pass = True
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check}")
        if not passed:
            all_pass = False
    
    return all_pass

def check_eval_gse_plugin():
    """Check eval_gse_plugin.py has all improvements"""
    print("\n" + "="*60)
    print("Checking eval_gse_plugin.py...")
    print("="*60)
    
    with open('src/train/eval_gse_plugin.py', 'r') as f:
        content = f.read()
    
    checks = {
        "✅ Temperature helper function":
            "def apply_per_expert_temperature" in content,
        
        "✅ Temperature in get_mixture_posteriors":
            "logits = apply_per_expert_temperature(logits, expert_names, temperatures)" in content,
        
        "✅ Temperature in recalibrate_thresholds_on_val":
            "logits_val_scaled = apply_per_expert_temperature(logits_val, expert_names, temperatures)" in content,
        
        "✅ Recalibration uses predicted groups":
            "pred_groups_val = cg_cpu[yhat_val]" in content,
        
        "✅ Load val_data function":
            "def load_val_data" in content,
        
        "✅ Recalibration integrated in main":
            "t_recalibrated = recalibrate_thresholds_on_val(" in content,
    }
    
    all_pass = True
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check}")
        if not passed:
            all_pass = False
    
    return all_pass

def check_config_consistency():
    """Check CONFIG consistency across files"""
    print("\n" + "="*60)
    print("Checking CONFIG consistency...")
    print("="*60)
    
    # Import and check config
    sys.path.insert(0, str(Path.cwd()))
    from src.train.gse_balanced_plugin import CONFIG
    
    checks = {
        "✅ alpha_min = 0.80": CONFIG['plugin_params']['alpha_min'] == 0.80,
        "✅ alpha_max = 1.40": CONFIG['plugin_params']['alpha_max'] == 1.40,
        "✅ alpha_steps = 7": CONFIG['plugin_params']['alpha_steps'] == 7,
        "✅ use_eg_outer = True": CONFIG['plugin_params']['use_eg_outer'] == True,
        "✅ use_conditional_alpha = True": CONFIG['plugin_params']['use_conditional_alpha'] == True,
    }
    
    all_pass = True
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check}")
        if not passed:
            all_pass = False
    
    return all_pass

def main():
    print("\n" + "="*60)
    print("VERIFYING AURC IMPROVEMENTS INTEGRATION")
    print("="*60)
    
    results = []
    results.append(("gse_worst_eg.py", check_gse_worst_eg()))
    results.append(("gse_balanced_plugin.py", check_gse_balanced_plugin()))
    results.append(("eval_gse_plugin.py", check_eval_gse_plugin()))
    results.append(("CONFIG consistency", check_config_consistency()))
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_pass = all(passed for _, passed in results)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")
    
    if all_pass:
        print("\n🎉 All improvements are properly integrated!")
        print("\nReady to run:")
        print("  python run_improved_eg_outer.py")
        print("  python -m src.train.eval_gse_plugin")
        return 0
    else:
        print("\n⚠️  Some improvements are missing. Please check the failed items above.")
        return 1

if __name__ == '__main__':
    exit(main())
