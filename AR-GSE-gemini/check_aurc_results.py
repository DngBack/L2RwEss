#!/usr/bin/env python3
"""
Quick checker: Show current AURC results from your checkpoints.
Run this to see what numbers you have for paper writing.
"""

import json
from pathlib import Path
import sys

def check_checkpoint_results(checkpoint_path, result_dir):
    """Check if evaluation results exist for a checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    result_dir = Path(result_dir)
    
    print(f"\n{'='*80}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Results dir: {result_dir}")
    print(f"{'='*80}")
    
    # Check if checkpoint exists
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found!")
        return False
    
    print(f"✅ Checkpoint exists")
    
    # Load checkpoint info
    try:
        ckpt = json.loads(checkpoint_path.read_text()) if checkpoint_path.suffix == '.json' else None
        if ckpt:
            print(f"   • Format: JSON config")
    except:
        print(f"   • Format: PyTorch checkpoint")
    
    # Check for evaluation results
    metrics_file = result_dir / 'metrics.json'
    aurc_file = result_dir / 'aurc_detailed_results.csv'
    
    if not metrics_file.exists():
        print(f"❌ No evaluation results found!")
        print(f"   Run: python -m src.train.eval_gse_plugin")
        return False
    
    print(f"✅ Evaluation results found")
    
    # Load and display results
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    # AURC results
    if 'aurc_results' in metrics:
        aurc = metrics['aurc_results']
        print("\n📊 COMPREHENSIVE AURC RESULTS (Cost Sweep [0.0, 0.8]):")
        
        # Full range [0, 1]
        print("\n   🔵 AURC (Full Range 0-1):")
        print("   ┌─────────────────────────────────┐")
        for key, value in aurc.items():
            if '_02_10' not in key:
                print(f"   │ {key.upper():12} AURC: {value:.6f} │")
        print("   └─────────────────────────────────┘")
        
        # Range [0.2, 1.0]
        print("\n   🟢 AURC (Practical Range 0.2-1):")
        print("   ┌─────────────────────────────────┐")
        for key, value in aurc.items():
            if '_02_10' in key:
                metric_name = key.replace('_02_10', '')
                print(f"   │ {metric_name.upper():12} AURC: {value:.6f} │")
        print("   └─────────────────────────────────┘")
        
        print("\n   📝 These are the numbers to report in paper!")
    
    # Traditional AURC (from RC curve)
    if 'aurc_balanced' in metrics:
        print(f"\n📈 TRADITIONAL AURC (Single margin threshold):")
        print(f"   • Balanced: {metrics['aurc_balanced']:.6f}")
        print(f"   • Worst:    {metrics.get('aurc_worst', 'N/A')}")
    
    # Plugin metrics at optimal threshold
    if 'plugin_metrics_at_threshold' in metrics:
        pm = metrics['plugin_metrics_at_threshold']
        print(f"\n🎯 METRICS AT OPTIMAL THRESHOLD:")
        print(f"   • Coverage:       {pm.get('coverage', 'N/A'):.3f}")
        print(f"   • Balanced Error: {pm.get('balanced_error', 'N/A'):.4f}")
        print(f"   • Worst Error:    {pm.get('worst_error', 'N/A'):.4f}")
        print(f"   • Overall Error:  {pm.get('overall_error', 'N/A'):.4f}")
        
        if 'group_errors' in pm:
            print(f"   • Group errors:   {[f'{e:.4f}' for e in pm['group_errors']]}")
    
    # ECE
    if 'ece' in metrics:
        print(f"\n📐 CALIBRATION:")
        print(f"   • ECE: {metrics['ece']:.4f}")
    
    # Check for detailed data
    if aurc_file.exists():
        print(f"\n✅ Detailed RC points available: {aurc_file}")
    
    return True

def main():
    """Check all configured experiments."""
    print("\n" + "="*80)
    print("AR-GSE AURC RESULTS CHECKER")
    print("="*80)
    print("\nThis script checks what evaluation results you currently have.")
    print("Use these numbers for paper comparison with baselines.\n")
    
    # Define experiments to check
    experiments = [
        {
            'name': 'AR-GSE Worst-case + EG-Outer (Improved)',
            'checkpoint': './checkpoints/argse_worst_eg_improved/cifar100_lt_if100/gse_balanced_plugin.ckpt',
            'results': './results_worst_eg_improved/cifar100_lt_if100'
        },
        {
            'name': 'AR-GSE Balanced',
            'checkpoint': './checkpoints/argse_balanced_plugin/cifar100_lt_if100/gse_balanced_plugin.ckpt',
            'results': './checkpoints/argse_balanced_plugin/cifar100_lt_if100'
        },
        {
            'name': 'AR-GSE Worst-case',
            'checkpoint': './checkpoints/argse_worst/cifar100_lt_if100/gse_balanced_plugin.ckpt',
            'results': './checkpoints/argse_worst/cifar100_lt_if100'
        },
    ]
    
    results_found = []
    
    for exp in experiments:
        print(f"\n{'#'*80}")
        print(f"# {exp['name']}")
        print(f"{'#'*80}")
        
        found = check_checkpoint_results(exp['checkpoint'], exp['results'])
        if found:
            results_found.append(exp['name'])
    
    # Summary
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    if results_found:
        print(f"\n✅ Found evaluation results for {len(results_found)} experiment(s):")
        for name in results_found:
            print(f"   • {name}")
        
        print(f"\n📝 TO USE IN PAPER:")
        print(f"   1. Copy the 'COMPREHENSIVE AURC RESULTS' numbers above")
        print(f"   2. These are computed using cost sweep [0.0, 0.8]")
        print(f"   3. Compare with baseline AURC from literature or your implementation")
        print(f"   4. Lower AURC = better performance")
        
        print(f"\n📊 TO GENERATE COMPARISON TABLE:")
        print(f"   Run: python analyze_aurc_results.py")
        
    else:
        print(f"\n❌ No evaluation results found!")
        print(f"\n📋 TODO:")
        print(f"   1. Train your model: python run_improved_eg_outer.py")
        print(f"   2. Evaluate with AURC: python -m src.train.eval_gse_plugin")
        print(f"   3. Run this script again to check results")
    
    print(f"\n{'='*80}")
    
    return len(results_found) > 0

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
