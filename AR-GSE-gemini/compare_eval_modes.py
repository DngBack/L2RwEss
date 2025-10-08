"""
Compare different evaluation modes for AR-GSE.

This script runs evaluation with 3 different methodologies:
1. threshold_shift: AR-GSE native (uniform threshold adjustment)
2. rejection_cost: Paper methodology (sweep rejection cost c)
3. alpha_scale: Alternative AR-GSE (scale alpha parameters)

Purpose: Compare AR-GSE native approach vs paper approach fairly.
"""

import subprocess
import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Configure evaluation modes to test
MODES = {
    'threshold_shift': {
        'name': 'AR-GSE Native (Threshold Shift)',
        'description': 'Uniform shift of per-group thresholds: t_k → t_k + δ',
        'color': 'blue',
        'marker': 'o'
    },
    'rejection_cost': {
        'name': 'Paper Methodology (Rejection Cost)',
        'description': 'Sweep rejection cost c in margin: m = α·η + μ - c',
        'color': 'red',
        'marker': 's'
    },
    'alpha_scale': {
        'name': 'AR-GSE Alternative (Alpha Scale)',
        'description': 'Scale alpha parameters: α_k → α_k × scale',
        'color': 'green',
        'marker': '^'
    }
}

OUTPUT_DIR = Path('./comparison_results')
OUTPUT_DIR.mkdir(exist_ok=True)

def run_evaluation(mode):
    """Run evaluation with specified mode."""
    print(f"\n{'='*80}")
    print(f"🚀 Running evaluation: {MODES[mode]['name']}")
    print(f"   {MODES[mode]['description']}")
    print(f"{'='*80}\n")
    
    # Modify config file temporarily
    config_file = Path('./src/train/eval_gse_plugin_auth.py')
    
    # Read current config
    with open(config_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace eval_mode
    import re
    pattern = r"'eval_mode': '[^']+'"
    replacement = f"'eval_mode': '{mode}'"
    modified_content = re.sub(pattern, replacement, content)
    
    # Write back
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(modified_content)
    
    # Run evaluation
    result = subprocess.run(
        ['python', '-m', 'src.train.eval_gse_plugin_auth'],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"❌ Evaluation failed!")
        print(result.stderr)
        return None
    
    print(f"✅ Evaluation completed!")
    
    # Load results
    results_file = Path('./results_worst_eg_improved/cifar100_lt_if100/argse_native_results.json')
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Load RC curve
    rc_file = Path('./results_worst_eg_improved/cifar100_lt_if100/argse_rc_curve.csv')
    rc_df = pd.read_csv(rc_file)
    
    # Save mode-specific results
    mode_dir = OUTPUT_DIR / mode
    mode_dir.mkdir(exist_ok=True)
    
    with open(mode_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    rc_df.to_csv(mode_dir / 'rc_curve.csv', index=False)
    
    return {
        'mode': mode,
        'results': results,
        'rc_df': rc_df
    }

def compare_results(all_results):
    """Compare results across all modes."""
    print("\n" + "="*80)
    print("📊 COMPARISON SUMMARY")
    print("="*80)
    
    comparison_data = []
    
    for data in all_results:
        mode = data['mode']
        results = data['results']
        
        comparison_data.append({
            'Mode': MODES[mode]['name'],
            'Balanced AURC': results['aurc']['balanced'],
            'Balanced CI Lower': results['aurc']['balanced_ci'][0],
            'Balanced CI Upper': results['aurc']['balanced_ci'][1],
            'Worst AURC': results['aurc']['worst'],
            'Worst CI Lower': results['aurc']['worst_ci'][0],
            'Worst CI Upper': results['aurc']['worst_ci'][1],
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    print("\n" + df_comparison.to_string(index=False))
    
    # Save comparison
    df_comparison.to_csv(OUTPUT_DIR / 'aurc_comparison.csv', index=False)
    
    # Plot comparison
    plot_comparison(all_results)
    
    return df_comparison

def plot_comparison(all_results):
    """Plot RC curves for all modes."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Balanced Error
    for data in all_results:
        mode = data['mode']
        rc_df = data['rc_df']
        
        axes[0].plot(
            rc_df['coverage'], 
            rc_df['balanced_error'],
            color=MODES[mode]['color'],
            marker=MODES[mode]['marker'],
            markersize=3,
            linewidth=2,
            label=MODES[mode]['name'],
            alpha=0.7
        )
    
    axes[0].set_xlabel('Coverage', fontsize=13)
    axes[0].set_ylabel('Balanced Error', fontsize=13)
    axes[0].set_title('Balanced Error vs Coverage\n(Comparison of Evaluation Methodologies)', 
                     fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)
    
    # Plot 2: Worst-Group Error
    for data in all_results:
        mode = data['mode']
        rc_df = data['rc_df']
        
        axes[1].plot(
            rc_df['coverage'], 
            rc_df['worst_error'],
            color=MODES[mode]['color'],
            marker=MODES[mode]['marker'],
            markersize=3,
            linewidth=2,
            label=MODES[mode]['name'],
            alpha=0.7
        )
    
    axes[1].set_xlabel('Coverage', fontsize=13)
    axes[1].set_ylabel('Worst-Group Error', fontsize=13)
    axes[1].set_title('Worst-Group Error vs Coverage\n(Comparison of Evaluation Methodologies)', 
                     fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'methodology_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'methodology_comparison.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Saved comparison plot to {OUTPUT_DIR / 'methodology_comparison.png'}")

def main():
    print("="*80)
    print("🔬 AR-GSE EVALUATION METHODOLOGY COMPARISON")
    print("="*80)
    print("\nThis script compares 3 evaluation approaches:")
    print("1. threshold_shift: AR-GSE native (post-hoc threshold adjustment)")
    print("2. rejection_cost: Paper methodology (sweep rejection cost)")
    print("3. alpha_scale: Alternative AR-GSE (scale calibration parameters)")
    print("\nPurpose: Fair comparison between AR-GSE native and paper approaches")
    print("="*80)
    
    all_results = []
    
    # Run each mode
    for mode in MODES.keys():
        result = run_evaluation(mode)
        if result:
            all_results.append(result)
    
    # Compare results
    if all_results:
        df_comparison = compare_results(all_results)
        
        print("\n" + "="*80)
        print("✅ COMPARISON COMPLETE!")
        print("="*80)
        print(f"📁 Results saved to: {OUTPUT_DIR}")
        print(f"   • AURC comparison: aurc_comparison.csv")
        print(f"   • Visualization: methodology_comparison.png/pdf")
        print(f"   • Individual results: {', '.join(MODES.keys())}/")
        print("="*80)

if __name__ == '__main__':
    main()
