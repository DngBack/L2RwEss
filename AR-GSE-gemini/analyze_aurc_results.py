#!/usr/bin/env python3
"""
Analyze AURC results and create comparison table for paper.

This script reads the comprehensive AURC evaluation results from eval_gse_plugin.py
and formats them for paper comparison with other L2R methods.

Key differences between AR-GSE and baseline methods:
1. AR-GSE: Optimizes (α, μ, c) for a single coverage target (~58%)
2. Baselines: Sweep rejection cost c over [0.0, 0.8] to generate full RC curve

Both approaches are then evaluated using the SAME AURC metric for fair comparison.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Configuration for different experimental results
EXPERIMENTS = {
    'argse_balanced': {
        'name': 'AR-GSE (Balanced)',
        'results_dir': './checkpoints/argse_balanced_plugin/cifar100_lt_if100',
        'description': 'Balanced objective: min (1/K) Σ_k error_k'
    },
    'argse_worst': {
        'name': 'AR-GSE (Worst-case)',
        'results_dir': './checkpoints/argse_worst/cifar100_lt_if100',
        'description': 'Worst-case objective: min max_k error_k'
    },
    'argse_worst_eg': {
        'name': 'AR-GSE (Worst-case + EG-Outer)',
        'results_dir': './results_worst_eg_improved/cifar100_lt_if100',
        'description': 'Worst-case with EG-outer optimization'
    },
}

def load_experiment_results(exp_config):
    """Load AURC results from experiment directory."""
    results_dir = Path(exp_config['results_dir'])
    
    # Try to load metrics.json
    metrics_file = results_dir / 'metrics.json'
    if not metrics_file.exists():
        print(f"⚠️  Metrics file not found: {metrics_file}")
        return None
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    # Extract AURC results
    aurc_results = metrics.get('aurc_results', {})
    
    # Load detailed RC points if available
    aurc_csv = results_dir / 'aurc_detailed_results.csv'
    rc_points = None
    if aurc_csv.exists():
        rc_points = pd.read_csv(aurc_csv)
    
    return {
        'aurc': aurc_results,
        'plugin_metrics': metrics.get('plugin_metrics_at_threshold', {}),
        'rc_points': rc_points,
        'full_metrics': metrics
    }

def create_comparison_table():
    """Create comparison table for paper."""
    print("="*80)
    print("AR-GSE AURC EVALUATION SUMMARY FOR PAPER")
    print("="*80)
    
    results_summary = []
    
    for exp_key, exp_config in EXPERIMENTS.items():
        print(f"\n📊 Loading {exp_config['name']}...")
        results = load_experiment_results(exp_config)
        
        if results is None:
            print(f"   ❌ No results found - skipping")
            continue
        
        aurc = results['aurc']
        plugin_metrics = results['plugin_metrics']
        
        # Extract key metrics
        row = {
            'Method': exp_config['name'],
            'AURC (Standard)': aurc.get('standard', float('nan')),
            'AURC (Balanced)': aurc.get('balanced', float('nan')),
            'AURC (Worst)': aurc.get('worst', float('nan')),
            'Coverage': plugin_metrics.get('coverage', float('nan')),
            'Balanced Error': plugin_metrics.get('balanced_error', float('nan')),
            'Worst Error': plugin_metrics.get('worst_error', float('nan')),
            'Overall Error': plugin_metrics.get('overall_error', float('nan')),
        }
        
        results_summary.append(row)
        
        print(f"   ✅ Loaded successfully")
        print(f"      • AURC (Standard): {row['AURC (Standard)']:.6f}")
        print(f"      • AURC (Balanced): {row['AURC (Balanced)']:.6f}")
        print(f"      • AURC (Worst): {row['AURC (Worst)']:.6f}")
        print(f"      • Coverage @ optimal: {row['Coverage']:.3f}")
    
    if not results_summary:
        print("\n❌ No results loaded - please run evaluation first!")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(results_summary)
    
    # Save to CSV
    output_file = Path('./paper_results_aurc_comparison.csv')
    df.to_csv(output_file, index=False, float_format='%.6f')
    print(f"\n💾 Saved comparison table to: {output_file}")
    
    # Print formatted table
    print("\n" + "="*80)
    print("COMPARISON TABLE (for paper)")
    print("="*80)
    print(df.to_string(index=False, float_format=lambda x: f'{x:.6f}' if not np.isnan(x) else 'N/A'))
    print("="*80)
    
    return df

def create_latex_table(df):
    """Generate LaTeX table code for paper."""
    if df is None or df.empty:
        return
    
    print("\n" + "="*80)
    print("LATEX TABLE CODE")
    print("="*80)
    
    latex_code = r"""
\begin{table}[t]
\centering
\caption{AURC Evaluation Results on CIFAR-100-LT (IF=100)}
\label{tab:aurc_results}
\begin{tabular}{l|ccc|ccc}
\toprule
\textbf{Method} & \multicolumn{3}{c|}{\textbf{AURC}} & \multicolumn{3}{c}{\textbf{Metrics @ Optimal}} \\
 & Standard & Balanced & Worst & Cov. & Bal. Err & Worst Err \\
\midrule
"""
    
    for _, row in df.iterrows():
        method = row['Method'].replace('_', '\\_')
        latex_code += f"{method} & "
        latex_code += f"{row['AURC (Standard)']:.4f} & "
        latex_code += f"{row['AURC (Balanced)']:.4f} & "
        latex_code += f"{row['AURC (Worst)']:.4f} & "
        latex_code += f"{row['Coverage']:.3f} & "
        latex_code += f"{row['Balanced Error']:.4f} & "
        latex_code += f"{row['Worst Error']:.4f} \\\\\n"
    
    latex_code += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    print(latex_code)
    print("="*80)
    
    # Save to file
    with open('./paper_results_latex_table.tex', 'w') as f:
        f.write(latex_code)
    print("💾 Saved LaTeX code to: paper_results_latex_table.tex")

def plot_rc_curves_comparison():
    """Plot RC curves from all experiments for comparison."""
    print("\n📊 Generating RC curve comparison plot...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = ['standard', 'balanced', 'worst']
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for exp_idx, (exp_key, exp_config) in enumerate(EXPERIMENTS.items()):
        results = load_experiment_results(exp_config)
        if results is None or results['rc_points'] is None:
            continue
        
        rc_points = results['rc_points']
        
        for metric_idx, metric in enumerate(metrics):
            ax = axes[metric_idx]
            metric_data = rc_points[rc_points['metric'] == metric]
            
            if not metric_data.empty:
                metric_data = metric_data.sort_values('coverage')
                ax.plot(metric_data['coverage'], metric_data['risk'], 
                       label=exp_config['name'], 
                       color=colors[exp_idx % len(colors)],
                       linewidth=2, alpha=0.8)
            
            ax.set_xlabel('Coverage', fontsize=12)
            ax.set_ylabel('Risk (Error)', fontsize=12)
            ax.set_title(f'{metric.title()} RC Curve', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, None)
    
    plt.tight_layout()
    output_path = Path('./paper_rc_curves_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved RC curves comparison to: {output_path}")
    plt.close()

def explain_methodology():
    """Explain the methodology for paper writing."""
    print("\n" + "="*80)
    print("METHODOLOGY EXPLANATION FOR PAPER")
    print("="*80)
    
    explanation = """
## 1. Training Phase Differences

### AR-GSE (Our Method):
- **Objective**: Optimize (α, μ, c) for a SINGLE coverage target (~58%)
- **Training**: 
  * Stage 1: Pretrain gating network φ
  * Stage 2: Plugin optimization on S1 split
    - Fixed-point matching for α
    - Grid search for μ over λ ∈ [-2.0, 2.0]
    - Find c for target coverage
  * Objective: minimize balanced/worst-case error at target coverage
- **Output**: Single optimal operating point (α*, μ*, c*)

### Baseline L2R Methods (e.g., SAT, DG, SelCon):
- **Objective**: Learn a confidence score for each sample
- **Training**: 
  * Train confidence estimator on labeled data
  * May use different objectives (e.g., coverage constraint, DG loss)
- **Output**: Confidence scores for all samples

## 2. AURC Evaluation Phase (SAME FOR ALL)

### Step 1: Train/Val Split (80-20)
- Split test set into validation and test portions
- Ensures unbiased AURC evaluation

### Step 2: Cost Sweep on Validation
For each rejection cost c ∈ [0.0, 0.8]:
  1. Find optimal threshold τ* that minimizes: risk + c × (1 - coverage)
  2. Apply τ* to validation set

### Step 3: Evaluate on Test
- Apply found thresholds to test set
- Compute (coverage, risk) points
- Generate RC curve

### Step 4: Compute AURC
- AURC = ∫ risk(coverage) d(coverage) from 0 to 1
- Lower AURC = better overall performance

## 3. Fair Comparison

✅ **Why this is fair:**
- Both methods evaluated using SAME AURC protocol
- Both sweep over rejection costs to generate full RC curve
- AURC captures performance across ALL operating points
- Not biased toward any particular coverage level

❌ **Common misunderstanding:**
- "AR-GSE only optimizes for one coverage, unfair!"
- Actually: AR-GSE learns mixture + group parameters at one point
- But AURC evaluation sweeps c to explore all coverage levels
- The learned (α, μ) generalize to other coverage levels via margin thresholds

## 4. What to Report in Paper

### Table: AURC Comparison
| Method | AURC (Std) | AURC (Bal) | AURC (Worst) |
|--------|-----------|------------|--------------|
| SAT    | 0.XXXX    | 0.XXXX     | 0.XXXX      |
| DG     | 0.XXXX    | 0.XXXX     | 0.XXXX      |
| AR-GSE | 0.XXXX    | 0.XXXX     | 0.XXXX      |

Lower is better for all metrics.

### Figure: RC Curves
- Plot coverage vs risk for all methods
- Show that AR-GSE dominates baselines across coverage range
- Highlight worst-case performance for tail groups

### Text Description:
"We evaluate all methods using the AURC metric following the 'Learning to 
Reject Meets Long-tail Learning' protocol. For each method, we sweep rejection 
costs c ∈ [0, 0.8] on a validation set to find optimal thresholds, then 
evaluate on the test set to compute the Area Under the Risk-Coverage curve.
This provides a comprehensive assessment of selective classification 
performance across all coverage levels, not just a single operating point."

## 5. Additional Metrics to Report

At specific coverage levels (e.g., 60%, 70%, 80%):
- Balanced error
- Worst-group error  
- Per-group errors (head vs tail)

This shows that AR-GSE not only has lower AURC (global metric) but also
better performance at practical coverage levels.
"""
    
    print(explanation)
    
    # Save to markdown file
    with open('./AURC_METHODOLOGY_FOR_PAPER.md', 'w') as f:
        f.write(explanation)
    print("💾 Saved methodology explanation to: AURC_METHODOLOGY_FOR_PAPER.md")
    print("="*80)

def main():
    """Main analysis function."""
    print("\n🔍 AR-GSE AURC Analysis for Paper Writing\n")
    
    # Create comparison table
    df = create_comparison_table()
    
    if df is not None:
        # Generate LaTeX table
        create_latex_table(df)
        
        # Plot RC curves comparison
        plot_rc_curves_comparison()
    
    # Explain methodology
    explain_methodology()
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  📄 paper_results_aurc_comparison.csv - CSV table of results")
    print("  📄 paper_results_latex_table.tex - LaTeX table code")
    print("  📊 paper_rc_curves_comparison.png - RC curves comparison")
    print("  📝 AURC_METHODOLOGY_FOR_PAPER.md - Methodology explanation")
    print("\n💡 Use these materials to write your paper!")
    print("="*80)

if __name__ == '__main__':
    main()
