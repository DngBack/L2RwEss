# src/train/eval_paper_benchmark.py
"""
Paper Benchmark Evaluation - Direct comparison with published results

Compares AR-GSE with:
1. SelectiveNet (Geifman & El-Yaniv, 2019)
2. Learning to Reject Meets Long-tail Learning (Cao et al., 2024)

Standard metrics:
- AURC (Area Under Risk-Coverage curve) from coverage 0.2 to 1.0
- E-AURC (Excess AURC over random baseline)
- Coverage-Accuracy trade-offs at specific points
- Per-group analysis for fairness
"""

import torch
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple

from src.train.eval_comprehensive import ComprehensiveEvaluator

class PaperBenchmarkEvaluator(ComprehensiveEvaluator):
    """
    Evaluator specifically designed for paper benchmarks.
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.paper_results = self._load_paper_results()
    
    def _load_paper_results(self) -> Dict:
        """
        Load published paper results for comparison.
        
        Note: Fill in actual paper results when available.
        """
        return {
            'selectivenet': {
                'cifar100': {
                    'aurc_0.2_1.0': None,  # Fill with paper values
                    'accuracy_at_90': None,
                }
            },
            'l2r_longtail': {
                'cifar100_lt': {
                    'aurc_standard': None,
                    'aurc_balanced': None,
                    'aurc_worst': None,
                }
            }
        }
    
    def evaluate_at_specific_coverages(self, confidence_scores: torch.Tensor) -> Dict:
        """
        Evaluate at specific coverage points commonly used in papers.
        
        Standard coverage points: 0.6, 0.7, 0.8, 0.9, 0.95
        """
        target_coverages = [0.6, 0.7, 0.8, 0.9, 0.95]
        results = {}
        
        # Sort by confidence
        sorted_indices = torch.argsort(confidence_scores, descending=True)
        
        for target_cov in target_coverages:
            num_to_accept = max(1, int(self.num_test_samples * target_cov))
            num_to_accept = min(num_to_accept, self.num_test_samples)
            
            accepted_mask = torch.zeros_like(self.test_labels, dtype=torch.bool)
            accepted_mask[sorted_indices[:num_to_accept]] = True
            
            metrics = self._compute_error_metrics(accepted_mask)
            
            results[f'cov_{target_cov:.2f}'] = {
                'actual_coverage': metrics['coverage'],
                'standard_accuracy': 1 - metrics['standard_error'],
                'balanced_accuracy': 1 - metrics['balanced_error'],
                'worst_accuracy': 1 - metrics['worst_error'],
                'standard_error': metrics['standard_error'],
                'balanced_error': metrics['balanced_error'],
                'worst_error': metrics['worst_error'],
                'head_error': metrics['group_errors'][0],
                'tail_error': metrics['group_errors'][1],
            }
        
        return results
    
    def compute_selective_risk_metrics(self, rc_df: pd.DataFrame) -> Dict:
        """
        Compute comprehensive selective risk metrics following paper standards.
        """
        metrics = {}
        
        # Standard range: 0.2 to 1.0
        for error_type in ['standard_error', 'balanced_error', 'worst_error']:
            # AURC
            aurc = self.compute_aurc(rc_df, error_type, 0.2, 1.0)
            
            # E-AURC
            eaurc, aurc_method, aurc_random = self.compute_eaurc(rc_df, error_type, 0.2, 1.0)
            
            # Normalized E-AURC (percentage reduction)
            if aurc_random > 0:
                normalized_eaurc = eaurc / aurc_random
            else:
                normalized_eaurc = 0.0
            
            metrics[error_type] = {
                'aurc': aurc,
                'eaurc': eaurc,
                'aurc_random': aurc_random,
                'normalized_eaurc': normalized_eaurc,
            }
        
        # Additional range: 0.0 to 1.0 (full range)
        for error_type in ['standard_error', 'balanced_error', 'worst_error']:
            aurc_full = self.compute_aurc(rc_df, error_type, 0.0, 1.0)
            metrics[error_type]['aurc_full'] = aurc_full
        
        return metrics
    
    def create_paper_comparison_table(self, our_results: Dict) -> pd.DataFrame:
        """
        Create a comparison table with paper benchmarks.
        """
        rows = []
        
        # Our method
        rows.append({
            'Method': 'AR-GSE (Ours)',
            'AURC-Std': our_results['aurc_standard'],
            'AURC-Bal': our_results['aurc_balanced'],
            'AURC-Worst': our_results['aurc_worst'],
            'E-AURC-Std': our_results['eaurc_standard'],
            'E-AURC-Bal': our_results['eaurc_balanced'],
            'E-AURC-Worst': our_results['eaurc_worst'],
        })
        
        # Add paper baselines (if available)
        if self.paper_results['selectivenet']['cifar100']['aurc_0.2_1.0'] is not None:
            rows.append({
                'Method': 'SelectiveNet',
                'AURC-Std': self.paper_results['selectivenet']['cifar100']['aurc_0.2_1.0'],
                'AURC-Bal': '-',
                'AURC-Worst': '-',
                'E-AURC-Std': '-',
                'E-AURC-Bal': '-',
                'E-AURC-Worst': '-',
            })
        
        if self.paper_results['l2r_longtail']['cifar100_lt']['aurc_standard'] is not None:
            rows.append({
                'Method': 'L2R-LongTail',
                'AURC-Std': self.paper_results['l2r_longtail']['cifar100_lt']['aurc_standard'],
                'AURC-Bal': self.paper_results['l2r_longtail']['cifar100_lt']['aurc_balanced'],
                'AURC-Worst': self.paper_results['l2r_longtail']['cifar100_lt']['aurc_worst'],
                'E-AURC-Std': '-',
                'E-AURC-Bal': '-',
                'E-AURC-Worst': '-',
            })
        
        df = pd.DataFrame(rows)
        return df
    
    def plot_paper_style_figures(self, results: Dict):
        """
        Create publication-quality figures following paper styles.
        """
        # Main method for comparison
        main_method = 'gse_margin'
        rc_df = results[main_method]['rc_curve']
        
        # Create multi-panel figure
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # ===== Panel 1: Standard RC Curve (0.2-1.0) =====
        ax1 = fig.add_subplot(gs[0, 0])
        mask = (rc_df['coverage'] >= 0.2) & (rc_df['coverage'] <= 1.0)
        filtered = rc_df[mask]
        
        ax1.plot(filtered['coverage'], filtered['standard_error'], 'b-', linewidth=3, label='Standard')
        ax1.fill_between(filtered['coverage'], 0, filtered['standard_error'], alpha=0.2, color='blue')
        
        ax1.set_xlabel('Coverage', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Selective Risk', fontsize=14, fontweight='bold')
        ax1.set_title('(a) Standard Error RC Curve', fontsize=16, fontweight='bold', loc='left')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(fontsize=12)
        ax1.set_xlim(0.2, 1.0)
        ax1.set_ylim(0, max(filtered['standard_error'].max() * 1.1, 0.5))
        
        # Add AURC annotation
        aurc_std = results[main_method]['aurc_standard']
        ax1.text(0.95, 0.95, f'AURC = {aurc_std:.4f}', 
                transform=ax1.transAxes, fontsize=12, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # ===== Panel 2: Balanced RC Curve =====
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(filtered['coverage'], filtered['balanced_error'], 'g-', linewidth=3, label='Balanced')
        ax2.fill_between(filtered['coverage'], 0, filtered['balanced_error'], alpha=0.2, color='green')
        
        ax2.set_xlabel('Coverage', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Selective Risk', fontsize=14, fontweight='bold')
        ax2.set_title('(b) Balanced Error RC Curve', fontsize=16, fontweight='bold', loc='left')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend(fontsize=12)
        ax2.set_xlim(0.2, 1.0)
        ax2.set_ylim(0, max(filtered['balanced_error'].max() * 1.1, 0.5))
        
        aurc_bal = results[main_method]['aurc_balanced']
        ax2.text(0.95, 0.95, f'AURC = {aurc_bal:.4f}', 
                transform=ax2.transAxes, fontsize=12,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # ===== Panel 3: Worst-Group RC Curve =====
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(filtered['coverage'], filtered['worst_error'], 'r-', linewidth=3, label='Worst-Group')
        ax3.fill_between(filtered['coverage'], 0, filtered['worst_error'], alpha=0.2, color='red')
        
        ax3.set_xlabel('Coverage', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Selective Risk', fontsize=14, fontweight='bold')
        ax3.set_title('(c) Worst-Group Error RC Curve', fontsize=16, fontweight='bold', loc='left')
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.legend(fontsize=12)
        ax3.set_xlim(0.2, 1.0)
        ax3.set_ylim(0, max(filtered['worst_error'].max() * 1.1, 0.5))
        
        aurc_wst = results[main_method]['aurc_worst']
        ax3.text(0.95, 0.95, f'AURC = {aurc_wst:.4f}',
                transform=ax3.transAxes, fontsize=12,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # ===== Panel 4: Per-Group Comparison =====
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(filtered['coverage'], filtered['head_error'], 'b-', linewidth=3, label='Head Group', marker='o', markersize=4, markevery=10)
        ax4.plot(filtered['coverage'], filtered['tail_error'], 'r-', linewidth=3, label='Tail Group', marker='s', markersize=4, markevery=10)
        
        ax4.set_xlabel('Coverage', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Group Error Rate', fontsize=14, fontweight='bold')
        ax4.set_title('(d) Per-Group Selective Errors', fontsize=16, fontweight='bold', loc='left')
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.legend(fontsize=12, loc='upper right')
        ax4.set_xlim(0.2, 1.0)
        
        # ===== Panel 5: AURC Comparison Bar Chart =====
        ax5 = fig.add_subplot(gs[1, 1])
        
        metrics = ['Standard', 'Balanced', 'Worst-Group']
        aurc_values = [aurc_std, aurc_bal, aurc_wst]
        eaurc_values = [
            results[main_method]['eaurc_standard'],
            results[main_method]['eaurc_balanced'],
            results[main_method]['eaurc_worst']
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax5.bar(x - width/2, aurc_values, width, label='AURC', 
                       color=['blue', 'green', 'red'], alpha=0.7, edgecolor='black', linewidth=2)
        bars2 = ax5.bar(x + width/2, eaurc_values, width, label='E-AURC',
                       color=['lightblue', 'lightgreen', 'lightcoral'], alpha=0.7, edgecolor='black', linewidth=2)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add oracle line if available
        if 'oracle' in results:
            oracle_values = [
                results['oracle']['aurc_standard'],
                results['oracle']['aurc_balanced'],
                results['oracle']['aurc_worst']
            ]
            ax5.plot(x, oracle_values, 'k--', linewidth=2, marker='*', markersize=12, label='Oracle')
        
        ax5.set_ylabel('AURC Score', fontsize=14, fontweight='bold')
        ax5.set_title('(e) AURC Metrics Comparison', fontsize=16, fontweight='bold', loc='left')
        ax5.set_xticks(x)
        ax5.set_xticklabels(metrics, fontsize=12)
        ax5.legend(fontsize=11, loc='upper left')
        ax5.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # ===== Panel 6: Coverage-Accuracy Trade-off =====
        ax6 = fig.add_subplot(gs[1, 2])
        
        # Convert errors to accuracies
        standard_acc = 1 - filtered['standard_error']
        balanced_acc = 1 - filtered['balanced_error']
        worst_acc = 1 - filtered['worst_error']
        
        ax6.plot(filtered['coverage'], standard_acc, 'b-', linewidth=3, label='Standard', marker='o', markersize=4, markevery=10)
        ax6.plot(filtered['coverage'], balanced_acc, 'g-', linewidth=3, label='Balanced', marker='s', markersize=4, markevery=10)
        ax6.plot(filtered['coverage'], worst_acc, 'r-', linewidth=3, label='Worst-Group', marker='^', markersize=4, markevery=10)
        
        ax6.set_xlabel('Coverage', fontsize=14, fontweight='bold')
        ax6.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
        ax6.set_title('(f) Coverage-Accuracy Trade-off', fontsize=16, fontweight='bold', loc='left')
        ax6.grid(True, alpha=0.3, linestyle='--')
        ax6.legend(fontsize=12, loc='lower right')
        ax6.set_xlim(0.2, 1.0)
        ax6.set_ylim(min(worst_acc.min() * 0.95, 0.5), 1.0)
        
        plt.savefig(self.output_dir / 'paper_benchmark_figures.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'paper_benchmark_figures.pdf', bbox_inches='tight')
        print(f"\n📊 Saved paper-style figures to {self.output_dir}")
    
    def generate_latex_table(self, results: Dict) -> str:
        """
        Generate LaTeX table for paper submission.
        """
        main_method = 'gse_margin'
        
        latex = r"""\begin{table}[t]
\centering
\caption{Selective Classification Performance on CIFAR-100-LT}
\label{tab:main_results}
\begin{tabular}{lccc}
\toprule
\textbf{Method} & \textbf{AURC-Std} $\downarrow$ & \textbf{AURC-Bal} $\downarrow$ & \textbf{AURC-Worst} $\downarrow$ \\
\midrule
"""
        
        # Add our method
        aurc_std = results[main_method]['aurc_standard']
        aurc_bal = results[main_method]['aurc_balanced']
        aurc_wst = results[main_method]['aurc_worst']
        
        latex += f"AR-GSE (Ours) & {aurc_std:.4f} & {aurc_bal:.4f} & \\textbf{{{aurc_wst:.4f}}} \\\\\n"
        
        # Add baseline (placeholder)
        latex += r"SelectiveNet & - & - & - \\" + "\n"
        latex += r"L2R-LongTail & - & - & - \\" + "\n"
        
        latex += r"""\midrule
Oracle & """
        
        if 'oracle' in results:
            oracle_std = results['oracle']['aurc_standard']
            oracle_bal = results['oracle']['aurc_balanced']
            oracle_wst = results['oracle']['aurc_worst']
            latex += f"{oracle_std:.4f} & {oracle_bal:.4f} & {oracle_wst:.4f} \\\\\n"
        else:
            latex += "- & - & - \\\\\n"
        
        latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
        
        # Save to file
        with open(self.output_dir / 'latex_table.tex', 'w') as f:
            f.write(latex)
        
        print(f"\n📝 Saved LaTeX table to {self.output_dir / 'latex_table.tex'}")
        
        return latex
    
    def run_paper_benchmark(self):
        """Run complete paper benchmark evaluation."""
        print("\n" + "="*80)
        print("PAPER BENCHMARK EVALUATION")
        print("Comparison with SelectiveNet & Learning to Reject Meets Long-tail Learning")
        print("="*80)
        
        # Evaluate main method
        main_method = 'gse_margin'
        print(f"\n🔄 Computing confidence scores using: {main_method}")
        confidence_scores = self.compute_confidence_scores(main_method)
        
        # Generate RC curve
        print("🔄 Generating Risk-Coverage curve...")
        rc_df = self.generate_rc_curve(confidence_scores, 0.2, 1.0, num_points=81)
        
        # Compute comprehensive metrics
        print("🔄 Computing selective risk metrics...")
        selective_metrics = self.compute_selective_risk_metrics(rc_df)
        
        # Evaluate at specific coverages
        print("🔄 Evaluating at specific coverage points...")
        coverage_metrics = self.evaluate_at_specific_coverages(confidence_scores)
        
        # Compute oracle
        print("🔄 Computing oracle baseline...")
        oracle_aurc_std = self.compute_oracle_aurc('standard_error', 0.2, 1.0)
        oracle_aurc_bal = self.compute_oracle_aurc('balanced_error', 0.2, 1.0)
        oracle_aurc_wst = self.compute_oracle_aurc('worst_error', 0.2, 1.0)
        
        # Compile results
        results = {
            main_method: {
                'rc_curve': rc_df,
                'aurc_standard': selective_metrics['standard_error']['aurc'],
                'aurc_balanced': selective_metrics['balanced_error']['aurc'],
                'aurc_worst': selective_metrics['worst_error']['aurc'],
                'eaurc_standard': selective_metrics['standard_error']['eaurc'],
                'eaurc_balanced': selective_metrics['balanced_error']['eaurc'],
                'eaurc_worst': selective_metrics['worst_error']['eaurc'],
                'selective_metrics': selective_metrics,
                'coverage_metrics': coverage_metrics,
            },
            'oracle': {
                'aurc_standard': oracle_aurc_std,
                'aurc_balanced': oracle_aurc_bal,
                'aurc_worst': oracle_aurc_wst,
            }
        }
        
        # Print summary
        self._print_summary(results)
        
        # Create visualizations
        self.plot_paper_style_figures(results)
        
        # Generate LaTeX table
        self.generate_latex_table(results)
        
        # Save detailed results
        self._save_detailed_results(results)
        
        print("\n" + "="*80)
        print("✅ PAPER BENCHMARK EVALUATION COMPLETE")
        print("="*80)
        
        return results
    
    def _print_summary(self, results: Dict):
        """Print evaluation summary."""
        main_method = 'gse_margin'
        data = results[main_method]
        
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        
        print("\n📊 AURC Metrics (Coverage 0.2-1.0):")
        print(f"  Standard Error:")
        print(f"    AURC:    {data['aurc_standard']:.6f}")
        print(f"    E-AURC:  {data['eaurc_standard']:.6f}")
        
        print(f"\n  Balanced Error:")
        print(f"    AURC:    {data['aurc_balanced']:.6f}")
        print(f"    E-AURC:  {data['eaurc_balanced']:.6f}")
        
        print(f"\n  Worst-Group Error:")
        print(f"    AURC:    {data['aurc_worst']:.6f}")
        print(f"    E-AURC:  {data['eaurc_worst']:.6f}")
        
        print("\n📊 Oracle Baseline:")
        print(f"  Standard:    {results['oracle']['aurc_standard']:.6f}")
        print(f"  Balanced:    {results['oracle']['aurc_balanced']:.6f}")
        print(f"  Worst-Group: {results['oracle']['aurc_worst']:.6f}")
        
        print("\n📊 Metrics at Specific Coverages:")
        for cov_key, metrics in data['coverage_metrics'].items():
            cov = float(cov_key.split('_')[1])
            print(f"\n  Coverage ≈ {metrics['actual_coverage']:.3f} (target {cov:.2f}):")
            print(f"    Standard Accuracy:    {metrics['standard_accuracy']:.4f}")
            print(f"    Balanced Accuracy:    {metrics['balanced_accuracy']:.4f}")
            print(f"    Worst-Group Accuracy: {metrics['worst_accuracy']:.4f}")
        
        print("\n" + "="*80)
    
    def _save_detailed_results(self, results: Dict):
        """Save all results."""
        main_method = 'gse_margin'
        
        # Save RC curve
        results[main_method]['rc_curve'].to_csv(
            self.output_dir / 'rc_curve_paper_benchmark.csv', index=False
        )
        
        # Prepare JSON summary (remove DataFrame)
        summary = {
            'dataset': self.config['dataset']['name'],
            'num_test_samples': self.num_test_samples,
            'method': main_method,
            'aurc_metrics': {
                'standard': {
                    'aurc': results[main_method]['aurc_standard'],
                    'eaurc': results[main_method]['eaurc_standard'],
                },
                'balanced': {
                    'aurc': results[main_method]['aurc_balanced'],
                    'eaurc': results[main_method]['eaurc_balanced'],
                },
                'worst': {
                    'aurc': results[main_method]['aurc_worst'],
                    'eaurc': results[main_method]['eaurc_worst'],
                }
            },
            'oracle': results['oracle'],
            'coverage_metrics': results[main_method]['coverage_metrics'],
            'selective_metrics': results[main_method]['selective_metrics'],
        }
        
        with open(self.output_dir / 'paper_benchmark_results.json', 'w') as f:
            json.dump(summary, f, indent=4)
        
        print(f"\n💾 Saved detailed results to {self.output_dir}")


def main():
    """Main function for paper benchmark evaluation."""
    
    CONFIG = {
        'dataset': {
            'name': 'cifar100_lt_if100',
            'splits_dir': './data/cifar100_lt_if100_splits',
            'num_classes': 100,
        },
        'experts': {
            'names': ['ce_baseline', 'logitadjust_baseline', 'balsoftmax_baseline'],
            'logits_dir': './outputs/logits',
        },
        'checkpoint_path': './checkpoints/argse_worst_eg_improved/cifar100_lt_if100/gse_balanced_plugin.ckpt',
        'output_dir': './paper_benchmark_results',
        'seed': 42
    }
    
    # Set seeds
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    # Run evaluation
    evaluator = PaperBenchmarkEvaluator(CONFIG)
    results = evaluator.run_paper_benchmark()
    
    return results


if __name__ == '__main__':
    main()
