# src/train/eval_comprehensive.py
"""
Comprehensive Evaluation Framework for AR-GSE
Compatible with Selective Net and "Learning to Reject Meets Long-tail Learning" benchmarks

Key Features:
1. Standard RC (Risk-Coverage) curves
2. Multiple confidence scoring methods
3. AURC, E-AURC metrics
4. Comparison with baselines (Oracle, Random)
5. Bootstrap confidence intervals
6. Per-group analysis
"""
import torch
import torchvision
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

from src.models.argse import AR_GSE
from src.metrics.selective_metrics import calculate_selective_errors
from src.metrics.calibration import calculate_ece
from src.train.gse_balanced_plugin import compute_margin, compute_raw_margin

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class ComprehensiveEvaluator:
    """
    Comprehensive evaluator for selective classification on long-tail data.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = DEVICE
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load test data
        self.test_logits, self.test_labels = self._load_test_data()
        self.num_test_samples = len(self.test_labels)
        
        # Load model and parameters
        self.model, self.alpha, self.mu, self.class_to_group, self.num_groups = self._load_model()
        
        # Get predictions and posteriors
        self.eta_mix, self.preds = self._get_predictions()
        
        print(f"✅ Initialized evaluator with {self.num_test_samples} test samples")
    
    def _load_test_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load test logits and labels."""
        logits_root = Path(self.config['experts']['logits_dir']) / self.config['dataset']['name']
        splits_dir = Path(self.config['dataset']['splits_dir'])
        
        with open(splits_dir / 'test_lt_indices.json', 'r') as f:
            test_indices = json.load(f)
        num_test_samples = len(test_indices)
        
        # Load expert logits
        num_experts = len(self.config['experts']['names'])
        stacked_logits = torch.zeros(num_test_samples, num_experts, self.config['dataset']['num_classes'])
        
        for i, expert_name in enumerate(self.config['experts']['names']):
            logits_path = logits_root / expert_name / "test_lt_logits.pt"
            if not logits_path.exists():
                raise FileNotFoundError(f"Logits file not found: {logits_path}")
            stacked_logits[:, i, :] = torch.load(logits_path, map_location='cpu', weights_only=False)
        
        # Load test labels
        full_test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False)
        test_labels = torch.tensor(np.array(full_test_dataset.targets)[test_indices])
        
        return stacked_logits, test_labels
    
    def _load_model(self) -> Tuple[AR_GSE, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Load model and optimal parameters."""
        checkpoint_path = Path(self.config['checkpoint_path'])
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"📂 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        alpha = checkpoint['alpha'].to(self.device)
        mu = checkpoint['mu'].to(self.device)
        class_to_group = checkpoint['class_to_group'].to(self.device)
        num_groups = checkpoint['num_groups']
        
        # Initialize model
        num_experts = len(self.config['experts']['names'])
        with torch.no_grad():
            dummy_logits = torch.zeros(2, num_experts, self.config['dataset']['num_classes']).to(self.device)
            temp_model = AR_GSE(num_experts, self.config['dataset']['num_classes'], num_groups, 1).to(self.device)
            gating_feature_dim = temp_model.feature_builder(dummy_logits).size(-1)
            del temp_model
        
        model = AR_GSE(num_experts, self.config['dataset']['num_classes'], num_groups, gating_feature_dim).to(self.device)
        
        if 'gating_net_state_dict' in checkpoint:
            model.gating_net.load_state_dict(checkpoint['gating_net_state_dict'])
        
        with torch.no_grad():
            model.alpha.copy_(alpha)
            model.mu.copy_(mu)
        
        model.eval()
        
        print(f"✅ Loaded model with α=[{alpha[0]:.4f}, {alpha[1]:.4f}], μ=[{mu[0]:.4f}, {mu[1]:.4f}]")
        
        return model, alpha, mu, class_to_group, num_groups
    
    def _get_predictions(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get mixture posteriors and predictions."""
        with torch.no_grad():
            logits = self.test_logits.to(self.device)
            
            # Expert posteriors
            expert_posteriors = torch.softmax(logits, dim=-1)  # [B, E, C]
            
            # Gating weights
            gating_features = self.model.feature_builder(logits)
            gating_weights = torch.softmax(self.model.gating_net(gating_features), dim=1)  # [B, E]
            
            # Mixture posteriors
            eta_mix = torch.einsum('be,bec->bc', gating_weights, expert_posteriors)  # [B, C]
            
            # GSE predictions
            alpha_cpu = self.alpha.cpu()
            class_to_group_cpu = self.class_to_group.cpu()
            eta_mix_cpu = eta_mix.cpu()
            scores = alpha_cpu[class_to_group_cpu] * eta_mix_cpu
            preds = scores.argmax(dim=1)
        
        return eta_mix_cpu, preds
    
    def compute_confidence_scores(self, method: str = 'gse_margin') -> torch.Tensor:
        """
        Compute confidence scores using different methods.
        
        Args:
            method: Confidence scoring method
                - 'gse_margin': GSE margin (score - threshold - c)
                - 'gse_raw_margin': GSE raw margin (score - threshold)
                - 'max_posterior': Maximum posterior probability
                - 'entropy': Negative entropy (higher = more confident)
                - 'least_confident': 1 - max_posterior
                - 'margin': Difference between top 2 posteriors
                
        Returns:
            confidence_scores: [N] confidence scores (higher = more confident)
        """
        alpha_cpu = self.alpha.cpu()
        mu_cpu = self.mu.cpu()
        class_to_group_cpu = self.class_to_group.cpu()
        
        if method == 'gse_margin':
            # Use c=0 for fair comparison
            scores = compute_margin(self.eta_mix, alpha_cpu, mu_cpu, 0.0, class_to_group_cpu)
        
        elif method == 'gse_raw_margin':
            scores = compute_raw_margin(self.eta_mix, alpha_cpu, mu_cpu, class_to_group_cpu)
        
        elif method == 'max_posterior':
            scores = self.eta_mix.max(dim=1).values
        
        elif method == 'entropy':
            # Negative entropy (higher = more confident)
            eps = 1e-10
            entropy = -(self.eta_mix * torch.log(self.eta_mix + eps)).sum(dim=1)
            scores = -entropy
        
        elif method == 'least_confident':
            # 1 - max posterior
            scores = 1.0 - self.eta_mix.max(dim=1).values
            scores = -scores  # Invert so higher is more confident
        
        elif method == 'margin':
            # Difference between top 2 posteriors
            top2 = torch.topk(self.eta_mix, k=2, dim=1).values
            scores = top2[:, 0] - top2[:, 1]
        
        else:
            raise ValueError(f"Unknown confidence method: {method}")
        
        return scores
    
    def generate_rc_curve(self, 
                         confidence_scores: torch.Tensor,
                         coverage_min: float = 0.0,
                         coverage_max: float = 1.0,
                         num_points: int = 101) -> pd.DataFrame:
        """
        Generate Risk-Coverage curve by sweeping coverage levels.
        
        Args:
            confidence_scores: [N] confidence scores
            coverage_min: Minimum coverage
            coverage_max: Maximum coverage
            num_points: Number of coverage points
            
        Returns:
            DataFrame with coverage and error metrics
        """
        # Sort by confidence (descending)
        sorted_indices = torch.argsort(confidence_scores, descending=True)
        
        rc_points = []
        
        for i in range(num_points):
            target_coverage = coverage_min + (coverage_max - coverage_min) * i / (num_points - 1)
            num_to_accept = max(1, int(self.num_test_samples * target_coverage))
            num_to_accept = min(num_to_accept, self.num_test_samples)
            
            # Accept top-k confident samples
            accepted_mask = torch.zeros_like(self.test_labels, dtype=torch.bool)
            accepted_mask[sorted_indices[:num_to_accept]] = True
            
            # Compute metrics
            metrics = self._compute_error_metrics(accepted_mask)
            
            rc_points.append({
                'coverage': metrics['coverage'],
                'standard_error': metrics['standard_error'],
                'balanced_error': metrics['balanced_error'],
                'worst_error': metrics['worst_error'],
                'head_error': metrics['group_errors'][0],
                'tail_error': metrics['group_errors'][1],
            })
        
        df = pd.DataFrame(rc_points)
        return df
    
    def _compute_error_metrics(self, accepted_mask: torch.Tensor) -> Dict:
        """Compute error metrics on accepted samples."""
        if accepted_mask.sum() == 0:
            return {
                'coverage': 0.0,
                'standard_error': 1.0,
                'balanced_error': 1.0,
                'worst_error': 1.0,
                'group_errors': [1.0] * self.num_groups
            }
        
        accepted_preds = self.preds[accepted_mask]
        accepted_labels = self.test_labels[accepted_mask]
        
        # Standard error
        correct = (accepted_preds == accepted_labels)
        standard_error = 1.0 - correct.float().mean().item()
        
        # Per-group errors
        y_groups = self.class_to_group.cpu()[accepted_labels]
        group_errors = []
        
        for k in range(self.num_groups):
            group_mask = (y_groups == k)
            if group_mask.sum() == 0:
                group_errors.append(1.0)
            else:
                group_correct = (accepted_preds[group_mask] == accepted_labels[group_mask])
                group_error = 1.0 - group_correct.float().mean().item()
                group_errors.append(group_error)
        
        balanced_error = float(np.mean(group_errors))
        worst_error = float(np.max(group_errors))
        
        coverage = accepted_mask.float().mean().item()
        
        return {
            'coverage': coverage,
            'standard_error': standard_error,
            'balanced_error': balanced_error,
            'worst_error': worst_error,
            'group_errors': group_errors
        }
    
    def compute_aurc(self, rc_df: pd.DataFrame, 
                     error_column: str = 'balanced_error',
                     coverage_min: float = 0.0,
                     coverage_max: float = 1.0) -> float:
        """
        Compute Area Under Risk-Coverage Curve.
        
        Args:
            rc_df: RC curve dataframe
            error_column: Error metric to use
            coverage_min: Lower integration bound
            coverage_max: Upper integration bound
            
        Returns:
            AURC value (lower is better)
        """
        mask = (rc_df['coverage'] >= coverage_min) & (rc_df['coverage'] <= coverage_max)
        filtered = rc_df[mask].sort_values('coverage')
        
        if len(filtered) < 2:
            return 1.0
        
        coverages = filtered['coverage'].values
        errors = filtered[error_column].values
        
        # Trapezoidal integration
        aurc = np.trapz(errors, coverages)
        
        # Normalize by coverage range
        coverage_range = coverage_max - coverage_min
        if coverage_range > 0:
            aurc = aurc / coverage_range
        
        return aurc
    
    def compute_eaurc(self, rc_df: pd.DataFrame,
                      error_column: str = 'balanced_error',
                      coverage_min: float = 0.0,
                      coverage_max: float = 1.0) -> Tuple[float, float, float]:
        """
        Compute Excess AURC (E-AURC).
        
        E-AURC = AURC(method) - AURC(random)
        
        Returns:
            (eaurc, aurc_method, aurc_random)
        """
        # Method AURC
        aurc_method = self.compute_aurc(rc_df, error_column, coverage_min, coverage_max)
        
        # Random baseline: constant error (overall error at full coverage)
        full_cov_rows = rc_df[rc_df['coverage'] >= 0.99]
        if len(full_cov_rows) > 0:
            random_error = full_cov_rows[error_column].values[-1]
        else:
            random_error = rc_df[error_column].iloc[-1]
        
        aurc_random = random_error
        eaurc = aurc_method - aurc_random
        
        return eaurc, aurc_method, aurc_random
    
    def compute_oracle_aurc(self, error_column: str = 'balanced_error',
                           coverage_min: float = 0.0,
                           coverage_max: float = 1.0) -> float:
        """
        Compute Oracle AURC (perfect confidence = correctness).
        
        Oracle always rejects incorrect predictions first.
        """
        # Oracle confidence: 1 if correct, 0 if incorrect
        correctness = (self.preds == self.test_labels).float()
        oracle_scores = correctness + torch.rand_like(correctness) * 0.01  # Add noise for ties
        
        rc_df = self.generate_rc_curve(oracle_scores, coverage_min, coverage_max, num_points=101)
        aurc_oracle = self.compute_aurc(rc_df, error_column, coverage_min, coverage_max)
        
        return aurc_oracle
    
    def evaluate_all_methods(self, 
                            coverage_min: float = 0.2,
                            coverage_max: float = 1.0,
                            num_points: int = 81) -> Dict:
        """
        Evaluate all confidence scoring methods.
        
        Returns:
            Dictionary with results for all methods
        """
        methods = [
            'gse_margin',
            'gse_raw_margin',
            'max_posterior',
            'entropy',
            'margin'
        ]
        
        results = {}
        
        print(f"\n{'='*70}")
        print("EVALUATING ALL CONFIDENCE SCORING METHODS")
        print(f"{'='*70}\n")
        
        for method in methods:
            print(f"🔄 Method: {method}")
            
            # Compute confidence scores
            confidence_scores = self.compute_confidence_scores(method)
            
            # Generate RC curve
            rc_df = self.generate_rc_curve(confidence_scores, coverage_min, coverage_max, num_points)
            
            # Compute AURC metrics
            aurc_std = self.compute_aurc(rc_df, 'standard_error', coverage_min, coverage_max)
            aurc_bal = self.compute_aurc(rc_df, 'balanced_error', coverage_min, coverage_max)
            aurc_wst = self.compute_aurc(rc_df, 'worst_error', coverage_min, coverage_max)
            
            eaurc_std, _, _ = self.compute_eaurc(rc_df, 'standard_error', coverage_min, coverage_max)
            eaurc_bal, _, _ = self.compute_eaurc(rc_df, 'balanced_error', coverage_min, coverage_max)
            eaurc_wst, _, _ = self.compute_eaurc(rc_df, 'worst_error', coverage_min, coverage_max)
            
            results[method] = {
                'rc_curve': rc_df,
                'aurc_standard': aurc_std,
                'aurc_balanced': aurc_bal,
                'aurc_worst': aurc_wst,
                'eaurc_standard': eaurc_std,
                'eaurc_balanced': eaurc_bal,
                'eaurc_worst': eaurc_wst,
            }
            
            print(f"   AURC: std={aurc_std:.5f}, bal={aurc_bal:.5f}, worst={aurc_wst:.5f}")
            print(f"   E-AURC: std={eaurc_std:.5f}, bal={eaurc_bal:.5f}, worst={eaurc_wst:.5f}\n")
        
        # Oracle baseline
        print("🔄 Computing Oracle baseline...")
        aurc_oracle_std = self.compute_oracle_aurc('standard_error', coverage_min, coverage_max)
        aurc_oracle_bal = self.compute_oracle_aurc('balanced_error', coverage_min, coverage_max)
        aurc_oracle_wst = self.compute_oracle_aurc('worst_error', coverage_min, coverage_max)
        
        results['oracle'] = {
            'aurc_standard': aurc_oracle_std,
            'aurc_balanced': aurc_oracle_bal,
            'aurc_worst': aurc_oracle_wst,
        }
        
        print(f"   Oracle AURC: std={aurc_oracle_std:.5f}, bal={aurc_oracle_bal:.5f}, worst={aurc_oracle_wst:.5f}\n")
        
        return results
    
    def plot_comparison(self, results: Dict, coverage_min: float = 0.2, coverage_max: float = 1.0):
        """Plot comparison of all methods."""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        methods = ['gse_margin', 'gse_raw_margin', 'max_posterior', 'entropy', 'margin']
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        
        error_types = [
            ('standard_error', 'Standard Error', axes[0, 0]),
            ('balanced_error', 'Balanced Error', axes[0, 1]),
            ('worst_error', 'Worst-Group Error', axes[0, 2])
        ]
        
        # Plot RC curves
        for error_col, title, ax in error_types:
            for method, color in zip(methods, colors):
                rc_df = results[method]['rc_curve']
                mask = (rc_df['coverage'] >= coverage_min) & (rc_df['coverage'] <= coverage_max)
                filtered = rc_df[mask]
                
                ax.plot(filtered['coverage'], filtered[error_col], 
                       label=method, color=color, linewidth=2, alpha=0.8)
            
            ax.set_xlabel('Coverage', fontsize=12)
            ax.set_ylabel('Selective Risk', fontsize=12)
            ax.set_title(f'RC Curve - {title}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
            ax.set_xlim(coverage_min, coverage_max)
        
        # Plot AURC comparison
        metric_names = ['Standard', 'Balanced', 'Worst-Group']
        
        for idx, (metric_name, error_key) in enumerate(zip(metric_names, ['standard', 'balanced', 'worst'])):
            ax = axes[1, idx]
            
            method_names = []
            aurc_values = []
            eaurc_values = []
            
            for method in methods:
                method_names.append(method)
                aurc_values.append(results[method][f'aurc_{error_key}'])
                eaurc_values.append(results[method][f'eaurc_{error_key}'])
            
            x = np.arange(len(method_names))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, aurc_values, width, label='AURC', alpha=0.8)
            bars2 = ax.bar(x + width/2, eaurc_values, width, label='E-AURC', alpha=0.8)
            
            # Add oracle line
            if 'oracle' in results:
                oracle_aurc = results['oracle'][f'aurc_{error_key}']
                ax.axhline(y=oracle_aurc, color='black', linestyle='--', 
                          linewidth=2, label='Oracle', alpha=0.7)
            
            ax.set_ylabel('Score', fontsize=12)
            ax.set_title(f'{metric_name} Error - AURC Comparison', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(method_names, rotation=45, ha='right', fontsize=9)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'comprehensive_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 Saved comparison plot to {self.output_dir / 'comprehensive_comparison.png'}")
    
    def save_results(self, results: Dict):
        """Save all results to JSON and CSV."""
        # Prepare summary
        summary = {
            'dataset': self.config['dataset']['name'],
            'num_test_samples': self.num_test_samples,
            'checkpoint': self.config['checkpoint_path'],
            'methods': {}
        }
        
        for method, data in results.items():
            if method == 'oracle':
                summary['methods'][method] = data
            else:
                # Save RC curve
                rc_df = data['rc_curve']
                rc_df.to_csv(self.output_dir / f'rc_curve_{method}.csv', index=False)
                
                # Add metrics to summary
                summary['methods'][method] = {
                    'aurc_standard': data['aurc_standard'],
                    'aurc_balanced': data['aurc_balanced'],
                    'aurc_worst': data['aurc_worst'],
                    'eaurc_standard': data['eaurc_standard'],
                    'eaurc_balanced': data['eaurc_balanced'],
                    'eaurc_worst': data['eaurc_worst'],
                }
        
        # Save summary
        with open(self.output_dir / 'evaluation_summary.json', 'w') as f:
            json.dump(summary, f, indent=4)
        
        print(f"\n💾 Saved evaluation summary to {self.output_dir / 'evaluation_summary.json'}")
        
        # Create comparison table
        self._create_comparison_table(results)
    
    def _create_comparison_table(self, results: Dict):
        """Create a comparison table of all methods."""
        methods = ['gse_margin', 'gse_raw_margin', 'max_posterior', 'entropy', 'margin']
        
        rows = []
        for method in methods:
            data = results[method]
            rows.append({
                'Method': method,
                'AURC-Std': data['aurc_standard'],
                'AURC-Bal': data['aurc_balanced'],
                'AURC-Worst': data['aurc_worst'],
                'E-AURC-Std': data['eaurc_standard'],
                'E-AURC-Bal': data['eaurc_balanced'],
                'E-AURC-Worst': data['eaurc_worst'],
            })
        
        # Add oracle
        if 'oracle' in results:
            rows.append({
                'Method': 'Oracle',
                'AURC-Std': results['oracle']['aurc_standard'],
                'AURC-Bal': results['oracle']['aurc_balanced'],
                'AURC-Worst': results['oracle']['aurc_worst'],
                'E-AURC-Std': '-',
                'E-AURC-Bal': '-',
                'E-AURC-Worst': '-',
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(self.output_dir / 'comparison_table.csv', index=False)
        
        print("\n" + "="*100)
        print("COMPARISON TABLE")
        print("="*100)
        print(df.to_string(index=False))
        print("="*100)
    
    def run_full_evaluation(self):
        """Run complete evaluation pipeline."""
        print("\n" + "="*70)
        print("COMPREHENSIVE EVALUATION FOR AR-GSE")
        print("="*70)
        
        # Configuration
        coverage_min = self.config.get('coverage_min', 0.2)
        coverage_max = self.config.get('coverage_max', 1.0)
        num_points = self.config.get('num_points', 81)
        
        print(f"\nConfiguration:")
        print(f"  Dataset: {self.config['dataset']['name']}")
        print(f"  Test samples: {self.num_test_samples}")
        print(f"  Coverage range: [{coverage_min:.0%}, {coverage_max:.0%}]")
        print(f"  Number of points: {num_points}")
        
        # Evaluate all methods
        results = self.evaluate_all_methods(coverage_min, coverage_max, num_points)
        
        # Plot comparison
        self.plot_comparison(results, coverage_min, coverage_max)
        
        # Save results
        self.save_results(results)
        
        # ECE
        ece = calculate_ece(self.eta_mix, self.test_labels)
        print(f"\n📊 Expected Calibration Error (ECE): {ece:.4f}")
        
        print("\n" + "="*70)
        print("✅ EVALUATION COMPLETE")
        print("="*70)
        
        return results


def main():
    """Main evaluation function."""
    
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
        'output_dir': './comprehensive_evaluation_results',
        'coverage_min': 0.2,
        'coverage_max': 1.0,
        'num_points': 81,
        'seed': 42
    }
    
    # Set seeds
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    # Run evaluation
    evaluator = ComprehensiveEvaluator(CONFIG)
    results = evaluator.run_full_evaluation()
    
    return results


if __name__ == '__main__':
    main()
