"""
AURC Evaluation Demo for AR-GSE
================================

This script demonstrates how to compute AURC (Area Under Risk-Coverage curve)
by sweeping multiple rejection cost values c, similar to "Learning to Reject 
Meets Long-tail Learning" paper methodology.

Key Steps:
1. Load model predictions and confidence scores
2. Define a grid of cost values c 
3. For each c, find optimal threshold on validation set
4. Apply optimal threshold to test set → get (coverage, risk) points
5. Compute AURC using trapezoidal integration
6. Repeat for multiple trials and report mean ± std
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

# Set device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🔧 Using device: {DEVICE}")

def create_toy_data(n_samples: int = 1000, n_classes: int = 10, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create toy prediction data for AURC demo.
    
    Returns:
        logits: [N, C] model logits
        labels: [N] true labels  
        groups: [N] group assignments (0=head, 1=tail)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create imbalanced dataset (head vs tail classes)
    head_classes = list(range(n_classes // 2))  # First half are head classes
    tail_classes = list(range(n_classes // 2, n_classes))  # Second half are tail classes
    
    # Sample more from head classes
    head_prob = 0.8
    is_head = np.random.random(n_samples) < head_prob
    
    labels = []
    groups = []
    for i in range(n_samples):
        if is_head[i]:
            labels.append(np.random.choice(head_classes))
            groups.append(0)  # Head group
        else:
            labels.append(np.random.choice(tail_classes))
            groups.append(1)  # Tail group
    
    labels = torch.tensor(labels)
    groups = torch.tensor(groups)
    
    # Create logits with some noise (head classes get higher confidence)
    logits = torch.randn(n_samples, n_classes) * 0.5
    for i in range(n_samples):
        true_class = labels[i]
        if groups[i] == 0:  # Head class - higher confidence
            logits[i, true_class] += np.random.normal(2.0, 0.5)
        else:  # Tail class - lower confidence
            logits[i, true_class] += np.random.normal(1.0, 0.8)
    
    return logits, labels, groups

def compute_confidence_score(logits: torch.Tensor, method: str = 'max_prob') -> torch.Tensor:
    """Compute confidence scores from logits."""
    probs = torch.softmax(logits, dim=1)
    
    if method == 'max_prob':
        return probs.max(dim=1)[0]
    elif method == 'entropy':
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
        return -entropy  # Higher confidence = lower entropy
    elif method == 'margin':
        sorted_probs = torch.sort(probs, dim=1, descending=True)[0]
        return sorted_probs[:, 0] - sorted_probs[:, 1]
    else:
        raise ValueError(f"Unknown confidence method: {method}")

def compute_group_metrics(preds: torch.Tensor, labels: torch.Tensor, 
                         accepted: torch.Tensor, groups: torch.Tensor, 
                         metric_type: str = 'balanced') -> float:
    """
    Compute group-aware risk metrics.
    
    Args:
        preds: [N] predictions
        labels: [N] true labels
        accepted: [N] acceptance mask
        groups: [N] group assignments  
        metric_type: 'standard', 'balanced', or 'worst'
    """
    if accepted.sum() == 0:
        return 1.0  # All rejected -> max risk
    
    # Standard error (overall accuracy on accepted)
    if metric_type == 'standard':
        correct = (preds[accepted] == labels[accepted])
        return 1.0 - correct.float().mean().item()
    
    # Group-aware metrics
    unique_groups = groups.unique()
    group_errors = []
    
    for g in unique_groups:
        group_mask = (groups == g)
        group_accepted = accepted & group_mask
        
        if group_accepted.sum() == 0:
            group_errors.append(1.0)  # No accepted samples in this group
        else:
            group_correct = (preds[group_accepted] == labels[group_accepted])
            group_error = 1.0 - group_correct.float().mean().item()
            group_errors.append(group_error)
    
    if metric_type == 'balanced':
        return np.mean(group_errors)
    elif metric_type == 'worst':
        return np.max(group_errors)
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")

def find_optimal_threshold(confidence_scores: torch.Tensor, preds: torch.Tensor,
                          labels: torch.Tensor, groups: torch.Tensor,
                          cost_c: float, metric_type: str = 'balanced') -> float:
    """
    Find optimal threshold that minimizes: risk + c * (1 - coverage)
    """
    # Create candidate thresholds
    unique_scores = torch.unique(confidence_scores)
    thresholds = torch.cat([torch.tensor([0.0]), unique_scores, torch.tensor([1.0])])
    thresholds = torch.sort(thresholds, descending=True)[0]  # High to low
    
    best_cost = float('inf')
    best_threshold = 0.0
    
    for threshold in thresholds:
        accepted = confidence_scores >= threshold
        coverage = accepted.float().mean().item()
        risk = compute_group_metrics(preds, labels, accepted, groups, metric_type)
        
        # Objective: risk + c * rejection_rate
        cost = risk + cost_c * (1.0 - coverage)
        
        if cost < best_cost:
            best_cost = cost
            best_threshold = threshold.item()
    
    return best_threshold

def sweep_cost_values(confidence_scores_val: torch.Tensor, preds_val: torch.Tensor,
                     labels_val: torch.Tensor, groups_val: torch.Tensor,
                     confidence_scores_test: torch.Tensor, preds_test: torch.Tensor,
                     labels_test: torch.Tensor, groups_test: torch.Tensor,
                     cost_values: np.ndarray, metric_type: str = 'balanced') -> List[Tuple[float, float, float]]:
    """
    Sweep cost values and return (cost, coverage, risk) points on test set.
    """
    rc_points = []
    
    print(f"🔄 Sweeping {len(cost_values)} cost values for {metric_type} metric...")
    
    for i, cost_c in enumerate(cost_values):
        # Find optimal threshold on validation
        optimal_threshold = find_optimal_threshold(
            confidence_scores_val, preds_val, labels_val, groups_val, cost_c, metric_type
        )
        
        # Apply to test set
        accepted_test = confidence_scores_test >= optimal_threshold
        coverage_test = accepted_test.float().mean().item()
        risk_test = compute_group_metrics(preds_test, labels_test, accepted_test, groups_test, metric_type)
        
        rc_points.append((cost_c, coverage_test, risk_test))
        
        if (i + 1) % 20 == 0:
            print(f"   Progress: {i+1}/{len(cost_values)} - Current: c={cost_c:.3f}, cov={coverage_test:.3f}, risk={risk_test:.3f}")
    
    return rc_points

def compute_aurc(rc_points: List[Tuple[float, float, float]]) -> float:
    """
    Compute AURC using trapezoidal integration.
    
    Args:
        rc_points: List of (cost, coverage, risk) tuples
    """
    # Sort by coverage
    rc_points = sorted(rc_points, key=lambda x: x[1])
    
    coverages = [p[1] for p in rc_points]
    risks = [p[2] for p in rc_points]
    
    # Ensure we have endpoints for proper integration
    if coverages[0] > 0.0:
        coverages = [0.0] + coverages
        risks = [0.0] + risks  # Risk is 0 when coverage is 0 (no predictions made)
    
    if coverages[-1] < 1.0:
        coverages = coverages + [1.0]
        risks = risks + [risks[-1]]  # Extend last risk to coverage=1
    
    # Trapezoidal integration
    aurc = np.trapz(risks, coverages)
    return aurc

def plot_risk_coverage_curves(results_dict: Dict[str, List[Tuple[float, float, float]]], 
                             save_path: str = None):
    """Plot risk-coverage curves for different metrics."""
    plt.figure(figsize=(12, 4))
    
    colors = ['blue', 'red', 'green']
    linestyles = ['-', '--', '-.']
    
    for i, (metric, rc_points) in enumerate(results_dict.items()):
        # Sort by coverage
        rc_points = sorted(rc_points, key=lambda x: x[1])
        coverages = [p[1] for p in rc_points]
        risks = [p[2] for p in rc_points]
        
        aurc = compute_aurc(rc_points)
        
        plt.subplot(1, 2, 1)
        plt.plot(coverages, risks, color=colors[i], linestyle=linestyles[i], 
                linewidth=2, label=f'{metric.title()} (AURC={aurc:.4f})')
    
    plt.subplot(1, 2, 1)
    plt.xlabel('Coverage (Fraction Accepted)')
    plt.ylabel('Risk (Error on Accepted)')
    plt.title('Risk-Coverage Curves')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, None)
    
    # Zoomed view for practical range
    plt.subplot(1, 2, 2)
    for i, (metric, rc_points) in enumerate(results_dict.items()):
        rc_points = sorted(rc_points, key=lambda x: x[1])
        coverages = [p[1] for p in rc_points if p[1] >= 0.2]
        risks = [p[2] for p in rc_points if p[1] >= 0.2]
        
        plt.plot(coverages, risks, color=colors[i], linestyle=linestyles[i], 
                linewidth=2, label=f'{metric.title()}')
    
    plt.xlabel('Coverage (Fraction Accepted)')
    plt.ylabel('Risk (Error on Accepted)')
    plt.title('Risk-Coverage Curves (Focused View)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0.2, 1.0)
    plt.ylim(0, None)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Saved plot to {save_path}")
    
    plt.show()

def main():
    """Main AURC evaluation demo."""
    print("="*60)
    print("AURC EVALUATION DEMO FOR AR-GSE")
    print("="*60)
    
    # Create toy data
    print("📊 Creating toy dataset...")
    logits, labels, groups = create_toy_data(n_samples=2000, seed=42)
    
    # Split into train/val/test
    n_train = 800
    n_val = 600
    n_test = 600
    
    val_idx = torch.arange(n_train, n_train + n_val)
    test_idx = torch.arange(n_train + n_val, n_train + n_val + n_test)
    
    # Compute predictions and confidence scores
    print("🔮 Computing predictions and confidence scores...")
    preds = logits.argmax(dim=1)
    confidence_scores = compute_confidence_score(logits, method='max_prob')
    
    # Split data
    confidence_val = confidence_scores[val_idx]
    preds_val = preds[val_idx]
    labels_val = labels[val_idx]
    groups_val = groups[val_idx]
    
    confidence_test = confidence_scores[test_idx]
    preds_test = preds[test_idx]
    labels_test = labels[test_idx]
    groups_test = groups[test_idx]
    
    print(f"✅ Data splits - Val: {len(val_idx)}, Test: {len(test_idx)}")
    print(f"   Head group samples in test: {(groups_test == 0).sum().item()}")
    print(f"   Tail group samples in test: {(groups_test == 1).sum().item()}")
    
    # Define cost grid
    cost_values = np.linspace(0.0, 0.8, 81)  # 81 points from 0 to 0.8
    print(f"🎯 Cost grid: {len(cost_values)} values from {cost_values[0]:.1f} to {cost_values[-1]:.1f}")
    
    # Sweep costs for different metrics
    metrics = ['standard', 'balanced', 'worst']
    results = {}
    
    for metric in metrics:
        print(f"\n🔄 Processing {metric} metric...")
        rc_points = sweep_cost_values(
            confidence_val, preds_val, labels_val, groups_val,
            confidence_test, preds_test, labels_test, groups_test,
            cost_values, metric
        )
        results[metric] = rc_points
        
        # Compute AURC
        aurc = compute_aurc(rc_points)
        print(f"✅ {metric.upper()} AURC: {aurc:.6f}")
    
    print("\n" + "="*60)
    print("FINAL AURC RESULTS")
    print("="*60)
    for metric in metrics:
        aurc = compute_aurc(results[metric])
        print(f"• {metric.upper():>12} AURC: {aurc:.6f}")
    print("="*60)
    print("📝 Lower AURC is better (less area under risk-coverage curve)")
    
    # Plot results
    print("\n📈 Plotting risk-coverage curves...")
    plot_save_path = Path("aurc_demo_curves.png")
    plot_risk_coverage_curves(results, save_path=str(plot_save_path))
    
    # Save detailed results
    print("\n💾 Saving detailed results...")
    results_df = []
    for metric, rc_points in results.items():
        for cost_c, coverage, risk in rc_points:
            results_df.append({
                'metric': metric,
                'cost': cost_c,
                'coverage': coverage,
                'risk': risk
            })
    
    results_df = pd.DataFrame(results_df)
    results_df.to_csv('aurc_demo_results.csv', index=False)
    print("✅ Saved results to aurc_demo_results.csv")
    
    # Summary statistics
    print("\n📊 Summary Statistics:")
    coverage_ranges = [(0.0, 0.5), (0.5, 0.8), (0.8, 1.0)]
    
    for metric in metrics:
        print(f"\n{metric.upper()} Metric:")
        rc_points = results[metric]
        
        for cov_min, cov_max in coverage_ranges:
            filtered_points = [(c, cov, r) for c, cov, r in rc_points 
                             if cov_min <= cov <= cov_max]
            if filtered_points:
                avg_risk = np.mean([r for _, _, r in filtered_points])
                print(f"  Coverage [{cov_min:.1f}-{cov_max:.1f}]: Avg Risk = {avg_risk:.4f}")
    
    print("\n🎉 AURC evaluation demo completed!")
    print("📝 This demonstrates the 'sweep multiple c values' methodology")
    print("   used in 'Learning to Reject Meets Long-tail Learning' paper.")

if __name__ == '__main__':
    main()