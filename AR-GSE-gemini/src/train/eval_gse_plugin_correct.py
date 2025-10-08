# src/train/eval_gse_plugin_correct.py
"""
CORRECTED Evaluation script following "Learning to Reject Meets Long-tail Learning" methodology.

Key fixes:
1. Standard RC curve: sweep coverage (not cost)
2. No data leakage: use full test set
3. Proper AURC: integrate over coverage from 0.2 to 1.0
4. Comparable with paper benchmarks
"""
import torch
import torchvision
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path

from src.models.argse import AR_GSE
from src.metrics.selective_metrics import calculate_selective_errors
from src.metrics.calibration import calculate_ece
from src.train.gse_balanced_plugin import compute_margin

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CONFIG = {
    'dataset': {
        'name': 'cifar100_lt_if100',
        'splits_dir': './data/cifar100_lt_if100_splits',
        'num_classes': 100,
    },
    'grouping': {
        'threshold': 20,
    },
    'experts': {
        'names': ['ce_baseline', 'logitadjust_baseline', 'balsoftmax_baseline'],
        'logits_dir': './outputs/logits',
    },
    'eval_params': {
        # Coverage range following paper
        'coverage_min': 0.2,
        'coverage_max': 1.0,
        'coverage_step': 0.01,  # 81 points from 0.2 to 1.0
    },
    'plugin_checkpoint': './checkpoints/argse_worst_eg_improved/cifar100_lt_if100/gse_balanced_plugin.ckpt',
    'output_dir': './results_worst_eg_improved/cifar100_lt_if100',
    'seed': 42
}

def load_test_data():
    """Load test logits and labels."""
    logits_root = Path(CONFIG['experts']['logits_dir']) / CONFIG['dataset']['name']
    splits_dir = Path(CONFIG['dataset']['splits_dir'])
    
    with open(splits_dir / 'test_lt_indices.json', 'r') as f:
        test_indices = json.load(f)
    num_test_samples = len(test_indices)
    
    # Load expert logits for test set
    num_experts = len(CONFIG['experts']['names'])
    stacked_logits = torch.zeros(num_test_samples, num_experts, CONFIG['dataset']['num_classes'])
    
    for i, expert_name in enumerate(CONFIG['experts']['names']):
        logits_path = logits_root / expert_name / "test_lt_logits.pt"
        if not logits_path.exists():
            raise FileNotFoundError(f"Logits file not found: {logits_path}")
        stacked_logits[:, i, :] = torch.load(logits_path, map_location='cpu', weights_only=False)
    
    # Load test labels
    full_test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False)
    test_labels = torch.tensor(np.array(full_test_dataset.targets)[test_indices])
    
    return stacked_logits, test_labels

def get_mixture_posteriors_and_predictions(model, logits, alpha, mu, class_to_group):
    """Get mixture posteriors and GSE predictions."""
    model.eval()
    with torch.no_grad():
        logits = logits.to(DEVICE)
        
        # Get expert posteriors
        expert_posteriors = torch.softmax(logits, dim=-1)  # [B, E, C]
        
        # Get gating weights
        gating_features = model.feature_builder(logits)
        gating_weights = torch.softmax(model.gating_net(gating_features), dim=1)  # [B, E]
        
        # Mixture posteriors
        eta_mix = torch.einsum('be,bec->bc', gating_weights, expert_posteriors)  # [B, C]
        
        # GSE predictions (using alpha-weighted scores)
        alpha_cpu = alpha.cpu()
        class_to_group_cpu = class_to_group.cpu()
        eta_mix_cpu = eta_mix.cpu()
        scores = alpha_cpu[class_to_group_cpu] * eta_mix_cpu  # [B, C]
        preds = scores.argmax(dim=1)  # [B]
        
    return eta_mix_cpu, preds

def compute_error_metrics(preds, labels, accepted_mask, class_to_group, K):
    """
    Compute standard, balanced, and worst-group error rates.
    
    Returns:
        dict with 'standard', 'balanced', 'worst' error rates
    """
    if accepted_mask.sum() == 0:
        return {'standard': 1.0, 'balanced': 1.0, 'worst': 1.0}
    
    accepted_preds = preds[accepted_mask]
    accepted_labels = labels[accepted_mask]
    
    # Standard error (overall accuracy)
    correct = (accepted_preds == accepted_labels)
    standard_error = 1.0 - correct.float().mean().item()
    
    # Per-group errors
    y_groups = class_to_group[accepted_labels]
    group_errors = []
    
    for k in range(K):
        group_mask = (y_groups == k)
        if group_mask.sum() == 0:
            group_errors.append(1.0)
        else:
            group_correct = (accepted_preds[group_mask] == accepted_labels[group_mask])
            group_error = 1.0 - group_correct.float().mean().item()
            group_errors.append(group_error)
    
    balanced_error = float(np.mean(group_errors))
    worst_error = float(np.max(group_errors))
    
    return {
        'standard': standard_error,
        'balanced': balanced_error,
        'worst': worst_error,
        'group_errors': group_errors
    }

def generate_rc_curve_correct(confidence_scores, preds, labels, class_to_group, K,
                              coverage_min=0.2, coverage_max=1.0, coverage_step=0.01):
    """
    Generate RC curve by sweeping coverage levels (CORRECT methodology).
    
    This follows the standard selective classification evaluation:
    - For each target coverage c ∈ [coverage_min, coverage_max]
    - Find threshold t such that fraction of accepted samples ≈ c
    - Compute error metrics on accepted samples
    - Plot (coverage, error)
    
    Args:
        confidence_scores: [N] confidence scores (e.g., GSE margins)
        preds: [N] predictions
        labels: [N] true labels
        class_to_group: [C] class to group mapping
        K: number of groups
        coverage_min: minimum coverage to evaluate
        coverage_max: maximum coverage to evaluate
        coverage_step: coverage increment
        
    Returns:
        DataFrame with columns: coverage, standard_error, balanced_error, worst_error
    """
    print(f"\n🔄 Generating RC Curve (coverage {coverage_min:.0%} to {coverage_max:.0%})...")
    
    # Generate coverage points
    coverage_targets = np.arange(coverage_min, coverage_max + coverage_step/2, coverage_step)
    
    rc_points = []
    
    for i, target_cov in enumerate(coverage_targets):
        # Find threshold for this coverage
        # Threshold at (1 - target_cov) quantile → accept top target_cov fraction
        quantile_value = max(0.0, min(1.0, 1.0 - target_cov))  # Clamp to [0, 1]
        threshold = torch.quantile(confidence_scores, quantile_value)
        
        # Apply threshold
        accepted_mask = confidence_scores >= threshold
        actual_cov = accepted_mask.float().mean().item()
        
        # Compute error metrics
        metrics = compute_error_metrics(preds, labels, accepted_mask, class_to_group, K)
        
        rc_points.append({
            'coverage': actual_cov,
            'standard_error': metrics['standard'],
            'balanced_error': metrics['balanced'],
            'worst_error': metrics['worst'],
            'head_error': metrics['group_errors'][0],
            'tail_error': metrics['group_errors'][1]
        })
        
        # Progress
        if (i + 1) % 20 == 0 or i == 0 or i == len(coverage_targets) - 1:
            print(f"   {i+1}/{len(coverage_targets)}: cov={actual_cov:.3f}, "
                  f"std={metrics['standard']:.3f}, bal={metrics['balanced']:.3f}, "
                  f"worst={metrics['worst']:.3f}")
    
    df = pd.DataFrame(rc_points)
    
    # Verification: errors should be monotonic
    print("\n✅ RC Curve Statistics:")
    print(f"   Coverage range: [{df['coverage'].min():.3f}, {df['coverage'].max():.3f}]")
    print(f"   Standard error range: [{df['standard_error'].min():.3f}, {df['standard_error'].max():.3f}]")
    print(f"   Balanced error range: [{df['balanced_error'].min():.3f}, {df['balanced_error'].max():.3f}]")
    print(f"   Worst error range: [{df['worst_error'].min():.3f}, {df['worst_error'].max():.3f}]")
    
    # Sanity check: worst >= balanced >= standard (approximately)
    avg_worst = df['worst_error'].mean()
    avg_balanced = df['balanced_error'].mean()
    avg_standard = df['standard_error'].mean()
    
    if avg_worst >= avg_balanced >= avg_standard:
        print(f"   ✅ Sanity check PASSED: worst({avg_worst:.3f}) >= balanced({avg_balanced:.3f}) >= standard({avg_standard:.3f})")
    else:
        print(f"   ⚠️ Sanity check WARNING: worst({avg_worst:.3f}), balanced({avg_balanced:.3f}), standard({avg_standard:.3f})")
    
    return df

def compute_aurc(rc_df, error_column='balanced_error', coverage_min=0.2, coverage_max=1.0):
    """
    Compute AURC (Area Under Risk-Coverage Curve).
    
    AURC = ∫[coverage_min to coverage_max] error(c) dc
    
    Lower is better.
    
    Args:
        rc_df: DataFrame with 'coverage' and error columns
        error_column: which error metric to use
        coverage_min: lower integration bound
        coverage_max: upper integration bound
        
    Returns:
        aurc: scalar AURC value
    """
    # Filter to desired coverage range
    mask = (rc_df['coverage'] >= coverage_min) & (rc_df['coverage'] <= coverage_max)
    filtered = rc_df[mask].sort_values('coverage')
    
    if len(filtered) < 2:
        print(f"⚠️ Warning: Only {len(filtered)} points in coverage range [{coverage_min}, {coverage_max}]")
        return 1.0
    
    coverages = filtered['coverage'].values
    errors = filtered[error_column].values
    
    # Trapezoidal integration
    aurc = np.trapz(errors, coverages)
    
    # Normalize by coverage range
    coverage_range = coverage_max - coverage_min
    aurc_normalized = aurc / coverage_range
    
    return aurc_normalized

def compute_eaurc(rc_df, error_column='balanced_error', coverage_min=0.2, coverage_max=1.0):
    """
    Compute E-AURC (Excess AURC over random baseline).
    
    E-AURC = AURC(method) - AURC(random)
    
    Random baseline: accept randomly → error = overall_error for all coverages
    
    Lower is better.
    """
    # Method AURC
    aurc_method = compute_aurc(rc_df, error_column, coverage_min, coverage_max)
    
    # Random baseline AURC
    # For random: error is constant (overall error at coverage=1.0)
    overall_error = rc_df[rc_df['coverage'] >= 0.99][error_column].values
    if len(overall_error) > 0:
        random_error = overall_error[-1]
    else:
        random_error = rc_df[error_column].iloc[-1]
    
    aurc_random = random_error  # Constant error → area = error × range
    
    eaurc = aurc_method - aurc_random
    
    return eaurc, aurc_method, aurc_random

def plot_rc_curves(rc_df, output_path):
    """Plot Risk-Coverage curves."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Full range - all metrics
    ax = axes[0, 0]
    ax.plot(rc_df['coverage'], rc_df['standard_error'], 'b-', linewidth=2, label='Standard Error')
    ax.plot(rc_df['coverage'], rc_df['balanced_error'], 'g-', linewidth=2, label='Balanced Error')
    ax.plot(rc_df['coverage'], rc_df['worst_error'], 'r-', linewidth=2, label='Worst-Group Error')
    ax.set_xlabel('Coverage (Fraction Accepted)', fontsize=12)
    ax.set_ylabel('Selective Risk (Error Rate)', fontsize=12)
    ax.set_title('Risk-Coverage Curves (Full Range)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Focused range 0.2-1.0
    ax = axes[0, 1]
    mask = rc_df['coverage'] >= 0.2
    ax.plot(rc_df[mask]['coverage'], rc_df[mask]['standard_error'], 'b-', linewidth=2, label='Standard')
    ax.plot(rc_df[mask]['coverage'], rc_df[mask]['balanced_error'], 'g-', linewidth=2, label='Balanced')
    ax.plot(rc_df[mask]['coverage'], rc_df[mask]['worst_error'], 'r-', linewidth=2, label='Worst-Group')
    ax.set_xlabel('Coverage (Fraction Accepted)', fontsize=12)
    ax.set_ylabel('Selective Risk (Error Rate)', fontsize=12)
    ax.set_title('Risk-Coverage Curves (0.2-1.0 Range)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_xlim(0.2, 1.0)
    
    # Per-group errors
    ax = axes[1, 0]
    ax.plot(rc_df['coverage'], rc_df['head_error'], 'b-', linewidth=2, label='Head Group')
    ax.plot(rc_df['coverage'], rc_df['tail_error'], 'r-', linewidth=2, label='Tail Group')
    ax.set_xlabel('Coverage (Fraction Accepted)', fontsize=12)
    ax.set_ylabel('Group Error Rate', fontsize=12)
    ax.set_title('Per-Group Selective Errors', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # AURC comparison bar plot
    ax = axes[1, 1]
    aurc_standard = compute_aurc(rc_df, 'standard_error', 0.2, 1.0)
    aurc_balanced = compute_aurc(rc_df, 'balanced_error', 0.2, 1.0)
    aurc_worst = compute_aurc(rc_df, 'worst_error', 0.2, 1.0)
    
    metrics = ['Standard', 'Balanced', 'Worst-Group']
    aurcs = [aurc_standard, aurc_balanced, aurc_worst]
    colors = ['blue', 'green', 'red']
    
    bars = ax.bar(metrics, aurcs, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('AURC (0.2-1.0 Coverage)', fontsize=12)
    ax.set_title('AURC Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, aurc in zip(bars, aurcs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.005,
                f'{aurc:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved RC curve plots to {output_path}")

def main():
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    print("=" * 70)
    print("GSE-Balanced Plugin Evaluation (CORRECTED)")
    print("Following 'Learning to Reject Meets Long-tail Learning' methodology")
    print("=" * 70)
    
    # Create output directory
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load plugin checkpoint
    plugin_ckpt_path = Path(CONFIG['plugin_checkpoint'])
    if not plugin_ckpt_path.exists():
        raise FileNotFoundError(f"Plugin checkpoint not found: {plugin_ckpt_path}")
    
    print(f"\n📂 Loading plugin checkpoint: {plugin_ckpt_path}")
    checkpoint = torch.load(plugin_ckpt_path, map_location=DEVICE, weights_only=False)
    
    alpha_star = checkpoint['alpha'].to(DEVICE)
    mu_star = checkpoint['mu'].to(DEVICE)
    class_to_group = checkpoint['class_to_group'].to(DEVICE)
    num_groups = checkpoint['num_groups']
    
    print("\n✅ Loaded optimal parameters:")
    print(f"   α* = [{alpha_star[0]:.4f}, {alpha_star[1]:.4f}]")
    print(f"   μ* = [{mu_star[0]:.4f}, {mu_star[1]:.4f}]")
    
    # 2. Set up model
    num_experts = len(CONFIG['experts']['names'])
    
    with torch.no_grad():
        dummy_logits = torch.zeros(2, num_experts, CONFIG['dataset']['num_classes']).to(DEVICE)
        temp_model = AR_GSE(num_experts, CONFIG['dataset']['num_classes'], num_groups, 1).to(DEVICE)
        gating_feature_dim = temp_model.feature_builder(dummy_logits).size(-1)
        del temp_model
    
    model = AR_GSE(num_experts, CONFIG['dataset']['num_classes'], num_groups, gating_feature_dim).to(DEVICE)
    
    if 'gating_net_state_dict' in checkpoint:
        model.gating_net.load_state_dict(checkpoint['gating_net_state_dict'])
        print("✅ Gating network weights loaded")
    
    with torch.no_grad():
        model.alpha.copy_(alpha_star)
        model.mu.copy_(mu_star)
    
    # 3. Load test data
    print("\n📊 Loading test data...")
    test_logits, test_labels = load_test_data()
    num_test_samples = len(test_labels)
    print(f"✅ Loaded {num_test_samples} test samples")
    
    # 4. Get predictions and confidence scores
    print("\n🔮 Computing predictions and confidence scores...")
    eta_mix, preds = get_mixture_posteriors_and_predictions(
        model, test_logits, alpha_star, mu_star, class_to_group
    )
    
    # Use GSE margins as confidence scores
    alpha_star_cpu = alpha_star.cpu()
    mu_star_cpu = mu_star.cpu()
    class_to_group_cpu = class_to_group.cpu()
    
    confidence_scores = compute_margin(eta_mix, alpha_star_cpu, mu_star_cpu, 0.0, class_to_group_cpu)
    
    print(f"✅ Confidence scores computed")
    print(f"   Range: [{confidence_scores.min():.3f}, {confidence_scores.max():.3f}]")
    print(f"   Mean: {confidence_scores.mean():.3f}, Std: {confidence_scores.std():.3f}")
    
    # 5. Generate RC Curve (CORRECT methodology)
    print("\n" + "=" * 70)
    print("GENERATING RISK-COVERAGE CURVE")
    print("=" * 70)
    
    rc_df = generate_rc_curve_correct(
        confidence_scores, preds, test_labels, class_to_group_cpu, num_groups,
        coverage_min=CONFIG['eval_params']['coverage_min'],
        coverage_max=CONFIG['eval_params']['coverage_max'],
        coverage_step=CONFIG['eval_params']['coverage_step']
    )
    
    # Save RC curve data
    rc_df.to_csv(output_dir / 'rc_curve_correct.csv', index=False)
    print(f"\n💾 Saved RC curve data to {output_dir / 'rc_curve_correct.csv'}")
    
    # 6. Compute AURC metrics
    print("\n" + "=" * 70)
    print("AURC METRICS (Coverage 0.2-1.0)")
    print("=" * 70)
    
    results = {}
    
    for metric_name, error_col in [
        ('Standard', 'standard_error'),
        ('Balanced', 'balanced_error'),
        ('Worst-Group', 'worst_error')
    ]:
        aurc = compute_aurc(rc_df, error_col, 0.2, 1.0)
        eaurc, aurc_method, aurc_random = compute_eaurc(rc_df, error_col, 0.2, 1.0)
        
        results[f'aurc_{error_col.split("_")[0]}'] = {
            'aurc': aurc,
            'eaurc': eaurc,
            'aurc_random': aurc_random
        }
        
        print(f"\n{metric_name} Error:")
        print(f"   AURC: {aurc:.6f}")
        print(f"   E-AURC: {eaurc:.6f} (vs random: {aurc_random:.6f})")
    
    # Sanity check
    print("\n✅ Sanity Check:")
    aurc_std = results['aurc_standard']['aurc']
    aurc_bal = results['aurc_balanced']['aurc']
    aurc_wst = results['aurc_worst']['aurc']
    
    if aurc_wst >= aurc_bal >= aurc_std:
        print(f"   ✅ PASSED: AURC(worst={aurc_wst:.4f}) >= AURC(balanced={aurc_bal:.4f}) >= AURC(standard={aurc_std:.4f})")
    else:
        print(f"   ⚠️ WARNING: Expected worst >= balanced >= standard")
        print(f"      Got: worst={aurc_wst:.4f}, balanced={aurc_bal:.4f}, standard={aurc_std:.4f}")
    
    # 7. Metrics at specific coverages (for comparison)
    print("\n" + "=" * 70)
    print("METRICS AT SPECIFIC COVERAGES")
    print("=" * 70)
    
    specific_coverages = [0.6, 0.7, 0.8, 0.9]
    results['metrics_at_coverage'] = {}
    
    for target_cov in specific_coverages:
        # Find closest coverage in RC curve
        idx = (rc_df['coverage'] - target_cov).abs().idxmin()
        row = rc_df.iloc[idx]
        
        results['metrics_at_coverage'][f'cov_{target_cov}'] = {
            'actual_coverage': row['coverage'],
            'standard_error': row['standard_error'],
            'balanced_error': row['balanced_error'],
            'worst_error': row['worst_error'],
            'head_error': row['head_error'],
            'tail_error': row['tail_error']
        }
        
        print(f"\nCoverage ≈ {row['coverage']:.3f} (target {target_cov:.1f}):")
        print(f"   Standard Error: {row['standard_error']:.4f}")
        print(f"   Balanced Error: {row['balanced_error']:.4f}")
        print(f"   Worst-Group Error: {row['worst_error']:.4f}")
        print(f"   Head Error: {row['head_error']:.4f}, Tail Error: {row['tail_error']:.4f}")
    
    # 8. ECE
    ece = calculate_ece(eta_mix, test_labels)
    results['ece'] = ece
    print(f"\n📊 Expected Calibration Error (ECE): {ece:.4f}")
    
    # 9. Save all results
    results['config'] = CONFIG
    results['checkpoint_info'] = {
        'alpha': alpha_star.cpu().tolist(),
        'mu': mu_star.cpu().tolist(),
        'source': checkpoint.get('source', 'unknown')
    }
    
    with open(output_dir / 'metrics_correct.json', 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\n💾 Saved all metrics to {output_dir / 'metrics_correct.json'}")
    
    # 10. Plot RC curves
    plot_rc_curves(rc_df, output_dir / 'rc_curves_correct.png')
    
    # Final summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Dataset: {CONFIG['dataset']['name']}")
    print(f"Test samples: {num_test_samples}")
    print(f"Coverage range: {CONFIG['eval_params']['coverage_min']:.0%} to {CONFIG['eval_params']['coverage_max']:.0%}")
    print(f"\n🎯 AURC Results (Lower is Better):")
    print(f"   Standard AURC: {results['aurc_standard']['aurc']:.6f}")
    print(f"   Balanced AURC: {results['aurc_balanced']['aurc']:.6f}")
    print(f"   Worst-Group AURC: {results['aurc_worst']['aurc']:.6f}")
    print(f"\n📝 These results are comparable with 'Learning to Reject Meets Long-tail Learning'")
    print("=" * 70)

if __name__ == '__main__':
    main()
