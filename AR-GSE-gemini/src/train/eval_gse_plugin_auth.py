"""
Evaluation script tailored for AR-GSE methodology.
Properly uses per-group α, μ, t parameters from plugin optimization.
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from tqdm import tqdm

from src.models.argse import AR_GSE
from src.metrics.selective_metrics import calculate_selective_errors
from src.metrics.rc_curve import generate_rc_curve, calculate_aurc
from src.metrics.calibration import calculate_ece
from src.metrics.bootstrap import bootstrap_ci
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
        'bootstrap_n': 500,
    },
    'argse_eval': {
        # AR-GSE specific: use optimal per-group parameters
        'use_optimal_params': True,
        # Sweep α scaling factor to generate RC curve
        'alpha_scale_values': np.linspace(0.5, 2.0, 31),  # Scale α_k → explore coverage
        # Or sweep per-group thresholds
        'threshold_shift_values': np.linspace(-1.5, 0.5, 61),  # Shift t_k - EXPANDED RANGE
        # Or sweep rejection cost (paper methodology)
        'rejection_cost_values': np.linspace(-2.0, 1.0, 61),  # Sweep c - EXPANDED for full coverage
        # Evaluation modes
        'eval_mode': 'rejection_cost',  # 'alpha_scale', 'threshold_shift', or 'rejection_cost'
    },
    'plugin_checkpoint': './checkpoints/argse_worst_eg_improved/cifar100_lt_if100/gse_balanced_plugin.ckpt',
    'output_dir': './results_worst_eg_improved/cifar100_lt_if100',
    'seed': 42
}


def load_test_data():
    """Load test data and expert logits."""
    print("📊 Loading test data...")
    import torchvision
    import json
    
    dataset = CONFIG['dataset']['name']
    logits_root = Path(CONFIG['experts']['logits_dir']) / dataset
    splits_dir = Path(CONFIG['dataset']['splits_dir'])
    
    # Load test indices
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
    
    # Load test labels from CIFAR100 dataset
    full_test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False)
    labels = torch.tensor(np.array(full_test_dataset.targets)[test_indices])
    
    print(f"✅ Loaded {len(labels)} test samples")
    print(f"   Expert logits shape: {stacked_logits.shape}")
    
    return stacked_logits, labels


def get_mixture_posteriors(model, logits):
    """Get mixture posteriors η̃(x) from AR-GSE model."""
    model.eval()
    with torch.no_grad():
        logits_dev = logits.to(DEVICE)
        
        # Get expert posteriors
        expert_posteriors = torch.softmax(logits_dev, dim=-1)  # [B, E, C]
        
        # Get gating weights
        gating_features = model.feature_builder(logits_dev)
        gating_weights = torch.softmax(model.gating_net(gating_features), dim=1)  # [B, E]
        
        # Mixture posteriors
        eta_mix = torch.einsum('be,bec->bc', gating_weights, expert_posteriors)  # [B, C]
        
    return eta_mix.cpu()


def evaluate_at_alpha_scale(eta_mix, labels, alpha_opt, mu_opt, t_opt, class_to_group, 
                            alpha_scale, K):
    """
    Evaluate by scaling α_k → α_k * scale.
    This explores different acceptance rates while preserving group structure.
    """
    alpha_scaled = alpha_opt * alpha_scale
    
    # Compute margins with scaled alpha
    margins_raw = compute_margin(eta_mix, alpha_scaled, mu_opt, 0.0, class_to_group)
    
    # Predictions using scaled alpha
    preds = (alpha_scaled[class_to_group] * eta_mix).argmax(dim=1)
    pred_groups = class_to_group[preds]
    
    # Apply per-group thresholds
    t_group_tensor = torch.tensor([t_opt[k] for k in range(K)], dtype=torch.float32)
    thresholds_per_sample = t_group_tensor[pred_groups]
    
    accepted = margins_raw >= thresholds_per_sample
    coverage = accepted.float().mean().item()
    
    # Compute errors
    errors = calculate_selective_errors(preds, labels, accepted, class_to_group, K)
    
    # Calculate per-group coverage
    y_groups = class_to_group[labels]
    group_coverages = []
    for k in range(K):
        group_mask = (y_groups == k)
        if group_mask.sum() > 0:
            group_cov = (accepted & group_mask).float().sum().item() / group_mask.sum().item()
        else:
            group_cov = 0.0
        group_coverages.append(group_cov)
    
    return {
        'alpha_scale': alpha_scale,
        'coverage': coverage,
        'balanced_error': errors['balanced_error'],
        'worst_error': errors['worst_error'],
        'head_error': errors['group_errors'][0],
        'tail_error': errors['group_errors'][1],
        'head_coverage': group_coverages[0],
        'tail_coverage': group_coverages[1],
    }


def evaluate_at_threshold_shift(eta_mix, labels, alpha_opt, mu_opt, t_opt, class_to_group,
                                threshold_shift, K):
    """
    Evaluate by shifting all t_k → t_k + shift.
    This directly controls coverage while using optimal α, μ.
    """
    t_shifted = [t_opt[k] + threshold_shift for k in range(K)]
    
    # Compute margins with optimal alpha, mu
    margins_raw = compute_margin(eta_mix, alpha_opt, mu_opt, 0.0, class_to_group)
    
    # Predictions using optimal alpha
    preds = (alpha_opt[class_to_group] * eta_mix).argmax(dim=1)
    pred_groups = class_to_group[preds]
    
    # Apply shifted per-group thresholds
    t_group_tensor = torch.tensor([t_shifted[k] for k in range(K)], dtype=torch.float32)
    thresholds_per_sample = t_group_tensor[pred_groups]
    
    accepted = margins_raw >= thresholds_per_sample
    coverage = accepted.float().mean().item()
    
    # Compute errors
    errors = calculate_selective_errors(preds, labels, accepted, class_to_group, K)
    
    # Calculate per-group coverage
    y_groups = class_to_group[labels]
    group_coverages = []
    for k in range(K):
        group_mask = (y_groups == k)
        if group_mask.sum() > 0:
            group_cov = (accepted & group_mask).float().sum().item() / group_mask.sum().item()
        else:
            group_cov = 0.0
        group_coverages.append(group_cov)
    
    return {
        'threshold_shift': threshold_shift,
        'coverage': coverage,
        'balanced_error': errors['balanced_error'],
        'worst_error': errors['worst_error'],
        'head_error': errors['group_errors'][0],
        'tail_error': errors['group_errors'][1],
        'head_coverage': group_coverages[0],
        'tail_coverage': group_coverages[1],
    }


def evaluate_at_rejection_cost(eta_mix, labels, alpha_opt, mu_opt, t_opt, class_to_group,
                               rejection_cost, K):
    """
    Evaluate by varying rejection cost c (paper methodology).
    
    Margin computation: m_k(x) = α_k * η̃_ŷ(x) + μ_k - c
    
    This differs from threshold_shift:
    - rejection_cost: Modifies margin calculation itself (affects confidence scoring)
    - threshold_shift: Post-hoc adjustment of decision boundary
    
    Args:
        eta_mix: Mixture posteriors [N, C]
        labels: True labels [N]
        alpha_opt: Optimal alpha parameters [K]
        mu_opt: Optimal mu parameters [K]
        t_opt: Optimal thresholds [K] (can be list or single value)
        class_to_group: Class to group mapping [C]
        rejection_cost: Rejection cost c
        K: Number of groups
        
    Returns:
        Dict with metrics
    """
    # Compute margins WITH rejection cost
    margins_raw = compute_margin(eta_mix, alpha_opt, mu_opt, rejection_cost, class_to_group)
    
    # Predictions using optimal alpha
    preds = (alpha_opt[class_to_group] * eta_mix).argmax(dim=1)
    pred_groups = class_to_group[preds]
    
    # Apply per-group thresholds (use optimal t* from training)
    if isinstance(t_opt, list):
        t_group_tensor = torch.tensor(t_opt, dtype=torch.float32)
    else:
        t_group_tensor = torch.tensor([t_opt] * K, dtype=torch.float32)
    
    thresholds_per_sample = t_group_tensor[pred_groups]
    
    # Accept if margin >= threshold
    accepted = margins_raw >= thresholds_per_sample
    coverage = accepted.float().mean().item()
    
    # Compute errors
    errors = calculate_selective_errors(preds, labels, accepted, class_to_group, K)
    
    # Calculate per-group coverage
    y_groups = class_to_group[labels]
    group_coverages = []
    for k in range(K):
        group_mask = (y_groups == k)
        if group_mask.sum() > 0:
            group_cov = (accepted & group_mask).float().sum().item() / group_mask.sum().item()
        else:
            group_cov = 0.0
        group_coverages.append(group_cov)
    
    return {
        'rejection_cost': rejection_cost,
        'coverage': coverage,
        'balanced_error': errors['balanced_error'],
        'worst_error': errors['worst_error'],
        'head_error': errors['group_errors'][0],
        'tail_error': errors['group_errors'][1],
        'head_coverage': group_coverages[0],
        'tail_coverage': group_coverages[1],
    }


def generate_argse_rc_curve(eta_mix, labels, alpha_opt, mu_opt, t_opt, class_to_group, K,
                           eval_mode='threshold_shift'):
    """
    Generate Risk-Coverage curve using AR-GSE native methodology.
    
    Three modes:
    1. 'alpha_scale': Scale α_k to explore coverage (preserves group structure)
    2. 'threshold_shift': Shift t_k to control coverage (direct control)
    3. 'rejection_cost': Sweep rejection cost c (paper methodology)
    """
    print(f"📈 Generating AR-GSE RC curve (mode: {eval_mode})...")
    
    rc_points = []
    
    if eval_mode == 'alpha_scale':
        scale_values = CONFIG['argse_eval']['alpha_scale_values']
        for scale in tqdm(scale_values, desc="Sweeping α scale"):
            result = evaluate_at_alpha_scale(eta_mix, labels, alpha_opt, mu_opt, t_opt,
                                            class_to_group, scale, K)
            rc_points.append(result)
    
    elif eval_mode == 'threshold_shift':
        shift_values = CONFIG['argse_eval']['threshold_shift_values']
        for shift in tqdm(shift_values, desc="Sweeping threshold shift"):
            result = evaluate_at_threshold_shift(eta_mix, labels, alpha_opt, mu_opt, t_opt,
                                                class_to_group, shift, K)
            rc_points.append(result)
    
    elif eval_mode == 'rejection_cost':
        cost_values = CONFIG['argse_eval']['rejection_cost_values']
        for cost in tqdm(cost_values, desc="Sweeping rejection cost"):
            result = evaluate_at_rejection_cost(eta_mix, labels, alpha_opt, mu_opt, t_opt,
                                               class_to_group, cost, K)
            rc_points.append(result)
    
    # Convert to DataFrame and sort by coverage
    rc_df = pd.DataFrame(rc_points)
    rc_df = rc_df.sort_values('coverage').reset_index(drop=True)
    
    return rc_df


def compute_argse_aurc(rc_df, metric='balanced_error'):
    """Compute AURC from AR-GSE RC curve using trapezoidal integration."""
    coverages = rc_df['coverage'].values
    errors = rc_df[metric].values
    
    # Sort by coverage
    sort_idx = np.argsort(coverages)
    coverages = coverages[sort_idx]
    errors = errors[sort_idx]
    
    # Trapezoidal integration
    aurc = np.trapz(errors, coverages)
    
    return aurc


def plot_argse_rc_curves(rc_df, output_dir):
    """Plot AR-GSE specific RC curves."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Balanced Error
    axes[0].plot(rc_df['coverage'], rc_df['balanced_error'], 'b-', linewidth=2, label='Balanced Error')
    axes[0].set_xlabel('Coverage', fontsize=12)
    axes[0].set_ylabel('Balanced Error', fontsize=12)
    axes[0].set_title('AR-GSE: Balanced Error vs Coverage', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot 2: Worst-Group Error
    axes[1].plot(rc_df['coverage'], rc_df['worst_error'], 'r-', linewidth=2, label='Worst-Group Error')
    axes[1].set_xlabel('Coverage', fontsize=12)
    axes[1].set_ylabel('Worst-Group Error', fontsize=12)
    axes[1].set_title('AR-GSE: Worst-Group Error vs Coverage', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Plot 3: Group-wise comparison
    axes[2].plot(rc_df['coverage'], rc_df['head_error'], 'g-', linewidth=2, label='Head Error')
    axes[2].plot(rc_df['coverage'], rc_df['tail_error'], 'm-', linewidth=2, label='Tail Error')
    axes[2].set_xlabel('Coverage', fontsize=12)
    axes[2].set_ylabel('Error Rate', fontsize=12)
    axes[2].set_title('AR-GSE: Group-wise Error vs Coverage', fontsize=13, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'argse_rc_curves.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'argse_rc_curves.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved AR-GSE RC curves to {output_dir}")


def main():
    print("=" * 70)
    print("=== AR-GSE Native Evaluation ===")
    print("=" * 70)
    
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    ckpt_path = Path(CONFIG['plugin_checkpoint'])
    print(f"📂 Loading plugin checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    alpha_opt = checkpoint['alpha'].cpu()
    mu_opt = checkpoint['mu'].cpu()
    
    # Handle per-group thresholds
    if 't_group' in checkpoint:
        t_opt = checkpoint['t_group']
        source = "per-group thresholds"
    else:
        t_opt = [checkpoint.get('threshold', checkpoint.get('c', 0.0))] * 2
        source = "single threshold (fallback)"
    
    K = len(alpha_opt)
    
    print(f"✅ Loaded optimal AR-GSE parameters:")
    print(f"   α* = {alpha_opt.tolist()}")
    print(f"   μ* = {mu_opt.tolist()}")
    print(f"   t* = {t_opt}")
    print(f"   Source: {source}")
    
    # Load test data
    test_logits, test_labels = load_test_data()
    
    # Get class-to-group mapping from checkpoint (already loaded above)
    class_to_group = checkpoint['class_to_group'].cpu()
    num_classes = CONFIG['dataset']['num_classes']
    print(f"📊 Group division: {(class_to_group == 0).sum()} head, {(class_to_group == 1).sum()} tail")
    
    # Build AR-GSE model
    num_experts = len(CONFIG['experts']['names'])
    feature_dim = checkpoint.get('gating_feature_dim', 24)
    
    model = AR_GSE(
        num_experts=num_experts,
        num_classes=num_classes,
        num_groups=K,
        gating_feature_dim=feature_dim
    ).to(DEVICE)
    
    # Load gating weights
    if 'gating_state' in checkpoint:
        model.gating_net.load_state_dict(checkpoint['gating_state'])
        print("✅ Gating network loaded")
    
    # Set optimal parameters
    model.alpha.data = alpha_opt.to(DEVICE)
    model.mu.data = mu_opt.to(DEVICE)
    
    # Get mixture posteriors
    print("🔮 Computing mixture posteriors...")
    eta_mix = get_mixture_posteriors(model, test_logits)
    
    # Evaluate at optimal parameters
    print("\n" + "=" * 70)
    print("📊 PERFORMANCE AT OPTIMAL AR-GSE PARAMETERS")
    print("=" * 70)
    
    margins_raw = compute_margin(eta_mix, alpha_opt, mu_opt, 0.0, class_to_group)
    preds = (alpha_opt[class_to_group] * eta_mix).argmax(dim=1)
    pred_groups = class_to_group[preds]
    
    t_group_tensor = torch.tensor([t_opt[k] for k in range(K)], dtype=torch.float32)
    thresholds_per_sample = t_group_tensor[pred_groups]
    accepted_opt = margins_raw >= thresholds_per_sample
    
    errors_opt = calculate_selective_errors(preds, test_labels, accepted_opt, class_to_group, K)
    
    # Calculate per-group coverage for display
    y_groups = class_to_group[test_labels]
    group_coverages_opt = []
    for k in range(K):
        group_mask = (y_groups == k)
        if group_mask.sum() > 0:
            group_cov = (accepted_opt & group_mask).float().sum().item() / group_mask.sum().item()
        else:
            group_cov = 0.0
        group_coverages_opt.append(group_cov)
    
    print(f"✅ Coverage: {errors_opt['coverage']:.1%}")
    print(f"✅ Balanced Error: {errors_opt['balanced_error']:.4f}")
    print(f"✅ Worst-Group Error: {errors_opt['worst_error']:.4f}")
    print(f"   • Head Error: {errors_opt['group_errors'][0]:.4f} (coverage: {group_coverages_opt[0]:.1%})")
    print(f"   • Tail Error: {errors_opt['group_errors'][1]:.4f} (coverage: {group_coverages_opt[1]:.1%})")
    
    # Generate AR-GSE RC curve
    print("\n" + "=" * 70)
    print("📈 GENERATING AR-GSE RISK-COVERAGE CURVES")
    print("=" * 70)
    
    eval_mode = CONFIG['argse_eval']['eval_mode']
    rc_df = generate_argse_rc_curve(eta_mix, test_labels, alpha_opt, mu_opt, t_opt,
                                    class_to_group, K, eval_mode=eval_mode)
    
    # Compute AURC
    aurc_balanced = compute_argse_aurc(rc_df, 'balanced_error')
    aurc_worst = compute_argse_aurc(rc_df, 'worst_error')
    
    print(f"\n🏆 AR-GSE AURC RESULTS:")
    print(f"   • Balanced Error AURC: {aurc_balanced:.4f}")
    print(f"   • Worst-Group Error AURC: {aurc_worst:.4f}")
    
    # Save RC curve data
    rc_df.to_csv(output_dir / 'argse_rc_curve.csv', index=False)
    print(f"💾 Saved RC curve to {output_dir / 'argse_rc_curve.csv'}")
    
    # Plot
    plot_argse_rc_curves(rc_df, output_dir)
    
    # Bootstrap CI for AURC
    print("\n🔄 Computing bootstrap confidence intervals...")
    
    def aurc_metric_balanced(eta, labels):
        rc_boot = generate_argse_rc_curve(eta, labels, alpha_opt, mu_opt, t_opt,
                                         class_to_group, K, eval_mode=eval_mode)
        return compute_argse_aurc(rc_boot, 'balanced_error')
    
    def aurc_metric_worst(eta, labels):
        rc_boot = generate_argse_rc_curve(eta, labels, alpha_opt, mu_opt, t_opt,
                                         class_to_group, K, eval_mode=eval_mode)
        return compute_argse_aurc(rc_boot, 'worst_error')
    
    n_bootstrap = CONFIG['eval_params']['bootstrap_n']
    
    mean_bal, lower_bal, upper_bal = bootstrap_ci(
        (eta_mix, test_labels), aurc_metric_balanced, n_bootstraps=n_bootstrap
    )
    mean_worst, lower_worst, upper_worst = bootstrap_ci(
        (eta_mix, test_labels), aurc_metric_worst, n_bootstraps=n_bootstrap
    )
    
    print(f"\n📊 Bootstrap 95% Confidence Intervals (N={n_bootstrap}):")
    print(f"   • Balanced AURC: {mean_bal:.4f} [{lower_bal:.4f}, {upper_bal:.4f}]")
    print(f"   • Worst-Group AURC: {mean_worst:.4f} [{lower_worst:.4f}, {upper_worst:.4f}]")
    
    # Save final results
    results = {
        'optimal_performance': {
            'coverage': float(errors_opt['coverage']),
            'balanced_error': float(errors_opt['balanced_error']),
            'worst_error': float(errors_opt['worst_error']),
            'head_error': float(errors_opt['group_errors'][0]),
            'tail_error': float(errors_opt['group_errors'][1]),
            'head_coverage': float(group_coverages_opt[0]),
            'tail_coverage': float(group_coverages_opt[1]),
        },
        'aurc': {
            'balanced': float(aurc_balanced),
            'worst': float(aurc_worst),
            'balanced_ci': [float(lower_bal), float(upper_bal)],
            'worst_ci': [float(lower_worst), float(upper_worst)],
        },
        'parameters': {
            'alpha': alpha_opt.tolist(),
            'mu': mu_opt.tolist(),
            't': t_opt,
            'eval_mode': eval_mode,
        }
    }
    
    with open(output_dir / 'argse_native_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Saved results to {output_dir / 'argse_native_results.json'}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("🏆 AR-GSE NATIVE EVALUATION SUMMARY")
    print("=" * 70)
    print(f"📊 Dataset: {CONFIG['dataset']['name']}")
    print(f"📊 Evaluation mode: {eval_mode}")
    
    # Explain methodology
    if eval_mode == 'rejection_cost':
        print(f"📊 Methodology: Paper-style (sweep rejection cost c)")
    elif eval_mode == 'threshold_shift':
        print(f"📊 Methodology: AR-GSE native (uniform threshold shift δ)")
    elif eval_mode == 'alpha_scale':
        print(f"📊 Methodology: AR-GSE native (alpha scaling)")
    
    print(f"📊 Optimal parameters: α*={alpha_opt.tolist()}, μ*={mu_opt.tolist()}, t*={t_opt}")
    print(f"\n🎯 PERFORMANCE AT OPTIMAL THRESHOLD:")
    print(f"   • Coverage: {errors_opt['coverage']:.1%}")
    print(f"   • Balanced Error: {errors_opt['balanced_error']:.4f}")
    print(f"   • Worst-Group Error: {errors_opt['worst_error']:.4f}")
    print(f"\n🏆 AR-GSE AURC (Native Methodology):")
    print(f"   • Balanced: {aurc_balanced:.4f} [{lower_bal:.4f}, {upper_bal:.4f}]")
    print(f"   • Worst-Group: {aurc_worst:.4f} [{lower_worst:.4f}, {upper_worst:.4f}]")
    print("=" * 70)


if __name__ == '__main__':
    main()
