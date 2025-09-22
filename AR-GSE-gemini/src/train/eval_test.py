# src/train/eval_test.py
import torch
import torchvision
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Import our custom modules
from src.models.argse import AR_GSE
from src.data.groups import get_class_to_group
from src.data.datasets import get_cifar100_lt_counts
from src.metrics.selective_metrics import calculate_selective_errors
from src.metrics.rc_curve import generate_rc_curve, calculate_aurc
from src.metrics.calibration import calculate_ece
from src.metrics.bootstrap import bootstrap_ci

# --- CONFIGURATION (sẽ được thay thế bằng Hydra) ---
CONFIG = {
    'dataset': {
        'name': 'cifar100_lt',
        'num_classes': 100,
    },
    'model_name': 'argse_balanced', # 'argse_balanced' or 'argse_worst'
    'checkpoint_path': './checkpoints/argse/cifar100_lt/argse_balanced.ckpt',
    'experts': {
        'names': ['ce', 'logitadjust', 'balsoftmax'],
        'logits_dir': './outputs/logits',
    },
    'eval_params': {
        'coverage_points': [0.7, 0.8, 0.9],
        'bootstrap_n': 1000, # Giảm xuống 100 để chạy nhanh hơn
        'bootstrap_ci': 0.95,
    },
    'output_dir': './results/cifar100_lt',
    'seed': 42
}
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- MAIN EVALUATION SCRIPT ---
def main():
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    output_dir = Path(CONFIG['output_dir']) / CONFIG['model_name']
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Group Info
    class_counts = get_cifar100_lt_counts(imb_factor=100)
    class_to_group = get_class_to_group(class_counts, K=2, head_ratio=0.5)
    num_groups = class_to_group.max().item() + 1
    
    # 2. Load Model
    num_experts = len(CONFIG['experts']['names'])
    gating_feature_dim = 4 * num_experts
    model = AR_GSE(num_experts, CONFIG['dataset']['num_classes'], num_groups, gating_feature_dim).to(DEVICE)
    model.load_state_dict(torch.load(CONFIG['checkpoint_path'], map_location=DEVICE))
    model.eval()
    print(f"Loaded model from {CONFIG['checkpoint_path']}")

    # 3. Load Test Data (Logits and Labels)
    print("Loading test logits and labels...")
    logits_root = Path(CONFIG['experts']['logits_dir']) / CONFIG['dataset']['name']
    
    stacked_logits = torch.zeros(10000, num_experts, CONFIG['dataset']['num_classes'])
    for i, expert_name in enumerate(CONFIG['experts']['names']):
        logits_path = logits_root / expert_name / "test_logits.pt"
        stacked_logits[:, i, :] = torch.load(logits_path, map_location='cpu')
    
    test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False)
    test_labels = torch.tensor(test_dataset.targets)

    # 4. Get Model Predictions on Test Set
    with torch.no_grad():
        # AR-GSE model expects a batch dimension, so add one
        outputs = model(stacked_logits.to(DEVICE), c=0.05, tau=10.0, class_to_group=class_to_group)
    
    margins = outputs['margin'].cpu()
    eta_mix = outputs['eta_mix'].cpu()
    _, preds = torch.max(eta_mix, 1)

    # 5. Calculate All Metrics
    results = {}
    print("\nCalculating metrics...")

    # 5.1 RC Curve and AURC
    rc_df = generate_rc_curve(margins, preds, test_labels, class_to_group, num_groups)
    rc_df.to_csv(output_dir / 'rc_curve.csv', index=False)
    print(f"Saved RC curve data to {output_dir / 'rc_curve.csv'}")
    
    aurc_bal = calculate_aurc(rc_df, 'balanced_error')
    aurc_wst = calculate_aurc(rc_df, 'worst_error')
    results['aurc_balanced'] = aurc_bal
    results['aurc_worst'] = aurc_wst
    print(f"AURC (Balanced): {aurc_bal:.4f}, AURC (Worst): {aurc_wst:.4f}")

    # 5.2 Bootstrap CI for AURC (Balanced)
    def aurc_metric_func(m, p, l):
        rc_df = generate_rc_curve(m, p, l, class_to_group, num_groups, num_points=51) # Use fewer points for speed
        return calculate_aurc(rc_df, 'balanced_error')

    mean_aurc, lower, upper = bootstrap_ci((margins, preds, test_labels), aurc_metric_func, n_bootstraps=CONFIG['eval_params']['bootstrap_n'])
    results['aurc_balanced_bootstrap'] = {'mean': mean_aurc, '95ci_lower': lower, '95ci_upper': upper}
    print(f"AURC (Balanced) Bootstrap 95% CI: [{lower:.4f}, {upper:.4f}]")

    # 5.3 Metrics @ Fixed Coverage
    results_at_coverage = {}
    for cov_target in CONFIG['eval_params']['coverage_points']:
        # Find the margin threshold for this coverage
        threshold = torch.quantile(margins, 1.0 - cov_target)
        accepted_mask = margins >= threshold
        
        metrics = calculate_selective_errors(preds, test_labels, accepted_mask, class_to_group, num_groups)
        results_at_coverage[f'cov_{cov_target}'] = metrics
        print(f"Metrics @ {metrics['coverage']:.2f} coverage: Bal. Err={metrics['balanced_error']:.4f}, Worst Err={metrics['worst_error']:.4f}")
    results['metrics_at_coverage'] = results_at_coverage

    # 5.4 Calibration (ECE)
    ece = calculate_ece(eta_mix, test_labels)
    results['ece'] = ece
    print(f"Expected Calibration Error (ECE): {ece:.4f}")

    # 6. Save Results
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved all metrics to {output_dir / 'metrics.json'}")

    # 7. Plot and Save RC Curve
    plt.figure()
    plt.plot(rc_df['coverage'], rc_df['balanced_error'], label='Balanced Error')
    plt.plot(rc_df['coverage'], rc_df['worst_error'], label='Worst-Group Error', linestyle='--')
    plt.xlabel('Coverage')
    plt.ylabel('Selective Risk (Error)')
    plt.title(f'Risk-Coverage Curve for {CONFIG["model_name"]}')
    plt.grid(True, linestyle=':')
    plt.legend()
    plt.savefig(output_dir / 'rc_curve.png')
    print(f"Saved RC curve plot to {output_dir / 'rc_curve.png'}")

if __name__ == '__main__':
    main()