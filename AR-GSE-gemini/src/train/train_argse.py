# src/train/train_argse.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import collections

# Import our custom modules
from src.models.argse import AR_GSE
from src.models.primal_dual import primal_dual_step
from src.models.surrogate_losses import selective_cls_loss
from src.data.groups import get_class_to_group, get_class_to_group_by_threshold
from src.data.datasets import get_cifar100_lt_counts

# --- CONFIGURATION (sẽ được thay thế bằng Hydra) ---
CONFIG = {
    'dataset': {
        'name': 'cifar100_lt_if100',
        'splits_dir': './data/cifar100_lt_if100_splits',
        'num_classes': 100,
    },
    'grouping': {
        'method': 'threshold',   # 'threshold' or 'ratio'
        'threshold': 20,         # classes with >threshold samples are head
        'head_ratio': 0.69,      # if using 'ratio' method
        'K': 2,                  # head/tail
    },
    'experts': {
        'names': ['ce_baseline', 'logitadjust_baseline', 'balsoftmax_baseline'],
        'logits_dir': './outputs/logits/',
    },
    'argse_params': {
        'mode': 'balanced',      # 'balanced' or 'worst'
        'epochs': 100,
        'batch_size': 256,
        'c': 0.7,                # reject cost (cao -> ít reject hơn)
        'alpha_clip': 5e-2,      # α_min
        'lambda_ent': 0.0,       # TẮT entropy reg để tránh ép gating ~ uniform
    },
    'optimizers': {
        'phi_lr': 1e-2,          # LR gating nhẹ để ổn định
        'alpha_lr': 5e-3,        # LR cho α (riêng)
        'mu_lr': 1e-2,           # LR cho μ (riêng)
        'rho': 5e-3,             # dual step cho Λ
    },
    'scheduler': {
        'tau_start': 1.0,        # cố định τ=1.0 để tránh bão hoà sigmoid
        'tau_end': 1.0,
        'tau_warmup_epochs': 0,
    },
    'worst_group_params': {
        'eg_xi': 1.0,            # cho chế độ 'worst'
    },
    'output': {
        'checkpoints_dir': './checkpoints/argse_balanced/',
    },
    'seed': 42
}
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- HELPER FUNCTIONS ---

def load_data_from_logits(config):
    """
    Loads pre-computed logits and labels for tuneV and val_lt splits.
    """
    logits_root = Path(config['experts']['logits_dir']) / config['dataset']['name']
    splits_dir = Path(config['dataset']['splits_dir'])
    expert_names = config['experts']['names']
    num_experts = len(expert_names)
    num_classes = config['dataset']['num_classes']
    
    dataloaders = {}
    
    # Base datasets
    cifar_train_full = torchvision.datasets.CIFAR100(root='./data', train=True, download=False)
    cifar_test_full  = torchvision.datasets.CIFAR100(root='./data', train=False, download=False)
    
    # Use tuneV (train split) and val_lt (test split)
    splits_config = [
        {'split_name': 'tuneV', 'base_dataset': cifar_train_full, 'indices_file': 'tuneV_indices.json'},
        {'split_name': 'val_lt', 'base_dataset': cifar_test_full,  'indices_file': 'val_lt_indices.json'}
    ]
    
    for split in splits_config:
        split_name = split['split_name']
        base_dataset = split['base_dataset']
        indices_path = splits_dir / split['indices_file']
        print(f"Loading data for split: {split_name}")
        if not indices_path.exists():
            raise FileNotFoundError(f"Missing indices file: {indices_path}")
        indices = json.loads(indices_path.read_text())

        # Stack expert logits
        stacked_logits = torch.zeros(len(indices), num_experts, num_classes)
        for i, expert_name in enumerate(expert_names):
            lp = logits_root / expert_name / f"{split_name}_logits.pt"
            if not lp.exists():
                raise FileNotFoundError(f"Missing logits file: {lp}")
            stacked_logits[:, i, :] = torch.load(lp, map_location='cpu')

        labels = torch.tensor(np.array(base_dataset.targets)[indices])
        dataset = TensorDataset(stacked_logits, labels)
        dataloaders[split_name] = DataLoader(
            dataset,
            batch_size=config['argse_params']['batch_size'],
            shuffle=(split_name == 'tuneV'),
            num_workers=4
        )

    return dataloaders['tuneV'], dataloaders['val_lt']


def update_beta_eg(current_beta, group_errors, xi):
    """Exponentiated-Gradient update for β (worst-group mode)."""
    if isinstance(group_errors, list):
        group_errors = torch.tensor(group_errors, device=current_beta.device)
    new_beta = current_beta * torch.exp(xi * group_errors)
    return new_beta / new_beta.sum()


def eval_epoch(model, loader, c, class_to_group, tau_eval: float = 1.0):
    """Evaluate with hard decisions at a fixed τ (default 1.0)."""
    model.eval()
    all_margins, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for logits, labels in loader:
            logits, labels = logits.to(DEVICE), labels.to(DEVICE)
            out = model(logits, c, tau_eval, class_to_group)

            # re-weighted scores: α_{grp(y)} * η̃_y(x)
            alpha = model.alpha.to(logits.device)
            ctg   = class_to_group.to(logits.device)
            # Clamp để an toàn số
            eta_mix = out['eta_mix'].clamp(1e-6, 1-1e-6)
            reweighted = alpha[ctg] * eta_mix
            _, preds = torch.max(reweighted, 1)

            all_margins.append(out['margin'])
            all_preds.append(preds)
            all_labels.append(labels)

    margins = torch.cat(all_margins)
    preds   = torch.cat(all_preds)
    labels  = torch.cat(all_labels)

    accepted = (margins > 0)
    n_acc = accepted.sum().item()
    n_tot = len(labels)
    coverage = n_acc / n_tot if n_tot > 0 else 0.0

    if n_acc == 0:
        return {'coverage': 0, 'balanced_error': 1.0, 'worst_error': 1.0, 'group_errors': [1.0]*model.num_groups}

    acc_preds  = preds[accepted]
    acc_labels = labels[accepted]
    acc_groups = class_to_group[acc_labels]
    group_errors = []
    for k in range(model.num_groups):
        mask = (acc_groups == k)
        if mask.sum() == 0:
            group_errors.append(1.0)
            continue
        acc_k = (acc_preds[mask] == acc_labels[mask]).float().mean().item()
        group_errors.append(1.0 - acc_k)

    return {
        'coverage': coverage,
        'balanced_error': float(np.mean(group_errors)),
        'worst_error': float(np.max(group_errors)),
        'group_errors': group_errors,
    }

# --- MAIN TRAINING SCRIPT ---
def main():
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])

    # 1) Dataloaders
    train_loader, val_loader = load_data_from_logits(CONFIG)
    print(f"✅ Loaded training data: {len(train_loader)} batches")
    print(f"✅ Loaded validation data: {len(val_loader)} batches")

    # 2) Grouping
    class_counts = get_cifar100_lt_counts(imb_factor=100)
    if CONFIG['grouping']['method'] == 'threshold':
        print(f"Using threshold-based grouping with threshold={CONFIG['grouping']['threshold']}")
        class_to_group = get_class_to_group_by_threshold(
            class_counts, threshold=CONFIG['grouping']['threshold']
        ).to(DEVICE)
        head = (class_to_group == 0).sum().item()
        tail = (class_to_group == 1).sum().item()
        print(f"Groups created: {head} head classes (>{CONFIG['grouping']['threshold']} samples), "
              f"{tail} tail classes (<= {CONFIG['grouping']['threshold']})")
    else:
        print(f"Using ratio-based grouping with head_ratio={CONFIG['grouping']['head_ratio']}")
        class_to_group = get_class_to_group(
            class_counts, K=CONFIG['grouping']['K'], head_ratio=CONFIG['grouping']['head_ratio']
        ).to(DEVICE)
        head = int(CONFIG['dataset']['num_classes'] * CONFIG['grouping']['head_ratio'])
        tail = CONFIG['dataset']['num_classes'] - head
        print(f"Groups created: {head} head classes, {tail} tail classes")
    num_groups = class_to_group.max().item() + 1

    # 3) Model & optimizers
    num_experts = len(CONFIG['experts']['names'])
    gating_feature_dim = 4 * num_experts
    model = AR_GSE(num_experts, CONFIG['dataset']['num_classes'], num_groups, gating_feature_dim).to(DEVICE)

    # init α, μ ổn định hơn
    with torch.no_grad():
        model.mu.add_(0.01 * torch.randn_like(model.mu))
        model.mu.sub_(model.mu.mean())  # zero-mean
        noise = 0.05 * torch.randn_like(model.alpha)
        model.alpha.copy_(torch.exp(noise))
        log_alpha = model.alpha.log()
        model.alpha.copy_(torch.exp(log_alpha - log_alpha.mean()))  # geomean(α)=1

    optimizers = {
        'phi': optim.Adam(model.gating_net.parameters(), lr=CONFIG['optimizers']['phi_lr']),
        'alpha_mu': optim.Adam([
            {'params': [model.alpha], 'lr': CONFIG['optimizers']['alpha_lr']},
            {'params': [model.mu],    'lr': CONFIG['optimizers']['mu_lr']},
        ])
    }

    # 4) Training state
    tau = CONFIG['scheduler']['tau_start']  # = tau_end = 1.0
    beta = torch.ones(num_groups, device=DEVICE) / num_groups

    # 5) Train loop
    best_val_metric = float('inf')
    epochs_no_improve = 0

    for epoch in range(CONFIG['argse_params']['epochs']):
        print(f"\n--- Epoch {epoch+1}/{CONFIG['argse_params']['epochs']} ---")
        model.train()
        epoch_stats = collections.defaultdict(list)

        for batch in tqdm(train_loader, desc="Training"):
            params = {
                'device': DEVICE,
                'c': CONFIG['argse_params']['c'],
                'tau': tau,
                'beta': beta,
                'class_to_group': class_to_group,
                'lambda_ent': CONFIG['argse_params']['lambda_ent'],
                'alpha_clip': CONFIG['argse_params']['alpha_clip'],
                'rho': CONFIG['optimizers']['rho'],
            }
            stats, _ = primal_dual_step(model, batch, optimizers, selective_cls_loss, params)

            # NaN guard
            if (not np.isfinite(stats['loss_total'])) or (not np.isfinite(stats['mean_margin'])):
                print("⚠️  Detected NaN/Inf in stats; consider lowering LRs or checking inputs.")
                # Bạn có thể break hoặc continue tuỳ ý. Ở đây mình break để an toàn.
                break

            for k, v in stats.items():
                epoch_stats[k].append(v)

        # Log
        cov = np.mean(epoch_stats['mean_coverage']) if epoch_stats['mean_coverage'] else float('nan')
        mrg = np.mean(epoch_stats['mean_margin'])   if epoch_stats['mean_margin']   else float('nan')
        tot = np.mean(epoch_stats['loss_total'])    if epoch_stats['loss_total']    else float('nan')
        print(f"Epoch {epoch+1} | Tau: {tau:.2f} | Loss: {tot:.4f} | Coverage: {cov:.3f} | Margin: {mrg:.3f}")
        print(f"[alpha] mean={model.alpha.mean().item():.3f} "
              f"std={model.alpha.std().item():.3f} "
              f"min={model.alpha.min().item():.3f} "
              f"max={model.alpha.max().item():.3f}")
        print(f"[mu]    mean={model.mu.mean().item():.3f} "
              f"std={model.mu.std().item():.3f} "
              f"min={model.mu.min().item():.3f} "
              f"max={model.mu.max().item():.3f})")

        # 6) Eval
        val = eval_epoch(model, val_loader, CONFIG['argse_params']['c'], class_to_group, tau_eval=1.0)
        key = 'worst_error' if CONFIG['argse_params']['mode'] == 'worst' else 'balanced_error'
        cur = val[key]
        print(f"Validation | Coverage: {val['coverage']:.3f} | Bal. Err: {val['balanced_error']:.4f} | Worst Err: {val['worst_error']:.4f}")

        # worst-group EG (nếu dùng)
        if CONFIG['argse_params']['mode'] == 'worst':
            beta = update_beta_eg(beta, val['group_errors'], CONFIG['worst_group_params']['eg_xi'])
            print(f"Updated Beta: {[f'{b:.3f}' for b in beta.tolist()]}")

        # 7) Early-stopping & checkpoint
        if cur < best_val_metric and np.isfinite(cur):
            best_val_metric = cur
            epochs_no_improve = 0
            outdir = Path(CONFIG['output']['checkpoints_dir']) / CONFIG['dataset']['name']
            outdir.mkdir(parents=True, exist_ok=True)
            ckpt = outdir / f"argse_{CONFIG['argse_params']['mode']}.ckpt"
            print(f"💾 New best! Saving to {ckpt}")
            torch.save(model.state_dict(), ckpt)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= 15:
            print("Early stopping after 15 epochs with no improvement.")
            break

if __name__ == '__main__':
    main()
