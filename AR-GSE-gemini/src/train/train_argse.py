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
from src.data.groups import get_class_to_group
from src.data.datasets import get_cifar100_lt_counts

# --- CONFIGURATION (sẽ được thay thế bằng Hydra) ---
CONFIG = {
    'dataset': {
        'name': 'cifar100_lt_if100',
        'splits_dir': './data/cifar100_lt_if100_splits',
        'num_classes': 100,
    },
    'experts': {
        'names': ['ce_baseline', 'logitadjust_baseline', 'balsoftmax_baseline'], # Cần khớp với output của M2
        'logits_dir': './outputs/logits/',
    },
    'argse_params': {
        'mode': 'balance',
        'epochs': 100,
        'batch_size': 256,
        'c': 0.8,  # Increase coverage requirement to encourage acceptance
        'alpha_clip': 1e-2,  # Increase minimum alpha
        'lambda_ent': 1e-2,  # Increase entropy regularization
    },
    'optimizers': {
        'phi_lr': 1e-3,       # Increase gating network learning rate
        'alpha_mu_lr': 5e-3,  # Increase threshold learning rate
        'rho': 1e-2,          # Increase dual variable update rate
    },
    'scheduler': {
        'tau_start': 1.0,     # Start with lower temperature
        'tau_end': 5.0,       # Lower final temperature
        'tau_warmup_epochs': 20,  # Slower warmup
    },
    'worst_group_params': {
        'eg_xi': 1.0, # Tốc độ học của Exponentiated Gradient
    },
    'output': {
        'checkpoints_dir': './checkpoints/argse_balance',
    },
    'seed': 42
}
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- HELPER FUNCTIONS ---

def load_data_from_logits(config):
    """
    Loads pre-computed logits and labels for tuneV and val_small splits.
    """
    logits_root = Path(config['experts']['logits_dir']) / config['dataset']['name']
    splits_dir = Path(config['dataset']['splits_dir'])
    expert_names = config['experts']['names']
    num_experts = len(expert_names)
    num_classes = config['dataset']['num_classes']
    
    dataloaders = {}
    
    # We need original labels to create datasets
    # Load from the full original training dataset
    cifar_train_full = torchvision.datasets.CIFAR100(root='./data', train=True, download=False)
    
    for split in ['tuneV', 'val_small']:
        print(f"Loading data for split: {split}")
        
        # Load indices for the split
        with open(splits_dir / f"{split}_indices.json", 'r') as f:
            indices = json.load(f)
        
        # Load logits from each expert and stack them
        stacked_logits = torch.zeros(len(indices), num_experts, num_classes)
        for i, expert_name in enumerate(expert_names):
            logits_path = logits_root / expert_name / f"{split}_logits.pt"
            stacked_logits[:, i, :] = torch.load(logits_path, map_location='cpu')

        # Get corresponding labels
        labels = torch.tensor(np.array(cifar_train_full.targets)[indices])
        
        dataset = TensorDataset(stacked_logits, labels)
        dataloader = DataLoader(dataset, batch_size=config['argse_params']['batch_size'], shuffle=(split=='tuneV'), num_workers=4)
        dataloaders[split] = dataloader

    return dataloaders['tuneV'], dataloaders['val_small']

def update_beta_eg(current_beta, group_errors, xi):
    """Updates group cost weights using Exponentiated Gradient."""
    # Ensure group_errors is a tensor
    if isinstance(group_errors, list):
        group_errors = torch.tensor(group_errors, device=current_beta.device)
    
    # EG update rule
    new_beta = current_beta * torch.exp(xi * group_errors)
    return new_beta / new_beta.sum()

def eval_epoch(model, loader, c, class_to_group):
    """Evaluates the model with hard rejection decisions. (Phiên bản đã sửa lỗi device)"""
    model.eval()
    
    all_margins = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for logits, labels in loader:
            logits, labels = logits.to(DEVICE), labels.to(DEVICE)
            
            # Forward pass to get margin
            outputs = model(logits, c, 1.0, class_to_group)
            
            # Get prediction from mixed posterior
            _, preds = torch.max(outputs['eta_mix'], 1)
            
            # --- THAY ĐỔI Ở ĐÂY ---
            # Giữ tất cả các tensor trên GPU để tính toán nhất quán
            all_margins.append(outputs['margin'])
            all_preds.append(preds)
            all_labels.append(labels)

    # Nối các tensor lại, tất cả đều đang ở trên GPU
    all_margins = torch.cat(all_margins)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Hard acceptance rule
    accepted_mask = all_margins > 0
    num_accepted = accepted_mask.sum().item()
    total_samples = len(all_labels)
    coverage = num_accepted / total_samples if total_samples > 0 else 0

    if num_accepted == 0:
        return {'coverage': 0, 'balanced_error': 1.0, 'worst_error': 1.0, 'group_errors': [1.0] * model.num_groups}

    # Filter for accepted samples (vẫn ở trên GPU)
    accepted_preds = all_preds[accepted_mask]
    accepted_labels = all_labels[accepted_mask]
    
    # Calculate per-group errors on the accepted set (tất cả đều trên GPU, không còn lỗi)
    accepted_groups = class_to_group[accepted_labels]
    group_errors = []
    for k in range(model.num_groups):
        group_mask = (accepted_groups == k)
        if group_mask.sum() == 0:
            # If no samples from this group were accepted, assign maximum error
            group_errors.append(1.0)
            continue
        
        correct_in_group = (accepted_preds[group_mask] == accepted_labels[group_mask]).sum().item()
        total_in_group = group_mask.sum().item()
        
        # Handle edge case where total_in_group is 0 (shouldn't happen due to check above)
        if total_in_group == 0:
            group_errors.append(1.0)
        else:
            accuracy = correct_in_group / total_in_group
            group_errors.append(1.0 - accuracy)

    return {
        'coverage': coverage,
        'balanced_error': np.mean(group_errors),
        'worst_error': np.max(group_errors),
        'group_errors': group_errors,
    }

# --- MAIN TRAINING SCRIPT ---
def main():
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])

    # 1. Load Data
    train_loader, val_loader = load_data_from_logits(CONFIG)
    
    # 2. Get class/group info
    class_counts = get_cifar100_lt_counts(imb_factor=100) # Assuming IF=100 from M1
    # class_to_group = get_class_to_group(class_counts, K=2, head_ratio=0.5) # Assuming K=2 from M1
    class_to_group = get_class_to_group(class_counts, K=2, head_ratio=0.5).to(DEVICE)
    num_groups = class_to_group.max().item() + 1
    
    # 3. Initialize Model and Optimizers
    num_experts = len(CONFIG['experts']['names'])
    # From M3, gating feature dim = 4 * num_experts
    gating_feature_dim = 4 * num_experts 
    model = AR_GSE(num_experts, CONFIG['dataset']['num_classes'], num_groups, gating_feature_dim).to(DEVICE)

    optimizers = {
        'phi': optim.Adam(model.gating_net.parameters(), lr=CONFIG['optimizers']['phi_lr']),
        'alpha_mu': optim.Adam([model.alpha, model.mu], lr=CONFIG['optimizers']['alpha_mu_lr'])
    }

    # 4. Initialize Training State
    tau = CONFIG['scheduler']['tau_start']
    # Initial beta weights
    if CONFIG['argse_params']['mode'] == 'balanced':
        beta = torch.ones(num_groups, device=DEVICE) / num_groups
    else: # worst
        beta = torch.ones(num_groups, device=DEVICE) / num_groups
    
    # 5. Training Loop
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
            # Primal-dual step now handles dual update internally
            stats, cons_violation_batch = primal_dual_step(model, batch, optimizers, selective_cls_loss, params)
            
            for k, v in stats.items():
                epoch_stats[k].append(v)
        
        # Log training stats for the epoch
        log_str = f"Epoch {epoch+1} | Tau: {tau:.2f} | Loss: {np.mean(epoch_stats['loss_total']):.4f}"
        log_str += f" | Coverage: {np.mean(epoch_stats['mean_coverage']):.3f}"
        log_str += f" | Margin: {np.mean(epoch_stats['mean_margin']):.3f}"
        
        # Add debug info for first few epochs
        if epoch < 5:
            log_str += f" | Alpha: {model.alpha.mean().item():.3f}"
            log_str += f" | Mu: {model.mu.mean().item():.3f}"
            log_str += f" | Lambda: {model.Lambda.mean().item():.3f}"
        
        print(log_str)

        # Evaluation
        val_metrics = eval_epoch(model, val_loader, CONFIG['argse_params']['c'], class_to_group)
        val_metric_key = 'worst_error' if CONFIG['argse_params']['mode'] == 'worst' else 'balanced_error'
        current_val_metric = val_metrics[val_metric_key]
        
        print(f"Validation | Coverage: {val_metrics['coverage']:.3f} | Bal. Err: {val_metrics['balanced_error']:.4f} | Worst Err: {val_metrics['worst_error']:.4f}")

        # Update beta for worst-group mode
        if CONFIG['argse_params']['mode'] == 'worst':
            beta = update_beta_eg(beta, val_metrics['group_errors'], CONFIG['worst_group_params']['eg_xi'])
            print(f"Updated Beta: {[f'{b:.3f}' for b in beta.tolist()]}")

        # Update tau (warm-up)
        if epoch < CONFIG['scheduler']['tau_warmup_epochs']:
            tau += (CONFIG['scheduler']['tau_end'] - CONFIG['scheduler']['tau_start']) / CONFIG['scheduler']['tau_warmup_epochs']

        # Early stopping and checkpointing
        if current_val_metric < best_val_metric:
            best_val_metric = current_val_metric
            epochs_no_improve = 0
            
            output_dir = Path(CONFIG['output']['checkpoints_dir']) / CONFIG['dataset']['name']
            output_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = output_dir / f"argse_{CONFIG['argse_params']['mode']}.ckpt"
            
            print(f"New best model! Saving to {ckpt_path}")
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= 15: # Early stop patience
            print("Early stopping after 15 epochs with no improvement.")
            break
            
if __name__ == '__main__':
    main()