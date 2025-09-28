# src/train/train_gating_only.py
"""
Pre-train the gating network only (freeze α,μ) for GSE-Balanced plugin approach.
This learns a good mixture of experts before applying the plugin algorithm.
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

from src.models.argse import AR_GSE
from src.data.groups import get_class_to_group_by_threshold
from src.data.datasets import get_cifar100_lt_counts

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
        'logits_dir': './outputs/logits/',
    },
    'gating_params': {
        'epochs': 20,
        'batch_size': 256,
        'lr': 1e-3,
        'weight_decay': 1e-4,
    },
    'output': {
        'checkpoints_dir': './checkpoints/gating_pretrained/',
    },
    'seed': 42
}

def load_data_from_logits(config):
    """Load pre-computed logits for training gating."""
    logits_root = Path(config['experts']['logits_dir']) / config['dataset']['name']
    splits_dir = Path(config['dataset']['splits_dir'])
    expert_names = config['experts']['names']
    num_experts = len(expert_names)
    num_classes = config['dataset']['num_classes']
    
    # Use tuneV for gating training 
    cifar_train_full = torchvision.datasets.CIFAR100(root='./data', train=True, download=False)
    indices_path = splits_dir / 'tuneV_indices.json'
    indices = json.loads(indices_path.read_text())

    # Stack expert logits
    stacked_logits = torch.zeros(len(indices), num_experts, num_classes)
    for i, expert_name in enumerate(expert_names):
        logits_path = logits_root / expert_name / "tuneV_logits.pt"
        stacked_logits[:, i, :] = torch.load(logits_path, map_location='cpu')

    labels = torch.tensor(np.array(cifar_train_full.targets)[indices])
    dataset = TensorDataset(stacked_logits, labels)
    
    return DataLoader(dataset, batch_size=config['gating_params']['batch_size'], 
                     shuffle=True, num_workers=4)

def mixture_cross_entropy_loss(expert_logits, labels, gating_weights):
    """
    Cross-entropy loss on mixture of expert predictions.
    L = -log(Σ_e w_e * softmax(logits_e)[y])
    """
    # expert_logits: [B, E, C]
    # gating_weights: [B, E]  
    # labels: [B]
    
    expert_probs = torch.softmax(expert_logits, dim=-1)  # [B, E, C]
    mixture_probs = torch.einsum('be,bec->bc', gating_weights, expert_probs)  # [B, C]
    
    # Clamp for numerical stability
    mixture_probs = torch.clamp(mixture_probs, min=1e-7, max=1.0-1e-7)
    
    # Cross-entropy: -log(p_y)
    log_probs = torch.log(mixture_probs)  # [B, C]
    loss = torch.nn.functional.nll_loss(log_probs, labels)
    
    return loss

def train_gating_only():
    """Train only the gating network with fixed α=1, μ=0."""
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    print("=== Training Gating Network Only ===")
    
    # Load data
    train_loader = load_data_from_logits(CONFIG)
    print(f"✅ Loaded training data: {len(train_loader)} batches")
    
    # Set up grouping (for model creation, but α/μ won't be used)
    class_counts = get_cifar100_lt_counts(imb_factor=100)
    class_to_group = get_class_to_group_by_threshold(class_counts, threshold=CONFIG['grouping']['threshold'])
    num_groups = class_to_group.max().item() + 1
    
    # Create model
    num_experts = len(CONFIG['experts']['names'])
    gating_feature_dim = 4 * num_experts
    model = AR_GSE(num_experts, CONFIG['dataset']['num_classes'], num_groups, gating_feature_dim).to(DEVICE)
    
    # Freeze α, μ at reasonable values
    with torch.no_grad():
        model.alpha.fill_(1.0)  # α = 1 for all groups
        model.mu.fill_(0.0)     # μ = 0 for all groups
        
    # Only optimize gating network parameters (feature_builder has no parameters)
    optimizer = optim.Adam(
        model.gating_net.parameters(),
        lr=CONFIG['gating_params']['lr'],
        weight_decay=CONFIG['gating_params']['weight_decay']
    )
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(CONFIG['gating_params']['epochs']):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (expert_logits, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            expert_logits = expert_logits.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass: get gating weights
            gating_features = model.feature_builder(expert_logits)
            raw_weights = model.gating_net(gating_features)
            gating_weights = torch.softmax(raw_weights, dim=1)  # [B, E]
            
            # Mixture cross-entropy loss
            loss = mixture_cross_entropy_loss(expert_logits, labels, gating_weights)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.gating_net.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{CONFIG['gating_params']['epochs']} | Loss: {avg_loss:.4f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            
            output_dir = Path(CONFIG['output']['checkpoints_dir']) / CONFIG['dataset']['name']
            output_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint = {
                'gating_net_state_dict': model.gating_net.state_dict(),
                'num_experts': num_experts,
                'num_classes': CONFIG['dataset']['num_classes'],
                'num_groups': num_groups,
                'gating_feature_dim': gating_feature_dim,
                'config': CONFIG,
            }
            
            ckpt_path = output_dir / 'gating_pretrained.ckpt'
            torch.save(checkpoint, ckpt_path)
            print(f"💾 New best! Saved to {ckpt_path}")
    
    print(f"✅ Gating training complete. Best loss: {best_loss:.4f}")

if __name__ == '__main__':
    train_gating_only()