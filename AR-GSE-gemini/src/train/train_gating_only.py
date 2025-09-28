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
        'epochs': 25,         # More epochs for better convergence
        'batch_size': 256,
        'lr': 8e-4,          # Slightly reduced for stability
        'weight_decay': 2e-4, # Increased weight decay 
        'balanced_training': True,  # Enable tail-aware training
        'tail_weight': 1.0,  # Even softer weighting for optimal balance
        'use_freq_weighting': True,  # Use frequency-based soft weighting
        'entropy_penalty': 0.0000,  # Giảm entropy_penalty về 0 để tránh ép uniform
        'diversity_penalty': 0.002,  # usage_balance nhỏ để tránh collapse
        'gradient_clip': 0.5,  # NEW: Gradient clipping for stability
    },
    'output': {
        'checkpoints_dir': './checkpoints/gating_pretrained/',
    },
    'seed': 42
}

def gating_diversity_regularizer(gating_weights, mode="usage_balance"):
    """
    Gating diversity regularizer with gradient for gating weights.
    usage_balance: khuyến khích tần suất dùng expert gần đều để tránh collapse
    """
    p_bar = gating_weights.mean(dim=0) + 1e-8   # [E]
    # KL(p_bar || Uniform) = sum p_bar * log(p_bar * E) >= 0 (min=0 tại đều)
    return torch.sum(p_bar * torch.log(p_bar * gating_weights.size(1)))

def mixture_cross_entropy_loss(expert_logits, labels, gating_weights, sample_weights=None, 
                            entropy_penalty=0.0, diversity_penalty=0.0):
    """
    Enhanced cross-entropy loss with diversity promotion.
    L = -log(Σ_e w_e * softmax(logits_e)[y]) + entropy_penalty * H(gating_weights) + diversity_penalty * D(gating)
    """
    # expert_logits: [B, E, C]
    # gating_weights: [B, E]  
    # labels: [B]
    # sample_weights: [B] optional
    
    expert_probs = torch.softmax(expert_logits, dim=-1)  # [B, E, C]
    mixture_probs = torch.einsum('be,bec->bc', gating_weights, expert_probs)  # [B, C]
    
    # Clamp for numerical stability
    mixture_probs = torch.clamp(mixture_probs, min=1e-7, max=1.0-1e-7)
    
    # Cross-entropy: -log(p_y)
    log_probs = torch.log(mixture_probs)  # [B, C]
    nll = torch.nn.functional.nll_loss(log_probs, labels, reduction='none')  # [B]
    
    # Apply sample weights if provided
    if sample_weights is not None:
        nll = nll * sample_weights
    
    ce_loss = nll.mean()
    
    # Add entropy penalty to encourage diversity in gating weights
    entropy_loss = 0.0
    if entropy_penalty > 0:
        # H(p) = -Σ p*log(p), we want to maximize entropy (minimize negative entropy)
        gating_log_probs = torch.log(gating_weights + 1e-8)  # [B, E]
        entropy = -(gating_weights * gating_log_probs).sum(dim=1).mean()  # [B] -> scalar
        entropy_loss = -entropy_penalty * entropy  # Negative because we want to maximize entropy
    
    # Add diversity penalty to promote usage balance of gating (có gradient)
    diversity_loss = 0.0
    if diversity_penalty > 0:
        div_reg = gating_diversity_regularizer(gating_weights, mode="usage_balance")
        diversity_loss = diversity_penalty * div_reg

    return ce_loss + entropy_loss + diversity_loss

def compute_frequency_weights(labels, class_counts, smoothing=0.5):
    """
    Compute frequency-based soft weights: w_i = (freq(y_i))^(-smoothing)
    """
    # Get frequencies for each class
    unique_labels = labels.unique()
    freq_weights = torch.ones_like(labels, dtype=torch.float)
    
    for label in unique_labels:
        class_freq = class_counts[label.item()]
        weight = (class_freq + 1) ** (-smoothing)  # +1 for smoothing
        freq_weights[labels == label] = weight
    
    # Normalize so mean weight = 1
    freq_weights = freq_weights / freq_weights.mean()
    return freq_weights

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

def train_gating_only():
    """Train only the gating network with fixed α=1, μ=0."""
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    print("=== Training Gating Network Only ===")
    
    # Load data
    train_loader = load_data_from_logits(CONFIG)
    print(f"✅ Loaded training data: {len(train_loader)} batches")
    
    # Get split counts from tuneV for sample weighting (không phải full counts)
    cifar_train_full = torchvision.datasets.CIFAR100(root='./data', train=True, download=False)
    indices_path = Path(CONFIG['dataset']['splits_dir']) / 'tuneV_indices.json'
    indices = json.loads(indices_path.read_text())
    split_labels = torch.tensor(np.array(cifar_train_full.targets)[indices])
    split_counts = torch.bincount(split_labels, minlength=CONFIG['dataset']['num_classes']).float()
    print("✅ Using tuneV split counts (not full counts) for sample weighting")
    
    # Set up grouping (for model creation, but α/μ won't be used)
    class_counts = get_cifar100_lt_counts(imb_factor=100)  # class_to_group still uses threshold from full LT
    class_to_group = get_class_to_group_by_threshold(class_counts, threshold=CONFIG['grouping']['threshold'])
    num_groups = class_to_group.max().item() + 1
    
    # Create model with dynamic feature dimension
    num_experts = len(CONFIG['experts']['names'])
    
    # Compute gating feature dimension dynamically
    with torch.no_grad():
        dummy = torch.zeros(2, num_experts, CONFIG['dataset']['num_classes']).to(DEVICE)
        # Create temporary model to get feature dimension
        temp_model = AR_GSE(num_experts, CONFIG['dataset']['num_classes'], num_groups, 1).to(DEVICE)
        gating_feature_dim = temp_model.feature_builder(dummy).size(-1)
        del temp_model
    
    model = AR_GSE(num_experts, CONFIG['dataset']['num_classes'], num_groups, gating_feature_dim).to(DEVICE)
    print(f"✅ Dynamic gating feature dimension: {gating_feature_dim}")
    
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
    
    # Training loop with optional balanced training
    best_loss = float('inf')
    
    # Set up class grouping for balanced training
    if CONFIG['gating_params']['balanced_training']:
        # class_counts dùng cho weighting => đổi sang split_counts từ tuneV
        class_counts = split_counts  # Sử dụng tần suất từ tuneV thay vì full counts
        # class_to_group giữ nguyên theo threshold toàn tập, KHÔNG đổi.
        tail_weight = CONFIG['gating_params']['tail_weight']
        use_freq_weighting = CONFIG['gating_params']['use_freq_weighting']
        entropy_penalty = CONFIG['gating_params']['entropy_penalty']
        
        print("✅ Balanced training enabled:")
        print(f"   - Tail weight: {tail_weight}")
        print(f"   - Frequency weighting: {use_freq_weighting}")
        print(f"   - Entropy penalty: {entropy_penalty}")
        print("   - Using tuneV split frequencies for weighting")
    
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
            
            # Compute sample weights for balanced training
            sample_weights = None
            if CONFIG['gating_params']['balanced_training']:
                if use_freq_weighting:
                    # Use soft frequency-based weighting
                    sample_weights = compute_frequency_weights(labels.cpu(), class_counts, smoothing=0.5).to(DEVICE)
                else:
                    # Use hard group-based weighting
                    with torch.no_grad():
                        g = class_to_group[labels.cpu()].to(DEVICE)  # 0=head, 1=tail, move to device
                        sample_weights = torch.where(g == 0, 
                                                    torch.tensor(1.0, device=DEVICE),
                                                    torch.tensor(tail_weight, device=DEVICE))
            
            # Mixture cross-entropy loss with sample weights, entropy penalty, and diversity penalty
            loss = mixture_cross_entropy_loss(expert_logits, labels, gating_weights, 
                                            sample_weights, 
                                            entropy_penalty=CONFIG['gating_params']['entropy_penalty'],
                                            diversity_penalty=CONFIG['gating_params']['diversity_penalty'])
            
            loss.backward()
            # Apply gradient clipping if specified
            if CONFIG['gating_params'].get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(model.gating_net.parameters(), 
                                             max_norm=CONFIG['gating_params']['gradient_clip'])
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