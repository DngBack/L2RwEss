# src/train/gse_balanced_plugin.py
"""
GSE-Balanced implementation using plug-in approach with S1/S2 splits.
This avoids the complex primal-dual training and uses fixed-point matching instead.
"""
import torch
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import torchvision

# Import our custom modules
from src.models.argse import AR_GSE
from src.data.groups import get_class_to_group_by_threshold
from src.data.datasets import get_cifar100_lt_counts

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- CONFIGURATION ---
CONFIG = {
    'dataset': {
        'name': 'cifar100_lt_if100',
        'splits_dir': './data/cifar100_lt_if100_splits',
        'num_classes': 100,
    },
    'grouping': {
        'threshold': 20,  # classes with >threshold samples are head
    },
    'experts': {
        'names': ['ce_baseline', 'logitadjust_baseline', 'balsoftmax_baseline'],
        'logits_dir': './outputs/logits/',
    },
    'plugin_params': {
        'c': 0.7,  # rejection cost
        'M': 10,   # number of plugin iterations
        'gamma': 0.3,  # EMA factor for alpha updates
        'alpha_min': 5e-2,
        'alpha_max': 2.0,
        'lambda_grid': [-1.0, -0.75, -0.5, -0.35, -0.25, -0.15, 0.0, 0.15, 0.25, 0.35, 0.5, 0.75, 1.0],  # lambda values to sweep
        'cov_target': 0.6,  # target coverage for fitting c
    },
    'output': {
        'checkpoints_dir': './checkpoints/argse_balanced_plugin/',
    },
    'seed': 42
}

@torch.no_grad()
def cache_eta_mix(gse_model, loader, class_to_group):
    """
    Cache mixture posteriors η̃(x) for S1, S2 by freezing all GSE components.
    
    Args:
        gse_model: Trained AR_GSE model (frozen)
        loader: DataLoader for split 
        class_to_group: [C] class to group mapping
        
    Returns:
        eta_mix: [N, C] mixture posteriors
        labels: [N] ground truth labels
    """
    gse_model.eval()
    etas, labels = [], []
    
    for logits, y in tqdm(loader, desc="Caching η̃"):
        logits = logits.to(DEVICE)
        
        # Get mixture posterior (no margin computation needed)
        expert_posteriors = torch.softmax(logits, dim=-1)  # [B, E, C]
        
        # Get gating weights
        gating_features = gse_model.feature_builder(logits)
        w = torch.softmax(gse_model.gating_net(gating_features), dim=1)  # [B, E]
        
        # Mixture: η̃_y(x) = Σ_e w^(e)(x) * p^(e)(y|x)
        eta = torch.einsum('be,bec->bc', w, expert_posteriors)  # [B, C]
        
        etas.append(eta.cpu())
        labels.append(y.cpu())
    
    return torch.cat(etas), torch.cat(labels)

def compute_raw_margin(eta, alpha, mu, class_to_group):
    # score = max_y α_{g(y)} * η̃_y
    score = (alpha[class_to_group] * eta).max(dim=1).values  # [N]
    # threshold = Σ_y (1/α_{g(y)} - μ_{g(y)}) * η̃_y
    coeff = 1.0 / alpha[class_to_group] - mu[class_to_group]
    threshold = (coeff.unsqueeze(0) * eta).sum(dim=1)        # [N]
    return score - threshold         

def compute_margin(eta, alpha, mu, c, class_to_group):
    raw = compute_raw_margin(eta, alpha, mu, class_to_group)
    return raw - c

def c_for_target_coverage_from_raw(raw_margins, target_cov=0.6):
    # c = quantile(raw_margin, 1 - target_cov)
    return torch.quantile(raw_margins, 1.0 - target_cov).item()


def accepted_and_pred(eta, alpha, mu, c, class_to_group):
    """
    Compute acceptance decisions and predictions.
    
    Returns:
        accepted: [N] boolean mask for accepted samples
        preds: [N] predicted class labels
        margin: [N] margin scores
    """
    margin = compute_margin(eta, alpha, mu, c, class_to_group)
    accepted = (margin > 0)
    preds = (alpha[class_to_group] * eta).argmax(dim=1)
    return accepted, preds, margin

def balanced_error_on_S(eta, y, alpha, mu, c, class_to_group, K):
    """
    Compute balanced error rate on a split S.
    
    Returns:
        bal_error: balanced error (average of per-group errors)
        group_errors: list of per-group error rates
    """
    accepted, preds, _ = accepted_and_pred(eta, alpha, mu, c, class_to_group)
    
    if accepted.sum() == 0:
        # No samples accepted -> return worst possible error
        return 1.0, [1.0] * K
    
    y_groups = class_to_group[y]  # [N]
    group_errors = []
    
    for k in range(K):
        mask_k = (y_groups == k) & accepted
        if mask_k.sum() == 0:
            # No accepted samples in group k
            group_errors.append(1.0)
        else:
            group_acc = (preds[mask_k] == y[mask_k]).float().mean().item()
            group_errors.append(1.0 - group_acc)
    
    return float(np.mean(group_errors)), group_errors

def update_alpha_fixed_point(eta_S1, y_S1, alpha, mu, c, class_to_group, K, 
                           gamma=0.3, alpha_min=5e-2, alpha_max=2.0):
    """
    Fixed-point alpha update based on acceptance frequency matching.
    α_k ← Π_geomean=1,[α_min,α_max]((1-γ)α_k + γ·K·Ê_S1[1{accept}·1{y∈G_k}])
    """
    accepted, _, _ = accepted_and_pred(eta_S1, alpha, mu, c, class_to_group)
    y_groups = class_to_group[y_S1]  # [N]
    N = y_S1.numel()
    
    # Joint acceptance per group: (1/N) * sum 1{accept}1{y∈G_k}
    joint = torch.zeros(K, dtype=torch.float32, device=eta_S1.device)
    for k in range(K):
        joint[k] = (accepted & (y_groups == k)).sum().float() / float(N)
    
    alpha_hat = K * joint  # target α_k
    
    # EMA update
    new_alpha = (1 - gamma) * alpha + gamma * alpha_hat
    
    # Project: geomean=1, then clamp
    new_alpha = new_alpha.clamp_min(alpha_min)
    log_alpha = new_alpha.log()
    new_alpha = torch.exp(log_alpha - log_alpha.mean())
    new_alpha = new_alpha.clamp(min=alpha_min, max=alpha_max)
    
    return new_alpha

def mu_from_lambda_grid(lambdas, K):
    """
    Convert lambda grid to mu vectors with constraint Σμ_k = 0.
    For K=2: μ = [λ/2, -λ/2]
    """
    mus = []
    for lam in lambdas:
        if K == 2:
            mus.append(torch.tensor([+lam/2.0, -lam/2.0], dtype=torch.float32))
        else:
            # For K>2: could use orthogonal vectors, but not implemented here
            raise NotImplementedError("Provide a mu grid for K>2")
    return mus

def gse_balanced_plugin(eta_S1, y_S1, eta_S2, y_S2, class_to_group, K,
                    c, M=10, lambda_grid=(-0.5, -0.25, 0.0, 0.25, 0.5),
                    alpha_init=None, gamma=0.3, cov_target=0.6):
    """
    Main GSE-Balanced plugin algorithm.
    
    Args:
        eta_S1, y_S1: cached posteriors and labels for S1 (tuning split)
        eta_S2, y_S2: cached posteriors and labels for S2 (validation split)
        class_to_group: [C] class to group mapping
        K: number of groups
        c: rejection cost
        M: number of plugin iterations
        lambda_grid: values of λ to sweep over
        alpha_init: initial alpha values
        gamma: EMA factor for alpha updates
        
    Returns:
        best_alpha: optimal α* 
        best_mu: optimal μ*
        best_score: best balanced error on S2
    """
    device = eta_S1.device
    y_S1 = y_S1.to(device)
    y_S2 = y_S2.to(device)
    class_to_group = class_to_group.to(device)
    
    # Initialize α
    if alpha_init is None:
        alpha = torch.ones(K, dtype=torch.float32, device=device)
    else:
        alpha = alpha_init.clone().float().to(device)
    
    best_alpha = alpha.clone()
    best_mu = torch.zeros(K, dtype=torch.float32, device=device)
    best_score = float('inf')
    
    mu_candidates = [mu.to(device) for mu in mu_from_lambda_grid(lambda_grid, K)]
    
    print(f"Starting GSE-Balanced plugin with {M} iterations, {len(mu_candidates)} μ candidates")
    
    best_c = None
    for m in range(M):
        print(f"\n--- Plugin Iteration {m+1}/{M} ---")
        for i, mu in enumerate(mu_candidates):
            alpha_cur = alpha.clone()
            c_cur = None

            # vài bước fixed-point cho α; mỗi bước re-fit c bằng raw margin S1
            for step in range(3):
                raw_S1 = compute_raw_margin(eta_S1, alpha_cur, mu, class_to_group)
                c_cur = c_for_target_coverage_from_raw(raw_S1, cov_target)  # fit c trên S1
                # update α bằng accept mask tại c_cur
                alpha_cur = update_alpha_fixed_point(
                    eta_S1, y_S1, alpha_cur, mu, c_cur, class_to_group, K,
                    gamma=gamma
                )

            # Đánh giá trên S2 với cùng c_cur
            bal_err, group_errs = balanced_error_on_S(
                eta_S2, y_S2, alpha_cur, mu, c_cur, class_to_group, K
            )

            if bal_err < best_score:
                best_score = bal_err
                best_alpha = alpha_cur.clone()
                best_mu = mu.clone()
                best_c = c_cur
                print(f"  λ={lambda_grid[i]:.2f}: NEW BEST! bal_err={bal_err:.4f}, "
                    f"α=[{alpha_cur[0]:.3f},{alpha_cur[1]:.3f}], "
                    f"μ=[{mu[0]:.3f},{mu[1]:.3f}], c={c_cur:.3f}")
            else:
                print(f"  λ={lambda_grid[i]:.2f}: bal_err={bal_err:.4f} (c={c_cur:.3f})")

        alpha = (0.5 * alpha + 0.5 * best_alpha).clone()
        print(f"[Iter {m+1}] Current best: bal_err={best_score:.4f}, c*={best_c:.3f}")

    return best_alpha.cpu(), best_mu.cpu(), best_score, float(best_c)

def calculate_optimal_c_from_eta(eta, target_coverage=0.6):
    """
    Calculate optimal c based on quantile of max_y η̃_y to achieve target coverage.
    c = 1 - quantile_q(max η̃)
    """
    max_probs = eta.max(dim=1).values  # [N]
    optimal_c = 1.0 - torch.quantile(max_probs, target_coverage)
    return optimal_c.item()

def load_data_from_logits(config):
    """Load pre-computed logits for tuneV (S1) and val_lt (S2) splits."""
    logits_root = Path(config['experts']['logits_dir']) / config['dataset']['name']
    splits_dir = Path(config['dataset']['splits_dir'])
    expert_names = config['experts']['names']
    num_experts = len(expert_names)
    num_classes = config['dataset']['num_classes']
    
    dataloaders = {}
    
    # Base datasets
    cifar_train_full = torchvision.datasets.CIFAR100(root='./data', train=True, download=False)
    cifar_test_full = torchvision.datasets.CIFAR100(root='./data', train=False, download=False)
    
    # Use tuneV (S1) and val_lt (S2) splits
    splits_config = [
        {'split_name': 'tuneV', 'base_dataset': cifar_train_full, 'indices_file': 'tuneV_indices.json'},
        {'split_name': 'val_lt', 'base_dataset': cifar_test_full, 'indices_file': 'val_lt_indices.json'}
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
            logits_path = logits_root / expert_name / f"{split_name}_logits.pt"
            if not logits_path.exists():
                raise FileNotFoundError(f"Missing logits file: {logits_path}")
            stacked_logits[:, i, :] = torch.load(logits_path, map_location='cpu')

        labels = torch.tensor(np.array(base_dataset.targets)[indices])
        dataset = TensorDataset(stacked_logits, labels)
        dataloaders[split_name] = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)

    return dataloaders['tuneV'], dataloaders['val_lt']

def main():
    """Main GSE-Balanced plugin training."""
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    print("=== GSE-Balanced Plugin Training ===")
    
    # 1) Load data
    S1_loader, S2_loader = load_data_from_logits(CONFIG)
    print(f"✅ Loaded S1 (tuneV): {len(S1_loader)} batches")
    print(f"✅ Loaded S2 (val_lt): {len(S2_loader)} batches")
    
    # 2) Set up grouping
    class_counts = get_cifar100_lt_counts(imb_factor=100)
    class_to_group = get_class_to_group_by_threshold(class_counts, threshold=CONFIG['grouping']['threshold'])
    num_groups = class_to_group.max().item() + 1
    head = (class_to_group == 0).sum().item()
    tail = (class_to_group == 1).sum().item()
    print(f"✅ Groups: {head} head classes, {tail} tail classes")
    
    # 3) Load frozen GSE model (with pre-trained gating if available)
    num_experts = len(CONFIG['experts']['names'])
    gating_feature_dim = 4 * num_experts
    model = AR_GSE(num_experts, CONFIG['dataset']['num_classes'], num_groups, gating_feature_dim).to(DEVICE)
    
    # Try to load pre-trained gating weights
    gating_ckpt_path = Path('./checkpoints/gating_pretrained/') / CONFIG['dataset']['name'] / 'gating_pretrained.ckpt'
    
    if gating_ckpt_path.exists():
        print(f"📂 Loading pre-trained gating from {gating_ckpt_path}")
        gating_ckpt = torch.load(gating_ckpt_path, map_location=DEVICE)
        model.gating_net.load_state_dict(gating_ckpt['gating_net_state_dict'])
        # Note: feature_builder has no parameters, so no need to load
        print("✅ Pre-trained gating loaded successfully!")
    else:
        print("⚠️  No pre-trained gating found. Using random initialization.")
        print("   Consider running train_gating_only.py first for better results.")
    
    # Initialize α, μ with reasonable values (will be optimized by plugin)
    with torch.no_grad():
        model.alpha.fill_(1.0)
        model.mu.fill_(0.0)
    
    # 4) Cache η̃ for both splits
    print("\n=== Caching mixture posteriors ===")
    eta_S1, y_S1 = cache_eta_mix(model, S1_loader, class_to_group)
    eta_S2, y_S2 = cache_eta_mix(model, S2_loader, class_to_group) 
    
    print(f"✅ Cached η̃_S1: {eta_S1.shape}, y_S1: {y_S1.shape}")
    print(f"✅ Cached η̃_S2: {eta_S2.shape}, y_S2: {y_S2.shape}")
    
    # Optional: save cached posteriors for multiple experiments
    cache_dir = Path('./cache/eta_mix')
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'eta': eta_S1, 'y': y_S1}, cache_dir / 'S1_tuneV.pt')
    torch.save({'eta': eta_S2, 'y': y_S2}, cache_dir / 'S2_val_lt.pt')
    print(f"💾 Saved cached posteriors to {cache_dir}")
    
    # 5) Auto-calibrate c if needed
    current_c = CONFIG['plugin_params']['c']
    optimal_c = calculate_optimal_c_from_eta(eta_S1, target_coverage=0.6)
    print(f"📊 Current c={current_c:.3f}, Optimal c for 60% coverage={optimal_c:.3f}")
    
    # Use current c or switch to optimal
    c_to_use = optimal_c # or optimal_c
    
    # 6) Run GSE-Balanced plugin
    cov_target = CONFIG['plugin_params']['cov_target']
    print(f"\n=== Running GSE-Balanced Plugin (target_coverage={cov_target:.2f}) ===")
    alpha_star, mu_star, best_score, t_star = gse_balanced_plugin(
        eta_S1=eta_S1.to(DEVICE),
        y_S1=y_S1.to(DEVICE),
        eta_S2=eta_S2.to(DEVICE), 
        y_S2=y_S2.to(DEVICE),
        class_to_group=class_to_group.to(DEVICE),
        K=num_groups,
        c=None,  # biến này không còn dùng bên trong plugin
        M=CONFIG['plugin_params']['M'],
        lambda_grid=CONFIG['plugin_params']['lambda_grid'],
        gamma=CONFIG['plugin_params']['gamma'],
        cov_target=CONFIG['plugin_params']['cov_target'],
    )
    print(f"Best raw-margin threshold t* (fitted on S1): {t_star:.3f}")

    
    # 7) Save results
    print("\n🎉 GSE-Balanced Plugin Complete!")
    print(f"α* = [{alpha_star[0]:.4f}, {alpha_star[1]:.4f}]")
    print(f"μ* = [{mu_star[0]:.4f}, {mu_star[1]:.4f}]")
    print(f"Best balanced error on S2 = {best_score:.4f}")
    
    # Save checkpoint with optimal parameters
    output_dir = Path(CONFIG['output']['checkpoints_dir']) / CONFIG['dataset']['name']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'alpha': alpha_star,
        'mu': mu_star,
        'class_to_group': class_to_group,
        'num_groups': num_groups,
        'threshold': t_star,  # Đổi tên từ 'c' thành 'threshold'
        'best_score': best_score,
        'config': CONFIG,
        'gating_net_state_dict': model.gating_net.state_dict(),
    }
    
    ckpt_path = output_dir / 'gse_balanced_plugin.ckpt'
    torch.save(checkpoint, ckpt_path)
    print(f"💾 Saved checkpoint to {ckpt_path}")

if __name__ == '__main__':
    main()