# src/train/eval_gse_plugin.py
"""
Evaluation script for GSE-Balanced plugin results.
Loads optimal (α*, μ*) and evaluates on test set.
"""
import torch
import torchvision
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path

# Import our custom modules
from src.models.argse import AR_GSE
from src.metrics.selective_metrics import calculate_selective_errors
from src.metrics.rc_curve import generate_rc_curve, generate_rc_curve_from_02, calculate_aurc, calculate_aurc_from_02
from src.metrics.calibration import calculate_ece
from src.metrics.bootstrap import bootstrap_ci

# Import plugin functions
from src.train.gse_balanced_plugin import compute_margin

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- HELPER FUNCTIONS ---

def apply_per_expert_temperature(logits, expert_names, temp_dict):
    """
    Apply per-expert temperature scaling to logits.
    
    Args:
        logits: [B, E, C] expert logits
        expert_names: list of expert names
        temp_dict: dict mapping expert name -> temperature
        
    Returns:
        scaled_logits: [B, E, C] temperature-scaled logits
    """
    if not temp_dict:
        return logits
    
    scaled = logits.clone()
    for i, name in enumerate(expert_names):
        T = float(temp_dict.get(name, 1.0))
        if abs(T - 1.0) > 1e-6:
            scaled[:, i, :] = scaled[:, i, :] / T
    return scaled

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
        'coverage_points': [0.7, 0.8, 0.9],
        'bootstrap_n': 1000,
    },
    'plugin_checkpoint': './checkpoints/argse_worst_eg_improved_v2/cifar100_lt_if100/gse_balanced_plugin.ckpt',
    'output_dir': './results_worst_eg_improved_v2/cifar100_lt_if100',
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

def load_val_data():
    """Load validation (val_lt) logits and labels for threshold recalibration."""
    logits_root = Path(CONFIG['experts']['logits_dir']) / CONFIG['dataset']['name']
    splits_dir = Path(CONFIG['dataset']['splits_dir'])
    
    with open(splits_dir / 'val_lt_indices.json', 'r') as f:
        val_indices = json.load(f)
    num_val_samples = len(val_indices)
    
    # Load expert logits for val set
    num_experts = len(CONFIG['experts']['names'])
    stacked_logits = torch.zeros(num_val_samples, num_experts, CONFIG['dataset']['num_classes'])
    
    for i, expert_name in enumerate(CONFIG['experts']['names']):
        logits_path = logits_root / expert_name / "val_lt_logits.pt"
        if not logits_path.exists():
            raise FileNotFoundError(f"Logits file not found: {logits_path}")
        stacked_logits[:, i, :] = torch.load(logits_path, map_location='cpu', weights_only=False)
    
    # Load val labels
    full_test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False)
    val_labels = torch.tensor(np.array(full_test_dataset.targets)[val_indices])
    
    return stacked_logits, val_labels

def get_mixture_posteriors(model, logits, expert_names=None, temperatures=None):
    """Get mixture posteriors η̃(x) from expert logits with optional temperature scaling."""
    model.eval()
    with torch.no_grad():
        logits = logits.to(DEVICE)
        
        # Apply per-expert temperature scaling if provided
        if expert_names is not None and temperatures is not None:
            logits = apply_per_expert_temperature(logits, expert_names, temperatures)
        
        # Get expert posteriors (with temperature-scaled logits)
        expert_posteriors = torch.softmax(logits, dim=-1)  # [B, E, C]
        
        # Get gating weights
        gating_features = model.feature_builder(logits)
        gating_weights = torch.softmax(model.gating_net(gating_features), dim=1)  # [B, E]
        
        # Mixture posteriors
        eta_mix = torch.einsum('be,bec->bc', gating_weights, expert_posteriors)  # [B, C]
        
    return eta_mix.cpu()

def analyze_group_performance(eta_mix, preds, labels, accepted, alpha, mu, threshold, class_to_group, K):
    """
    Analyze detailed per-group performance metrics.
    Uses per-sample thresholds based on PREDICTED groups for accurate overlap analysis.
    """
    print("\n" + "="*50)
    print("DETAILED GROUP-WISE ANALYSIS (by Ground-Truth Groups)")
    print("="*50)
    
    # Ensure all tensors are on same device
    device = eta_mix.device
    class_to_group = class_to_group.to(device)
    y_groups = class_to_group[labels]
    
    # Compute per-sample thresholds based on PREDICTED groups (for overlap analysis)
    alpha_per_class = alpha[class_to_group].to(device)
    yhat = (alpha_per_class.unsqueeze(0) * eta_mix).argmax(dim=1)
    pred_groups = class_to_group[yhat]
    
    # Convert threshold to tensor
    if isinstance(threshold, (list, torch.Tensor)):
        t_group_tensor = torch.tensor(threshold, device=device) if isinstance(threshold, list) else threshold.to(device)
    else:
        # Single threshold -> replicate for all groups
        t_group_tensor = torch.full((K,), float(threshold), device=device)
    
    thr_per_sample = t_group_tensor[pred_groups]  # Per-sample threshold by predicted group
    
    for k in range(K):
        group_name = "Head" if k == 0 else "Tail"
        group_mask = (y_groups == k)
        group_size = group_mask.sum().item()
        
        if group_size == 0:
            continue
            
        # Coverage and error for this GT group
        group_accepted = accepted[group_mask]
        group_coverage = group_accepted.float().mean().item()
        
        # TPR/FPR analysis for this group
        group_preds_all = preds[group_mask]
        group_labels_all = labels[group_mask]
        group_correct = (group_preds_all == group_labels_all)
        
        # True Positive Rate (TPR): fraction of correct predictions that are accepted
        correct_accepted = group_accepted & group_correct
        tpr = correct_accepted.sum().item() / group_correct.sum().item() if group_correct.sum() > 0 else 0.0
        
        # False Positive Rate (FPR): fraction of incorrect predictions that are accepted  
        incorrect_accepted = group_accepted & (~group_correct)
        fpr = incorrect_accepted.sum().item() / (~group_correct).sum().item() if (~group_correct).sum() > 0 else 0.0
        
        if group_accepted.sum() > 0:
            group_preds = preds[group_mask & accepted]
            group_labels = labels[group_mask & accepted]
            group_accuracy = (group_preds == group_labels).float().mean().item()
            group_error = 1.0 - group_accuracy
        else:
            group_error = 1.0
            
        # Raw margin statistics for this GT group
        raw_margins = compute_margin(eta_mix[group_mask], alpha, mu, 0.0, class_to_group)
        margin_mean = raw_margins.mean().item()
        margin_std = raw_margins.std().item()
        margin_min = raw_margins.min().item()
        margin_max = raw_margins.max().item()
        
        print(f"\n{group_name} Group (k={k}, by ground-truth):")
        print(f"  • Size: {group_size} samples")
        print(f"  • Coverage: {group_coverage:.3f}")
        print(f"  • Error: {group_error:.3f}")
        print(f"  • TPR (correct accepted): {tpr:.3f}")
        print(f"  • FPR (incorrect accepted): {fpr:.3f}")
        print(f"  • α_k: {alpha[k]:.3f}")
        print(f"  • μ_k: {mu[k]:.3f}")
        # Show the threshold for this group
        group_threshold_val = t_group_tensor[k].item()
        print(f"  • τ_k (from config): {group_threshold_val:.3f}")
        print(f"  • Raw margin stats: μ={margin_mean:.3f}, σ={margin_std:.3f}, range=[{margin_min:.3f}, {margin_max:.3f}]")
        
        # Check separation quality using per-sample thresholds (by predicted group)
        group_accepted_margins = raw_margins[group_accepted]
        group_rejected_margins = raw_margins[~group_accepted]
        group_thr_accepted = thr_per_sample[group_mask][group_accepted]
        group_thr_rejected = thr_per_sample[group_mask][~group_accepted]
        
        if len(group_accepted_margins) > 0 and len(group_rejected_margins) > 0:
            separation = group_accepted_margins.min().item() - group_rejected_margins.max().item()
            # Overlap: rejected samples with margin > their per-sample threshold
            overlap_ratio = (group_rejected_margins > group_thr_rejected).float().mean().item()
            print(f"  • Margin separation: {separation:.3f}")
            print(f"  • Overlap ratio (w.r.t per-sample t by pred-group): {overlap_ratio:.3f}")
    
    # Additional: Breakdown by PREDICTED groups
    print("\n" + "="*50)
    print("BREAKDOWN BY PREDICTED GROUPS")
    print("="*50)
    for k in range(K):
        pred_mask = (pred_groups == k)
        if pred_mask.sum() == 0:
            continue
        pred_cov = accepted[pred_mask].float().mean().item()
        pred_name = "Head" if k == 0 else "Tail"
        print(f"{pred_name} predictions (k={k}): {pred_mask.sum().item()} samples, coverage={pred_cov:.3f}, threshold={t_group_tensor[k].item():.3f}")
    
    print("\n" + "="*50)

def recalibrate_thresholds_on_val(logits_val, labels_val, model, alpha, mu, class_to_group, 
                                   tau_by_group, expert_names, temperatures, checkpoint):
    """
    Post-hoc recalibrate per-group thresholds on validation set (val_lt).
    Fits t_k on ALL predictions (not just correct ones) to achieve target coverage by predicted group.
    
    This is the deployable approach: thresholds based on predicted groups, not ground-truth groups.
    
    Args:
        logits_val: validation expert logits [N, E, C]
        labels_val: validation labels [N] (used only for analysis, not for threshold fitting)
        model: AR_GSE model with trained gating
        alpha, mu: optimal parameters from plugin
        class_to_group: class -> group mapping
        tau_by_group: target coverage per group [τ_head, τ_tail]
        expert_names: list of expert names
        temperatures: per-expert temperature dict
        checkpoint: full checkpoint dict (for gating weights)
        
    Returns:
        t_recalibrated: list of per-group thresholds [K]
    """
    print("\n" + "="*50)
    print("POST-HOC THRESHOLD RECALIBRATION ON VAL_LT")
    print("="*50)
    
    device = DEVICE
    K = int(class_to_group.max().item() + 1)
    
    # 1) Apply temperature scaling
    logits_val_scaled = apply_per_expert_temperature(logits_val, expert_names, temperatures)
    
    # 2) Get mixture posteriors
    model.eval()
    with torch.no_grad():
        logits_val_scaled = logits_val_scaled.to(device)
        expert_posteriors = torch.softmax(logits_val_scaled, dim=-1)  # [N, E, C]
        
        # Get gating weights
        gating_features = model.feature_builder(logits_val_scaled)
        w = torch.softmax(model.gating_net(gating_features), dim=1)  # [N, E]
        
        # Mixture posteriors
        eta_val = torch.einsum('be,bec->bc', w, expert_posteriors).cpu()  # [N, C]
    
    # 3) Compute raw margins and predictions with α-reweighting
    alpha_cpu = alpha.cpu()
    mu_cpu = mu.cpu()
    cg_cpu = class_to_group.cpu()
    
    from src.train.gse_balanced_plugin import compute_margin
    margins_val = compute_margin(eta_val, alpha_cpu, mu_cpu, 0.0, cg_cpu)  # [N]
    
    # Prediction with α
    alpha_per_class = alpha_cpu[cg_cpu]  # [C]
    yhat_val = (alpha_per_class.unsqueeze(0) * eta_val).argmax(dim=1)  # [N]
    pred_groups_val = cg_cpu[yhat_val]  # [N] - groups of predicted classes
    
    # 4) Fit per-group thresholds on ALL predictions (not just correct ones)
    # This is deployable: we don't filter by correctness at test-time
    tau_head, tau_tail = tau_by_group if len(tau_by_group) == 2 else (0.56, 0.44)
    quantile_targets = [1 - tau_head, 1 - tau_tail] if K == 2 else [1 - np.mean(tau_by_group)] * K
    
    t_recalibrated = []
    print(f"\nTarget coverage: head={tau_head:.2f}, tail={tau_tail:.2f}")
    print("Fitting thresholds on val_lt by PREDICTED groups (deployable):\n")
    
    for k in range(K):
        pred_mask = (pred_groups_val == k)
        group_name = "head" if k == 0 else "tail"
        
        if pred_mask.sum() == 0:
            # No predictions in this group - use min margin as fallback
            t_k = float(margins_val.min())
            print(f"  ⚠️  {group_name} (k={k}): No predictions, using t_k={t_k:.3f}")
        else:
            # Compute quantile on margins of samples predicted as group k
            t_k = float(torch.quantile(margins_val[pred_mask], quantile_targets[k]))
            
            # Verify achieved coverage
            accepted_k = margins_val[pred_mask] >= t_k
            achieved_cov = accepted_k.float().mean().item()
            
            print(f"  ✅ {group_name} (k={k}): {pred_mask.sum().item()} predictions, "
                  f"t_k={t_k:.3f}, achieved coverage={achieved_cov:.3f} (target={tau_by_group[k]:.2f})")
        
        t_recalibrated.append(t_k)
    
    # 5) Diagnostic: Show coverage by ground-truth groups (for comparison)
    print("\n  Diagnostic - Coverage by ground-truth groups:")
    y_groups_val = cg_cpu[labels_val]
    for k in range(K):
        gt_mask = (y_groups_val == k)
        if gt_mask.sum() > 0:
            # Use per-sample thresholds based on predicted groups
            t_per_sample = torch.tensor([t_recalibrated[pred_groups_val[i].item()] 
                                        for i in range(len(pred_groups_val))])
            gt_accepted = margins_val[gt_mask] >= t_per_sample[gt_mask]
            gt_cov = gt_accepted.float().mean().item()
            group_name = "head" if k == 0 else "tail"
            print(f"    {group_name} GT (k={k}): {gt_mask.sum().item()} samples, coverage={gt_cov:.3f}")
    
    print("="*50)
    return t_recalibrated

def selective_risk_from_mask(preds, labels, accepted_mask, c_cost, class_to_group, K, kind="balanced"):
    """
    Compute selective risk with rejection cost c_cost.
    Risk = error_rate * coverage + c_cost * (1 - coverage) for each group
    """
    y = labels
    g = class_to_group[y]
    
    if kind == "balanced":
        vals = []
        for k in range(K):
            mk = (g == k)
            if mk.sum() == 0:
                vals.append(c_cost)  # No samples in group k
                continue
                
            acc_k = accepted_mask[mk].float().mean().item()
            
            if accepted_mask[mk].sum() == 0:
                err_k = 1.0  # No accepted samples, assume error = 1.0
            else:
                err_k = (preds[mk & accepted_mask] != y[mk & accepted_mask]).float().mean().item()
            
            risk_k = err_k * acc_k + c_cost * (1.0 - acc_k)
            vals.append(risk_k)
        return float(np.mean(vals))
    else:  # worst-group risk
        worst = 0.0
        for k in range(K):
            mk = (g == k)
            if mk.sum() == 0:
                worst = max(worst, c_cost)
                continue
                
            acc_k = accepted_mask[mk].float().mean().item()
            
            if accepted_mask[mk].sum() == 0:
                err_k = 1.0
            else:
                err_k = (preds[mk & accepted_mask] != y[mk & accepted_mask]).float().mean().item()
            
            risk_k = err_k * acc_k + c_cost * (1.0 - acc_k)
            worst = max(worst, risk_k)
        return worst

def main():
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    print("=== GSE-Balanced Plugin Evaluation ===")
    
    # Create output directory
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load plugin checkpoint
    plugin_ckpt_path = Path(CONFIG['plugin_checkpoint'])
    if not plugin_ckpt_path.exists():
        raise FileNotFoundError(f"Plugin checkpoint not found: {plugin_ckpt_path}")
    
    print(f"📂 Loading plugin checkpoint: {plugin_ckpt_path}")
    checkpoint = torch.load(plugin_ckpt_path, map_location=DEVICE, weights_only=False)
    
    alpha_star = checkpoint['alpha'].to(DEVICE)
    mu_star = checkpoint['mu'].to(DEVICE)
    class_to_group = checkpoint['class_to_group'].to(DEVICE)
    num_groups = checkpoint['num_groups']
    plugin_threshold = checkpoint.get('threshold', checkpoint.get('c'))  # Backward compatibility
    
    print("✅ Loaded optimal parameters:")
    print(f"   α* = [{alpha_star[0]:.4f}, {alpha_star[1]:.4f}]")
    print(f"   μ* = [{mu_star[0]:.4f}, {mu_star[1]:.4f}]")
    
    # Handle both single threshold and per-group thresholds
    if isinstance(plugin_threshold, list):
        print(f"   per-group thresholds t* = {plugin_threshold}")
    else:
        print(f"   raw-margin threshold t* = {plugin_threshold:.3f}")
    if 'best_score' in checkpoint:
        print(f"   Best S2 score = {checkpoint['best_score']:.4f}")
    if 'source' in checkpoint:
        print(f"   Source: {checkpoint['source']}")
    if 'improvement' in checkpoint:
        print(f"   Expected improvement: {checkpoint['improvement']:.1f}%")
    
    # 2. Set up model with optimal parameters
    num_experts = len(CONFIG['experts']['names'])
    
    # Compute dynamic gating feature dimension (enriched features)
    with torch.no_grad():
        dummy_logits = torch.zeros(2, num_experts, CONFIG['dataset']['num_classes']).to(DEVICE)
        temp_model = AR_GSE(num_experts, CONFIG['dataset']['num_classes'], num_groups, 1).to(DEVICE)
        gating_feature_dim = temp_model.feature_builder(dummy_logits).size(-1)
        del temp_model
    print(f"✅ Dynamic gating feature dim: {gating_feature_dim}")
    
    model = AR_GSE(num_experts, CONFIG['dataset']['num_classes'], num_groups, gating_feature_dim).to(DEVICE)
    
    # Load gating network weights with dimension compatibility check
    if 'gating_net_state_dict' in checkpoint:
        saved_state = checkpoint['gating_net_state_dict']
        current_state = model.gating_net.state_dict()
        
        compatible = True
        for key in saved_state.keys():
            if key in current_state and saved_state[key].shape != current_state[key].shape:
                print(f"⚠️  Dimension mismatch for {key}: saved {saved_state[key].shape} vs current {current_state[key].shape}")
                compatible = False
        
        if compatible:
            model.gating_net.load_state_dict(saved_state)
            print("✅ Gating network weights loaded successfully")
        else:
            print("❌ Gating checkpoint incompatible with enriched features. Using random weights.")
    else:
        print("⚠️ No gating network weights found in checkpoint")
    
    # Set optimal α*, μ*
    with torch.no_grad():
        model.alpha.copy_(alpha_star)
        model.mu.copy_(mu_star)
    
    print("✅ Model configured with optimal parameters")
    
    # 3. Load validation data for threshold recalibration
    print("\n📊 Loading validation data for threshold recalibration...")
    try:
        val_logits, val_labels = load_val_data()
        print(f"✅ Loaded {len(val_labels)} val_lt samples")
        
        # Recalibrate thresholds on val_lt (post-hoc, no retraining)
        target_cov_by_group = checkpoint.get('target_cov_by_group', [0.56, 0.44])
        t_recalibrated = recalibrate_thresholds_on_val(
            val_logits, val_labels, model, alpha_star, mu_star, class_to_group,
            tau_by_group=target_cov_by_group,
            expert_names=CONFIG['experts']['names'],
            temperatures=checkpoint.get('temperatures', None),
            checkpoint=checkpoint
        )
        print(f"✅ Recalibrated thresholds: {[f'{t:.3f}' for t in t_recalibrated]}")
        use_recalibrated = True
    except FileNotFoundError as e:
        print(f"⚠️  Could not load val_lt data: {e}")
        print("   Falling back to checkpoint thresholds")
        t_recalibrated = None
        use_recalibrated = False
    
    # 4. Load test data
    print("\n📊 Loading test data...")
    test_logits, test_labels = load_test_data()
    num_test_samples = len(test_labels)
    print(f"✅ Loaded {num_test_samples} test samples")
    
    # Load temperatures if available
    temperatures = checkpoint.get('temperatures', None)
    if temperatures:
        print(f"🌡️  Per-expert temperatures: {temperatures}")
    
    # 5. Get test predictions with temperature scaling
    print("🔮 Computing test predictions...")
    eta_mix = get_mixture_posteriors(model, test_logits, 
                                      expert_names=CONFIG['experts']['names'],
                                      temperatures=temperatures)
    
    # Ensure all tensors are on CPU for consistent computation
    alpha_star_cpu = alpha_star.cpu()
    mu_star_cpu = mu_star.cpu()
    class_to_group_cpu = class_to_group.cpu()
    
    # Compute raw margins and predictions with α-reweighting
    margins_raw = compute_margin(eta_mix, alpha_star_cpu, mu_star_cpu, 0.0, class_to_group_cpu)
    alpha_per_class = alpha_star_cpu[class_to_group_cpu]  # [C]
    preds = (alpha_per_class.unsqueeze(0) * eta_mix).argmax(dim=1)  # [N] - α-reweighted prediction
    
    # ✅ CORRECT: Use per-group thresholds with PREDICTED groups (deployable rule)
    # Priority: use recalibrated thresholds if available, else use checkpoint thresholds
    if use_recalibrated and t_recalibrated is not None:
        t_group_list = t_recalibrated
        t_group_tensor = torch.tensor(t_recalibrated, dtype=margins_raw.dtype)
        threshold_source = "recalibrated on val_lt"
    else:
        t_group = checkpoint.get('t_group', None)
        if t_group is not None:
            # Convert to tensor if it's a list
            t_group_tensor = torch.tensor(t_group, dtype=margins_raw.dtype) if isinstance(t_group, list) else t_group.cpu()
            t_group_list = t_group if isinstance(t_group, list) else t_group.tolist()
            threshold_source = "from checkpoint"
        else:
            # Fallback to global threshold
            plugin_threshold = checkpoint.get('threshold', checkpoint.get('c', 0.0))
            t_group_tensor = torch.tensor([plugin_threshold, plugin_threshold], dtype=margins_raw.dtype)
            t_group_list = [plugin_threshold, plugin_threshold]
            threshold_source = "global (fallback)"
    
    # Per-sample threshold based on PREDICTED group (test-time deployable!)
    pred_groups = class_to_group_cpu[preds]  # Group of predicted class
    thresholds_per_sample = t_group_tensor[pred_groups]  # [N]
    accepted = margins_raw >= thresholds_per_sample
    
    print(f"\n✅ Using per-group thresholds by PREDICTED group ({threshold_source}):")
    print(f"   {[f'{t:.3f}' for t in t_group_list]}")
    print(f"✅ Test coverage: {accepted.float().mean():.3f}")
    
    # Per-group coverage breakdown by PREDICTED groups
    for k in range(len(t_group_tensor)):
        pred_mask = (pred_groups == k)
        if pred_mask.sum() > 0:
            pred_cov = accepted[pred_mask].float().mean().item()
            group_name = "head" if k == 0 else "tail" 
            print(f"   📊 {group_name} predictions (group {k}): {pred_mask.sum().item()} samples, coverage={pred_cov:.3f}, threshold={t_group_list[k]:.3f}")
    
    # Also show GT-group breakdown for comparison
    y_groups = class_to_group_cpu[test_labels]
    print(f"\n   Comparison - Coverage by ground-truth groups:")
    for k in range(len(t_group_tensor)):
        gt_mask = (y_groups == k)
        if gt_mask.sum() > 0:
            gt_cov = accepted[gt_mask].float().mean().item()
            group_name = "head" if k == 0 else "tail"
            print(f"   📊 {group_name} GT (group {k}): {gt_mask.sum().item()} samples, coverage={gt_cov:.3f}")
    
    # 5. Calculate metrics
    print("📈 Calculating metrics...")
    results = {}
    
    # 5.1 RC Curve and AURC (using raw margins for fair comparison)
    rc_df = generate_rc_curve(margins_raw, preds, test_labels, class_to_group_cpu, num_groups)
    rc_df.to_csv(output_dir / 'rc_curve.csv', index=False)
    
    aurc_bal = calculate_aurc(rc_df, 'balanced_error')
    aurc_wst = calculate_aurc(rc_df, 'worst_error')
    results['aurc_balanced'] = aurc_bal
    results['aurc_worst'] = aurc_wst
    print(f"AURC (Balanced): {aurc_bal:.4f}, AURC (Worst): {aurc_wst:.4f}")
    
    # 5.2 RC Curve 0.2-1.0 range
    rc_df_02 = generate_rc_curve_from_02(margins_raw, preds, test_labels, class_to_group_cpu, num_groups)
    rc_df_02.to_csv(output_dir / 'rc_curve_02_10.csv', index=False)
    
    aurc_bal_02 = calculate_aurc_from_02(rc_df_02, 'balanced_error')
    aurc_wst_02 = calculate_aurc_from_02(rc_df_02, 'worst_error')
    results['aurc_balanced_02_10'] = aurc_bal_02
    results['aurc_worst_02_10'] = aurc_wst_02
    print(f"AURC 0.2-1.0 (Balanced): {aurc_bal_02:.4f}, AURC 0.2-1.0 (Worst): {aurc_wst_02:.4f}")
    
    # 5.3 Bootstrap CI for AURC
    def aurc_metric_func(m, p, labels):
        rc_df_boot = generate_rc_curve(m, p, labels, class_to_group_cpu, num_groups, num_points=51)
        return calculate_aurc(rc_df_boot, 'balanced_error')

    mean_aurc, lower, upper = bootstrap_ci(
        (margins_raw, preds, test_labels), aurc_metric_func, n_bootstraps=CONFIG['eval_params']['bootstrap_n']
    )
    results['aurc_balanced_bootstrap'] = {'mean': mean_aurc, '95ci_lower': lower, '95ci_upper': upper}
    print(f"AURC Bootstrap 95% CI: [{lower:.4f}, {upper:.4f}]")
    
    # 5.4 Metrics at fixed coverage points (using raw margins)
    results_at_coverage = {}
    for cov_target in CONFIG['eval_params']['coverage_points']:
        thr_cov = torch.quantile(margins_raw, 1.0 - cov_target)
        accepted_mask = margins_raw >= thr_cov   # >= giúp bền vững khi có ties
        
        metrics = calculate_selective_errors(preds, test_labels, accepted_mask, class_to_group_cpu, num_groups)
        results_at_coverage[f'cov_{cov_target}'] = metrics
        print(f"Metrics @ {metrics['coverage']:.2f} coverage: "
              f"Bal.Err={metrics['balanced_error']:.4f}, Worst.Err={metrics['worst_error']:.4f}")
    results['metrics_at_coverage'] = results_at_coverage
    
    # 5.5 Plugin-specific metrics (at optimal threshold)
    plugin_metrics = calculate_selective_errors(preds, test_labels, accepted, class_to_group_cpu, num_groups)
    results['plugin_metrics_at_threshold'] = plugin_metrics
    
    # Better messaging based on threshold type used
    t_group = checkpoint.get('t_group', None)
    if t_group is not None:
        # Convert to tensor if it's a list
        if isinstance(t_group, list):
            t_group_list = t_group
            t_group = torch.tensor(t_group)
        else:
            t_group_list = t_group.tolist()
            
        print(f"Plugin metrics @ per-group thresholds {[f'{t:.3f}' for t in t_group_list]}: "
              f"Coverage={plugin_metrics['coverage']:.3f}, "
              f"Bal.Err={plugin_metrics['balanced_error']:.4f}, Worst.Err={plugin_metrics['worst_error']:.4f}")
    else:
        print(f"Plugin metrics @ global threshold t*={plugin_threshold:.3f}: "
              f"Coverage={plugin_metrics['coverage']:.3f}, "
              f"Bal.Err={plugin_metrics['balanced_error']:.4f}, Worst.Err={plugin_metrics['worst_error']:.4f}")
    
    # 5.5a Detailed Group-wise Analysis
    threshold_param = t_group if t_group is not None else plugin_threshold
    analyze_group_performance(eta_mix, preds, test_labels, accepted,
                             alpha_star_cpu, mu_star_cpu, threshold_param, class_to_group_cpu, num_groups)
    
    # 5.6 ECE
    ece = calculate_ece(eta_mix, test_labels)
    results['ece'] = ece
    print(f"ECE: {ece:.4f}")
    
    # 5.7 Selective risk with different rejection costs (for comparison)
    print("\nSelective Risk with different rejection costs:")
    selective_risks = {}
    for c_cost in [0.3, 0.5, 0.7]:
        bal_risk = selective_risk_from_mask(preds, test_labels, accepted, c_cost,
                                            class_to_group_cpu, num_groups, "balanced")
        worst_risk = selective_risk_from_mask(preds, test_labels, accepted, c_cost,
                                              class_to_group_cpu, num_groups, "worst")
        selective_risks[f'cost_{c_cost}'] = {'balanced': bal_risk, 'worst': worst_risk}
        print(f"  Cost c={c_cost:.2f}: Balanced={bal_risk:.4f}, Worst={worst_risk:.4f}")
    results['selective_risks'] = selective_risks
    
    # 6. Save results
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(results, f, indent=4)
    print(f"💾 Saved metrics to {output_dir / 'metrics.json'}")
    
    # 7. Plot RC curves
    plt.figure(figsize=(12, 5))
    
    # Full range
    plt.subplot(1, 2, 1)
    plt.plot(rc_df['coverage'], rc_df['balanced_error'], label='Balanced Error', linewidth=2)
    plt.plot(rc_df['coverage'], rc_df['worst_error'], label='Worst-Group Error', linestyle='--', linewidth=2)
    plt.xlabel('Coverage')
    plt.ylabel('Selective Risk (Error)')
    plt.title('GSE-Balanced Plugin RC Curve (Full Range)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Focused range
    plt.subplot(1, 2, 2)
    plt.plot(rc_df_02['coverage'], rc_df_02['balanced_error'], label='Balanced Error', linewidth=2)
    plt.plot(rc_df_02['coverage'], rc_df_02['worst_error'], label='Worst-Group Error', linestyle='--', linewidth=2)
    plt.xlabel('Coverage')
    plt.ylabel('Selective Risk (Error)')
    plt.title('GSE-Balanced Plugin RC Curve (0.2-1.0)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0.2, 1.0)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rc_curve_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'rc_curve_comparison.pdf', bbox_inches='tight')
    print(f"📊 Saved RC curve plots to {output_dir}")
    
    # Summary
    print("\n" + "="*60)
    print("GSE-BALANCED PLUGIN EVALUATION SUMMARY")
    print("="*60)
    print(f"Dataset: {CONFIG['dataset']['name']}")
    print(f"Test samples: {num_test_samples}")
    print(f"Optimal parameters: α*={alpha_star.cpu().tolist()}, μ*={mu_star.cpu().tolist()}")
    
    # Handle threshold display
    t_group = checkpoint.get('t_group', None)
    if t_group is not None:
        if isinstance(t_group, list):
            t_group_display = [f'{t:.3f}' for t in t_group]
        else:
            t_group_display = [f'{t:.3f}' for t in t_group.tolist()]
        print(f"Per-group thresholds (fitted on S1): t* = {t_group_display}")
    else:
        print(f"Raw-margin threshold (fitted on S1): t* = {plugin_threshold:.3f}")
    
    print()
    print("Key Results:")
    print(f"• AURC (Balanced): {aurc_bal:.4f}")
    print(f"• AURC (Worst): {aurc_wst:.4f}") 
    
    if t_group is not None:
        print(f"• Plugin @ per-group thresholds: Coverage={plugin_metrics['coverage']:.3f}, "
              f"Bal.Err={plugin_metrics['balanced_error']:.4f}")
    else:
        print(f"• Plugin @ t*={plugin_threshold:.3f}: Coverage={plugin_metrics['coverage']:.3f}, "
              f"Bal.Err={plugin_metrics['balanced_error']:.4f}")
    
    print(f"• ECE: {ece:.4f}")
    print("="*60)

if __name__ == '__main__':
    main()