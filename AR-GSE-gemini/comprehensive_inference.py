#!/usr/bin/env python3
"""
Comprehensive AR-GSE Inference Script
Performs inference on 50 randomly selected samples (30 Head + 20 Tail) with detailed analysis.
"""

import sys
sys.path.append('.')

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import random
from collections import defaultdict
import pandas as pd

# Import our modules
from src.models.argse import AR_GSE
from src.data.groups import get_class_to_group_by_threshold
from src.data.datasets import get_cifar100_lt_counts

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

def load_test_data_with_stratified_sampling():
    """Load test data and perform stratified sampling: 30 Head + 20 Tail samples."""
    
    # Configuration
    config = {
        'dataset': {'name': 'cifar100_lt_if100', 'splits_dir': './data/cifar100_lt_if100_splits'},
        'experts': {'names': ['ce_baseline', 'logitadjust_baseline', 'balsoftmax_baseline'], 
                'logits_dir': './outputs/logits'},
    }
    
    logits_root = Path(config['experts']['logits_dir']) / config['dataset']['name']
    splits_dir = Path(config['dataset']['splits_dir'])
    
    # Load all test indices
    with open(splits_dir / 'test_lt_indices.json', 'r') as f:
        all_test_indices = json.load(f)
    
    print(f"📊 Total available test samples: {len(all_test_indices)}")
    
    # Load labels for all test samples
    full_test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False)
    all_test_labels = torch.tensor(np.array(full_test_dataset.targets))[all_test_indices]
    
    # Setup class grouping to determine Head/Tail
    class_counts = get_cifar100_lt_counts(imb_factor=100, num_classes=100)
    class_to_group = get_class_to_group_by_threshold(class_counts, threshold=20)
    
    # Separate samples by group
    head_samples = []  # Group 0
    tail_samples = []  # Group 1
    
    for idx, label in enumerate(all_test_labels):
        if class_to_group[label.item()] == 0:  # Head
            head_samples.append(idx)
        else:  # Tail
            tail_samples.append(idx)
    
    print(f"📈 Available Head samples: {len(head_samples)}")
    print(f"📉 Available Tail samples: {len(tail_samples)}")
    
    # Random sampling
    if len(head_samples) < 30:
        print(f"⚠️  Warning: Only {len(head_samples)} Head samples available, using all")
        selected_head = head_samples
    else:
        selected_head = random.sample(head_samples, 30)
    
    if len(tail_samples) < 20:
        print(f"⚠️  Warning: Only {len(tail_samples)} Tail samples available, using all")
        selected_tail = tail_samples
    else:
        selected_tail = random.sample(tail_samples, 20)
    
    # Combine and shuffle
    selected_indices = selected_head + selected_tail
    random.shuffle(selected_indices)
    
    print(f"✅ Selected {len(selected_head)} Head + {len(selected_tail)} Tail = {len(selected_indices)} total samples")
    
    # Convert to actual test indices
    actual_test_indices = [all_test_indices[i] for i in selected_indices]
    selected_labels = all_test_labels[selected_indices]
    
    # Load expert logits for selected samples
    num_experts = len(config['experts']['names'])
    stacked_logits = torch.zeros(len(selected_indices), num_experts, 100)
    
    print("🔄 Loading expert logits...")
    for i, expert_name in enumerate(config['experts']['names']):
        logits_path = logits_root / expert_name / "test_lt_logits.pt"
        if not logits_path.exists():
            raise FileNotFoundError(f"Logits file not found: {logits_path}")
        
        expert_logits = torch.load(logits_path, map_location='cpu', weights_only=False)
        # Select only the chosen samples
        stacked_logits[:, i, :] = expert_logits[selected_indices]
        print(f"   ✅ Loaded {expert_name}")
    
    return stacked_logits, selected_labels, class_to_group, actual_test_indices

def load_argse_model(device):
    """Load the trained AR-GSE model."""
    
    num_experts = 3
    gating_feature_dim = 7 * num_experts + 3  # 7*3 + 3 = 24
    
    model = AR_GSE(num_experts=num_experts, num_classes=100, num_groups=2, 
                  gating_feature_dim=gating_feature_dim).to(device)
    
    # Load trained weights
    checkpoint_path = './checkpoints/argse_worst_eg_improved/cifar100_lt_if100/gse_balanced_plugin.ckpt'
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    print("🔍 Loading model parameters...")
    
    # Load α and μ parameters
    if 'alpha' in checkpoint and 'mu' in checkpoint:
        model.alpha.data = checkpoint['alpha'].to(device)
        model.mu.data = checkpoint['mu'].to(device)
        print("✅ Loaded α, μ parameters")
        
        alpha = checkpoint['alpha'].cpu().numpy()
        mu = checkpoint['mu'].cpu().numpy()
        print(f"   α parameters: Head={alpha[0]:.3f}, Tail={alpha[1]:.3f}")
        print(f"   μ parameters: Head={mu[0]:.3f}, Tail={mu[1]:.3f}")
    else:
        raise ValueError("α, μ parameters not found in checkpoint")
    
    # Load gating network weights
    if 'gating_net_state_dict' in checkpoint:
        try:
            model.gating_net.load_state_dict(checkpoint['gating_net_state_dict'])
            print("✅ Loaded gating network weights")
        except RuntimeError as e:
            print(f"⚠️  Gating network dimension mismatch: {e}")
            print("   Using randomly initialized gating network")
    else:
        # Try separate gating checkpoint
        gating_path = './checkpoints/gating_pretrained/cifar100_lt_if100/gating_selective.ckpt'
        if Path(gating_path).exists():
            try:
                gating_checkpoint = torch.load(gating_path, map_location=device, weights_only=False)
                if 'gating_net_state_dict' in gating_checkpoint:
                    model.gating_net.load_state_dict(gating_checkpoint['gating_net_state_dict'])
                    print("✅ Loaded gating network from separate checkpoint")
                else:
                    print("⚠️  Warning: No gating network weights found, using randomly initialized")
            except RuntimeError as e:
                print(f"⚠️  Gating network loading failed: {e}")
                print("   Using randomly initialized gating network")
        else:
            print("⚠️  Warning: No gating network weights found, using randomly initialized")
    
    # Initialize buffers
    if not hasattr(model, 'Lambda') or model.Lambda is None:
        model.register_buffer('Lambda', torch.zeros(2).to(device))
    if not hasattr(model, 'alpha_ema') or model.alpha_ema is None:
        model.register_buffer('alpha_ema', torch.ones(2).to(device))
    if not hasattr(model, 'm_std') or model.m_std is None:
        model.register_buffer('m_std', torch.tensor(1.0).to(device))
    
    print("✅ Model loaded successfully")
    return model

def run_comprehensive_inference():
    """Run inference on all selected samples."""
    
    print("🚀 AR-GSE Comprehensive Inference")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Using device: {device}")
    
    # Load data and model
    expert_logits, labels, class_to_group, test_indices = load_test_data_with_stratified_sampling()
    model = load_argse_model(device)
    
    # Move data to device
    expert_logits = expert_logits.to(device)
    labels = labels.to(device)
    class_to_group = class_to_group.to(device)
    
    # Run inference with loaded data
    results = run_inference_with_data(expert_logits, labels, class_to_group, test_indices, model, device)
    
    return results, expert_logits, labels, class_to_group, model, device

def run_inference_with_data(expert_logits, labels, class_to_group, test_indices, model, device):
    """Run inference with pre-loaded data."""
    
    print(f"\n📊 Running inference on {len(labels)} samples...")
    
    # Storage for results
    results = []
    expert_names = ['CE', 'LogitAdj', 'BalSoftmax']
    c_cost = 0.2
    tau = 25.0
    
    model.eval()
    with torch.no_grad():
        # Process all samples at once for efficiency
        batch_size = len(labels)
        
        # Expert posteriors
        expert_posteriors = torch.softmax(expert_logits, dim=-1)  # [N, E, C]
        
        # Gating weights
        gating_features = model.feature_builder(expert_logits)
        gating_raw = model.gating_net(gating_features)
        gating_weights = torch.softmax(gating_raw, dim=1)  # [N, E]
        
        # Mixture distribution
        mixture_probs = torch.einsum('ne,nec->nc', gating_weights, expert_posteriors)
        
        # Predictions
        predicted_classes = torch.argmax(mixture_probs, dim=1)
        is_correct = (predicted_classes == labels)
        
        # Selective margins and decisions
        raw_margins = model.selective_margin(mixture_probs, 0.0, class_to_group)
        margins_with_cost = raw_margins - c_cost
        accept_probs = torch.sigmoid(tau * margins_with_cost)
        is_accepted = accept_probs > 0.5
        
        # Store detailed results for each sample
        for i in range(batch_size):
            sample_result = {
                'sample_idx': i,
                'test_idx': test_indices[i],
                'true_label': labels[i].item(),
                'true_group': class_to_group[labels[i]].item(),
                'group_name': 'Head' if class_to_group[labels[i]].item() == 0 else 'Tail',
                'predicted_label': predicted_classes[i].item(),
                'is_correct': is_correct[i].item(),
                'raw_margin': raw_margins[i].item(),
                'margin_with_cost': margins_with_cost[i].item(),
                'accept_prob': accept_probs[i].item(),
                'is_accepted': is_accepted[i].item(),
                'mixture_prob_true': mixture_probs[i, labels[i]].item(),
                'mixture_prob_pred': mixture_probs[i, predicted_classes[i]].item(),
                'max_mixture_prob': mixture_probs[i].max().item(),
            }
            
            # Add expert information
            for e, expert_name in enumerate(expert_names):
                sample_result[f'{expert_name.lower()}_weight'] = gating_weights[i, e].item()
                sample_result[f'{expert_name.lower()}_prob_true'] = expert_posteriors[i, e, labels[i]].item()
                sample_result[f'{expert_name.lower()}_prob_pred'] = expert_posteriors[i, e, predicted_classes[i]].item()
                sample_result[f'{expert_name.lower()}_top_class'] = expert_posteriors[i, e].argmax().item()
                sample_result[f'{expert_name.lower()}_top_prob'] = expert_posteriors[i, e].max().item()
            
            results.append(sample_result)
    
    print("✅ Inference completed!")
    return results

def analyze_results(results):
    """Analyze and summarize the inference results."""
    
    print("\n📊 COMPREHENSIVE RESULTS ANALYSIS")
    print("="*60)
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    # Basic statistics
    total_samples = len(df)
    head_samples = len(df[df['group_name'] == 'Head'])
    tail_samples = len(df[df['group_name'] == 'Tail'])
    
    print(f"📈 Dataset Statistics:")
    print(f"   Total samples: {total_samples}")
    print(f"   Head samples: {head_samples}")
    print(f"   Tail samples: {tail_samples}")
    
    # Prediction accuracy
    correct_predictions = df['is_correct'].sum()
    accuracy = correct_predictions / total_samples
    
    head_accuracy = df[df['group_name'] == 'Head']['is_correct'].mean()
    tail_accuracy = df[df['group_name'] == 'Tail']['is_correct'].mean()
    
    print(f"\n🎯 Prediction Accuracy:")
    print(f"   Overall: {accuracy:.3f} ({correct_predictions}/{total_samples})")
    print(f"   Head group: {head_accuracy:.3f}")
    print(f"   Tail group: {tail_accuracy:.3f}")
    
    # Decision statistics
    accepted_samples = df['is_accepted'].sum()
    acceptance_rate = accepted_samples / total_samples
    
    head_acceptance = df[df['group_name'] == 'Head']['is_accepted'].mean()
    tail_acceptance = df[df['group_name'] == 'Tail']['is_accepted'].mean()
    
    print(f"\n⚖️  Decision Statistics:")
    print(f"   Overall acceptance rate: {acceptance_rate:.3f} ({accepted_samples}/{total_samples})")
    print(f"   Head acceptance rate: {head_acceptance:.3f}")
    print(f"   Tail acceptance rate: {tail_acceptance:.3f}")
    
    # Decision quality analysis
    correct_accepts = len(df[(df['is_accepted'] == True) & (df['is_correct'] == True)])
    incorrect_accepts = len(df[(df['is_accepted'] == True) & (df['is_correct'] == False)])
    correct_rejects = len(df[(df['is_accepted'] == False) & (df['is_correct'] == False)])
    incorrect_rejects = len(df[(df['is_accepted'] == False) & (df['is_correct'] == True)])
    
    print(f"\n🎭 Decision Quality:")
    print(f"   ✅ Correct Accepts: {correct_accepts} (Good: High confidence + Correct)")
    print(f"   ❌ Incorrect Accepts: {incorrect_accepts} (Bad: High confidence + Wrong)")
    print(f"   ✅ Correct Rejects: {correct_rejects} (Good: Low confidence + Wrong)")
    print(f"   ❌ Incorrect Rejects: {incorrect_rejects} (Bad: Low confidence + Correct)")
    
    good_decisions = correct_accepts + correct_rejects
    bad_decisions = incorrect_accepts + incorrect_rejects
    decision_quality = good_decisions / total_samples
    
    print(f"   Overall decision quality: {decision_quality:.3f} ({good_decisions}/{total_samples})")
    
    # Expert analysis
    print(f"\n🤖 Expert Analysis:")
    gating_weights_mean = df[['ce_weight', 'logitadj_weight', 'balsoftmax_weight']].mean()
    print("   Average gating weights:")
    for expert, weight in zip(['CE', 'LogitAdj', 'BalSoftmax'], gating_weights_mean):
        print(f"     {expert:12}: {weight:.3f}")
    
    # Margin analysis
    print(f"\n📏 Margin Analysis:")
    print(f"   Average raw margin: {df['raw_margin'].mean():.3f} (±{df['raw_margin'].std():.3f})")
    print(f"   Average margin with cost: {df['margin_with_cost'].mean():.3f} (±{df['margin_with_cost'].std():.3f})")
    print(f"   Average accept probability: {df['accept_prob'].mean():.3f} (±{df['accept_prob'].std():.3f})")
    
    # Group-specific margin analysis
    print(f"\n   Head group margins:")
    head_df = df[df['group_name'] == 'Head']
    print(f"     Raw margin: {head_df['raw_margin'].mean():.3f} (±{head_df['raw_margin'].std():.3f})")
    print(f"     Margin with cost: {head_df['margin_with_cost'].mean():.3f} (±{head_df['margin_with_cost'].std():.3f})")
    
    print(f"   Tail group margins:")
    tail_df = df[df['group_name'] == 'Tail']
    print(f"     Raw margin: {tail_df['raw_margin'].mean():.3f} (±{tail_df['raw_margin'].std():.3f})")
    print(f"     Margin with cost: {tail_df['margin_with_cost'].mean():.3f} (±{tail_df['margin_with_cost'].std():.3f})")
    
    return df

def save_detailed_results(results, df):
    """Save detailed results to files."""
    
    output_dir = Path('./comprehensive_inference_results')
    output_dir.mkdir(exist_ok=True)
    
    # Save raw results as JSON
    results_path = output_dir / 'inference_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved raw results: {results_path}")
    
    # Save DataFrame as CSV
    csv_path = output_dir / 'inference_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"💾 Saved CSV: {csv_path}")
    
    # Save detailed text summary
    summary_path = output_dir / 'comprehensive_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("AR-GSE COMPREHENSIVE INFERENCE ANALYSIS\n")
        f.write("="*60 + "\n\n")
        
        f.write("EXPERIMENT SETUP:\n")
        f.write("- Total samples: 50 (30 Head + 20 Tail)\n")
        f.write("- Random sampling with seed=42\n")
        f.write("- Model: AR-GSE with balanced plugin\n")
        f.write("- Experts: CE, LogitAdj, BalSoftmax\n")
        f.write("- Rejection cost: 0.2\n")
        f.write("- Temperature: 25.0\n\n")
        
        # Overall statistics
        total_samples = len(df)
        correct_predictions = df['is_correct'].sum()
        accepted_samples = df['is_accepted'].sum()
        
        f.write("OVERALL PERFORMANCE:\n")
        f.write(f"- Prediction Accuracy: {correct_predictions/total_samples:.3f} ({correct_predictions}/{total_samples})\n")
        f.write(f"- Acceptance Rate: {accepted_samples/total_samples:.3f} ({accepted_samples}/{total_samples})\n")
        
        # Group-wise performance
        head_df = df[df['group_name'] == 'Head']
        tail_df = df[df['group_name'] == 'Tail']
        
        f.write(f"- Head Group Accuracy: {head_df['is_correct'].mean():.3f}\n")
        f.write(f"- Tail Group Accuracy: {tail_df['is_correct'].mean():.3f}\n")
        f.write(f"- Head Group Acceptance: {head_df['is_accepted'].mean():.3f}\n")
        f.write(f"- Tail Group Acceptance: {tail_df['is_accepted'].mean():.3f}\n\n")
        
        # Decision quality
        correct_accepts = len(df[df['is_accepted'] & df['is_correct']])
        incorrect_accepts = len(df[df['is_accepted'] & ~df['is_correct']])
        correct_rejects = len(df[~df['is_accepted'] & ~df['is_correct']])
        incorrect_rejects = len(df[~df['is_accepted'] & df['is_correct']])
        
        f.write("DECISION QUALITY BREAKDOWN:\n")
        f.write(f"- Correct Accepts (Good): {correct_accepts}\n")
        f.write(f"- Incorrect Accepts (Bad): {incorrect_accepts}\n")
        f.write(f"- Correct Rejects (Good): {correct_rejects}\n")
        f.write(f"- Incorrect Rejects (Bad): {incorrect_rejects}\n")
        f.write(f"- Decision Quality: {(correct_accepts + correct_rejects)/total_samples:.3f}\n\n")
        
        # Expert weights
        f.write("EXPERT GATING WEIGHTS (Average):\n")
        gating_weights_mean = df[['ce_weight', 'logitadj_weight', 'balsoftmax_weight']].mean()
        for expert, weight in zip(['CE', 'LogitAdj', 'BalSoftmax'], gating_weights_mean):
            f.write(f"- {expert:12}: {weight:.3f}\n")
        f.write("\n")
        
        # Margin statistics
        f.write("MARGIN STATISTICS:\n")
        f.write(f"- Raw Margin (Mean±Std): {df['raw_margin'].mean():.3f}±{df['raw_margin'].std():.3f}\n")
        f.write(f"- Margin with Cost: {df['margin_with_cost'].mean():.3f}±{df['margin_with_cost'].std():.3f}\n")
        f.write(f"- Accept Probability: {df['accept_prob'].mean():.3f}±{df['accept_prob'].std():.3f}\n\n")
        
        # Sample-by-sample summary
        f.write("INDIVIDUAL SAMPLE RESULTS:\n")
        f.write("-" * 100 + "\n")
        f.write("ID | Group | True | Pred | Correct | Accept | Margin | AcceptProb | Decision Quality\n")
        f.write("-" * 100 + "\n")
        
        for _, row in df.iterrows():
            # Determine decision quality
            if row['is_accepted'] and row['is_correct']:
                quality = "Good (CA)"
            elif not row['is_accepted'] and not row['is_correct']:
                quality = "Good (CR)"
            elif row['is_accepted'] and not row['is_correct']:
                quality = "Bad (IA)"
            else:  # not accepted but correct
                quality = "Bad (IR)"
            
            f.write(f"{row['sample_idx']:2d} | {row['group_name']:4s} | {row['true_label']:3d} | "
                   f"{row['predicted_label']:3d} | {'✓' if row['is_correct'] else '✗':7s} | "
                   f"{'✓' if row['is_accepted'] else '✗':6s} | {row['margin_with_cost']:6.3f} | "
                   f"{row['accept_prob']:9.3f} | {quality}\n")
        
        f.write("-" * 100 + "\n")
        f.write("Legend: CA=Correct Accept, CR=Correct Reject, IA=Incorrect Accept, IR=Incorrect Reject\n")
    
    print(f"💾 Saved comprehensive summary: {summary_path}")

def create_visualizations(df):
    """Create comprehensive visualizations."""
    
    output_dir = Path('./comprehensive_inference_results')
    output_dir.mkdir(exist_ok=True)
    
    # Create a comprehensive dashboard
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('AR-GSE Comprehensive Inference Analysis (50 samples)', fontsize=16, fontweight='bold')
    
    # 1. Accuracy by Group
    ax1 = axes[0, 0]
    groups = ['Head', 'Tail', 'Overall']
    accuracies = [
        df[df['group_name'] == 'Head']['is_correct'].mean(),
        df[df['group_name'] == 'Tail']['is_correct'].mean(),
        df['is_correct'].mean()
    ]
    colors = ['lightblue', 'lightcoral', 'lightgreen']
    bars = ax1.bar(groups, accuracies, color=colors, alpha=0.8)
    ax1.set_title('Prediction Accuracy by Group')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Acceptance Rate by Group
    ax2 = axes[0, 1]
    acceptance_rates = [
        df[df['group_name'] == 'Head']['is_accepted'].mean(),
        df[df['group_name'] == 'Tail']['is_accepted'].mean(),
        df['is_accepted'].mean()
    ]
    bars = ax2.bar(groups, acceptance_rates, color=colors, alpha=0.8)
    ax2.set_title('Acceptance Rate by Group')
    ax2.set_ylabel('Acceptance Rate')
    ax2.set_ylim(0, 1)
    for bar, rate in zip(bars, acceptance_rates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{rate:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Decision Quality Matrix
    ax3 = axes[0, 2]
    decision_matrix = np.array([
        [len(df[(df['is_accepted'] == True) & (df['is_correct'] == True)]),   # Correct Accept
         len(df[(df['is_accepted'] == True) & (df['is_correct'] == False)])], # Incorrect Accept
        [len(df[(df['is_accepted'] == False) & (df['is_correct'] == True)]),  # Incorrect Reject
         len(df[(df['is_accepted'] == False) & (df['is_correct'] == False)])] # Correct Reject
    ])
    
    im = ax3.imshow(decision_matrix, cmap='RdYlGn', alpha=0.8)
    ax3.set_title('Decision Quality Matrix')
    ax3.set_xticks([0, 1])
    ax3.set_yticks([0, 1])
    ax3.set_xticklabels(['Correct Pred', 'Wrong Pred'])
    ax3.set_yticklabels(['Accept', 'Reject'])
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax3.text(j, i, decision_matrix[i, j], ha="center", va="center", 
                           color="black", fontweight='bold', fontsize=12)
    
    # 4. Expert Gating Weights Distribution
    ax4 = axes[1, 0]
    expert_weights = df[['ce_weight', 'logitadj_weight', 'balsoftmax_weight']].values
    ax4.boxplot(expert_weights, labels=['CE', 'LogitAdj', 'BalSoftmax'])
    ax4.set_title('Expert Gating Weights Distribution')
    ax4.set_ylabel('Weight')
    ax4.grid(True, alpha=0.3)
    
    # 5. Margin Distribution by Group
    ax5 = axes[1, 1]
    head_margins = df[df['group_name'] == 'Head']['margin_with_cost']
    tail_margins = df[df['group_name'] == 'Tail']['margin_with_cost']
    
    ax5.hist(head_margins, alpha=0.6, label='Head', bins=10, color='lightblue')
    ax5.hist(tail_margins, alpha=0.6, label='Tail', bins=10, color='lightcoral')
    ax5.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Decision Threshold')
    ax5.set_title('Margin Distribution by Group')
    ax5.set_xlabel('Margin with Cost')
    ax5.set_ylabel('Frequency')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Accept Probability vs Margin
    ax6 = axes[1, 2]
    head_data = df[df['group_name'] == 'Head']
    tail_data = df[df['group_name'] == 'Tail']
    
    ax6.scatter(head_data['margin_with_cost'], head_data['accept_prob'], 
               alpha=0.7, label='Head', color='blue', s=50)
    ax6.scatter(tail_data['margin_with_cost'], tail_data['accept_prob'], 
               alpha=0.7, label='Tail', color='red', s=50)
    ax6.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Accept Threshold')
    ax6.axvline(x=0, color='gray', linestyle='--', alpha=0.7, label='Margin=0')
    ax6.set_title('Accept Probability vs Margin')
    ax6.set_xlabel('Margin with Cost')
    ax6.set_ylabel('Accept Probability')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save visualization
    viz_path = output_dir / 'comprehensive_analysis.png'
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved comprehensive visualization: {viz_path}")
    plt.close(fig)
    
    # Create individual sample visualization for interesting cases
    create_sample_highlights(df, output_dir)

def create_individual_sample_analysis(results, expert_logits, labels, class_to_group, device, model):
    """Create detailed individual sample analysis like demo_inference.py."""
    
    print("\n🔍 Creating individual sample analysis...")
    
    # Create individual analysis directory
    individual_dir = Path('./comprehensive_inference_results/individual_samples')
    individual_dir.mkdir(exist_ok=True)
    
    # Select interesting samples for detailed analysis (max 10 to avoid clutter)
    interesting_indices = []
    
    # Get some samples from each category
    df = pd.DataFrame(results)
    
    # Correct samples (both accepted and rejected)
    correct_samples = df[df['is_correct'] == True]
    if len(correct_samples) > 0:
        interesting_indices.extend(correct_samples.head(3)['sample_idx'].tolist())
    
    # Incorrect samples
    incorrect_samples = df[df['is_correct'] == False]
    if len(incorrect_samples) > 0:
        interesting_indices.extend(incorrect_samples.head(3)['sample_idx'].tolist())
    
    # Samples with highest margins
    high_margin_samples = df.nlargest(2, 'raw_margin')
    interesting_indices.extend(high_margin_samples['sample_idx'].tolist())
    
    # Samples with lowest margins  
    low_margin_samples = df.nsmallest(2, 'raw_margin')
    interesting_indices.extend(low_margin_samples['sample_idx'].tolist())
    
    # Remove duplicates and limit to 10 samples
    interesting_indices = list(set(interesting_indices))[:10]
    
    print(f"   Analyzing {len(interesting_indices)} individual samples...")
    
    model.eval()
    with torch.no_grad():
        for sample_idx in interesting_indices:
            # Get sample data
            sample_logits = expert_logits[sample_idx:sample_idx+1].to(device)  # [1, E, C]
            sample_label = labels[sample_idx].to(device)
            
            # Get group info
            true_group = class_to_group[sample_label].item()
            group_name = "Head" if true_group == 0 else "Tail"
            
            # Inference process
            expert_posteriors = torch.softmax(sample_logits, dim=-1)  # [1, E, C]
            gating_features = model.feature_builder(sample_logits)
            gating_raw = model.gating_net(gating_features)
            gating_weights = torch.softmax(gating_raw, dim=1)  # [1, E]
            mixture_probs = torch.einsum('be,bec->bc', gating_weights, expert_posteriors)  # [1, C]
            
            predicted_class = torch.argmax(mixture_probs, dim=1).item()
            is_correct = predicted_class == sample_label.item()
            
            # Selective margin and decision
            c_cost = 0.2
            tau = 25.0
            raw_margin = model.selective_margin(mixture_probs, 0.0, class_to_group)
            margin_with_cost = raw_margin - c_cost
            accept_prob = torch.sigmoid(tau * margin_with_cost)
            is_accepted = accept_prob > 0.5
            
            # Create individual sample visualization
            create_individual_sample_visualization(
                sample_idx, sample_label.cpu().item(), group_name,
                expert_posteriors.cpu().numpy()[0],
                gating_weights.cpu().numpy()[0], 
                mixture_probs.cpu().numpy()[0],
                raw_margin.cpu().item(), margin_with_cost.cpu().item(),
                accept_prob.cpu().item(), is_accepted.cpu().item(),
                predicted_class, is_correct, individual_dir
            )
            
            # Create individual sample text summary
            create_individual_sample_text_summary(
                sample_idx, sample_label.cpu().item(), group_name,
                expert_posteriors.cpu().numpy()[0],
                gating_weights.cpu().numpy()[0], 
                mixture_probs.cpu().numpy()[0],
                raw_margin.cpu().item(), margin_with_cost.cpu().item(),
                accept_prob.cpu().item(), is_accepted.cpu().item(),
                predicted_class, is_correct, individual_dir
            )
    
    # Create overall individual analysis summary
    create_individual_overall_summary(interesting_indices, individual_dir)
    
    print(f"   ✅ Individual sample analysis completed!")
    return interesting_indices

def create_individual_sample_visualization(sample_idx, true_label, group_name, expert_posteriors, 
                               gating_weights, mixture_probs, raw_margin, margin_with_cost,
                               accept_prob, is_accepted, predicted_class, is_correct, output_dir):
    """Create detailed visualization for individual sample (like demo_inference.py)."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    decision_text = "ACCEPT" if is_accepted else "REJECT"
    prediction_text = "Pred Correct" if is_correct else "Pred Wrong"
    
    title_text = (f'Sample {sample_idx + 1}: True={true_label} | Pred={predicted_class} | Decision={decision_text} | {prediction_text}\n'
                 f'Raw Margin={raw_margin:.3f} | Margin-Cost={margin_with_cost:.3f} | Accept Prob={accept_prob:.4f}')
    
    fig.suptitle(title_text, fontsize=12, fontweight='bold')
    
    # 1. Expert posteriors (ALL 100 classes)
    ax1 = axes[0, 0]
    expert_names = ['CE', 'LogitAdj', 'BalSoftmax']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    class_indices = np.arange(100)
    
    for e, (expert_name, color) in enumerate(zip(expert_names, colors)):
        probs = expert_posteriors[e]
        ax1.plot(class_indices, probs, label=expert_name, color=color, alpha=0.8, linewidth=2)
        ax1.scatter(true_label, probs[true_label], color='red', s=100, zorder=5)
    
    ax1.axvline(x=true_label, color='red', linestyle='--', alpha=0.7, label=f'True Class {true_label}')
    ax1.set_title(f'Expert Posteriors (All 100 Classes)\nTrue Class {true_label} Probability: CE={expert_posteriors[0][true_label]:.3f}, LogitAdj={expert_posteriors[1][true_label]:.3f}, BalSoftmax={expert_posteriors[2][true_label]:.3f}')
    ax1.set_ylabel('Probability')
    ax1.set_xlabel('Class Index')
    ax1.legend()
    ax1.set_xlim(0, 99)
    ax1.grid(True, alpha=0.3)
    
    # 2. Gating weights
    ax2 = axes[0, 1]
    bars = ax2.bar(expert_names, gating_weights, color=colors, alpha=0.8)
    ax2.set_title('Gating Weights')
    ax2.set_ylabel('Weight')
    ax2.set_ylim(0, 1)
    
    for bar, weight in zip(bars, gating_weights):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{weight:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Mixture distribution (ALL 100 classes)
    ax3 = axes[1, 0] 
    ax3.plot(class_indices, mixture_probs, color='purple', linewidth=2, alpha=0.8, label='Mixture')
    ax3.scatter(true_label, mixture_probs[true_label], color='red', s=120, 
               label=f'True Class {true_label}', zorder=5, marker='o')
    
    if predicted_class != true_label:
        ax3.scatter(predicted_class, mixture_probs[predicted_class], color='black', s=120,
                   label=f'Predicted Class {predicted_class}', zorder=5, marker='s')
    
    ax3.axvline(x=true_label, color='red', linestyle='--', alpha=0.5)
    if predicted_class != true_label:
        ax3.axvline(x=predicted_class, color='black', linestyle='--', alpha=0.5)
    
    ax3.set_title(f'Mixture Distribution (All 100 Classes)\nTrue={true_label}(p={mixture_probs[true_label]:.3f}) | Pred={predicted_class}(p={mixture_probs[predicted_class]:.3f})')
    ax3.set_ylabel('Probability')
    ax3.set_xlabel('Class Index')
    ax3.set_xlim(0, 99)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. Decision process
    ax4 = axes[1, 1]
    
    # Calculate threshold based on group
    if group_name == "Head":
        threshold = 0.920  # approximation
    else:  # Tail
        threshold = 0.680  # approximation
        
    margin_data = ['Threshold', 'Raw Margin', 'Margin - Cost']
    margin_values = [threshold, raw_margin, margin_with_cost]
    colors_margin = ['red', 'blue', 'green']
    
    bars = ax4.bar(margin_data, margin_values, color=colors_margin, alpha=0.7)
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    decision_result = "ACCEPT" if is_accepted else "REJECT"
    ax4.set_title(f'Decision Process: {decision_result}\nAccept Prob: {accept_prob:.3f} | Margin: {margin_with_cost:.3f}')
    ax4.set_ylabel('Margin Value')
    
    for bar, value in zip(bars, margin_values):
        ax4.text(bar.get_x() + bar.get_width()/2, 
                bar.get_height() + (0.01 if value >= 0 else -0.02),
                f'{value:.3f}', ha='center', 
                va='bottom' if value >= 0 else 'top', fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    save_path = output_dir / f'sample_{sample_idx + 1}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def create_individual_sample_text_summary(sample_idx, true_label, group_name, expert_posteriors, 
                            gating_weights, mixture_probs, raw_margin, margin_with_cost,
                            accept_prob, is_accepted, predicted_class, is_correct, output_dir):
    """Create detailed text summary for individual sample (like demo_inference.py)."""
    
    text_path = output_dir / f'sample_{sample_idx + 1}_summary.txt'
    
    prediction_status = "CORRECT" if is_correct else "WRONG" 
    decision_status = "ACCEPT" if is_accepted else "REJECT"
    
    if is_accepted and is_correct:
        outcome = "✓ Correct Accept (Good decision)"
    elif not is_accepted and not is_correct:
        outcome = "✓ Correct Reject (Good decision)" 
    elif is_accepted and not is_correct:
        outcome = "✗ Incorrect Accept (Bad decision)"
    else:
        outcome = "✗ Incorrect Reject (Bad decision)"
    
    expert_names = ['CE', 'LogitAdj', 'BalSoftmax']
    expert_top_classes = [expert_posteriors[e].argmax() for e in range(3)]
    expert_top_probs = [expert_posteriors[e].max() for e in range(3)]
    expert_true_probs = [expert_posteriors[e][true_label] for e in range(3)]
    
    with open(text_path, 'w', encoding='utf-8') as f:
        f.write(f"AR-GSE INFERENCE ANALYSIS - SAMPLE {sample_idx + 1}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("GROUND TRUTH INFORMATION:\n")
        f.write(f"  True Label: Class {true_label}\n")
        f.write(f"  Group: {group_name}\n\n")
        
        f.write("EXPERT PREDICTIONS:\n")
        for i, expert_name in enumerate(expert_names):
            f.write(f"  {expert_name:12}: Top prediction = Class {expert_top_classes[i]} (prob={expert_top_probs[i]:.4f})\n")
            f.write(f"  {expert_name:12}: True class prob = {expert_true_probs[i]:.4f}\n")
        f.write("\n")
        
        f.write("GATING NETWORK WEIGHTS:\n")
        for i, expert_name in enumerate(expert_names):
            f.write(f"  {expert_name:12}: {gating_weights[i]:.4f}\n")
        f.write(f"  Dominant expert: {expert_names[gating_weights.argmax()]}\n\n")
        
        f.write("MIXTURE RESULT:\n")
        f.write(f"  Predicted Class: {predicted_class}\n")
        f.write(f"  Prediction Status: {prediction_status}\n")
        f.write(f"  Predicted Class Probability: {mixture_probs[predicted_class]:.4f}\n")
        f.write(f"  True Class Probability: {mixture_probs[true_label]:.4f}\n")
        f.write(f"  Max Probability: {mixture_probs.max():.4f}\n\n")
        
        f.write("SELECTIVE DECISION PROCESS:\n")
        f.write(f"  Raw Margin: {raw_margin:.4f}\n")
        f.write(f"  Margin - Cost: {margin_with_cost:.4f}\n")
        f.write(f"  Accept Probability: {accept_prob:.4f}\n")
        f.write(f"  Decision: {decision_status}\n\n")
        
        f.write("OVERALL EVALUATION:\n")
        f.write(f"  Prediction: {prediction_status} (True={true_label}, Pred={predicted_class})\n")
        f.write(f"  Decision: {decision_status}\n")
        f.write(f"  Outcome: {outcome}\n\n")
        
        f.write("DETAILED ANALYSIS:\n")
        if prediction_status == "WRONG":
            f.write(f"  - Model incorrectly predicted Class {predicted_class} instead of Class {true_label}\n")
        else:
            f.write(f"  - Model correctly predicted Class {predicted_class}\n")
            
        if decision_status == "REJECT":
            f.write("  - Model rejected this sample (low confidence/margin)\n")
            if prediction_status == "WRONG":
                f.write("  - This is a GOOD decision: rejecting wrong prediction\n")
            else:
                f.write("  - This is a BAD decision: rejecting correct prediction\n")
        else:
            f.write("  - Model accepted this sample (high confidence/margin)\n")
            if prediction_status == "CORRECT":
                f.write("  - This is a GOOD decision: accepting correct prediction\n")
            else:
                f.write("  - This is a BAD decision: accepting wrong prediction\n")

def create_individual_overall_summary(sample_indices, output_dir):
    """Create overall summary for individual sample analysis."""
    
    summary_path = output_dir / 'individual_analysis_summary.txt'
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("INDIVIDUAL SAMPLE ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("FILES GENERATED:\n")
        f.write("This folder contains detailed analysis for selected interesting samples.\n\n")
        
        f.write("For each sample, you will find:\n")
        f.write("- sample_X.png: Comprehensive visualization with 4 panels:\n")
        f.write("  • Expert Posteriors: How each expert assigns probabilities\n")
        f.write("  • Gating Weights: Which expert the model trusts most\n")
        f.write("  • Mixture Distribution: Final combined probabilities\n")
        f.write("  • Decision Process: Margin calculation and accept/reject decision\n\n")
        
        f.write("- sample_X_summary.txt: Detailed text analysis including:\n")
        f.write("  • Ground truth information\n")
        f.write("  • Expert predictions and confidences\n")
        f.write("  • Gating network weights\n")
        f.write("  • Mixture results\n")
        f.write("  • Selective decision process\n")
        f.write("  • Overall evaluation and outcome\n\n")
        
        f.write(f"SAMPLES ANALYZED: {len(sample_indices)}\n")
        f.write("Sample indices: " + ", ".join([str(i+1) for i in sample_indices]) + "\n\n")
        
        f.write("HOW TO USE:\n")
        f.write("1. Start with the visualizations (.png files) for quick overview\n")
        f.write("2. Read text summaries for detailed mathematical explanations\n")
        f.write("3. Compare different samples to understand model behavior\n")
        f.write("4. Focus on 'Bad' decisions to identify improvement areas\n")

def create_sample_highlights(df, output_dir):
    """Create visualizations for interesting individual samples."""
    
    # Find interesting samples:
    # 1. Correct Accept with high confidence
    # 2. Incorrect Accept (bad decision)
    # 3. Correct Reject (good decision) 
    # 4. Incorrect Reject (bad decision)
    
    interesting_samples = []
    
    # Best case: Correct Accept with highest margin
    correct_accepts = df[df['is_accepted'] & df['is_correct']]
    if len(correct_accepts) > 0:
        best_accept = correct_accepts.loc[correct_accepts['margin_with_cost'].idxmax()]
        interesting_samples.append(('Best Accept', best_accept))
    
    # Worst case: Incorrect Accept with highest confidence
    incorrect_accepts = df[df['is_accepted'] & ~df['is_correct']]
    if len(incorrect_accepts) > 0:
        worst_accept = incorrect_accepts.loc[incorrect_accepts['accept_prob'].idxmax()]
        interesting_samples.append(('Worst Accept', worst_accept))
    
    # Good reject: Correct Reject with lowest margin
    correct_rejects = df[~df['is_accepted'] & ~df['is_correct']]
    if len(correct_rejects) > 0:
        good_reject = correct_rejects.loc[correct_rejects['margin_with_cost'].idxmin()]
        interesting_samples.append(('Good Reject', good_reject))
    
    # Bad reject: Incorrect Reject with highest margin among rejects
    incorrect_rejects = df[~df['is_accepted'] & df['is_correct']]
    if len(incorrect_rejects) > 0:
        bad_reject = incorrect_rejects.loc[incorrect_rejects['margin_with_cost'].idxmax()]
        interesting_samples.append(('Bad Reject', bad_reject))
    
    # Create summary visualization
    if interesting_samples:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Interesting Sample Cases Analysis', fontsize=16, fontweight='bold')
        
        for i, (case_name, sample) in enumerate(interesting_samples[:4]):
            ax = axes[i//2, i%2]
            
            # Create bar chart for this sample's expert weights and key metrics
            categories = ['CE Weight', 'LogitAdj Weight', 'BalSoftmax Weight', 'True Prob', 'Pred Prob', 'Accept Prob']
            values = [
                sample['ce_weight'], 
                sample['logitadj_weight'], 
                sample['balsoftmax_weight'],
                sample['mixture_prob_true'],
                sample['mixture_prob_pred'],
                sample['accept_prob']
            ]
            
            colors = ['lightblue', 'orange', 'lightgreen', 'red', 'purple', 'gray']
            bars = ax.bar(categories, values, color=colors, alpha=0.7)
            
            # Add value labels
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9)
            
            decision = "ACCEPT" if sample['is_accepted'] else "REJECT"
            correctness = "CORRECT" if sample['is_correct'] else "WRONG"
            
            ax.set_title(f'{case_name}\nSample {sample["sample_idx"]} ({sample["group_name"]}): '
                        f'True={sample["true_label"]}, Pred={sample["predicted_label"]}\n'
                        f'Decision: {decision}, Prediction: {correctness}')
            ax.set_ylabel('Value')
            ax.set_ylim(0, 1.1)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Hide empty subplots
        for i in range(len(interesting_samples), 4):
            axes[i//2, i%2].set_visible(False)
        
        plt.tight_layout()
        
        highlights_path = output_dir / 'sample_highlights.png'
        plt.savefig(highlights_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved sample highlights: {highlights_path}")
        plt.close(fig)

def main():
    """Main function to run comprehensive inference."""
    
    try:
        print("🚀 Starting AR-GSE Comprehensive Inference")
        print("   Target: 50 samples (30 Head + 20 Tail)")
        print("   Random seed: 42")
        print()
        
        # Run inference (now returns additional data for individual analysis)
        results, expert_logits, labels, class_to_group, model, device = run_comprehensive_inference()
        
        # Analyze results
        df = analyze_results(results)
        
        # Save results
        save_detailed_results(results, df)
        
        # Create visualizations
        create_visualizations(df)
        
        # Create individual sample analysis (like demo_inference.py)
        interesting_samples = create_individual_sample_analysis(
            results, expert_logits, labels, class_to_group, device, model
        )
        
        print("\n🎉 Comprehensive inference completed!")
        print("📁 Results saved in: './comprehensive_inference_results/'")
        print("   - inference_results.json: Raw results")
        print("   - inference_results.csv: Tabular data") 
        print("   - comprehensive_summary.txt: Detailed text analysis")
        print("   - comprehensive_analysis.png: Main visualizations")
        print("   - sample_highlights.png: Interesting cases")
        print(f"   - individual_samples/: Detailed analysis for {len(interesting_samples)} samples")
        print("     • sample_X.png: Individual visualizations")
        print("     • sample_X_summary.txt: Detailed text analysis")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()