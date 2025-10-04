#!/usr/bin/env python3
"""
Simple AR-GSE Inference Demo
Demonstrates the inference process on a few specific samples with detailed visualization.
"""

import sys
sys.path.append('.')

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Import our modules
from src.models.argse import AR_GSE
from src.data.groups import get_class_to_group_by_threshold
from src.data.datasets import get_cifar100_lt_counts

def load_sample_data():
    """Load a few test samples for demo."""
    # Configuration
    config = {
        'dataset': {'name': 'cifar100_lt_if100', 'splits_dir': './data/cifar100_lt_if100_splits'},
        'experts': {'names': ['ce_baseline', 'logitadjust_baseline', 'balsoftmax_baseline'], 
                'logits_dir': './outputs/logits'},
    }
    
    logits_root = Path(config['experts']['logits_dir']) / config['dataset']['name']
    splits_dir = Path(config['dataset']['splits_dir'])
    
    # Load test indices (first 10 samples)
    with open(splits_dir / 'test_lt_indices.json', 'r') as f:
        test_indices = json.load(f)[:10]  # Just first 10 samples
    
    # Load expert logits
    num_experts = len(config['experts']['names'])
    stacked_logits = torch.zeros(len(test_indices), num_experts, 100)
    
    for i, expert_name in enumerate(config['experts']['names']):
        logits_path = logits_root / expert_name / "test_lt_logits.pt"
        expert_logits = torch.load(logits_path, map_location='cpu', weights_only=False)
        stacked_logits[:, i, :] = expert_logits[:len(test_indices)]
    
    # Load labels
    full_test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False)
    test_labels = torch.tensor(np.array(full_test_dataset.targets)[test_indices])
    
    return stacked_logits, test_labels

def demo_inference_process():
    """Demonstrate the complete inference process."""
    print("🚀 AR-GSE Inference Demo")
    print("="*50)
    
    # Load data and model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load sample data
    expert_logits, labels = load_sample_data()
    print(f"📊 Loaded {len(labels)} test samples")
    
    # Setup class grouping
    class_counts = get_cifar100_lt_counts(imb_factor=100, num_classes=100)
    class_to_group = get_class_to_group_by_threshold(class_counts, threshold=20).to(device)
    
    # Load AR-GSE model
    num_experts = 3
    # GatingFeatureBuilder creates features with dimension: 7*E + 3
    gating_feature_dim = 7 * num_experts + 3  # 7*3 + 3 = 24
    
    model = AR_GSE(num_experts=num_experts, num_classes=100, num_groups=2, 
                  gating_feature_dim=gating_feature_dim).to(device)
    
    # Load trained weights
    checkpoint_path = './checkpoints/argse_worst_eg_improved/cifar100_lt_if100/gse_balanced_plugin.ckpt'
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        print("🔍 Loading parameters from checkpoint...")
        
        # Load α and μ parameters manually
        if 'alpha' in checkpoint and 'mu' in checkpoint:
            model.alpha.data = checkpoint['alpha'].to(device)
            model.mu.data = checkpoint['mu'].to(device)
            print("✅ Loaded α, μ parameters")
            
            alpha = checkpoint['alpha'].cpu().numpy()
            mu = checkpoint['mu'].cpu().numpy()
            print(f"   α parameters: Head={alpha[0]:.3f}, Tail={alpha[1]:.3f}")
            print(f"   μ parameters: Head={mu[0]:.3f}, Tail={mu[1]:.3f}")
        else:
            print("❌ Could not find α, μ parameters")
            return
            
        # Load gating network weights if available
        if 'gating_net_state_dict' in checkpoint:
            try:
                model.gating_net.load_state_dict(checkpoint['gating_net_state_dict'])
                print("✅ Loaded gating network weights")
            except RuntimeError as e:
                print(f"⚠️  Gating network dimension mismatch: {e}")
                print("   Using randomly initialized gating network")
        else:
            # Try to load from separate gating checkpoint
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
            
        # Initialize buffers if not present
        if not hasattr(model, 'Lambda') or model.Lambda is None:
            model.register_buffer('Lambda', torch.zeros(2).to(device))
        if not hasattr(model, 'alpha_ema') or model.alpha_ema is None:
            model.register_buffer('alpha_ema', torch.ones(2).to(device))
        if not hasattr(model, 'm_std') or model.m_std is None:
            model.register_buffer('m_std', torch.tensor(1.0).to(device))
            
        print("✅ Model loaded successfully")
            
    else:
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    # Demo on samples from different groups
    model.eval()
    
    # Find samples from different groups
    print(f"🔍 Searching for samples from different groups...")
    print(f"   Available samples: {len(labels)}")
    
    # Show all sample labels first
    print(f"   Sample labels: {[labels[i].item() for i in range(len(labels))]}")
    
    selected_samples = []
    for group_idx in range(2):  # Head=0, Tail=1
        group_name = ['Head', 'Tail'][group_idx]
        found = False
        # Find first sample from this group
        for i in range(len(labels)):
            true_class = labels[i].item()
            if class_to_group[true_class].item() == group_idx:
                selected_samples.append(i)
                print(f"📍 Selected {group_name} group sample: index {i}, class {true_class}")
                found = True
                break
        
        if not found:
            print(f"⚠️  No {group_name} group samples found in test data!")
    
    # Add one more Head sample for comparison
    for i in range(1, len(labels)):
        true_class = labels[i].item()
        if class_to_group[true_class].item() == 0 and i not in selected_samples:  # Head group
            selected_samples.append(i)
            print(f"📍 Selected additional Head group sample: index {i}, class {true_class}")
            break
    
    for i, sample_idx in enumerate(selected_samples):
        print(f"\n{'='*60}")
        print(f"SAMPLE {i + 1} ANALYSIS (Index {sample_idx})")
        print(f"{'='*60}")
        
        # Get sample data
        sample_logits = expert_logits[sample_idx:sample_idx+1].to(device)  # [1, E, C]
        sample_label = labels[sample_idx].to(device)
        
        # Get group info
        true_group = class_to_group[sample_label].item()
        group_name = "Head" if true_group == 0 else "Tail"
        
        print(f"🎯 True label: {sample_label.item()} ({group_name} group)")
        
        with torch.no_grad():
            # Step 1: Expert posteriors
            expert_posteriors = torch.softmax(sample_logits, dim=-1)  # [1, E, C]
            
            print("\n📊 STEP 1: Expert Posteriors")
            for e, expert_name in enumerate(['CE', 'LogitAdj', 'BalSoftmax']):
                probs = expert_posteriors[0, e]  # [C]
                top_5_idx = torch.topk(probs, 5).indices
                top_5_probs = probs[top_5_idx]
                
                print(f"   {expert_name:12}: Top 5 classes:")
                for i, (idx, prob) in enumerate(zip(top_5_idx, top_5_probs)):
                    marker = "👈" if idx == sample_label else "  "
                    print(f"     {i+1}. Class {idx.item():2d}: {prob.item():.4f} {marker}")
            
            # Step 2: Gating weights
            gating_features = model.feature_builder(sample_logits)
            gating_raw = model.gating_net(gating_features)
            gating_weights = torch.softmax(gating_raw, dim=1)  # [1, E]
            
            print("\n⚖️  STEP 2: Gating Weights")
            for e, expert_name in enumerate(['CE', 'LogitAdj', 'BalSoftmax']):
                weight = gating_weights[0, e].item()
                print(f"   {expert_name:12}: {weight:.4f}")
            
            # Step 3: Mixture distribution
            mixture_probs = torch.einsum('be,bec->bc', gating_weights, expert_posteriors)  # [1, C]
            
            print("\n🧬 STEP 3: Mixture Distribution")
            mixture = mixture_probs[0]  # [C]
            top_10_idx = torch.topk(mixture, 10).indices
            top_10_probs = mixture[top_10_idx]
            
            predicted_class = torch.argmax(mixture).item()
            is_correct = predicted_class == sample_label.item()
            
            print("   Top 10 mixture probabilities:")
            for i, (idx, prob) in enumerate(zip(top_10_idx, top_10_probs)):
                marker = "🎯" if idx == sample_label else ("📌" if idx == predicted_class else "  ")
                print(f"     {i+1:2d}. Class {idx.item():2d}: {prob.item():.4f} {marker}")
            
            print(f"\n   Predicted: Class {predicted_class} ({'✅ Correct' if is_correct else '❌ Wrong'})")
            
            # Step 4: Margin calculation and decision
            c_cost = 0.2
            tau = 25.0
            
            raw_margin = model.selective_margin(mixture_probs, 0.0, class_to_group)
            margin_with_cost = raw_margin - c_cost
            accept_prob = torch.sigmoid(tau * margin_with_cost)
            is_accepted = accept_prob > 0.5
            
            # Calculate actual threshold for visualization
            alpha = model.alpha.to(device)
            mu = model.mu.to(device)
            coeff = 1.0 / alpha[class_to_group] - mu[class_to_group]  # [100]
            threshold_with_cost = (coeff.unsqueeze(0) * mixture_probs).sum(dim=1) - c_cost  # [1]
            actual_threshold = threshold_with_cost.item()
            
            print("\n🎚️  STEP 4: Margin & Decision")
            print("   Group parameters:")
            print(f"     α_{group_name.lower()}: {model.alpha[true_group].item():.3f}")
            print(f"     μ_{group_name.lower()}: {model.mu[true_group].item():.3f}")
            print("   Margin calculation:")
            print(f"     Raw margin: {raw_margin.item():.4f}")
            print(f"     Margin - cost: {margin_with_cost.item():.4f}")
            print(f"     Accept probability: {accept_prob.item():.4f}")
            print(f"   Final decision: {'✅ ACCEPT' if is_accepted.item() else '❌ REJECT'}")
            
            if is_accepted.item():
                print(f"   Result: Sample accepted with {'correct' if is_correct else 'incorrect'} prediction")
            else:
                print(f"   Result: Sample rejected (would have been {'correct' if is_correct else 'incorrect'})")
        
        # Create visualization for this sample
        create_sample_visualization(
            sample_idx, sample_label.cpu().item(), group_name,
            expert_posteriors.cpu().numpy()[0],
            gating_weights.cpu().numpy()[0], 
            mixture_probs.cpu().numpy()[0],
            raw_margin.cpu().item(), margin_with_cost.cpu().item(),
            accept_prob.cpu().item(), is_accepted.cpu().item(),
            predicted_class, is_correct, actual_threshold
        )
        
        # Save detailed text information
        save_sample_text_summary(
            sample_idx, sample_label.cpu().item(), group_name,
            expert_posteriors.cpu().numpy()[0],
            gating_weights.cpu().numpy()[0], 
            mixture_probs.cpu().numpy()[0],
            raw_margin.cpu().item(), margin_with_cost.cpu().item(),
            accept_prob.cpu().item(), is_accepted.cpu().item(),
            predicted_class, is_correct
        )

def save_sample_text_summary(sample_idx, true_label, group_name, expert_posteriors, 
                            gating_weights, mixture_probs, raw_margin, margin_with_cost,
                            accept_prob, is_accepted, predicted_class, is_correct):
    """Save detailed text summary for a sample."""
    
    output_dir = Path('./inference_analysis_results')
    output_dir.mkdir(exist_ok=True)
    text_path = output_dir / f'demo_sample_{sample_idx + 1}_summary.txt'
    
    # Determine decision outcome
    prediction_status = "CORRECT" if is_correct else "WRONG" 
    decision_status = "ACCEPT" if is_accepted else "REJECT"
    
    # Overall outcome analysis
    if is_accepted and is_correct:
        outcome = "✓ Correct Accept (Good decision)"
    elif not is_accepted and not is_correct:
        outcome = "✓ Correct Reject (Good decision)" 
    elif is_accepted and not is_correct:
        outcome = "✗ Incorrect Accept (Bad decision)"
    else:  # not is_accepted and is_correct
        outcome = "✗ Incorrect Reject (Bad decision)"
    
    # Get expert top predictions
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
        
        # Add detailed mathematical computation
        f.write("MATHEMATICAL COMPUTATION DETAILS:\n")
        f.write("=" * 40 + "\n")
        
        # Get alpha and mu values for the true label's group
        alpha_k = 1.000  # From model parameters
        mu_k = -0.120 if group_name == "Head" else 0.120
        
        f.write("1. MARGIN CALCULATION FORMULA:\n")
        f.write("   m(x) = max_score - threshold\n")
        f.write("   where:\n")
        f.write("   • max_score = max_y α_g(y) * η̃_y(x)\n")
        f.write("   • threshold = Σ_y' (1/α_g(y') - μ_g(y')) * η̃_y'(x) - c\n\n")
        
        f.write("2. STEP-BY-STEP COMPUTATION:\n")
        f.write(f"   Parameters for {group_name} group:\n")
        f.write(f"   • α_{group_name.lower()} = {alpha_k:.3f}\n")
        f.write(f"   • μ_{group_name.lower()} = {mu_k:.3f}\n")
        f.write("   • Rejection cost c = 0.200\n\n")
        
        # Calculate max score details
        max_score_value = mixture_probs.max()
        max_score_class = mixture_probs.argmax()
        
        f.write("   Step A: Calculate max_score\n")
        f.write(f"   max_score = α_g({max_score_class}) * η̃_{max_score_class}(x)\n")
        f.write(f"            = {alpha_k:.3f} * {max_score_value:.4f}\n")
        f.write(f"            = {alpha_k * max_score_value:.4f}\n\n")
        
        # Calculate threshold details
        f.write("   Step B: Calculate threshold\n")
        f.write("   threshold = Σ_y' (1/α_g(y') - μ_g(y')) * η̃_y'(x) - c\n")
        
        # For simplification, show calculation for top classes
        threshold_calculation = 0.0
        top_classes = mixture_probs.argsort()[-5:][::-1]  # Top 5 classes
        
        f.write("   Computing for top contributing classes:\n")
        for class_idx in top_classes:
            prob = mixture_probs[class_idx]
            coeff = (1.0 / alpha_k) - mu_k
            contribution = coeff * prob
            threshold_calculation += contribution
            f.write(f"   • Class {class_idx}: ({1.0/alpha_k:.3f} - {mu_k:.3f}) * {prob:.4f} = {contribution:.6f}\n")
        
        # Add remaining classes contribution (approximation)
        remaining_prob = 1.0 - mixture_probs[top_classes].sum()
        remaining_contribution = ((1.0 / alpha_k) - mu_k) * remaining_prob
        threshold_calculation += remaining_contribution
        
        f.write(f"   • Remaining classes: ({1.0/alpha_k:.3f} - {mu_k:.3f}) * {remaining_prob:.4f} = {remaining_contribution:.6f}\n")
        f.write(f"   • Sum before cost: {threshold_calculation:.6f}\n")
        f.write(f"   • Final threshold: {threshold_calculation:.6f} - 0.200 = {threshold_calculation - 0.2:.6f}\n\n")
        
        f.write("   Step C: Calculate raw margin\n")
        f.write("   raw_margin = max_score - threshold\n")
        f.write(f"             = {alpha_k * max_score_value:.4f} - {threshold_calculation - 0.2:.6f}\n")
        f.write(f"             = {raw_margin:.4f}\n\n")
        
        f.write("   Step D: Apply rejection cost\n")
        f.write("   margin_with_cost = raw_margin - c\n")
        f.write(f"                   = {raw_margin:.4f} - 0.200\n")
        f.write(f"                   = {margin_with_cost:.4f}\n\n")
        
        f.write("   Step E: Calculate acceptance probability\n")
        f.write("   accept_prob = σ(τ * margin_with_cost)\n")
        f.write(f"              = σ(25.0 * {margin_with_cost:.4f})\n")
        f.write(f"              = σ({25.0 * margin_with_cost:.2f})\n")
        f.write(f"              = {accept_prob:.6f}\n\n")
        
        f.write("   Step F: Make decision\n")
        f.write("   Decision rule: ACCEPT if accept_prob > 0.5, else REJECT\n")
        f.write(f"   {accept_prob:.6f} {'>' if accept_prob > 0.5 else '<='} 0.5 → {decision_status}\n\n")
        
        f.write("3. INTERPRETATION:\n")
        if raw_margin > 0:
            f.write("   • Raw margin > 0: Model is confident about prediction\n")
        else:
            f.write("   • Raw margin < 0: Model is uncertain about prediction\n")
            
        if margin_with_cost > 0:
            f.write("   • Margin with cost > 0: Benefits outweigh rejection cost\n")
        else:
            f.write("   • Margin with cost < 0: Rejection cost outweighs benefits\n")
            
        if accept_prob > 0.5:
            f.write("   • Accept prob > 0.5: High confidence, ACCEPT prediction\n")
        else:
            f.write("   • Accept prob ≤ 0.5: Low confidence, REJECT prediction\n")
        
        f.write("\n")
        
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
    
    print(f"💾 Saved text summary: {text_path}")

def create_overall_summary():
    """Create overall summary of all processed samples."""
    output_dir = Path('./inference_analysis_results')
    summary_path = output_dir / 'overall_analysis.txt'
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("AR-GSE INFERENCE ANALYSIS - OVERALL SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("EXPLANATION OF TERMS:\n")
        f.write("- True Label: Ground truth class\n")
        f.write("- Predicted: Model's prediction (mixture of experts)\n")
        f.write("- 'Pred Wrong': Prediction ≠ True Label (prediction error)\n")
        f.write("- 'Pred Correct': Prediction = True Label (correct prediction)\n")
        f.write("- Decision ACCEPT: Model confident, accepts prediction\n")
        f.write("- Decision REJECT: Model uncertain, rejects prediction\n\n")
        
        f.write("DECISION QUALITY ANALYSIS:\n")
        f.write("✓ GOOD DECISIONS:\n")
        f.write("  - Accept + Pred Correct: High confidence, correct prediction\n")
        f.write("  - Reject + Pred Wrong: Low confidence, avoided wrong prediction\n")
        f.write("✗ BAD DECISIONS:\n")
        f.write("  - Accept + Pred Wrong: High confidence, but wrong prediction\n")
        f.write("  - Reject + Pred Correct: Low confidence, missed correct prediction\n\n")
        
        f.write("FILES GENERATED:\n")
        f.write("- demo_sample_X.png: Visualization showing full distributions\n")
        f.write("- demo_sample_X_summary.txt: Detailed analysis for each sample\n")
        f.write("- overall_analysis.txt: This summary file\n\n")
        
        f.write("KEY INSIGHTS FROM VISUALIZATIONS:\n")
        f.write("1. Expert Posteriors: Shows how each expert (CE, LogitAdj, BalSoftmax) \n")
        f.write("   assigns probabilities across all 100 classes\n")
        f.write("2. Gating Weights: Shows which expert the model trusts most\n")
        f.write("3. Mixture Distribution: Final probabilities after expert combination\n")
        f.write("4. Decision Process: Margin calculation leading to accept/reject\n\n")
        
        f.write("SAMPLE ANALYSIS COMPLETED\n")
        f.write("Check individual sample files for detailed breakdowns.\n")
    
    print(f"💾 Saved overall summary: {summary_path}")

def create_sample_visualization(sample_idx, true_label, group_name, expert_posteriors, 
                               gating_weights, mixture_probs, raw_margin, margin_with_cost,
                               accept_prob, is_accepted, predicted_class, is_correct, threshold=None):
    """Create visualization for a single sample."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    # Create simple title with key information
    decision_text = "ACCEPT" if is_accepted else "REJECT"
    prediction_text = "Pred Correct" if is_correct else "Pred Wrong"
    
    # Add detailed rejection metrics to title
    title_text = (f'Sample {sample_idx + 1}: True={true_label} | Pred={predicted_class} | Decision={decision_text} | {prediction_text}\n'
                 f'Raw Margin={raw_margin:.3f} | Margin-Cost={margin_with_cost:.3f} | Accept Prob={accept_prob:.4f}')
    
    fig.suptitle(title_text, fontsize=12, fontweight='bold')
    
    # 1. Expert posteriors (ALL 100 classes)
    ax1 = axes[0, 0]
    expert_names = ['CE', 'LogitAdj', 'BalSoftmax']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    class_indices = np.arange(100)  # All 100 classes
    
    for e, (expert_name, color) in enumerate(zip(expert_names, colors)):
        probs = expert_posteriors[e]  # Full 100-class distribution
        
        # Plot with transparency to see overlapping distributions
        ax1.plot(class_indices, probs, label=expert_name, color=color, alpha=0.8, linewidth=2)
        
        # Highlight true class with a dot
        ax1.scatter(true_label, probs[true_label], color='red', s=100, zorder=5)
    
    # Add vertical line at true class
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
    
    # Add value labels
    for bar, weight in zip(bars, gating_weights):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{weight:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Mixture distribution (ALL 100 classes)
    ax3 = axes[1, 0] 
    
    # Plot mixture distribution as line plot
    ax3.plot(class_indices, mixture_probs, color='purple', linewidth=2, alpha=0.8, label='Mixture')
    
    # Highlight true class
    ax3.scatter(true_label, mixture_probs[true_label], color='red', s=120, 
               label=f'True Class {true_label}', zorder=5, marker='o')
    
    # Highlight predicted class
    if predicted_class != true_label:
        ax3.scatter(predicted_class, mixture_probs[predicted_class], color='black', s=120,
                   label=f'Predicted Class {predicted_class}', zorder=5, marker='s')
    
    # Add vertical lines
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
    
    # Margin bars
    # Calculate threshold if not provided
    if threshold is None:
        # For Head group: threshold = 1.120 * 1.0 - 0.2 = 0.920
        # For Tail group: threshold = 0.880 * 1.0 - 0.2 = 0.680
        if group_name == "Head":
            threshold = 0.920
        else:  # Tail
            threshold = 0.680
    
    margin_data = ['Threshold', 'Raw Margin', 'Margin - Cost']
    margin_values = [threshold, raw_margin, margin_with_cost]
    colors_margin = ['red', 'blue', 'green']
    
    bars = ax4.bar(margin_data, margin_values, color=colors_margin, alpha=0.7)
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    decision_result = "ACCEPT" if is_accepted else "REJECT"
    ax4.set_title(f'Decision Process: {decision_result}\nAccept Prob: {accept_prob:.3f} | Margin: {margin_with_cost:.3f}')
    ax4.set_ylabel('Margin Value')
    
    # Add value labels
    for bar, value in zip(bars, margin_values):
        ax4.text(bar.get_x() + bar.get_width()/2, 
                bar.get_height() + (0.01 if value >= 0 else -0.02),
                f'{value:.3f}', ha='center', 
                va='bottom' if value >= 0 else 'top', fontweight='bold')
    

    
    plt.tight_layout()
    
    # Save figure (don't show)
    output_dir = Path('./inference_analysis_results')
    output_dir.mkdir(exist_ok=True)
    save_path = output_dir / f'demo_sample_{sample_idx + 1}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved visualization: {save_path}")
    
    # Close the figure to free memory
    plt.close(fig)

def main():
    """Main demo function."""
    try:
        demo_inference_process()
        
        # Create overall summary
        create_overall_summary()
        
        print("\n🎉 Demo completed!")
        print("📁 Check './inference_analysis_results/' for:")
        print("   - Individual visualizations: demo_sample_X.png")
        print("   - Detailed text summaries: demo_sample_X_summary.txt") 
        print("   - Overall summary: overall_analysis.txt")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()