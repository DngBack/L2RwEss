#!/usr/bin/env python3
"""
Summary and Analysis of Comprehensive Inference Results
"""

import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def load_and_summarize_results():
    """Load and provide a clear summary of inference results."""
    
    results_dir = Path('./comprehensive_inference_results')
    
    # Load CSV data
    df = pd.read_csv(results_dir / 'inference_results.csv')
    
    print("🚀 AR-GSE COMPREHENSIVE INFERENCE SUMMARY")
    print("=" * 60)
    
    # Basic info
    total_samples = len(df)
    head_samples = len(df[df['group_name'] == 'Head'])
    tail_samples = len(df[df['group_name'] == 'Tail'])
    
    print(f"📊 DATASET OVERVIEW:")
    print(f"   • Total samples analyzed: {total_samples}")
    print(f"   • Head group samples: {head_samples}")
    print(f"   • Tail group samples: {tail_samples}")
    
    # Prediction performance
    correct_predictions = df['is_correct'].sum()
    overall_accuracy = correct_predictions / total_samples
    head_accuracy = df[df['group_name'] == 'Head']['is_correct'].mean()
    tail_accuracy = df[df['group_name'] == 'Tail']['is_correct'].mean()
    
    print(f"\n🎯 PREDICTION PERFORMANCE:")
    print(f"   • Overall accuracy: {overall_accuracy:.1%} ({correct_predictions}/{total_samples})")
    print(f"   • Head group accuracy: {head_accuracy:.1%} ({int(head_accuracy * head_samples)}/{head_samples})")
    print(f"   • Tail group accuracy: {tail_accuracy:.1%} ({int(tail_accuracy * tail_samples)}/{tail_samples})")
    
    # Decision analysis - the KEY INSIGHT
    accepted_samples = df['is_accepted'].sum()
    rejected_samples = total_samples - accepted_samples
    
    print(f"\n⚖️  SELECTIVE PREDICTION DECISIONS:")
    print(f"   • Samples ACCEPTED: {accepted_samples} ({accepted_samples/total_samples:.1%})")
    print(f"   • Samples REJECTED: {rejected_samples} ({rejected_samples/total_samples:.1%})")
    
    # This is the critical analysis - what happened to each category
    correct_accepts = len(df[df['is_accepted'] & df['is_correct']])
    incorrect_accepts = len(df[df['is_accepted'] & ~df['is_correct']])
    correct_rejects = len(df[~df['is_accepted'] & ~df['is_correct']])
    incorrect_rejects = len(df[~df['is_accepted'] & df['is_correct']])
    
    print(f"\n🎭 DECISION QUALITY BREAKDOWN:")
    print(f"   • ✅ Correct Accepts (Good): {correct_accepts} - High confidence + Right prediction")
    print(f"   • ❌ Incorrect Accepts (Bad): {incorrect_accepts} - High confidence + Wrong prediction")
    print(f"   • ✅ Correct Rejects (Good): {correct_rejects} - Low confidence + Wrong prediction")
    print(f"   • ❌ Incorrect Rejects (Bad): {incorrect_rejects} - Low confidence + Right prediction")
    
    good_decisions = correct_accepts + correct_rejects
    bad_decisions = incorrect_accepts + incorrect_rejects
    decision_quality = good_decisions / total_samples
    
    print(f"\n   📈 Decision Quality Score: {decision_quality:.1%} ({good_decisions}/{total_samples})")
    
    # Key observation
    print(f"\n🔍 KEY OBSERVATIONS:")
    if accepted_samples == 0:
        print(f"   • ⚠️  MODEL REJECTED ALL SAMPLES!")
        print(f"     - This means the model had very low confidence in all predictions")
        print(f"     - Raw margins were all negative (below acceptance threshold)")
        print(f"     - This is conservative behavior - avoids wrong predictions but misses correct ones")
    
    # Expert analysis
    print(f"\n🤖 EXPERT SYSTEM ANALYSIS:")
    ce_weight = df['ce_weight'].mean()
    logitadj_weight = df['logitadj_weight'].mean()
    balsoftmax_weight = df['balsoftmax_weight'].mean()
    
    print(f"   • Average Expert Weights:")
    print(f"     - CE (Cross-Entropy): {ce_weight:.1%}")
    print(f"     - LogitAdjust: {logitadj_weight:.1%}")
    print(f"     - BalancedSoftmax: {balsoftmax_weight:.1%}")
    
    dominant_expert = ['CE', 'LogitAdjust', 'BalancedSoftmax'][np.argmax([ce_weight, logitadj_weight, balsoftmax_weight])]
    print(f"   • Dominant Expert: {dominant_expert}")
    
    # Margin analysis
    print(f"\n📏 MARGIN ANALYSIS (Why all rejected?):")
    avg_raw_margin = df['raw_margin'].mean()
    avg_margin_cost = df['margin_with_cost'].mean()
    avg_accept_prob = df['accept_prob'].mean()
    
    print(f"   • Average Raw Margin: {avg_raw_margin:.3f}")
    print(f"   • Average Margin with Cost: {avg_margin_cost:.3f}")
    print(f"   • Average Accept Probability: {avg_accept_prob:.3f}")
    print(f"   • Rejection Cost: 0.200")
    
    if avg_margin_cost < 0:
        print(f"   • 💡 All margins with cost < 0 → All samples rejected")
        print(f"     - Model is very conservative")
        print(f"     - May need to adjust rejection cost or retrain")
    
    # Group-specific analysis
    print(f"\n📊 GROUP-SPECIFIC ANALYSIS:")
    
    head_df = df[df['group_name'] == 'Head']
    tail_df = df[df['group_name'] == 'Tail']
    
    print(f"   HEAD GROUP (Frequent Classes):")
    print(f"   • Accuracy: {head_df['is_correct'].mean():.1%}")
    print(f"   • Avg Raw Margin: {head_df['raw_margin'].mean():.3f}")
    print(f"   • Avg Margin w/ Cost: {head_df['margin_with_cost'].mean():.3f}")
    print(f"   • Accept Rate: {head_df['is_accepted'].mean():.1%}")
    
    print(f"\n   TAIL GROUP (Rare Classes):")
    print(f"   • Accuracy: {tail_df['is_correct'].mean():.1%}")
    print(f"   • Avg Raw Margin: {tail_df['raw_margin'].mean():.3f}")
    print(f"   • Avg Margin w/ Cost: {tail_df['margin_with_cost'].mean():.3f}")
    print(f"   • Accept Rate: {tail_df['is_accepted'].mean():.1%}")
    
    # Performance insight
    print(f"\n💡 PERFORMANCE INSIGHTS:")
    if head_accuracy > tail_accuracy:
        print(f"   • Model performs much better on Head (frequent) classes")
        print(f"   • This shows the imbalanced learning problem")
    
    if avg_margin_cost < -0.5:
        print(f"   • Very negative margins suggest model lacks confidence")
        print(f"   • May need hyperparameter tuning or retraining")
    
    # Recommendations
    print(f"\n🎯 RECOMMENDATIONS:")
    print(f"   1. If you want more accepts, consider:")
    print(f"      - Reducing rejection cost (currently 0.2)")
    print(f"      - Retraining with different parameters")
    print(f"      - Adjusting temperature parameter (currently 25.0)")
    print(f"   2. Current model is very conservative - good for safety-critical applications")
    print(f"   3. Tail class performance needs improvement - consider:")
    print(f"      - More data augmentation for rare classes")
    print(f"      - Different loss functions")
    print(f"      - Better expert training")
    
    return df

def create_simple_summary_visualization(df):
    """Create a simple summary chart."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('AR-GSE Inference Summary (50 Samples)', fontsize=16, fontweight='bold')
    
    # 1. Accuracy comparison
    ax1 = axes[0, 0]
    groups = ['Head\n(30 samples)', 'Tail\n(20 samples)', 'Overall\n(50 samples)']
    accuracies = [
        df[df['group_name'] == 'Head']['is_correct'].mean(),
        df[df['group_name'] == 'Tail']['is_correct'].mean(),
        df['is_correct'].mean()
    ]
    colors = ['lightblue', 'lightcoral', 'lightgreen']
    bars = ax1.bar(groups, accuracies, color=colors, alpha=0.8)
    ax1.set_title('Prediction Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Decision quality pie chart
    ax2 = axes[0, 1]
    correct_accepts = len(df[df['is_accepted'] & df['is_correct']])
    incorrect_accepts = len(df[df['is_accepted'] & ~df['is_correct']])
    correct_rejects = len(df[~df['is_accepted'] & ~df['is_correct']])
    incorrect_rejects = len(df[~df['is_accepted'] & df['is_correct']])
    
    sizes = [correct_accepts, incorrect_accepts, correct_rejects, incorrect_rejects]
    labels = [f'Correct Accept\n({correct_accepts})', f'Incorrect Accept\n({incorrect_accepts})', 
              f'Correct Reject\n({correct_rejects})', f'Incorrect Reject\n({incorrect_rejects})']
    colors_pie = ['lightgreen', 'red', 'yellow', 'orange']
    
    # Only show non-zero slices
    non_zero_sizes = [(size, label, color) for size, label, color in zip(sizes, labels, colors_pie) if size > 0]
    if non_zero_sizes:
        sizes_nz, labels_nz, colors_nz = zip(*non_zero_sizes)
        ax2.pie(sizes_nz, labels=labels_nz, colors=colors_nz, autopct='%1.0f%%', startangle=90)
    ax2.set_title('Decision Quality Breakdown')
    
    # 3. Expert weights
    ax3 = axes[1, 0]
    experts = ['CE', 'LogitAdj', 'BalSoftmax']
    weights = [df['ce_weight'].mean(), df['logitadj_weight'].mean(), df['balsoftmax_weight'].mean()]
    bars = ax3.bar(experts, weights, color=['lightblue', 'orange', 'lightgreen'], alpha=0.8)
    ax3.set_title('Average Expert Weights')
    ax3.set_ylabel('Weight')
    ax3.set_ylim(0, 1)
    
    for bar, weight in zip(bars, weights):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{weight:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Margin distribution
    ax4 = axes[1, 1]
    ax4.hist(df['margin_with_cost'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Accept Threshold')
    ax4.set_title('Margin Distribution (with Cost)')
    ax4.set_xlabel('Margin Value')
    ax4.set_ylabel('Number of Samples')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add text showing why all rejected
    rejected_count = len(df[df['margin_with_cost'] < 0])
    ax4.text(0.05, 0.95, f'{rejected_count}/50 samples\nhave margin < 0\n(All Rejected)', 
             transform=ax4.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save
    output_path = Path('./comprehensive_inference_results/summary_visualization.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved summary visualization: {output_path}")
    plt.close(fig)

def main():
    """Main function."""
    df = load_and_summarize_results()
    create_simple_summary_visualization(df)
    
    print(f"\n🎉 Summary analysis completed!")
    print(f"📁 Check './comprehensive_inference_results/' for all files")

if __name__ == '__main__':
    main()