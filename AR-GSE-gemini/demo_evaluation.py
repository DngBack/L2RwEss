#!/usr/bin/env python3
"""
Quick demo script to test evaluation framework.

This script provides a quick way to:
1. Check if all dependencies are installed
2. Verify data/checkpoint files exist
3. Run a fast evaluation (fewer points)
4. Display key metrics

Usage:
    python demo_evaluation.py
"""

import sys
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed."""
    print("\n🔍 Checking dependencies...")
    
    required = [
        'torch',
        'torchvision', 
        'numpy',
        'pandas',
        'matplotlib',
        'tqdm'
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    
    print("✅ All dependencies installed!")
    return True

def check_files():
    """Check if required files exist."""
    print("\n🔍 Checking required files...")
    
    files_to_check = [
        './data/cifar100_lt_if100_splits/test_lt_indices.json',
        './outputs/logits/cifar100_lt_if100/ce_baseline/test_lt_logits.pt',
        './outputs/logits/cifar100_lt_if100/logitadjust_baseline/test_lt_logits.pt',
        './outputs/logits/cifar100_lt_if100/balsoftmax_baseline/test_lt_logits.pt',
    ]
    
    all_exist = True
    for file_path in files_to_check:
        path = Path(file_path)
        if path.exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path}")
            all_exist = False
    
    if not all_exist:
        print("\n⚠️  Some required files are missing!")
        print("Make sure you have:")
        print("  1. Generated train/test splits")
        print("  2. Trained expert models and saved logits")
        return False
    
    print("✅ All required files found!")
    return True

def find_checkpoint():
    """Find available checkpoints."""
    print("\n🔍 Looking for checkpoints...")
    
    checkpoint_dirs = [
        './checkpoints/argse_worst_eg_improved/cifar100_lt_if100/',
        './checkpoints/argse_balanced/cifar100_lt_if100/',
        './checkpoints/argse_constrained_plugin/cifar100_lt_if100/',
    ]
    
    found_checkpoints = []
    for ckpt_dir in checkpoint_dirs:
        path = Path(ckpt_dir)
        if path.exists():
            for ckpt_file in path.glob('*.ckpt'):
                found_checkpoints.append(str(ckpt_file))
                print(f"  ✅ {ckpt_file}")
    
    if not found_checkpoints:
        print("  ❌ No checkpoints found!")
        print("\n⚠️  Train a model first using:")
        print("    python run_constrained_plugin.py")
        return None
    
    print(f"\n✅ Found {len(found_checkpoints)} checkpoint(s)")
    return found_checkpoints[0]  # Return first one

def run_fast_evaluation(checkpoint_path):
    """Run a fast evaluation with reduced number of points."""
    print("\n🚀 Running fast evaluation...")
    print(f"Using checkpoint: {checkpoint_path}\n")
    
    try:
        from src.train.eval_paper_benchmark import PaperBenchmarkEvaluator
        import torch
        import numpy as np
        
        # Configuration for fast evaluation
        config = {
            'dataset': {
                'name': 'cifar100_lt_if100',
                'splits_dir': './data/cifar100_lt_if100_splits',
                'num_classes': 100,
            },
            'experts': {
                'names': ['ce_baseline', 'logitadjust_baseline', 'balsoftmax_baseline'],
                'logits_dir': './outputs/logits',
            },
            'checkpoint_path': checkpoint_path,
            'output_dir': './demo_evaluation_results',
            'seed': 42
        }
        
        # Set seeds
        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])
        
        # Create evaluator
        evaluator = PaperBenchmarkEvaluator(config)
        
        # Quick evaluation (fewer points for speed)
        print("🔄 Computing confidence scores...")
        confidence_scores = evaluator.compute_confidence_scores('gse_margin')
        
        print("🔄 Generating RC curve (quick mode: 21 points)...")
        rc_df = evaluator.generate_rc_curve(confidence_scores, 0.2, 1.0, num_points=21)
        
        print("🔄 Computing metrics...")
        
        # AURC metrics
        aurc_std = evaluator.compute_aurc(rc_df, 'standard_error', 0.2, 1.0)
        aurc_bal = evaluator.compute_aurc(rc_df, 'balanced_error', 0.2, 1.0)
        aurc_wst = evaluator.compute_aurc(rc_df, 'worst_error', 0.2, 1.0)
        
        # E-AURC
        eaurc_std, _, _ = evaluator.compute_eaurc(rc_df, 'standard_error', 0.2, 1.0)
        eaurc_bal, _, _ = evaluator.compute_eaurc(rc_df, 'balanced_error', 0.2, 1.0)
        eaurc_wst, _, _ = evaluator.compute_eaurc(rc_df, 'worst_error', 0.2, 1.0)
        
        # Specific coverage
        coverage_metrics = evaluator.evaluate_at_specific_coverages(confidence_scores)
        
        # Print results
        print("\n" + "="*80)
        print("QUICK EVALUATION RESULTS")
        print("="*80)
        
        print("\n📊 AURC Metrics (Coverage 0.2-1.0):")
        print(f"  Standard Error:   AURC = {aurc_std:.6f}, E-AURC = {eaurc_std:.6f}")
        print(f"  Balanced Error:   AURC = {aurc_bal:.6f}, E-AURC = {eaurc_bal:.6f}")
        print(f"  Worst-Group Error: AURC = {aurc_wst:.6f}, E-AURC = {eaurc_wst:.6f}")
        
        print("\n📊 Performance at Key Coverages:")
        for cov_key in ['cov_0.70', 'cov_0.80', 'cov_0.90']:
            if cov_key in coverage_metrics:
                metrics = coverage_metrics[cov_key]
                cov = metrics['actual_coverage']
                std_acc = metrics['standard_accuracy']
                bal_acc = metrics['balanced_accuracy']
                wst_acc = metrics['worst_accuracy']
                
                print(f"\n  Coverage ≈ {cov:.1%}:")
                print(f"    Standard Accuracy:    {std_acc:.4f}")
                print(f"    Balanced Accuracy:    {bal_acc:.4f}")
                print(f"    Worst-Group Accuracy: {wst_acc:.4f}")
        
        print("\n" + "="*80)
        print("✅ Quick evaluation complete!")
        print(f"\nFor full evaluation with visualizations, run:")
        print(f"  python run_benchmark_evaluation.py --checkpoint {checkpoint_path}")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main demo function."""
    print("="*80)
    print("EVALUATION FRAMEWORK DEMO")
    print("="*80)
    
    # Step 1: Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Step 2: Check files
    if not check_files():
        sys.exit(1)
    
    # Step 3: Find checkpoint
    checkpoint = find_checkpoint()
    if checkpoint is None:
        sys.exit(1)
    
    # Step 4: Run evaluation
    success = run_fast_evaluation(checkpoint)
    
    if success:
        print("\n✨ Demo completed successfully!")
        print("\nNext steps:")
        print("  1. Run full evaluation: python run_benchmark_evaluation.py")
        print("  2. Read guide: EVALUATION_GUIDE.md")
        print("  3. Compare methods: use eval_comprehensive.py")
    else:
        print("\n❌ Demo failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == '__main__':
    main()
