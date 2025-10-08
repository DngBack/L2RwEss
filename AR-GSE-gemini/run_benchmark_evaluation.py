# run_benchmark_evaluation.py
"""
Quick script to run benchmark evaluation on trained AR-GSE model.

Usage:
    python run_benchmark_evaluation.py --checkpoint <path> --output <dir>
"""

import argparse
from pathlib import Path
from src.train.eval_paper_benchmark import PaperBenchmarkEvaluator

def main():
    parser = argparse.ArgumentParser(description='Run benchmark evaluation')
    parser.add_argument('--checkpoint', type=str, 
                       default='./checkpoints/argse_worst_eg_improved/cifar100_lt_if100/gse_balanced_plugin.ckpt',
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str,
                       default='./paper_benchmark_results',
                       help='Output directory')
    parser.add_argument('--dataset', type=str,
                       default='cifar100_lt_if100',
                       help='Dataset name')
    parser.add_argument('--experts', nargs='+',
                       default=['ce_baseline', 'logitadjust_baseline', 'balsoftmax_baseline'],
                       help='Expert names')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Verify checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        print("\nAvailable checkpoints:")
        checkpoints_dir = Path('./checkpoints')
        if checkpoints_dir.exists():
            for ckpt in checkpoints_dir.rglob('*.ckpt'):
                print(f"  - {ckpt}")
        return
    
    # Configuration
    config = {
        'dataset': {
            'name': args.dataset,
            'splits_dir': f'./data/{args.dataset}_splits',
            'num_classes': 100,
        },
        'experts': {
            'names': args.experts,
            'logits_dir': './outputs/logits',
        },
        'checkpoint_path': args.checkpoint,
        'output_dir': args.output,
        'seed': args.seed
    }
    
    print("\n" + "="*80)
    print("BENCHMARK EVALUATION")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset:    {args.dataset}")
    print(f"Experts:    {', '.join(args.experts)}")
    print(f"Output:     {args.output}")
    print("="*80)
    
    # Run evaluation
    evaluator = PaperBenchmarkEvaluator(config)
    results = evaluator.run_paper_benchmark()
    
    print(f"\n✅ Evaluation complete! Results saved to: {args.output}")
    print(f"\n📊 Key files generated:")
    print(f"  - paper_benchmark_results.json  (detailed metrics)")
    print(f"  - paper_benchmark_figures.png   (visualizations)")
    print(f"  - paper_benchmark_figures.pdf   (publication quality)")
    print(f"  - latex_table.tex               (for paper submission)")
    print(f"  - rc_curve_paper_benchmark.csv  (raw RC curve data)")


if __name__ == '__main__':
    main()
