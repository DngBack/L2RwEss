#!/usr/bin/env python3
"""
Run GSE Constrained Plugin optimization.
This implements constrained optimization with coverage and fairness constraints.
"""

import sys
sys.path.append('.')

from src.train.gse_constrained_plugin import main, CONFIG

if __name__ == '__main__':
    print("🚀 GSE Constrained Plugin Optimization")
    print("=" * 60)
    print("Features:")
    print("✅ Constrained optimization with Lagrangian formulation")
    print("✅ Coverage constraint: minimum average coverage ≥ τ")
    print("✅ Fairness constraint: per-group error ≤ δ")
    print("✅ Balanced risk minimization objective")
    print("✅ Dual ascent for constraint handling")
    print("✅ Adaptive constraint parameters")
    print("=" * 60)
    
    # Configuration summary
    print("\nConfiguration:")
    print(f"  Coverage constraint τ: {CONFIG['constrained_params']['tau']:.2f}")
    print(f"  Fairness multiplier: {CONFIG['constrained_params']['delta_multiplier']:.1f}× avg error")
    print(f"  Outer iterations: {CONFIG['constrained_params']['T']}")
    print(f"  Dual step size: {CONFIG['constrained_params']['eta_dual']:.3f}")
    print(f"  Warmup iterations: {CONFIG['constrained_params']['warmup_iters']}")
    print(f"  Output: {CONFIG['output']['checkpoints_dir']}")
    
    print("\nStarting constrained optimization...")
    main()
    
    print("\n" + "=" * 60)
    print("🎉 Constrained optimization complete!")
    print("Results saved with optimization history and plots.")
    print("Key benefits:")
    print("  • Principled trade-off between coverage and error")
    print("  • Guaranteed minimum coverage via constraints")
    print("  • Fairness between groups via per-group error bounds")
    print("  • Interpretable dual variables (constraint violations)")
    print("=" * 60)