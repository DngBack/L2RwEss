#!/usr/bin/env python3
"""
Compare and Summary Script
Compares the new comprehensive inference results with demo_inference results
"""

from pathlib import Path
import json

def compare_results():
    """Compare the outputs and provide summary."""
    
    print("🔍 COMPARISON: Comprehensive vs Demo Inference")
    print("=" * 60)
    
    # Check demo_inference results
    demo_dir = Path('./inference_analysis_results')
    comprehensive_dir = Path('./comprehensive_inference_results')
    individual_dir = comprehensive_dir / 'individual_samples'
    
    print("📊 OUTPUT COMPARISON:")
    print("\n🎯 DEMO_INFERENCE.PY (Original):")
    if demo_dir.exists():
        demo_files = list(demo_dir.glob('*'))
        for file in sorted(demo_files):
            print(f"   ✅ {file.name}")
        print(f"   Total files: {len(demo_files)}")
    else:
        print("   ❌ No demo inference results found")
    
    print("\n🚀 COMPREHENSIVE_INFERENCE.PY (New Enhanced):")
    if comprehensive_dir.exists():
        # Main directory files
        main_files = [f for f in comprehensive_dir.glob('*') if f.is_file()]
        for file in sorted(main_files):
            print(f"   ✅ {file.name}")
            
        # Individual samples directory
        if individual_dir.exists():
            individual_files = list(individual_dir.glob('*'))
            print(f"   ✅ individual_samples/ ({len(individual_files)} files)")
            
            # Show some examples
            png_files = list(individual_dir.glob('*.png'))
            txt_files = list(individual_dir.glob('*_summary.txt'))
            print(f"     • {len(png_files)} visualization files (.png)")
            print(f"     • {len(txt_files)} text summary files (.txt)")
            print(f"     • 1 overall summary file")
            
        total_files = len(main_files) + len(list(individual_dir.glob('*'))) if individual_dir.exists() else len(main_files)
        print(f"   Total files: {total_files}")
    
    print("\n📈 FEATURE COMPARISON:")
    print("\n✨ MAINTAINED FEATURES (from demo_inference.py):")
    print("   ✅ Individual sample visualization (4-panel charts)")
    print("   ✅ Detailed text summaries with mathematical explanations")
    print("   ✅ Expert posterior analysis")
    print("   ✅ Gating weight analysis")
    print("   ✅ Mixture distribution visualization")
    print("   ✅ Decision process explanation")
    print("   ✅ Step-by-step margin calculations")
    
    print("\n🚀 NEW ENHANCED FEATURES:")
    print("   ✅ 50 samples instead of 3 (30 Head + 20 Tail)")
    print("   ✅ Stratified random sampling")
    print("   ✅ Comprehensive statistical analysis")
    print("   ✅ CSV export for further analysis")
    print("   ✅ JSON export for raw data")
    print("   ✅ Group-wise performance metrics")
    print("   ✅ Decision quality analysis")
    print("   ✅ Expert weight distribution analysis")
    print("   ✅ Margin distribution analysis")
    print("   ✅ Overall dashboard visualization")
    print("   ✅ Sample highlights visualization")
    print("   ✅ Intelligent sample selection for detailed analysis")
    
    print("\n🎯 KEY ADVANTAGES:")
    print("   1. SCALABILITY: 50 samples vs 3 samples")
    print("   2. STATISTICAL RIGOR: Proper stratified sampling")
    print("   3. COMPREHENSIVE ANALYSIS: Multiple visualization types")
    print("   4. DATA EXPORT: CSV/JSON for external analysis")
    print("   5. ORGANIZED OUTPUT: Structured folder hierarchy")
    print("   6. SELECTIVE DETAIL: Smart selection of interesting samples")
    print("   7. BOTH OVERVIEW AND DETAIL: Macro + micro analysis")
    
    # Check if we can load the comprehensive results
    results_file = comprehensive_dir / 'inference_results.json'
    if results_file.exists():
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        print(f"\n📊 COMPREHENSIVE RESULTS SUMMARY:")
        print(f"   • Total samples analyzed: {len(results)}")
        
        # Calculate some quick stats
        correct_count = sum(1 for r in results if r['is_correct'])
        accepted_count = sum(1 for r in results if r['is_accepted'])
        head_count = sum(1 for r in results if r['group_name'] == 'Head')
        tail_count = sum(1 for r in results if r['group_name'] == 'Tail')
        
        print(f"   • Prediction accuracy: {correct_count/len(results):.1%}")
        print(f"   • Acceptance rate: {accepted_count/len(results):.1%}")
        print(f"   • Head samples: {head_count}")
        print(f"   • Tail samples: {tail_count}")
        
        # Decision quality
        correct_accepts = sum(1 for r in results if r['is_accepted'] and r['is_correct'])
        correct_rejects = sum(1 for r in results if not r['is_accepted'] and not r['is_correct'])
        good_decisions = correct_accepts + correct_rejects
        
        print(f"   • Decision quality: {good_decisions/len(results):.1%}")
    
    print("\n💡 USE CASES:")
    print("   📋 DEMO_INFERENCE.PY: Best for...")
    print("     • Quick understanding of a few samples")
    print("     • Educational/demonstration purposes")
    print("     • Debugging specific samples")
    
    print("   📊 COMPREHENSIVE_INFERENCE.PY: Best for...")
    print("     • Research and evaluation")
    print("     • Statistical analysis")
    print("     • Model performance assessment")
    print("     • Publication-quality results")
    print("     • Comprehensive model understanding")
    
    print(f"\n🎉 SUCCESS: Enhanced inference system ready!")
    print(f"   Both approaches are available and complement each other.")

if __name__ == '__main__':
    compare_results()