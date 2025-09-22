#!/usr/bin/env python3
"""
Enhanced dataset runner with easy-to-use interface for AR-GSE dataset generation.
"""

import argparse
import os
import sys
from pathlib import Path

def print_banner():
    print("=" * 50)
    print("🔬 AR-GSE Dataset Runner")
    print("=" * 50)
    print()

def run_dataset_interactive():
    """Interactive mode for running dataset splits."""
    print_banner()
    print("Available options:")
    print("1. CIFAR-10 with default settings (imb_factor=100)")
    print("2. CIFAR-100 with default settings (imb_factor=100)")
    print("3. Custom configuration")
    print("4. Quick test (small imbalance)")
    print("5. Show command help")
    print()
    
    choice = input("Enter your choice (1-5): ").strip()
    
    if choice == "1":
        run_command(name="cifar10", imb_factor=100.0, seed=42)
    elif choice == "2":
        run_command(name="cifar100", imb_factor=100.0, seed=42)
    elif choice == "3":
        custom_config()
    elif choice == "4":
        quick_test()
    elif choice == "5":
        show_help()
    else:
        print("❌ Invalid choice. Please try again.")

def custom_config():
    """Get custom configuration from user."""
    print("\n🔧 Custom Configuration")
    print("-" * 30)
    
    # Dataset selection
    while True:
        name = input("Dataset (cifar10/cifar100): ").strip().lower()
        if name in ["cifar10", "cifar100"]:
            break
        print("❌ Please enter 'cifar10' or 'cifar100'")
    
    # Imbalance factor
    while True:
        try:
            imb_factor = float(input("Imbalance factor (1.0=balanced, 100.0=very imbalanced): "))
            if imb_factor >= 1.0:
                break
            print("❌ Imbalance factor must be >= 1.0")
        except ValueError:
            print("❌ Please enter a valid number")
    
    # Seed
    while True:
        try:
            seed = int(input("Random seed (e.g., 42): "))
            break
        except ValueError:
            print("❌ Please enter a valid integer")
    
    # Head ratio
    while True:
        try:
            head_ratio = float(input("Head ratio (0.1-0.9, default 0.5): ") or "0.5")
            if 0.1 <= head_ratio <= 0.9:
                break
            print("❌ Head ratio must be between 0.1 and 0.9")
        except ValueError:
            print("❌ Please enter a valid number")
    
    run_command(name=name, imb_factor=imb_factor, seed=seed, head_ratio=head_ratio)

def quick_test():
    """Run a quick test with small imbalance."""
    print("\n🚀 Quick Test Mode")
    print("-" * 20)
    dataset = input("Dataset (cifar10/cifar100, default cifar10): ").strip().lower() or "cifar10"
    if dataset not in ["cifar10", "cifar100"]:
        dataset = "cifar10"
    
    print(f"Running {dataset} with light imbalance (factor=10.0)...")
    run_command(name=dataset, imb_factor=10.0, seed=42)

def run_command(name, imb_factor, seed, head_ratio=0.5):
    """Execute the dataset splits command."""
    # Create output filename
    output_file = f"./splits/{name}_lt_if{imb_factor}_hr{head_ratio}_seed{seed}.json"
    
    cmd = [
        sys.executable, "-m", "src.data.splits",
        "--root", "./data",
        "--name", name,
        "--imb_factor", str(imb_factor),
        "--seed", str(seed),
        "--head_ratio", str(head_ratio),
        "--save", output_file
    ]
    
    print(f"\n🔄 Running command:")
    print(" ".join(cmd))
    print("\n" + "=" * 50)
    
    # Run the command
    import subprocess
    try:
        result = subprocess.run(cmd, check=True)
        print("\n" + "=" * 50)
        print(f"✅ Success! Dataset splits saved to: {output_file}")
        print(f"📁 Check the ./splits/ folder for your files")
        
        # Show file info
        if os.path.exists(output_file):
            size = os.path.getsize(output_file)
            print(f"📊 File size: {size:,} bytes")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running command: {e}")
    except KeyboardInterrupt:
        print(f"\n⚠️  Operation cancelled by user")

def show_help():
    """Show detailed help information."""
    print("\n📖 Dataset Splits Help")
    print("=" * 30)
    print("""
🎯 Purpose:
   Generate long-tailed dataset splits for CIFAR-10/100

📋 Parameters:
   --root        : Directory to store CIFAR data (default: ./data)
   --name        : Dataset name (cifar10 or cifar100)
   --imb_factor  : Imbalance factor - higher = more imbalanced
                   • 1.0 = perfectly balanced
                   • 10.0 = light imbalance  
                   • 100.0 = heavy imbalance
   --seed        : Random seed for reproducibility
   --head_ratio  : Ratio of classes in the 'head' group (0.1-0.9)
   --save        : Output JSON file path

📊 What gets created:
   • train: 80% of long-tailed samples
   • tuneV: 8% for hyperparameter tuning
   • val_small: 6% for validation
   • calib: 6% for calibration
   • test: Original test set (unchanged)

🔍 Group Analysis:
   Classes are divided into 2 groups by frequency:
   • Group 0 (Head): Most frequent classes
   • Group 1 (Tail): Least frequent classes

💾 Output:
   JSON file with indices for each split + statistics

📁 File Structure:
   ./data/           - CIFAR datasets (auto-downloaded)
   ./splits/         - Generated split files
   
🚀 Quick Examples:
   python -m src.data.splits --name cifar10 --imb_factor 50.0
   python -m src.data.splits --name cifar100 --imb_factor 100.0 --seed 123
    """)

def list_existing_splits():
    """List existing split files."""
    splits_dir = Path("./splits")
    if splits_dir.exists():
        files = list(splits_dir.glob("*.json"))
        if files:
            print(f"\n📁 Existing split files ({len(files)}):")
            for f in sorted(files):
                size = f.stat().st_size
                print(f"   {f.name} ({size:,} bytes)")
        else:
            print("\n📁 No existing split files found")
    else:
        print("\n📁 Splits directory doesn't exist yet")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Enhanced AR-GSE Dataset Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--interactive", "-i", 
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true", 
        help="List existing split files"
    )
    
    # Add all the original parameters for direct usage
    parser.add_argument("--name", choices=["cifar10", "cifar100"], help="Dataset name")
    parser.add_argument("--imb_factor", type=float, help="Imbalance factor")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--head_ratio", type=float, default=0.5, help="Head ratio")
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs("./data", exist_ok=True)
    os.makedirs("./splits", exist_ok=True)
    
    if args.list:
        list_existing_splits()
        return
    
    if args.interactive or (not args.name and not args.imb_factor):
        run_dataset_interactive()
    else:
        # Direct mode
        if not all([args.name, args.imb_factor, args.seed]):
            print("❌ For direct mode, please provide --name, --imb_factor, and --seed")
            print("   Or use --interactive for guided setup")
            return
        run_command(args.name, args.imb_factor, args.seed, args.head_ratio)

if __name__ == "__main__":
    main()