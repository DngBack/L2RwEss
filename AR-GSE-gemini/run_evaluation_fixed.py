#!/usr/bin/env python3
"""
Fixed wrapper script to run AR-GSE evaluation with proper path setup.
"""

import sys
import os
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

# Change to project root directory
os.chdir(project_root)

# Now import and run the evaluation
try:
    from train.eval_test import main
    
    print("🔍 Starting AR-GSE Evaluation...")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print(f"Using logits from: {project_root}/outputs/logits/")
    print("=" * 60)
    
    main()
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure all dependencies are installed:")
    print("pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"❌ Evaluation Error: {e}")
    sys.exit(1)
