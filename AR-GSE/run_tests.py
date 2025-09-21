#!/usr/bin/env python3
"""
Simple test runner script for AR-GSE dataset tests.
Run with: python run_tests.py
"""

import os
import sys
import subprocess

def main():
    # Add src to Python path
    src_path = os.path.join(os.path.dirname(__file__), "src")
    sys.path.insert(0, os.path.abspath(src_path))
    
    # Check if pytest is available
    try:
        import pytest
    except ImportError:
        print("pytest not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pytest", "pytest-mock"])
        import pytest
    
    # Run tests
    print("Running dataset tests...")
    exit_code = pytest.main([
        "tests/data/test_dataset.py",
        "-v",
        "--tb=short"
    ])
    
    return exit_code

if __name__ == "__main__":
    sys.exit(main())