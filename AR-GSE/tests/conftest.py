"""
Pytest configuration and fixtures for AR-GSE tests.
"""
import sys
import os

# Add the src directory to the Python path so we can import our modules
src_path = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(src_path))