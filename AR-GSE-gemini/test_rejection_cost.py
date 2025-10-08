"""
Quick test: Run rejection_cost mode only
"""
import subprocess
import sys

print("="*80)
print("🧪 Testing rejection_cost mode (Paper Methodology)")
print("="*80)

# First, set mode to rejection_cost
import re
from pathlib import Path

config_file = Path('./src/train/eval_gse_plugin_auth.py')

with open(config_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace eval_mode
pattern = r"'eval_mode': '[^']+'"
replacement = "'eval_mode': 'rejection_cost'"
modified_content = re.sub(pattern, replacement, content)

with open(config_file, 'w', encoding='utf-8') as f:
    f.write(modified_content)

print("✅ Set eval_mode = 'rejection_cost'")
print("\n🚀 Running evaluation...\n")

# Run
result = subprocess.run(
    [sys.executable, '-m', 'src.train.eval_gse_plugin_auth'],
    text=True
)

print("\n" + "="*80)
if result.returncode == 0:
    print("✅ Evaluation completed successfully!")
else:
    print("❌ Evaluation failed!")
print("="*80)
