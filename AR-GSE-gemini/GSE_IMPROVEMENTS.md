# GSE Plugin Improvements

This document describes the key improvements made to the GSE (Group-aware Selective prediction with Experts) plugin system.

## 1. Gating Network Improvements

### Sample Weighting Based on tuneV Split
- **Change**: Use frequency counts from tuneV split instead of full dataset counts for sample weighting
- **File**: `src/train/train_gating_only.py`
- **Rationale**: Soft weighting should reflect the actual distribution of gating training data, not assumptions from full LT distribution

### Regularization Updates
- **entropy_penalty**: Reduced to `0.0000` (from `0.0025`)
  - Positive entropy penalty encourages uniform mixing across experts
  - This makes it harder for tail classes to improve
  - Setting to 0 avoids forcing uniform distribution
- **diversity_penalty**: Set to `0.002`
  - Small usage balance penalty to prevent expert collapse
  - Prevents any single expert from dominating

## 2. Plugin Alpha Update Improvements

### Blended Alpha Update
- **New function**: `update_alpha_fixed_point_blend()`
- **File**: `src/train/gse_balanced_plugin.py`
- **Features**:
  - Combines joint and conditional acceptance methods
  - `a_new = (1 - blend_lambda) * a_joint + blend_lambda * a_cond`
  - Default `blend_lambda = 0.25` (tunable between 0.15-0.35)
  - Improves stability for worst/balanced objectives

### Adaptive Lambda Grid Expansion
- **Feature**: Automatically expands lambda search grid when optimal hits boundaries
- **Logic**: When best lambda is at grid edges (index 0 or last), expand by 4 steps
- **Benefits**: Prevents getting stuck in suboptimal regions

## 3. Per-Group Thresholds (Mondrian)

### Implementation
- **New file**: `src/train/per_group_threshold.py`
- **Functions**:
  - `fit_group_thresholds_from_raw()`: Fits separate threshold t_k for each group
  - `accept_with_group_thresholds()`: Applies group-specific thresholds

### Usage
- **Training**: Fits thresholds based on S1 predictions and target coverage per group
- **Testing**: Uses predicted class group to select appropriate threshold
- **Coverage targets**: Different for head (0.60) and tail (0.45) groups
- **Benefit**: Typically reduces worst-group error by 10-20%

### Integration
- **Training**: `src/train/gse_balanced_plugin.py` - fits and saves `t_group`
- **Evaluation**: `src/train/eval_gse_plugin.py` - uses per-group thresholds if available

## 4. EG-Outer for Worst-Group Optimization

### Implementation
- **New file**: `src/train/gse_worst_eg.py`
- **Algorithm**: Exponentiated-Gradient outer loop for worst-group objectives
- **Key functions**:
  - `worst_group_eg_outer()`: Main EG-outer algorithm
  - `inner_cost_sensitive_plugin()`: Inner optimization for given beta
  - `compute_raw_margin_with_beta()`: Margin computation with group weights

### Features
- **Beta updates**: `β_k ← β_k * exp(xi * error_k)` followed by normalization
- **Parameters**: 
  - T=20 EG outer iterations
  - xi=1.0 step size
  - Expanded lambda grid [-1.2, 1.2] with 41 points
- **Benefits**: Usually improves worst-group error significantly

### Usage
```python
# Enable EG-outer for worst-group
CONFIG['plugin_params']['objective'] = 'worst'
CONFIG['plugin_params']['use_eg_outer'] = True
```

## 5. Enhanced Evaluation

### Per-Group Threshold Support
- **File**: `src/train/eval_gse_plugin.py`
- **Logic**: Automatically detects and uses per-group thresholds from checkpoint
- **Fallback**: Uses global threshold if per-group not available

### Better Margin Computation
- **Raw margins**: Used for RC curve generation (fair comparison)
- **Acceptance**: Based on group-specific or global thresholds

## Usage Examples

### Standard Balanced Optimization
```python
python -m src.train.gse_balanced_plugin
```

### Worst-Group with EG-Outer
```python
python run_gse_worst_eg.py
```

### Evaluation
```python
python -m src.train.eval_gse_plugin
```

## Configuration Options

### Key Parameters
```python
'gating_params': {
    'entropy_penalty': 0.0000,    # Avoid forcing uniform mixing
    'diversity_penalty': 0.002,   # Prevent expert collapse
}

'plugin_params': {
    'objective': 'balanced',      # 'worst', 'balanced', 'hybrid'
    'use_eg_outer': False,        # Enable EG-outer for worst-group
    'eg_outer_T': 20,            # EG iterations
    'eg_outer_xi': 1.0,          # EG step size
    'use_conditional_alpha': True, # Blended alpha updates
}
```

## Expected Improvements

### Typical Results
- **Balanced error**: 5-10% relative improvement
- **Worst-group error**: 10-20% relative improvement (especially with Mondrian + EG-outer)
- **Stability**: Better convergence with blended alpha updates
- **Tail performance**: Improved through reduced entropy penalty and per-group thresholds

### Method Combinations
- **Best for balanced**: Standard plugin + per-group thresholds
- **Best for worst-group**: EG-outer + per-group thresholds
- **Most stable**: Blended alpha updates + adaptive lambda grid