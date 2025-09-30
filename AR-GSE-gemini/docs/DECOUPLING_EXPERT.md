# Decoupling Expert for AR-GSE

## Overview

This document explains the implementation of the **Decoupling Expert** based on the paper ["Decoupling Representation and Classifier for Long-Tailed Recognition"](https://arxiv.org/abs/1910.09217) by Kang et al. (ICLR 2020).

## Key Concept

### Problem with Standard Training
When training with Cross-Entropy loss on imbalanced data:
- **Backbone (feature extractor)**: Learns good representations due to abundant head class data
- **Classifier (final linear layer)**: Becomes biased toward head classes due to softmax maximizing likelihood

### Solution: Two-Stage Decoupling

**Stage 1: Representation Learning**
- Train the entire model (backbone + classifier) with standard CE loss on imbalanced data
- Goal: Learn strong feature representations using all available data
- Result: Good backbone, but biased classifier

**Stage 2: Classifier Re-training**  
- **Freeze** the backbone to preserve learned representations
- **Reset & re-train** only the classifier using balanced sampling
- Goal: Learn fair decision boundaries without head/tail bias

## Mathematical Foundation

### Stage 1: Standard CE Loss
```
L_CE = -1/N ∑(i=1 to N) log(exp(w_yi^T f(xi)) / ∑_c exp(w_c^T f(xi)))
```

Where:
- `f(x)`: features from backbone
- `w_c`: classifier weights for class c
- Training uses original imbalanced distribution

### Stage 2: Balanced CE Loss
Same CE formula, but with balanced sampling:
- Each minibatch has approximately equal samples from all classes
- Effectively changes the prior π(y) from imbalanced to uniform
- Classifier learns p(y|x) ∝ p(x|y) without frequency bias

## Implementation Details

### Configuration
```python
'decoupling': {
    'name': 'decoupling_twostage',
    'loss_type': 'decoupling',
    # Stage 1: Representation Learning (imbalanced)
    'stage1_epochs': 180,
    'stage1_lr': 0.1,
    'stage1_weight_decay': 1e-4,
    'stage1_milestones': [80, 140, 160],
    'stage1_gamma': 0.1,
    # Stage 2: Classifier Re-training (balanced)
    'stage2_epochs': 76,
    'stage2_lr': 0.01,     # Lower LR for fine-tuning
    'stage2_weight_decay': 1e-4,
    'stage2_milestones': [40, 60],
    'stage2_gamma': 0.1,
    'dropout_rate': 0.1,
    'balanced_sampling': True
}
```

### Key Implementation Features

1. **Balanced Sampling**: Uses `WeightedRandomSampler` to ensure each class has equal probability of being selected
2. **Parameter Freezing**: Backbone parameters are frozen during Stage 2
3. **Classifier Reset**: Optional reset of classifier weights between stages
4. **Progressive Learning Rates**: Different LR schedules for each stage

### Balanced Dataloader
```python
def get_balanced_dataloader(original_train_loader, batch_size=None):
    # Calculate inverse frequency weights
    sample_weights = [1.0 / class_counts[target] for target in targets]
    
    # Create weighted sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=total_samples,
        replacement=True
    )
    
    return DataLoader(dataset, sampler=sampler, ...)
```

## Training Process

### Stage 1: Representation Learning (180 epochs)
1. Train entire model with imbalanced data
2. Use standard CE loss
3. Higher learning rate (0.1) for full model training
4. Save best model based on validation accuracy

### Stage 2: Classifier Re-training (76 epochs)  
1. Load best Stage 1 model
2. Freeze all backbone parameters
3. Reset classifier weights (optional)
4. Create balanced dataloader with WeightedRandomSampler
5. Train only classifier with lower LR (0.01)
6. Save final model

### Post-processing
1. Unfreeze all parameters
2. Apply temperature calibration
3. Export logits for all dataset splits

## Expected Benefits

1. **Better Tail Performance**: Balanced classifier improves tail class accuracy
2. **Preserved Head Performance**: Strong backbone maintains head class performance  
3. **Improved Overall Balance**: Better trade-off between head and tail accuracy
4. **Ensemble Diversity**: Different training process creates diverse expert for AR-GSE

## Comparison with Other Experts

| Expert | Training Method | Key Characteristic |
|--------|----------------|-------------------|
| CE | Single-stage CE | Baseline, head-biased |
| LogitAdjust | Single-stage with frequency adjustment | Compensates for frequency bias |
| BalancedSoftmax | Single-stage with balanced loss | Reweights softmax |
| **Decoupling** | **Two-stage: representation + classifier** | **Separates feature learning from classification** |

## Usage

### Training Single Decoupling Expert
```python
# Train only decoupling expert
python src/train/train_expert.py --expert decoupling

# Or using the training script
from src.train.train_expert import train_single_expert
model_path = train_single_expert('decoupling')
```

### Testing Implementation
```python
# Quick test with reduced epochs
python test_decoupling_expert.py
```

### Integration with AR-GSE
The decoupling expert integrates seamlessly with existing AR-GSE pipeline:
1. Trained experts are saved in `checkpoints/experts/`
2. Logits are exported to `outputs/logits/` for gating network training
3. Temperature calibration ensures proper probability calibration

## References

- Kang, B., Xie, S., Rohrbach, M., Yan, Z., Gordo, A., Feng, J., & Kalantidis, Y. (2019). Decoupling representation and classifier for long-tailed recognition. arXiv preprint arXiv:1910.09217.
- Original implementation ideas adapted for CIFAR-100-LT and AR-GSE framework.