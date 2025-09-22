# src/metrics/rc_curve.py
import torch
import numpy as np
import pandas as pd
from .selective_metrics import calculate_selective_errors

def generate_rc_curve(margins, preds, labels, class_to_group, num_groups, num_points=101):
    """
    Generates data points for the Risk-Coverage curve.
    """
    # Sort samples by margin, descending. Higher margin = higher confidence.
    sorted_indices = torch.argsort(margins, descending=True)
    sorted_margins = margins[sorted_indices]
    sorted_preds = preds[sorted_indices]
    sorted_labels = labels[sorted_indices]

    rc_data = []
    total_samples = len(labels)
    
    # Sweep through different coverage levels
    for i in range(1, num_points + 1):
        coverage_target = i / num_points
        num_to_accept = int(total_samples * coverage_target)
        
        if num_to_accept == 0:
            rc_data.append({'coverage': 0, 'balanced_error': 1.0, 'worst_error': 1.0})
            continue

        accepted_mask = torch.zeros_like(labels, dtype=torch.bool)
        accepted_mask[sorted_indices[:num_to_accept]] = True
        
        metrics = calculate_selective_errors(preds, labels, accepted_mask, class_to_group, num_groups)
        rc_data.append({
            'coverage': metrics['coverage'], 
            'balanced_error': metrics['balanced_error'], 
            'worst_error': metrics['worst_error']
        })

    return pd.DataFrame(rc_data)

def calculate_aurc(rc_dataframe, risk_key='balanced_error'):
    """Calculates the Area Under the Risk-Coverage Curve."""
    if rc_dataframe.empty:
        return 1.0
    
    coverages = rc_dataframe['coverage'].values
    risks = rc_dataframe[risk_key].values
    
    # Use trapezoidal rule for integration
    return np.trapz(risks, coverages)