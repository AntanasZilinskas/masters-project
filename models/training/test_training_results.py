#!/usr/bin/env python3
"""
Test recent training results with corrected ECE calculation.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from pathlib import Path

def calculate_corrected_ece(y_true: np.ndarray, y_probs: np.ndarray, n_bins: int = 15) -> float:
    """Calculate Expected Calibration Error with corrected binning."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0
    for j, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
        # Corrected binning logic
        if j == n_bins - 1:  # Last bin includes upper boundary
            in_bin = (y_probs >= bin_lower) & (y_probs <= bin_upper)
        else:
            in_bin = (y_probs >= bin_lower) & (y_probs < bin_upper)
        
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_probs[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece

def calculate_buggy_ece(y_true: np.ndarray, y_probs: np.ndarray, n_bins: int = 15) -> float:
    """Calculate ECE with the original buggy binning for comparison."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Original buggy logic (excludes left boundary)
        in_bin = (y_probs > bin_lower) & (y_probs <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_probs[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece

def test_training_results():
    """Test training results with corrected ECE."""
    results_dir = Path("results")
    
    # Find experiment directories
    experiment_dirs = list(results_dir.glob("everest_*_*h_seed*"))
    
    if not experiment_dirs:
        print("❌ No experiment directories found")
        return
    
    print(f"🔍 Found {len(experiment_dirs)} experiment directories")
    print("=" * 60)
    
    for exp_dir in experiment_dirs[:3]:  # Test first 3
        exp_name = exp_dir.name
        print(f"\n📊 Testing: {exp_name}")
        
        # Load predictions if available
        pred_file = exp_dir / "predictions.csv"
        if not pred_file.exists():
            print(f"   ⚠️  No predictions.csv found")
            continue
            
        try:
            pred_df = pd.read_csv(pred_file)
            
            if 'y_true' not in pred_df.columns or 'y_prob' not in pred_df.columns:
                print(f"   ⚠️  Missing required columns")
                continue
                
            y_true = pred_df['y_true'].values
            y_probs = pred_df['y_prob'].values
            
            # Calculate ECE with both methods
            corrected_ece = calculate_corrected_ece(y_true, y_probs)
            buggy_ece = calculate_buggy_ece(y_true, y_probs)
            
            print(f"   Corrected ECE:  {corrected_ece:.6f}")
            print(f"   Buggy ECE:      {buggy_ece:.6f}")
            print(f"   Difference:     {corrected_ece - buggy_ece:.6f}")
            
            # Check for zero probabilities
            zero_probs = np.sum(y_probs == 0.0)
            one_probs = np.sum(y_probs == 1.0)
            print(f"   Samples at 0.0: {zero_probs:,}")
            print(f"   Samples at 1.0: {one_probs:,}")
            print(f"   Min probability: {np.min(y_probs):.6f}")
            print(f"   Max probability: {np.max(y_probs):.6f}")
            
        except Exception as e:
            print(f"   ❌ Error processing: {e}")

if __name__ == "__main__":
    test_training_results() 