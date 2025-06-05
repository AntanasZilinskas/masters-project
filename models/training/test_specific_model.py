#!/usr/bin/env python3
"""
Test specific model weights with corrected ECE calculation and reliability diagram.
"""

import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, brier_score_loss
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import EVEREST components
from models.solarknowledge_ret_plus import RETPlusWrapper
from models.utils import get_testing_data

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

def calculate_tss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate True Skill Statistic."""
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        if (tp + fn) == 0 or (tn + fp) == 0:
            return 0.0
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        return sensitivity + specificity - 1
    return 0.0

def generate_reliability_diagram(y_true: np.ndarray, y_probs: np.ndarray, title: str, save_path: str):
    """Generate reliability diagram with corrected binning."""
    n_bins = 15
    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for j in range(n_bins):
        bin_lower = bin_boundaries[j]
        bin_upper = bin_boundaries[j + 1]

        # Use corrected binning
        if j == n_bins - 1:  # Include upper boundary for last bin
            in_bin = (y_probs >= bin_lower) & (y_probs <= bin_upper)
        else:
            in_bin = (y_probs >= bin_lower) & (y_probs < bin_upper)

        if np.sum(in_bin) > 0:  # Only process non-empty bins
            bin_accuracy = np.mean(y_true[in_bin])
            bin_confidence = np.mean(y_probs[in_bin])  # Actual mean, not bin center
            bin_count = np.sum(in_bin)
            
            bin_accuracies.append(bin_accuracy)
            bin_confidences.append(bin_confidence)
            bin_counts.append(bin_count)

    # Create plot
    plt.figure(figsize=(8, 6))
    
    # Plot reliability curve
    plt.plot([0, 1], [0, 1], "k--", alpha=0.7, linewidth=2, label="Perfect calibration")
    if bin_confidences:  # Only plot if we have data
        plt.plot(bin_confidences, bin_accuracies, "o-", linewidth=2, markersize=8, 
                label="Model", color='red')
        
        # Add bin counts as text annotations
        for i, (conf, acc, count) in enumerate(zip(bin_confidences, bin_accuracies, bin_counts)):
            plt.annotate(f'{count}', (conf, acc), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8, alpha=0.7)

    plt.xlabel("Mean Predicted Probability", fontsize=12)
    plt.ylabel("Fraction of Positives", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    # Add ECE to plot
    corrected_ece = calculate_corrected_ece(y_true, y_probs)
    buggy_ece = calculate_buggy_ece(y_true, y_probs)
    
    plt.text(0.02, 0.98, f'Corrected ECE: {corrected_ece:.4f}\nBuggy ECE: {buggy_ece:.4f}', 
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Reliability diagram saved to: {save_path}")

def load_and_evaluate_model():
    """Load specific model and evaluate with corrected ECE."""
    print("🔍 Testing model: tests/model_weights_EVEREST_72h_M5.pt")
    print("=" * 60)
    
    # Model parameters (M5, 72h)
    flare_class = "M5"
    time_window = "72"
    model_path = "../../tests/model_weights_EVEREST_72h_M5.pt"
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    print(f"📁 Loading model from: {model_path}")
    
    # Load test data
    print(f"📊 Loading test data for {flare_class}-class, {time_window}h...")
    X_test, y_test = get_testing_data(time_window, flare_class)
    
    if X_test is None or y_test is None:
        print(f"❌ Could not load test data for {flare_class}/{time_window}h")
        return
        
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    print(f"   Test samples: {len(X_test):,}")
    print(f"   Positive rate: {np.mean(y_test):.4f}")
    
    # Create model wrapper
    print("🚀 Creating model wrapper...")
    model = RETPlusWrapper(
        input_shape=(10, 9),
        early_stopping_patience=10,
        use_attention_bottleneck=True,
        use_evidential=True,
        use_evt=True,
        use_precursor=True,
    )
    
    # Load weights
    print("⚡ Loading model weights...")
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        model.model.load_state_dict(state_dict)
        model.model.eval()
        print("✅ Model weights loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model weights: {e}")
        return
    
    # Get predictions
    print("🔮 Generating predictions...")
    y_probs = model.predict_proba(X_test).flatten()
    
    # Test different thresholds to find a reasonable one
    print("\n🎯 Finding optimal threshold...")
    thresholds = np.arange(0.1, 1.0, 0.05)
    best_tss = -1
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)
        tss = calculate_tss(y_test, y_pred)
        if tss > best_tss:
            best_tss = tss
            best_threshold = threshold
    
    print(f"   Optimal threshold: {best_threshold:.3f}")
    print(f"   Best TSS: {best_tss:.4f}")
    
    # Generate final predictions with optimal threshold
    y_pred = (y_probs >= best_threshold).astype(int)
    
    # Calculate all metrics
    print("\n📊 Model Performance:")
    print("-" * 30)
    
    # Basic metrics
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    brier = brier_score_loss(y_test, y_probs)
    
    # ECE comparison
    corrected_ece = calculate_corrected_ece(y_test, y_probs)
    buggy_ece = calculate_buggy_ece(y_test, y_probs)
    
    print(f"TSS:                {best_tss:.4f}")
    print(f"Precision:          {precision:.4f}")
    print(f"Recall:             {recall:.4f}")
    print(f"F1:                 {f1:.4f}")
    print(f"Brier Score:        {brier:.4f}")
    print(f"Corrected ECE:      {corrected_ece:.4f}")
    print(f"Buggy ECE:          {buggy_ece:.4f}")
    print(f"ECE Difference:     {corrected_ece - buggy_ece:.4f}")
    
    # Probability distribution analysis
    print(f"\n📈 Probability Distribution:")
    print(f"   Min probability:  {np.min(y_probs):.4f}")
    print(f"   Max probability:  {np.max(y_probs):.4f}")
    print(f"   Mean probability: {np.mean(y_probs):.4f}")
    print(f"   Std probability:  {np.std(y_probs):.4f}")
    print(f"   Samples at 0.0:   {np.sum(y_probs == 0.0):,}")
    print(f"   Samples at 1.0:   {np.sum(y_probs == 1.0):,}")
    
    # Generate reliability diagram
    print("\n📊 Generating reliability diagram...")
    save_path = "model_weights_EVEREST_72h_M5_reliability.png"
    title = f"EVEREST M5-72h Reliability Diagram\n(Corrected ECE: {corrected_ece:.4f}, Buggy ECE: {buggy_ece:.4f})"
    generate_reliability_diagram(y_test, y_probs, title, save_path)
    
    print("\n✅ Evaluation complete!")

if __name__ == "__main__":
    load_and_evaluate_model() 