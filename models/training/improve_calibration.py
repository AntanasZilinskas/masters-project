#!/usr/bin/env python3
"""
Improve model calibration using post-hoc methods.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize_scalar
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt

def calculate_ece(y_true: np.ndarray, y_probs: np.ndarray, n_bins: int = 15) -> float:
    """Calculate Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0
    
    for j in range(n_bins):
        bin_lower = bin_boundaries[j]
        bin_upper = bin_boundaries[j + 1]
        
        if j == n_bins - 1:
            in_bin = (y_probs >= bin_lower) & (y_probs <= bin_upper)
        else:
            in_bin = (y_probs >= bin_lower) & (y_probs < bin_upper)
        
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_probs[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece

class TemperatureScaling:
    """Temperature Scaling for post-hoc calibration."""
    
    def __init__(self):
        self.temperature = 1.0
    
    def fit(self, y_true: np.ndarray, logits: np.ndarray):
        """Find optimal temperature on validation set."""
        
        def temperature_scale(temp):
            """Apply temperature scaling."""
            scaled_logits = logits / temp
            # Convert to probabilities (assuming logits are log-odds)
            probs = 1 / (1 + np.exp(-scaled_logits))
            return calculate_ece(y_true, probs)
        
        # Find optimal temperature
        result = minimize_scalar(temperature_scale, bounds=(0.1, 10.0), method='bounded')
        self.temperature = result.x
        
        print(f"Optimal temperature: {self.temperature:.3f}")
        return self
    
    def transform(self, logits: np.ndarray):
        """Apply temperature scaling to logits."""
        scaled_logits = logits / self.temperature
        return 1 / (1 + np.exp(-scaled_logits))

def platt_scaling(y_true: np.ndarray, y_scores: np.ndarray):
    """Platt scaling for calibration."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.base import BaseEstimator, ClassifierMixin
    
    class DummyClassifier(BaseEstimator, ClassifierMixin):
        def __init__(self, scores):
            self.scores = scores
        
        def fit(self, X, y):
            return self
        
        def decision_function(self, X):
            return self.scores[:len(X)]
    
    # Create dummy classifier with your scores
    dummy = DummyClassifier(y_scores)
    
    # Apply Platt scaling
    calibrated = CalibratedClassifierCV(dummy, method='sigmoid', cv='prefit')
    calibrated.fit(np.arange(len(y_true)).reshape(-1, 1), y_true)
    
    # Get calibrated probabilities
    calibrated_probs = calibrated.predict_proba(np.arange(len(y_true)).reshape(-1, 1))[:, 1]
    
    return calibrated_probs

def isotonic_regression(y_true: np.ndarray, y_scores: np.ndarray):
    """Isotonic regression for calibration."""
    from sklearn.isotonic import IsotonicRegression
    
    # Fit isotonic regression
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    calibrated_probs = iso_reg.fit_transform(y_scores, y_true)
    
    return calibrated_probs

def compare_calibration_methods(y_true: np.ndarray, y_probs: np.ndarray):
    """Compare different calibration methods."""
    print("🔧 Calibration Method Comparison")
    print("=" * 40)
    
    # Original
    original_ece = calculate_ece(y_true, y_probs)
    original_brier = brier_score_loss(y_true, y_probs)
    
    print(f"Original:")
    print(f"  ECE:   {original_ece:.4f}")
    print(f"  Brier: {original_brier:.4f}")
    
    # Platt Scaling
    try:
        platt_probs = platt_scaling(y_true, y_probs)
        platt_ece = calculate_ece(y_true, platt_probs)
        platt_brier = brier_score_loss(y_true, platt_probs)
        
        print(f"\nPlatt Scaling:")
        print(f"  ECE:   {platt_ece:.4f} ({'↓' if platt_ece < original_ece else '↑'}{abs(platt_ece - original_ece):.4f})")
        print(f"  Brier: {platt_brier:.4f} ({'↓' if platt_brier < original_brier else '↑'}{abs(platt_brier - original_brier):.4f})")
    except Exception as e:
        print(f"\nPlatt Scaling failed: {e}")
    
    # Isotonic Regression
    try:
        iso_probs = isotonic_regression(y_true, y_probs)
        iso_ece = calculate_ece(y_true, iso_probs)
        iso_brier = brier_score_loss(y_true, iso_probs)
        
        print(f"\nIsotonic Regression:")
        print(f"  ECE:   {iso_ece:.4f} ({'↓' if iso_ece < original_ece else '↑'}{abs(iso_ece - original_ece):.4f})")
        print(f"  Brier: {iso_brier:.4f} ({'↓' if iso_brier < original_brier else '↑'}{abs(iso_brier - original_brier):.4f})")
    except Exception as e:
        print(f"\nIsotonic Regression failed: {e}")

def generate_calibration_improvement_plot(y_true, y_probs, save_path="calibration_comparison.png"):
    """Generate before/after calibration plots."""
    
    # Apply calibration methods
    try:
        platt_probs = platt_scaling(y_true, y_probs)
        iso_probs = isotonic_regression(y_true, y_probs)
    except:
        print("⚠️  Calibration methods failed, plotting original only")
        platt_probs = y_probs
        iso_probs = y_probs
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    methods = [
        ("Original", y_probs),
        ("Platt Scaling", platt_probs), 
        ("Isotonic Regression", iso_probs)
    ]
    
    for i, (method_name, probs) in enumerate(methods):
        ax = axes[i]
        
        # Calculate reliability curve
        n_bins = 15
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        bin_accuracies = []
        bin_confidences = []
        
        for j in range(n_bins):
            bin_lower = bin_boundaries[j]
            bin_upper = bin_boundaries[j + 1]
            
            if j == n_bins - 1:
                in_bin = (probs >= bin_lower) & (probs <= bin_upper)
            else:
                in_bin = (probs >= bin_lower) & (probs < bin_upper)
            
            if np.sum(in_bin) > 0:
                bin_accuracy = np.mean(y_true[in_bin])
                bin_confidence = np.mean(probs[in_bin])
                bin_accuracies.append(bin_accuracy)
                bin_confidences.append(bin_confidence)
        
        # Plot
        ax.plot([0, 1], [0, 1], "k--", alpha=0.7, label="Perfect calibration")
        if bin_confidences:
            ax.plot(bin_confidences, bin_accuracies, "o-", linewidth=2, markersize=6)
        
        # Calculate metrics
        ece = calculate_ece(y_true, probs)
        ax.set_title(f"{method_name}\nECE: {ece:.4f}")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Calibration comparison plot saved to: {save_path}")

# Training-time improvements
def suggest_training_improvements():
    """Suggest training-time calibration improvements."""
    print("\n🚀 Training-Time Calibration Improvements:")
    print("=" * 50)
    
    print("1. 🎯 Focal Loss Tuning:")
    print("   - Current gamma_max: 2.803")
    print("   - Try gamma_max: 1.5-2.0 (less aggressive)")
    print("   - Lower gamma = better calibration, slightly lower accuracy")
    
    print("\n2. 📊 Label Smoothing:")
    print("   - Add to loss function: (1-ε)y + ε/K")
    print("   - Try ε = 0.05-0.1")
    print("   - Prevents overconfident predictions")
    
    print("\n3. 🎲 Mixup/CutMix:")
    print("   - Mix samples: x = λx₁ + (1-λ)x₂")
    print("   - Mix labels: y = λy₁ + (1-λ)y₂") 
    print("   - Improves calibration naturally")
    
    print("\n4. 🔧 Evidential Learning Tuning:")
    print("   - Increase evidential loss weight")
    print("   - Current: evid=0.1, try evid=0.15-0.2")
    print("   - Better uncertainty quantification")
    
    print("\n5. 📈 Learning Rate Schedule:")
    print("   - Use cosine annealing")
    print("   - Lower final LR improves calibration")
    print("   - Try final_lr = 0.1 * initial_lr")

if __name__ == "__main__":
    # Example usage with dummy data
    np.random.seed(42)
    n_samples = 1000
    
    # Create example data (similar to your model's characteristics)
    y_true = np.random.binomial(1, 0.014, n_samples)  # Similar to your 0.0014 positive rate
    y_probs = np.random.beta(1, 50, n_samples)  # Similar to your low probability distribution
    
    # Add some calibration error
    y_probs = np.clip(y_probs * 1.2, 0, 1)  # Slightly overconfident
    
    print("🔍 Example Calibration Analysis")
    print("=" * 40)
    print(f"Original ECE: {calculate_ece(y_true, y_probs):.4f}")
    
    # Compare methods
    compare_calibration_methods(y_true, y_probs)
    
    # Generate plots
    generate_calibration_improvement_plot(y_true, y_probs)
    
    # Training suggestions
    suggest_training_improvements() 