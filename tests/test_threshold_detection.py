"""
Test script to demonstrate dynamic over-confidence threshold detection.

This script shows how the threshold detection algorithm works across
different synthetic calibration scenarios.
"""

import numpy as np
from sklearn.calibration import calibration_curve


def detect_overconfidence_threshold(probs, labels, n_bins=15):
    """
    Detect the over-confidence threshold where gap >= 0.1.
    
    Returns:
        threshold: The detected threshold, or None if no over-confidence found
        mean_pred: Mean predicted probabilities per bin
        frac_pos: Empirical frequencies per bin
    """
    frac_pos, mean_pred = calibration_curve(labels, probs, n_bins=n_bins, strategy="uniform")
    
    threshold = None
    for pred, frac in zip(mean_pred, frac_pos):
        gap = pred - frac
        if gap >= 0.1 and threshold is None:
            threshold = pred
            break
    
    return threshold, mean_pred, frac_pos


def create_scenario(name, n_samples=1000):
    """Create different calibration scenarios for testing."""
    np.random.seed(42)
    
    if name == "well_calibrated":
        # Well-calibrated across all ranges
        probs = np.random.beta(2, 2, n_samples)
        labels = np.random.binomial(1, probs)
        
    elif name == "overconfident_high":
        # Over-confident only at high probabilities (like SolarKnowledge)
        probs = np.random.beta(2, 5, n_samples)  # Skewed low
        labels = np.random.binomial(1, probs)
        
        # Make high confidence predictions over-confident
        high_mask = probs > 0.6
        probs[high_mask] += 0.2  # Increase confidence
        probs = np.clip(probs, 0, 1)
        
    elif name == "overconfident_medium":
        # Over-confident starting from medium probabilities
        probs = np.random.beta(3, 3, n_samples)
        labels = np.random.binomial(1, probs)
        
        # Make medium+ confidence predictions over-confident
        medium_mask = probs > 0.4
        probs[medium_mask] += 0.15
        probs = np.clip(probs, 0, 1)
        
    elif name == "underconfident":
        # Under-confident (conservative)
        probs = np.random.beta(2, 2, n_samples)
        labels = np.random.binomial(1, probs)
        probs *= 0.8  # Make more conservative
        
    return probs, labels


def test_threshold_detection():
    """Test threshold detection across different scenarios."""
    
    scenarios = [
        "well_calibrated",
        "overconfident_high", 
        "overconfident_medium",
        "underconfident"
    ]
    
    print("=" * 70)
    print("Dynamic Over-Confidence Threshold Detection Test")
    print("=" * 70)
    
    for scenario in scenarios:
        print(f"\n📊 Scenario: {scenario.replace('_', ' ').title()}")
        print("-" * 50)
        
        probs, labels = create_scenario(scenario)
        threshold, mean_pred, frac_pos = detect_overconfidence_threshold(probs, labels)
        
        print(f"Samples: {len(probs)}")
        print(f"Prob range: [{probs.min():.3f}, {probs.max():.3f}]")
        print(f"Pos rate: {labels.mean():.3f}")
        
        if threshold is not None:
            print(f"✅ Over-confidence threshold: p ≳ {threshold:.3f}")
        else:
            print(f"❌ No over-confidence detected")
        
        # Show bins with gaps ≥ 0.1
        overconfident_bins = []
        for i, (pred, frac) in enumerate(zip(mean_pred, frac_pos)):
            gap = pred - frac
            if gap >= 0.1:
                overconfident_bins.append((i+1, pred, frac, gap))
        
        if overconfident_bins:
            print(f"Over-confident bins (gap ≥ 0.10):")
            for bin_num, pred, frac, gap in overconfident_bins:
                print(f"  Bin {bin_num}: pred={pred:.3f}, emp={frac:.3f}, gap={gap:+.3f}")
        else:
            print(f"No bins with gap ≥ 0.10")
    
    print("\n" + "=" * 70)
    print("Summary:")
    print("- The threshold is the FIRST bin where confidence exceeds accuracy by ≥0.1")
    print("- This represents the smallest predicted probability showing over-confidence")
    print("- For SolarKnowledge M5-72h, the canonical threshold is ~0.43 (p ≳ 0.40)")
    print("=" * 70)


if __name__ == "__main__":
    test_threshold_detection() 