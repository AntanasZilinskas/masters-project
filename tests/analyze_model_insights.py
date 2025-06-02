"""
Analysis of what the calibration results tell us about the SolarKnowledge model.

This script interprets the calibration analysis to understand model behavior,
reliability patterns, and practical implications for solar flare prediction.
"""

import numpy as np
from pathlib import Path


def analyze_model_behavior():
    """Analyze what the calibration results tell us about the model."""

    # Load calibration results
    results_file = Path("calibration_results/skn_calib_curve.npz")
    if not results_file.exists():
        print(
            "❌ No calibration results found. Run test_solarknowledge_calibration.py first."
        )
        return

    data = np.load(results_file)
    mean_pred = data["mean_pred"]
    frac_pos = data["frac_pos"]
    ece = data["ece"]
    probs = data["probs"]
    labels = data["labels"]

    print("=" * 70)
    print("SOLARKNOWLEDGE MODEL BEHAVIOR ANALYSIS")
    print("=" * 70)

    # 1. Overall Calibration Quality
    print(f"\n🎯 Overall Calibration Quality:")
    print(f"   ECE (15-bin): {ece:.3f}")

    if ece < 0.05:
        quality = "Excellent"
        interpretation = "Model predictions are highly reliable"
    elif ece < 0.1:
        quality = "Good"
        interpretation = "Model predictions are generally trustworthy"
    elif ece < 0.2:
        quality = "Moderate"
        interpretation = "Model predictions need some caution"
    else:
        quality = "Poor"
        interpretation = "Model predictions are unreliable"

    print(f"   Quality: {quality} calibration")
    print(f"   Interpretation: {interpretation}")

    # 2. Confidence Distribution Analysis
    print(f"\n📈 Confidence Distribution:")
    confidence_ranges = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    for low, high in confidence_ranges:
        in_range = ((probs >= low) & (probs < high)).sum()
        pct = 100 * in_range / len(probs)
        print(f"   {low:.1f}-{high:.1f}: {in_range:4d} predictions ({pct:5.1f}%)")

    # 3. Bin-by-bin Behavior Analysis
    print(f"\n🔍 Detailed Confidence vs Reality Analysis:")
    print(f"{'Bin':>3} {'Confidence':>10} {'Reality':>8} {'Gap':>8} {'Behavior':>15}")
    print("-" * 55)

    well_calibrated_bins = 0
    overconfident_bins = 0
    underconfident_bins = 0

    for i, (pred, frac) in enumerate(zip(mean_pred, frac_pos)):
        gap = pred - frac

        if abs(gap) < 0.05:
            behavior = "Well-calibrated"
            well_calibrated_bins += 1
        elif gap >= 0.1:
            behavior = "Over-confident"
            overconfident_bins += 1
        elif gap <= -0.1:
            behavior = "Under-confident"
            underconfident_bins += 1
        elif gap > 0:
            behavior = "Slightly over"
        else:
            behavior = "Slightly under"

        print(f"{i+1:3d} {pred:10.3f} {frac:8.3f} {gap:+8.3f} {behavior:>15}")

    # 4. Pattern Summary
    print(f"\n📊 Calibration Pattern Summary:")
    print(f"   Well-calibrated bins: {well_calibrated_bins}")
    print(f"   Over-confident bins: {overconfident_bins}")
    print(f"   Under-confident bins: {underconfident_bins}")

    # 5. Find over-confidence threshold
    threshold = None
    for pred, frac in zip(mean_pred, frac_pos):
        if (pred - frac) >= 0.1:
            threshold = pred
            break

    print(f"\n⚠️  Over-confidence Analysis:")
    if threshold is not None:
        print(f"   Threshold: p ≳ {threshold:.3f}")
        print(f"   Meaning: When model is >{threshold:.1%} confident, it's often wrong")

        # Calculate accuracy above threshold
        above_threshold = probs >= threshold
        if above_threshold.sum() > 0:
            accuracy_above = labels[above_threshold].mean()
            avg_confidence_above = probs[above_threshold].mean()
            print(f"   Above threshold: {above_threshold.sum()} predictions")
            print(f"   Average confidence: {avg_confidence_above:.1%}")
            print(f"   Actual accuracy: {accuracy_above:.1%}")
            print(
                f"   Over-confidence gap: {avg_confidence_above - accuracy_above:.1%}"
            )
    else:
        print(f"   No significant over-confidence detected")

    # 6. Practical Implications
    print(f"\n🚨 Practical Implications for Solar Flare Prediction:")

    if threshold is not None and threshold < 0.5:
        print(f"   ⚠️  HIGH RISK: Model over-confident at moderate probabilities")
        print(f"       When model says >{threshold:.0%} chance, be skeptical")
        print(f"       Consider probability thresholds below {threshold:.2f}")
    elif threshold is not None and threshold < 0.7:
        print(f"   ⚠️  MEDIUM RISK: Model over-confident at high probabilities")
        print(f"       When model says >{threshold:.0%} chance, apply caution")
        print(f"       High-confidence predictions may be inflated")
    elif threshold is not None:
        print(f"   ✅ LOW RISK: Over-confidence only at very high probabilities")
        print(f"       Model generally reliable except at extreme confidence")
    else:
        print(f"   ✅ WELL CALIBRATED: No systematic over-confidence")
        print(f"       Model predictions can be trusted at face value")

    # 7. Model Architecture Insights
    print(f"\n🏗️  Model Architecture Insights:")
    print(f"   This is a Transformer-based model for solar flare prediction")
    print(f"   Common issues with transformer models:")

    if ece > 0.15:
        print(f"   • Poor calibration (observed) - needs calibration techniques")
    if overconfident_bins > 0:
        print(f"   • Over-confidence (observed) - may need temperature scaling")
    if threshold is not None and threshold < 0.6:
        print(f"   • Early over-confidence - may benefit from label smoothing")

    print(f"\n💡 Recommendations:")
    if ece > 0.1:
        print(f"   1. Apply post-hoc calibration (Platt scaling, temperature scaling)")
    if threshold is not None:
        print(f"   2. Use threshold-aware decision making")
        print(f"   3. Consider ensemble methods to improve calibration")
    if overconfident_bins > 2:
        print(f"   4. Retrain with calibration-aware loss functions")

    # 8. Comparison to Expected Results
    print(f"\n📚 Comparison to Recipe Expectations:")
    print(f"   Expected ECE: ~0.087")
    print(f"   Observed ECE: {ece:.3f}")
    print(f"   Expected threshold: ~0.43 (p ≳ 0.40)")
    threshold_str = f"{threshold:.3f}" if threshold is not None else "None"
    print(f"   Observed threshold: {threshold_str}")

    if threshold is not None and abs(threshold - 0.43) < 0.1:
        print(f"   ✅ Threshold roughly matches expectation")
    elif threshold is not None:
        print(f"   ⚠️  Threshold differs significantly from expectation")
    else:
        print(f"   ⚠️  No threshold detected (expected ~0.43)")

    if abs(ece - 0.087) < 0.05:
        print(f"   ✅ ECE matches expectation")
    else:
        print(f"   ⚠️  ECE differs from expectation")

    print("=" * 70)


if __name__ == "__main__":
    analyze_model_behavior()
