"""Check the actual ECE and over-confidence values from current results."""

import numpy as np


def check_actual_results():
    """Load and display the actual calibration results."""

    data = np.load("calibration_results/skn_calib_curve.npz")
    mean_pred = data["mean_pred"]
    frac_pos = data["frac_pos"]
    ece = data["ece"]
    probs = data["probs"]
    labels = data["labels"]

    print("=" * 50)
    print("ACTUAL CALIBRATION RESULTS")
    print("=" * 50)

    print(f"ECE (15-bin): {ece:.3f}")

    # Find over-confidence threshold
    threshold = None
    for pred, frac in zip(mean_pred, frac_pos):
        if (pred - frac) >= 0.1:
            threshold = pred
            break

    if threshold is not None:
        print(f"Over-confidence threshold: p ≳ {threshold:.3f}")

        # Calculate stats above threshold
        above_threshold = probs >= threshold
        if above_threshold.sum() > 0:
            accuracy_above = labels[above_threshold].mean()
            avg_confidence_above = probs[above_threshold].mean()
            gap_above = avg_confidence_above - accuracy_above
            print(f"Predictions above threshold: {above_threshold.sum()}")
            print(f"Average confidence above threshold: {avg_confidence_above:.1%}")
            print(f"Actual accuracy above threshold: {accuracy_above:.1%}")
            print(f"Over-confidence gap: {gap_above:.1%}")
    else:
        print("No over-confidence threshold detected")

    print(f"\nBin-by-bin gaps:")
    overconfident_bins = 0
    for i, (pred, frac) in enumerate(zip(mean_pred, frac_pos)):
        gap = pred - frac
        if gap >= 0.1:
            status = "(OVER-CONFIDENT)"
            overconfident_bins += 1
        else:
            status = ""
        print(f"  Bin {i+1}: {gap:+.3f} {status}")

    print(f"\nSummary:")
    print(f"- Total over-confident bins: {overconfident_bins}")
    print(
        f'- ECE indicates: {"Poor" if ece > 0.2 else "Moderate" if ece > 0.1 else "Good"} calibration'
    )


if __name__ == "__main__":
    check_actual_results()
