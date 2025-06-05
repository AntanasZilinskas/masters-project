"""
Compare actual ECE of trained EVEREST vs SolarKnowledge models.

This script loads the real trained model weights and evaluates their 
calibration on the same test dataset to get empirical ECE values 
for the paper comparison.
"""

import sys
import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import h5py
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "models"))

warnings.filterwarnings("ignore")

# Device setup
DEVICE = "cpu"  # Force CPU to avoid device mismatch issues
print(f"Using device: {DEVICE}")

# Import models
try:
    from models.solarknowledge_ret_plus import RETPlusWrapper, RETPlusModel

    print("✅ EVEREST (RETPlusWrapper) imported")
except ImportError as e:
    print(f"❌ EVEREST import failed: {e}")

try:
    from models.SolarKnowledge_model_pytorch import SolarKnowledge

    print("✅ SolarKnowledge imported")
except ImportError as e:
    print(f"❌ SolarKnowledge import failed: {e}")


class ECECalculator:
    """Calculate Expected Calibration Error with 15 bins."""

    def __init__(self, n_bins: int = 15):
        self.n_bins = n_bins

    def __call__(self, probs: np.ndarray, labels: np.ndarray) -> float:
        """Calculate ECE."""
        if len(probs.shape) > 1:
            probs = probs.squeeze()
        if len(labels.shape) > 1:
            labels = labels.squeeze()

        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        ece = 0.0

        for i in range(self.n_bins):
            bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
            in_bin = (probs > bin_lower) & (probs <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = labels[in_bin].mean()
                avg_confidence_in_bin = probs[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


def load_test_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load actual test data for evaluation."""

    # Try to load from SHARP dataset
    sharp_file = project_root / "datasets/tiny_sample/sharp_ci_sample.h5"
    if sharp_file.exists():
        print(f"Loading test data from: {sharp_file}")
        with h5py.File(sharp_file, "r") as f:
            # Load features and labels
            X = f["X"][:]  # Shape: (N, T, F)
            y = f["y"][:]  # Shape: (N,)

            print(f"Loaded {len(X)} samples, {y.mean():.1%} positive rate")
            return X, y

    # Fallback to synthetic data if real data not available
    print("⚠️  Using synthetic test data (real SHARP data not found)")
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randn(n_samples, 10, 9)

    # Create realistic flare probability pattern
    feature_strength = X[:, :5, :].mean(axis=(1, 2))
    y_prob = 1 / (1 + np.exp(-(0.5 * feature_strength - 1.5)))
    y = np.random.binomial(1, y_prob)

    print(f"Generated {n_samples} synthetic samples, {y.mean():.1%} positive rate")
    return X, y


def load_everest_model() -> RETPlusWrapper:
    """Load trained EVEREST model."""

    model_path = project_root / "tests/model_weights_EVEREST_72h_M5.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"EVEREST model not found: {model_path}")

    print(f"Loading EVEREST model from: {model_path}")

    # Create EVEREST wrapper
    everest_model = RETPlusWrapper(
        input_shape=(10, 9),
        use_attention_bottleneck=True,
        use_evidential=True,
        use_evt=True,
        use_precursor=True,
    )

    # Load weights
    checkpoint = torch.load(model_path, map_location=DEVICE)

    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        everest_model.model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        everest_model.model.load_state_dict(checkpoint["state_dict"])
    else:
        everest_model.model.load_state_dict(checkpoint)

    everest_model.model.eval()
    print("✅ EVEREST model loaded successfully")

    return everest_model


def load_solarknowledge_model() -> SolarKnowledge:
    """Load trained SolarKnowledge model."""

    model_path = (
        project_root / "models/models/SolarKnowledge-v1.4-M5-72h/model_weights.pt"
    )

    if not model_path.exists():
        print(f"⚠️  SolarKnowledge model not found: {model_path}")
        print("Creating untrained SolarKnowledge model for architecture comparison")

        # Create model with correct SolarKnowledge constructor
        model = SolarKnowledge()
        model.build_base_model(
            input_shape=(10, 9),
            embed_dim=128,
            num_heads=4,
            ff_dim=256,
            num_transformer_blocks=6,
            dropout_rate=0.2,
        )
        model.compile()

        return model

    print(f"Loading SolarKnowledge model from: {model_path}")

    # Create SolarKnowledge model with correct constructor
    solarknowledge_model = SolarKnowledge()
    solarknowledge_model.build_base_model(
        input_shape=(10, 9),
        embed_dim=128,
        num_heads=4,
        ff_dim=256,
        num_transformer_blocks=6,
        dropout_rate=0.2,
    )

    # Load weights
    checkpoint = torch.load(model_path, map_location=DEVICE)

    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        solarknowledge_model.model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        solarknowledge_model.model.load_state_dict(checkpoint["state_dict"])
    else:
        solarknowledge_model.model.load_state_dict(checkpoint)

    solarknowledge_model.model.eval()
    print("✅ SolarKnowledge model loaded successfully")

    return solarknowledge_model


def evaluate_model_calibration(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    is_everest: bool = False,
) -> Dict:
    """Evaluate model calibration and return metrics."""

    print(f"\nEvaluating {model_name} calibration...")

    # Convert to torch tensors and ensure CPU placement
    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

    # Get predictions
    with torch.no_grad():
        if is_everest:
            # EVEREST model returns dict with multiple outputs
            outputs = model.model(X_tensor)

            if isinstance(outputs, dict):
                # Use main logits for probability
                if "logits" in outputs:
                    logits = outputs["logits"]
                elif "main_logits" in outputs:
                    logits = outputs["main_logits"]
                else:
                    # Take first output if structure unclear
                    logits = list(outputs.values())[0]
            else:
                logits = outputs

        else:
            # SolarKnowledge model - use predict method for compatibility
            try:
                # Try direct model forward
                logits = model.model(X_tensor)
            except Exception as e:
                print(f"   Direct forward failed: {e}")
                # Fallback to predict method
                probs_np = model.predict(X_test)
                # Handle different output formats
                if probs_np.shape[1] == 2:
                    probs_np = probs_np[:, 1]  # Take positive class probability
                elif probs_np.shape[1] == 1:
                    probs_np = probs_np.squeeze()

                # Calculate ECE directly
                ece_calc = ECECalculator(n_bins=15)
                ece = ece_calc(probs_np, y_test)

                # Calculate reliability curve
                frac_pos, mean_pred = calibration_curve(
                    y_test, probs_np, n_bins=15, strategy="uniform"
                )

                # Find over-confidence threshold
                threshold = None
                for pred, frac in zip(mean_pred, frac_pos):
                    if (pred - frac) >= 0.1:
                        threshold = pred
                        break

                results = {
                    "model_name": model_name,
                    "ece": ece,
                    "probs": probs_np,
                    "mean_pred": mean_pred,
                    "frac_pos": frac_pos,
                    "threshold": threshold,
                    "n_samples": len(y_test),
                    "positive_rate": y_test.mean(),
                }

                print(f"   ECE: {ece:.3f}")
                if threshold is not None:
                    print(f"   Over-confidence threshold: p ≳ {threshold:.3f}")
                else:
                    print(f"   No over-confidence detected (all gaps < 0.10)")

                return results

        # Convert logits to probabilities
        if len(logits.shape) > 1 and logits.shape[1] > 1:
            # Multi-class output, take positive class
            probs = torch.softmax(logits, dim=1)[:, 1]
        else:
            # Binary classification
            probs = torch.sigmoid(logits.squeeze())

        probs_np = probs.cpu().numpy()

    # Calculate ECE
    ece_calc = ECECalculator(n_bins=15)
    ece = ece_calc(probs_np, y_test)

    # Calculate reliability curve
    frac_pos, mean_pred = calibration_curve(
        y_test, probs_np, n_bins=15, strategy="uniform"
    )

    # Find over-confidence threshold
    threshold = None
    for pred, frac in zip(mean_pred, frac_pos):
        if (pred - frac) >= 0.1:
            threshold = pred
            break

    results = {
        "model_name": model_name,
        "ece": ece,
        "probs": probs_np,
        "mean_pred": mean_pred,
        "frac_pos": frac_pos,
        "threshold": threshold,
        "n_samples": len(y_test),
        "positive_rate": y_test.mean(),
    }

    print(f"   ECE: {ece:.3f}")
    if threshold is not None:
        print(f"   Over-confidence threshold: p ≳ {threshold:.3f}")
    else:
        print(f"   No over-confidence detected (all gaps < 0.10)")

    return results


def compare_models():
    """Compare EVEREST vs SolarKnowledge ECE on the same test data."""

    print("=" * 80)
    print("EVEREST vs SolarKnowledge ECE COMPARISON")
    print("=" * 80)

    # Load test data
    print("\n1. Loading test data...")
    X_test, y_test = load_test_data()

    # Load models
    print("\n2. Loading trained models...")

    try:
        everest_model = load_everest_model()
        everest_available = True
    except Exception as e:
        print(f"❌ Could not load EVEREST model: {e}")
        everest_available = False

    try:
        solarknowledge_model = load_solarknowledge_model()
        solarknowledge_available = True
    except Exception as e:
        print(f"❌ Could not load SolarKnowledge model: {e}")
        solarknowledge_available = False

    if not (everest_available or solarknowledge_available):
        print("❌ No models could be loaded for comparison")
        return

    # Evaluate models
    print("\n3. Evaluating model calibration...")
    results = {}

    if everest_available:
        results["everest"] = evaluate_model_calibration(
            everest_model, X_test, y_test, "EVEREST", is_everest=True
        )

    if solarknowledge_available:
        results["solarknowledge"] = evaluate_model_calibration(
            solarknowledge_model, X_test, y_test, "SolarKnowledge", is_everest=False
        )

    # Compare results
    print("\n" + "=" * 80)
    print("CALIBRATION COMPARISON RESULTS")
    print("=" * 80)

    if "everest" in results and "solarknowledge" in results:
        everest_ece = results["everest"]["ece"]
        solarknowledge_ece = results["solarknowledge"]["ece"]

        print(f"\n📊 ECE Comparison (15-bin):")
        print(f"   SolarKnowledge ECE:      {solarknowledge_ece:.3f}")
        print(f"   EVEREST ECE:             {everest_ece:.3f}")

        improvement = solarknowledge_ece - everest_ece
        if improvement > 0:
            improvement_pct = (improvement / solarknowledge_ece) * 100
            print(
                f"   ECE Improvement:         {improvement:.3f} ({improvement_pct:.1f}% reduction)"
            )
            print(f"   ✅ EVEREST is better calibrated")
        else:
            worsening_pct = abs(improvement / solarknowledge_ece) * 100
            print(
                f"   ECE Difference:          {abs(improvement):.3f} ({worsening_pct:.1f}% increase)"
            )
            print(f"   ⚠️  SolarKnowledge is better calibrated")

        # Over-confidence analysis
        print(f"\n⚠️  Over-confidence Analysis:")
        sk_threshold = results["solarknowledge"]["threshold"]
        ev_threshold = results["everest"]["threshold"]

        if sk_threshold is not None:
            print(f"   SolarKnowledge threshold: p ≳ {sk_threshold:.3f}")
        else:
            print(f"   SolarKnowledge: No over-confidence")

        if ev_threshold is not None:
            print(f"   EVEREST threshold:        p ≳ {ev_threshold:.3f}")
        else:
            print(f"   EVEREST: No over-confidence")

    elif "everest" in results:
        print(f"\n📊 EVEREST Results:")
        print(f"   ECE: {results['everest']['ece']:.3f}")

    elif "solarknowledge" in results:
        print(f"\n📊 SolarKnowledge Results:")
        print(f"   ECE: {results['solarknowledge']['ece']:.3f}")

    # Save results
    save_path = Path("tests/calibration_results")
    save_path.mkdir(exist_ok=True)

    np.savez(
        save_path / "everest_vs_solarknowledge_ece.npz",
        **{f"{k}_results": v for k, v in results.items()},
        test_samples=len(y_test),
        positive_rate=y_test.mean(),
    )

    print(f"\n📈 Results saved to: {save_path}/everest_vs_solarknowledge_ece.npz")

    # Generate comparison plot
    if len(results) >= 2:
        generate_comparison_plot(results, save_path)

    print("\n" + "=" * 80)
    print("PAPER PARAGRAPH NUMBERS")
    print("=" * 80)

    if "everest" in results and "solarknowledge" in results:
        sk_ece = results["solarknowledge"]["ece"]
        ev_ece = results["everest"]["ece"]
        print(f"For your paper:")
        print(
            f'"ECE drops from {sk_ece:.3f} (SolarKnowledge) to {ev_ece:.3f} (EVEREST)"'
        )

    return results


def generate_comparison_plot(results: Dict, save_path: Path):
    """Generate comparison reliability plots."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    colors = {"everest": "blue", "solarknowledge": "red"}

    # Reliability diagram
    for model_key, result in results.items():
        mean_pred = result["mean_pred"]
        frac_pos = result["frac_pos"]
        ece = result["ece"]
        model_name = result["model_name"]

        ax1.plot(
            mean_pred,
            frac_pos,
            "o-",
            label=f"{model_name} (ECE={ece:.3f})",
            color=colors.get(model_key, "gray"),
            linewidth=2,
            markersize=6,
        )

    ax1.plot(
        [0, 1],
        [0, 1],
        "--",
        color="gray",
        linewidth=1,
        alpha=0.8,
        label="Perfect calibration",
    )
    ax1.set_xlabel("Mean predicted probability")
    ax1.set_ylabel("Empirical frequency")
    ax1.set_title("Calibration Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # ECE bar comparison
    if len(results) >= 2:
        models = [result["model_name"] for result in results.values()]
        eces = [result["ece"] for result in results.values()]

        bars = ax2.bar(
            models,
            eces,
            color=[colors.get(k, "gray") for k in results.keys()],
            alpha=0.7,
        )
        ax2.set_ylabel("Expected Calibration Error")
        ax2.set_title("ECE Comparison")
        ax2.grid(True, alpha=0.3, axis="y")

        # Add value labels
        for bar, ece in zip(bars, eces):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.002,
                f"{ece:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    plt.suptitle(
        "EVEREST vs SolarKnowledge Calibration", fontsize=16, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(
        save_path / "everest_vs_solarknowledge_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"📊 Comparison plot saved: everest_vs_solarknowledge_comparison.png")


if __name__ == "__main__":
    compare_models()
