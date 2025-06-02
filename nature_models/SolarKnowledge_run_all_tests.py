#!/usr/bin/env python3
"""
 (c) Copyright 2023
 All rights reserved
 Programs written by Yasser Abduallah
 Department of Computer Science
 New Jersey Institute of Technology
 University Heights, Newark, NJ 07102, USA

 This script tests the transformer-based SolarKnowledge model by loading pre-trained
 weights and evaluating on the test datasets for various time windows and flare classes.
 It calculates comprehensive metrics including TSS, ECE, Brier score, ROC-AUC, and more.
 Additionally, it computes key metrics and writes out a JSON file that stores these results
 for each time window so that they can be automatically integrated into the final report.
 @author: Yasser Abduallah
"""

from SolarKnowledge_model import SolarKnowledge
from utils import get_testing_data, log, supported_flare_class
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    brier_score_loss,
    f1_score,
)
import json
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

# Import utility functions and configuration from your project

# This dictionary will hold your "This work" performance for each time window.
# Structure: { "24": { "C": {metrics}, "M": {metrics}, "M5": {metrics} },
#              "48": { ... },
#              "72": { ... } }
all_metrics = {}


def calculate_ece(y_true, y_prob, n_bins=15):
    """Calculate Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Select samples in bin
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


def calculate_tss(y_true, y_pred):
    """Calculate True Skill Statistic (TSS = Sensitivity + Specificity - 1)."""
    cm = confusion_matrix(y_true, y_pred)

    # Handle different confusion matrix shapes
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        # Multi-class case - use macro-averaged TSS
        n_classes = cm.shape[0]
        tss_scores = []

        for i in range(n_classes):
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            fp = np.sum(cm[:, i]) - tp
            tn = np.sum(cm) - tp - fn - fp

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            tss_scores.append(sensitivity + specificity - 1)

        return np.mean(tss_scores)

    return sensitivity + specificity - 1


def calculate_comprehensive_metrics(y_true, y_pred, y_prob):
    """Calculate comprehensive set of performance metrics."""
    metrics = {}

    try:
        # Basic classification metrics
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["precision"] = precision_score(
            y_true, y_pred, average="binary", zero_division=0
        )
        metrics["recall"] = recall_score(
            y_true, y_pred, average="binary", zero_division=0
        )
        metrics["f1_score"] = f1_score(
            y_true, y_pred, average="binary", zero_division=0
        )
        metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)

        # Domain-specific metrics
        metrics["TSS"] = calculate_tss(y_true, y_pred)

        # Probabilistic metrics (require probability scores)
        if y_prob is not None and len(y_prob) > 0:
            # Handle probability formatting
            if y_prob.ndim > 1:
                # If probabilities are one-hot or multi-class, take positive class
                if y_prob.shape[1] == 2:
                    y_prob_positive = y_prob[:, 1]  # Positive class probability
                else:
                    y_prob_positive = np.max(y_prob, axis=1)  # Max probability
            else:
                y_prob_positive = y_prob

            try:
                metrics["ROC_AUC"] = roc_auc_score(y_true, y_prob_positive)
            except ValueError as e:
                print(f"Warning: Could not calculate ROC-AUC: {e}")
                metrics["ROC_AUC"] = "N/A"

            try:
                metrics["Brier"] = brier_score_loss(y_true, y_prob_positive)
            except ValueError as e:
                print(f"Warning: Could not calculate Brier score: {e}")
                metrics["Brier"] = "N/A"

            try:
                metrics["ECE"] = calculate_ece(y_true, y_prob_positive)
            except Exception as e:
                print(f"Warning: Could not calculate ECE: {e}")
                metrics["ECE"] = "N/A"
        else:
            metrics["ROC_AUC"] = "N/A"
            metrics["Brier"] = "N/A"
            metrics["ECE"] = "N/A"

        # Confusion matrix statistics
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["true_positives"] = int(tp)
            metrics["false_positives"] = int(fp)
            metrics["true_negatives"] = int(tn)
            metrics["false_negatives"] = int(fn)

            # Additional derived metrics
            metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics["positive_predictive_value"] = (
                tp / (tp + fp) if (tp + fp) > 0 else 0
            )
            metrics["negative_predictive_value"] = (
                tn / (tn + fn) if (tn + fn) > 0 else 0
            )

    except Exception as e:
        print(f"Error calculating metrics: {e}")
        # Return basic metrics if comprehensive calculation fails
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "TSS": calculate_tss(y_true, y_pred),
            "error": str(e),
        }

    return metrics


def test_model(time_window, flare_class):
    log(
        "Testing initiated for time window: "
        + str(time_window)
        + " and flare class: "
        + flare_class,
        verbose=True,
    )

    # Load the testing data using your project's utility function
    X_test, y_test = get_testing_data(time_window, flare_class)
    # Convert y_test to a NumPy array so we can check its dimensions
    y_test = np.array(y_test)
    # If your test labels are one-hot encoded, convert them to class indices.
    if y_test.ndim > 1:
        y_true = np.argmax(y_test, axis=-1)
    else:
        y_true = y_test

    input_shape = (X_test.shape[1], X_test.shape[2])

    # Build and compile the model
    model = SolarKnowledge(early_stopping_patience=5)
    model.build_base_model(input_shape)
    model.compile()

    # Define the weights directory
    weight_dir = os.path.join("models", str(time_window), flare_class)
    if not os.path.exists(weight_dir):
        print(
            f"Warning: Model weights directory: {weight_dir} does not exist! Skipping test for time window {time_window} and flare class {flare_class}."
        )
        # Ensure we record placeholders for missing models.
        time_key = str(time_window)
        if time_key not in all_metrics:
            all_metrics[time_key] = {}
        all_metrics[time_key][flare_class] = {
            "accuracy": "N/A",
            "precision": "N/A",
            "recall": "N/A",
            "balanced_accuracy": "N/A",
            "TSS": "N/A",
            "ROC_AUC": "N/A",
            "Brier": "N/A",
            "ECE": "N/A",
            "f1_score": "N/A",
            "status": "Model weights not found",
        }
        return

    print("Loading weights from model dir:", weight_dir)
    model.load_weights(flare_class=flare_class, w_dir=weight_dir, verbose=True)

    # Run predictions on test data
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=-1)

    # Get probability scores for positive class
    if predictions.ndim > 1 and predictions.shape[1] >= 2:
        y_prob = predictions[:, 1]  # Positive class probability
    else:
        y_prob = predictions.flatten()  # Single output

    # Calculate comprehensive metrics
    metrics = calculate_comprehensive_metrics(y_true, predicted_classes, y_prob)

    print("==============================================")
    print(
        f"Testing results for flare class {flare_class} with time window {time_window}:"
    )
    print(f"  Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
    print(f"  TSS: {metrics.get('TSS', 'N/A'):.4f}")
    print(f"  ROC-AUC: {metrics.get('ROC_AUC', 'N/A')}")
    print(f"  Brier Score: {metrics.get('Brier', 'N/A')}")
    print(f"  ECE: {metrics.get('ECE', 'N/A')}")
    print(f"  F1 Score: {metrics.get('f1_score', 'N/A'):.4f}")
    print(f"  Precision: {metrics.get('precision', 'N/A'):.4f}")
    print(f"  Recall: {metrics.get('recall', 'N/A'):.4f}")

    print("\nDetailed Classification Report:")
    print(classification_report(y_true, predicted_classes))
    print("==============================================\n\n")

    # Store metrics for the given time window and flare class.
    time_key = str(time_window)
    if time_key not in all_metrics:
        all_metrics[time_key] = {}

    # Round numerical metrics for cleaner output
    rounded_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if key in [
                "true_positives",
                "false_positives",
                "true_negatives",
                "false_negatives",
            ]:
                rounded_metrics[key] = int(value)
            else:
                rounded_metrics[key] = round(float(value), 4)
        else:
            rounded_metrics[key] = value

    rounded_metrics["status"] = "Success"
    rounded_metrics["test_samples"] = len(y_true)
    rounded_metrics["positive_samples"] = int(np.sum(y_true))
    rounded_metrics["negative_samples"] = int(len(y_true) - np.sum(y_true))

    all_metrics[time_key][flare_class] = rounded_metrics


if __name__ == "__main__":
    print("🚀 Starting comprehensive SolarKnowledge model evaluation")
    print(
        "📊 Calculating metrics: Accuracy, TSS, ROC-AUC, Brier, ECE, F1, Precision, Recall"
    )
    print("=" * 80)

    # Loop over the desired time windows and flare classes.
    for time_window in [24, 48, 72]:
        for flare_class in ["C", "M", "M5"]:
            if flare_class not in supported_flare_class:
                print(
                    "Unsupported flare class:",
                    flare_class,
                    "It must be one of:",
                    ", ".join(supported_flare_class),
                )
                continue
            test_model(str(time_window), flare_class)
            log(
                "===========================================================\n\n",
                verbose=True,
            )

    # Save the comprehensive metrics for all time windows into a JSON file.
    output_file = "solarknowledge_comprehensive_results.json"
    with open(output_file, "w") as f:
        json.dump(all_metrics, f, indent=4)
    print(f"✅ Saved comprehensive test metrics into {output_file}")

    # Print summary table
    print("\n📋 SUMMARY TABLE")
    print("=" * 80)
    print(
        f"{'Config':<15} {'Accuracy':<10} {'TSS':<8} {'ROC-AUC':<9} {'Brier':<8} {'ECE':<8} {'F1':<8}"
    )
    print("-" * 80)

    for time_window in ["24", "48", "72"]:
        if time_window in all_metrics:
            for flare_class in ["C", "M", "M5"]:
                if flare_class in all_metrics[time_window]:
                    m = all_metrics[time_window][flare_class]
                    config = f"{flare_class}-{time_window}h"

                    # Format metrics for display
                    acc = (
                        f"{m.get('accuracy', 'N/A'):.3f}"
                        if isinstance(m.get("accuracy"), (int, float))
                        else "N/A"
                    )
                    tss = (
                        f"{m.get('TSS', 'N/A'):.3f}"
                        if isinstance(m.get("TSS"), (int, float))
                        else "N/A"
                    )
                    auc = (
                        f"{m.get('ROC_AUC', 'N/A'):.3f}"
                        if isinstance(m.get("ROC_AUC"), (int, float))
                        else "N/A"
                    )
                    brier = (
                        f"{m.get('Brier', 'N/A'):.4f}"
                        if isinstance(m.get("Brier"), (int, float))
                        else "N/A"
                    )
                    ece = (
                        f"{m.get('ECE', 'N/A'):.4f}"
                        if isinstance(m.get("ECE"), (int, float))
                        else "N/A"
                    )
                    f1 = (
                        f"{m.get('f1_score', 'N/A'):.3f}"
                        if isinstance(m.get("f1_score"), (int, float))
                        else "N/A"
                    )

                    print(
                        f"{config:<15} {acc:<10} {tss:<8} {auc:<9} {brier:<8} {ece:<8} {f1:<8}"
                    )

    print("=" * 80)
    print(f"🎯 Results saved to: {output_file}")
