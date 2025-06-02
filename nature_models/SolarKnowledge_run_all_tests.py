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
 It calculates comprehensive metrics for comparison with SolarFlareNet including:
 - True Skill Statistic (TSS)
 - Precision, Recall, Balanced Accuracy
 - Brier Score (BS) and Brier Skill Score (BSS)
 - Expected Calibration Error (ECE)
 - ROC-AUC, F1-Score, Accuracy
 Additionally, it computes confidence intervals and writes detailed results to JSON.
 @author: Yasser Abduallah
 Enhanced for comprehensive SolarFlareNet comparison
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
from datetime import datetime

warnings.filterwarnings("ignore")

# This dictionary will hold comprehensive performance metrics for each time window.
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
    """Calculate True Skill Statistic (TSS = Sensitivity + Specificity - 1).
    
    TSS is a key metric for space weather evaluation, ranging from -1 to 1.
    Values > 0 indicate skill above random chance.
    """
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


def calculate_brier_skill_score(y_true, y_prob):
    """Calculate Brier Skill Score (BSS) using climatological reference.
    
    BSS = 1 - (BS_forecast / BS_reference)
    where BS_reference is the Brier Score of climatological forecast
    """
    try:
        # Calculate actual Brier Score
        bs_forecast = brier_score_loss(y_true, y_prob)
        
        # Calculate reference Brier Score (climatological)
        # BS_ref = mean_event_rate * (1 - mean_event_rate)
        event_rate = np.mean(y_true)
        bs_reference = event_rate * (1 - event_rate)
        
        # Handle edge cases
        if bs_reference == 0:
            return 1.0 if bs_forecast == 0 else -np.inf
        
        bss = 1 - (bs_forecast / bs_reference)
        return bss
        
    except Exception as e:
        print(f"Warning: Could not calculate BSS: {e}")
        return "N/A"


def bootstrap_confidence_interval(y_true, y_pred, y_prob, metric_func, n_bootstrap=1000, confidence_level=0.95):
    """Calculate bootstrap confidence intervals for a metric.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels  
        y_prob: Predicted probabilities
        metric_func: Function to calculate metric
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
    
    Returns:
        tuple: (lower_bound, upper_bound)
    """
    try:
        np.random.seed(42)  # For reproducibility
        n_samples = len(y_true)
        bootstrap_scores = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            y_prob_boot = y_prob[indices] if y_prob is not None else None
            
            try:
                if y_prob_boot is not None:
                    score = metric_func(y_true_boot, y_pred_boot, y_prob_boot)
                else:
                    score = metric_func(y_true_boot, y_pred_boot)
                    
                if isinstance(score, (int, float)) and not np.isnan(score):
                    bootstrap_scores.append(score)
            except:
                continue
        
        if len(bootstrap_scores) < 10:  # Need sufficient samples
            return "N/A", "N/A"
            
        # Calculate confidence interval
        alpha = 1 - confidence_level
        lower_percentile = 100 * (alpha / 2)
        upper_percentile = 100 * (1 - alpha / 2)
        
        ci_lower = np.percentile(bootstrap_scores, lower_percentile)
        ci_upper = np.percentile(bootstrap_scores, upper_percentile)
        
        return ci_lower, ci_upper
        
    except Exception as e:
        print(f"Warning: Could not calculate confidence interval: {e}")
        return "N/A", "N/A"


def calculate_comprehensive_metrics(y_true, y_pred, y_prob):
    """Calculate comprehensive set of performance metrics for SolarFlareNet comparison."""
    metrics = {}
    
    try:
        # Convert to numpy arrays for consistency
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Handle probability formatting
        if y_prob is not None and len(y_prob) > 0:
            y_prob = np.array(y_prob)
            if y_prob.ndim > 1:
                # If probabilities are one-hot or multi-class, take positive class
                if y_prob.shape[1] == 2:
                    y_prob_positive = y_prob[:, 1]  # Positive class probability
                else:
                    y_prob_positive = np.max(y_prob, axis=1)  # Max probability
            else:
                y_prob_positive = y_prob
        else:
            y_prob_positive = None

        # =================================================================
        # TIER 1 METRICS: Most Important for SolarFlareNet Comparison
        # =================================================================
        
        # True Skill Statistic (TSS) - Primary metric for space weather
        metrics["TSS"] = calculate_tss(y_true, y_pred)
        
        # Precision, Recall, Balanced Accuracy
        metrics["precision"] = precision_score(y_true, y_pred, average="binary", zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, average="binary", zero_division=0)
        metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
        
        # Brier Score and Brier Skill Score
        if y_prob_positive is not None:
            try:
                metrics["Brier_Score"] = brier_score_loss(y_true, y_prob_positive)
                metrics["Brier_Skill_Score"] = calculate_brier_skill_score(y_true, y_prob_positive)
            except Exception as e:
                print(f"Warning: Could not calculate Brier metrics: {e}")
                metrics["Brier_Score"] = "N/A"
                metrics["Brier_Skill_Score"] = "N/A"
        else:
            metrics["Brier_Score"] = "N/A"
            metrics["Brier_Skill_Score"] = "N/A"
        
        # Expected Calibration Error (ECE)
        if y_prob_positive is not None:
            try:
                metrics["ECE"] = calculate_ece(y_true, y_prob_positive)
            except Exception as e:
                print(f"Warning: Could not calculate ECE: {e}")
                metrics["ECE"] = "N/A"
        else:
            metrics["ECE"] = "N/A"

        # =================================================================
        # TIER 2 METRICS: Highly Recommended for Comprehensive Comparison
        # =================================================================
        
        # F1-Score (harmonic mean of precision and recall)
        metrics["f1_score"] = f1_score(y_true, y_pred, average="binary", zero_division=0)
        
        # ROC-AUC
        if y_prob_positive is not None:
            try:
                metrics["ROC_AUC"] = roc_auc_score(y_true, y_prob_positive)
            except Exception as e:
                print(f"Warning: Could not calculate ROC-AUC: {e}")
                metrics["ROC_AUC"] = "N/A"
        else:
            metrics["ROC_AUC"] = "N/A"
        
        # =================================================================
        # TIER 3 METRICS: Standard ML metrics for completeness  
        # =================================================================
        
        # Accuracy (note: less meaningful for imbalanced datasets)
        metrics["accuracy"] = accuracy_score(y_true, y_pred)

        # =================================================================
        # CONFUSION MATRIX COMPONENTS AND DERIVED METRICS
        # =================================================================
        
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["true_positives"] = int(tp)
            metrics["false_positives"] = int(fp)
            metrics["true_negatives"] = int(tn)
            metrics["false_negatives"] = int(fn)

            # Additional derived metrics for detailed analysis
            metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0  # Same as recall
            metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics["positive_predictive_value"] = tp / (tp + fp) if (tp + fp) > 0 else 0  # Same as precision
            metrics["negative_predictive_value"] = tn / (tn + fn) if (tn + fn) > 0 else 0
            
            # Event rates and base rates
            metrics["event_rate"] = np.mean(y_true)
            metrics["prediction_rate"] = np.mean(y_pred)
            
        # =================================================================
        # CONFIDENCE INTERVALS (Bootstrap Method)
        # =================================================================
        
        print("  📊 Calculating bootstrap confidence intervals...")
        
        # Define metric functions for bootstrap CI calculation
        def tss_func(yt, yp, yprob=None): return calculate_tss(yt, yp)
        def precision_func(yt, yp, yprob=None): return precision_score(yt, yp, average="binary", zero_division=0)
        def recall_func(yt, yp, yprob=None): return recall_score(yt, yp, average="binary", zero_division=0)
        def bacc_func(yt, yp, yprob=None): return balanced_accuracy_score(yt, yp)
        def f1_func(yt, yp, yprob=None): return f1_score(yt, yp, average="binary", zero_division=0)
        
        # Calculate confidence intervals for key metrics
        ci_metrics = {}
        
        ci_lower, ci_upper = bootstrap_confidence_interval(y_true, y_pred, y_prob_positive, tss_func)
        ci_metrics["TSS_CI"] = {"lower": ci_lower, "upper": ci_upper}
        
        ci_lower, ci_upper = bootstrap_confidence_interval(y_true, y_pred, y_prob_positive, precision_func)
        ci_metrics["precision_CI"] = {"lower": ci_lower, "upper": ci_upper}
        
        ci_lower, ci_upper = bootstrap_confidence_interval(y_true, y_pred, y_prob_positive, recall_func)
        ci_metrics["recall_CI"] = {"lower": ci_lower, "upper": ci_upper}
        
        ci_lower, ci_upper = bootstrap_confidence_interval(y_true, y_pred, y_prob_positive, bacc_func)
        ci_metrics["balanced_accuracy_CI"] = {"lower": ci_lower, "upper": ci_upper}
        
        # Add CIs for probabilistic metrics if available
        if y_prob_positive is not None:
            def brier_func(yt, yp, yprob): return brier_score_loss(yt, yprob)
            def auc_func(yt, yp, yprob): return roc_auc_score(yt, yprob)
            
            ci_lower, ci_upper = bootstrap_confidence_interval(y_true, y_pred, y_prob_positive, brier_func)
            ci_metrics["Brier_Score_CI"] = {"lower": ci_lower, "upper": ci_upper}
            
            ci_lower, ci_upper = bootstrap_confidence_interval(y_true, y_pred, y_prob_positive, auc_func)
            ci_metrics["ROC_AUC_CI"] = {"lower": ci_lower, "upper": ci_upper}
        
        # Add confidence intervals to main metrics
        metrics.update(ci_metrics)

    except Exception as e:
        print(f"Error calculating comprehensive metrics: {e}")
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
            "Brier_Score": "N/A",
            "Brier_Skill_Score": "N/A",
            "ECE": "N/A",
            "f1_score": "N/A",
            "sensitivity": "N/A",
            "specificity": "N/A",
            "positive_predictive_value": "N/A",
            "negative_predictive_value": "N/A",
            "event_rate": "N/A",
            "prediction_rate": "N/A",
            "true_positives": "N/A",
            "false_positives": "N/A",
            "true_negatives": "N/A",
            "false_negatives": "N/A",
            "TSS_CI": {"lower": "N/A", "upper": "N/A"},
            "precision_CI": {"lower": "N/A", "upper": "N/A"},
            "recall_CI": {"lower": "N/A", "upper": "N/A"},
            "balanced_accuracy_CI": {"lower": "N/A", "upper": "N/A"},
            "Brier_Score_CI": {"lower": "N/A", "upper": "N/A"},
            "ROC_AUC_CI": {"lower": "N/A", "upper": "N/A"},
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
    print(f"Testing results for flare class {flare_class} with time window {time_window}:")
    print("==============================================")
    print("📊 TIER 1 METRICS (Primary for SolarFlareNet Comparison):")
    print(f"  TSS: {metrics.get('TSS', 'N/A'):.4f}")
    
    # Format precision with CI if available
    precision_val = metrics.get('precision', 'N/A')
    precision_ci = metrics.get('precision_CI', {})
    if isinstance(precision_val, (int, float)) and precision_ci.get('lower') != 'N/A':
        print(f"  Precision: {precision_val:.4f} [CI: {precision_ci['lower']:.4f}-{precision_ci['upper']:.4f}]")
    else:
        print(f"  Precision: {precision_val:.4f}" if isinstance(precision_val, (int, float)) else f"  Precision: {precision_val}")
    
    # Format recall with CI
    recall_val = metrics.get('recall', 'N/A')
    recall_ci = metrics.get('recall_CI', {})
    if isinstance(recall_val, (int, float)) and recall_ci.get('lower') != 'N/A':
        print(f"  Recall: {recall_val:.4f} [CI: {recall_ci['lower']:.4f}-{recall_ci['upper']:.4f}]")
    else:
        print(f"  Recall: {recall_val:.4f}" if isinstance(recall_val, (int, float)) else f"  Recall: {recall_val}")
    
    # Format balanced accuracy with CI
    bacc_val = metrics.get('balanced_accuracy', 'N/A')
    bacc_ci = metrics.get('balanced_accuracy_CI', {})
    if isinstance(bacc_val, (int, float)) and bacc_ci.get('lower') != 'N/A':
        print(f"  Balanced Accuracy: {bacc_val:.4f} [CI: {bacc_ci['lower']:.4f}-{bacc_ci['upper']:.4f}]")
    else:
        print(f"  Balanced Accuracy: {bacc_val:.4f}" if isinstance(bacc_val, (int, float)) else f"  Balanced Accuracy: {bacc_val}")
    
    # Format Brier Score with CI
    brier_val = metrics.get('Brier_Score', 'N/A')
    brier_ci = metrics.get('Brier_Score_CI', {})
    if isinstance(brier_val, (int, float)) and brier_ci.get('lower') != 'N/A':
        print(f"  Brier Score: {brier_val:.4f} [CI: {brier_ci['lower']:.4f}-{brier_ci['upper']:.4f}]")
    else:
        print(f"  Brier Score: {brier_val:.4f}" if isinstance(brier_val, (int, float)) else f"  Brier Score: {brier_val}")
    
    print(f"  Brier Skill Score: {metrics.get('Brier_Skill_Score', 'N/A'):.4f}" if isinstance(metrics.get('Brier_Skill_Score'), (int, float)) else f"  Brier Skill Score: {metrics.get('Brier_Skill_Score', 'N/A')}")
    print(f"  ECE: {metrics.get('ECE', 'N/A'):.4f}" if isinstance(metrics.get('ECE'), (int, float)) else f"  ECE: {metrics.get('ECE', 'N/A')}")
    
    print("\n📈 TIER 2 METRICS (Additional Performance Indicators):")
    print(f"  F1 Score: {metrics.get('f1_score', 'N/A'):.4f}" if isinstance(metrics.get('f1_score'), (int, float)) else f"  F1 Score: {metrics.get('f1_score', 'N/A')}")
    
    # Format ROC-AUC with CI
    auc_val = metrics.get('ROC_AUC', 'N/A')
    auc_ci = metrics.get('ROC_AUC_CI', {})
    if isinstance(auc_val, (int, float)) and auc_ci.get('lower') != 'N/A':
        print(f"  ROC-AUC: {auc_val:.4f} [CI: {auc_ci['lower']:.4f}-{auc_ci['upper']:.4f}]")
    else:
        print(f"  ROC-AUC: {auc_val:.4f}" if isinstance(auc_val, (int, float)) else f"  ROC-AUC: {auc_val}")
    
    print("\n📋 TIER 3 METRICS (Standard ML Metrics):")
    print(f"  Accuracy: {metrics.get('accuracy', 'N/A'):.4f}" if isinstance(metrics.get('accuracy'), (int, float)) else f"  Accuracy: {metrics.get('accuracy', 'N/A')}")
    
    print("\n🔍 DETAILED ANALYSIS:")
    print(f"  Sensitivity (Recall): {metrics.get('sensitivity', 'N/A'):.4f}" if isinstance(metrics.get('sensitivity'), (int, float)) else f"  Sensitivity: {metrics.get('sensitivity', 'N/A')}")
    print(f"  Specificity: {metrics.get('specificity', 'N/A'):.4f}" if isinstance(metrics.get('specificity'), (int, float)) else f"  Specificity: {metrics.get('specificity', 'N/A')}")
    print(f"  Event Rate: {metrics.get('event_rate', 'N/A'):.4f}" if isinstance(metrics.get('event_rate'), (int, float)) else f"  Event Rate: {metrics.get('event_rate', 'N/A')}")
    
    print("\n📊 CONFUSION MATRIX COMPONENTS:")
    print(f"  True Positives: {metrics.get('true_positives', 'N/A')}")
    print(f"  False Positives: {metrics.get('false_positives', 'N/A')}")
    print(f"  True Negatives: {metrics.get('true_negatives', 'N/A')}")
    print(f"  False Negatives: {metrics.get('false_negatives', 'N/A')}")

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
    print("📊 Calculating metrics for SolarFlareNet comparison:")
    print("   🎯 TIER 1: TSS, Precision, Recall, Balanced Accuracy, Brier Score, BSS, ECE")
    print("   📈 TIER 2: F1-Score, ROC-AUC")
    print("   📋 TIER 3: Accuracy")
    print("   🔢 CONFIDENCE INTERVALS: Bootstrap 95% CIs for key metrics")
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
    
    # Add metadata to the output
    metadata = {
        "generated_on": datetime.now().isoformat(),
        "description": "Comprehensive SolarKnowledge evaluation for SolarFlareNet comparison",
        "metrics_included": {
            "tier_1": ["TSS", "precision", "recall", "balanced_accuracy", "Brier_Score", "Brier_Skill_Score", "ECE"],
            "tier_2": ["f1_score", "ROC_AUC"],
            "tier_3": ["accuracy"],
            "confidence_intervals": "95% bootstrap CIs for key metrics",
            "detailed_analysis": ["sensitivity", "specificity", "confusion_matrix_components"]
        },
        "comparison_target": "SolarFlareNet paper results",
        "key_tasks": ["C-24h", "M-24h", "M5-24h", "C-48h"]
    }
    
    output_data = {
        "metadata": metadata,
        "results": all_metrics
    }
    
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=4)
    print(f"✅ Saved comprehensive test metrics into {output_file}")

    # Print comprehensive summary tables
    print("\n" + "=" * 100)
    print("📋 COMPREHENSIVE SOLARKNOWLEDGE EVALUATION SUMMARY")
    print("🎯 For Direct Comparison with SolarFlareNet Paper")
    print("=" * 100)

    # Table 1: Primary Metrics (Tier 1)
    print("\n🥇 TABLE 1: PRIMARY METRICS (Direct SolarFlareNet Comparison)")
    print("-" * 100)
    header1 = f"{'Config':<12} {'TSS':<8} {'Precision':<11} {'Recall':<9} {'BACC':<8} {'Brier':<8} {'BSS':<8} {'ECE':<8}"
    print(header1)
    print("-" * 100)

    for time_window in ["24", "48", "72"]:
        if time_window in all_metrics:
            for flare_class in ["C", "M", "M5"]:
                if flare_class in all_metrics[time_window]:
                    m = all_metrics[time_window][flare_class]
                    config = f"{flare_class}-{time_window}h"

                    # Format primary metrics
                    tss = f"{m.get('TSS', 'N/A'):.3f}" if isinstance(m.get("TSS"), (int, float)) else "N/A"
                    precision = f"{m.get('precision', 'N/A'):.3f}" if isinstance(m.get("precision"), (int, float)) else "N/A"
                    recall = f"{m.get('recall', 'N/A'):.3f}" if isinstance(m.get("recall"), (int, float)) else "N/A"
                    bacc = f"{m.get('balanced_accuracy', 'N/A'):.3f}" if isinstance(m.get("balanced_accuracy"), (int, float)) else "N/A"
                    brier = f"{m.get('Brier_Score', 'N/A'):.4f}" if isinstance(m.get("Brier_Score"), (int, float)) else "N/A"
                    bss = f"{m.get('Brier_Skill_Score', 'N/A'):.3f}" if isinstance(m.get("Brier_Skill_Score"), (int, float)) else "N/A"
                    ece = f"{m.get('ECE', 'N/A'):.4f}" if isinstance(m.get("ECE"), (int, float)) else "N/A"

                    print(f"{config:<12} {tss:<8} {precision:<11} {recall:<9} {bacc:<8} {brier:<8} {bss:<8} {ece:<8}")

    # Table 2: Secondary Metrics (Tier 2 & 3)
    print(f"\n🥈 TABLE 2: SECONDARY METRICS")
    print("-" * 80)
    header2 = f"{'Config':<12} {'F1-Score':<9} {'ROC-AUC':<9} {'Accuracy':<9} {'Sensitivity':<11} {'Specificity':<11}"
    print(header2)
    print("-" * 80)

    for time_window in ["24", "48", "72"]:
        if time_window in all_metrics:
            for flare_class in ["C", "M", "M5"]:
                if flare_class in all_metrics[time_window]:
                    m = all_metrics[time_window][flare_class]
                    config = f"{flare_class}-{time_window}h"

                    f1 = f"{m.get('f1_score', 'N/A'):.3f}" if isinstance(m.get("f1_score"), (int, float)) else "N/A"
                    auc = f"{m.get('ROC_AUC', 'N/A'):.3f}" if isinstance(m.get("ROC_AUC"), (int, float)) else "N/A"
                    acc = f"{m.get('accuracy', 'N/A'):.3f}" if isinstance(m.get("accuracy"), (int, float)) else "N/A"
                    sens = f"{m.get('sensitivity', 'N/A'):.3f}" if isinstance(m.get("sensitivity"), (int, float)) else "N/A"
                    spec = f"{m.get('specificity', 'N/A'):.3f}" if isinstance(m.get("specificity"), (int, float)) else "N/A"

                    print(f"{config:<12} {f1:<9} {auc:<9} {acc:<9} {sens:<11} {spec:<11}")

    # Table 3: Key Comparison Tasks (highlighted for paper)
    print(f"\n⭐ TABLE 3: KEY COMPARISON TASKS (Primary for SolarFlareNet Paper)")
    print("-" * 100)
    key_tasks = ["C-24h", "M-24h", "M5-24h", "C-48h"]
    
    print("Task      | TSS     | Precision | Recall  | BACC    | Brier   | BSS     | ECE     | ROC-AUC")
    print("-" * 100)
    
    for task in key_tasks:
        flare_class, time_str = task.split("-")
        time_window = time_str.replace("h", "")
        
        if time_window in all_metrics and flare_class in all_metrics[time_window]:
            m = all_metrics[time_window][flare_class]
            
            tss = f"{m.get('TSS', 'N/A'):.3f}" if isinstance(m.get("TSS"), (int, float)) else "N/A    "
            precision = f"{m.get('precision', 'N/A'):.3f}" if isinstance(m.get("precision"), (int, float)) else "N/A     "
            recall = f"{m.get('recall', 'N/A'):.3f}" if isinstance(m.get("recall"), (int, float)) else "N/A   "
            bacc = f"{m.get('balanced_accuracy', 'N/A'):.3f}" if isinstance(m.get("balanced_accuracy"), (int, float)) else "N/A   "
            brier = f"{m.get('Brier_Score', 'N/A'):.4f}" if isinstance(m.get("Brier_Score"), (int, float)) else "N/A   "
            bss = f"{m.get('Brier_Skill_Score', 'N/A'):.3f}" if isinstance(m.get("Brier_Skill_Score"), (int, float)) else "N/A   "
            ece = f"{m.get('ECE', 'N/A'):.4f}" if isinstance(m.get("ECE"), (int, float)) else "N/A   "
            auc = f"{m.get('ROC_AUC', 'N/A'):.3f}" if isinstance(m.get("ROC_AUC"), (int, float)) else "N/A    "
            
            print(f"{task:<9} | {tss:<7} | {precision:<9} | {recall:<7} | {bacc:<7} | {brier:<7} | {bss:<7} | {ece:<7} | {auc}")
        else:
            print(f"{task:<9} | N/A     | N/A       | N/A     | N/A     | N/A     | N/A     | N/A     | N/A")

    # Table 4: Confidence Intervals for Key Metrics
    print(f"\n📊 TABLE 4: CONFIDENCE INTERVALS (95% Bootstrap)")
    print("-" * 90)
    print("Task      | TSS CI              | Precision CI        | Recall CI           | BACC CI")
    print("-" * 90)
    
    for task in key_tasks:
        flare_class, time_str = task.split("-")
        time_window = time_str.replace("h", "")
        
        if time_window in all_metrics and flare_class in all_metrics[time_window]:
            m = all_metrics[time_window][flare_class]
            
            # Format confidence intervals
            def format_ci(ci_key):
                ci = m.get(ci_key, {})
                if ci.get('lower') != 'N/A' and ci.get('upper') != 'N/A':
                    return f"[{ci['lower']:.3f}-{ci['upper']:.3f}]"
                else:
                    return "N/A"
            
            tss_ci = format_ci('TSS_CI')
            prec_ci = format_ci('precision_CI')
            recall_ci = format_ci('recall_CI')
            bacc_ci = format_ci('balanced_accuracy_CI')
            
            print(f"{task:<9} | {tss_ci:<19} | {prec_ci:<19} | {recall_ci:<19} | {bacc_ci}")
        else:
            print(f"{task:<9} | N/A                 | N/A                 | N/A                 | N/A")

    print("=" * 100)
    print(f"🎯 Results saved to: {output_file}")
    print(f"📈 Total configurations evaluated: {sum(len(v) for v in all_metrics.values())}")
    
    # Count successful vs failed evaluations
    successful = 0
    failed = 0
    for time_window in all_metrics.values():
        for flare_class in time_window.values():
            if flare_class.get('status') == 'Success':
                successful += 1
            else:
                failed += 1
    
    print(f"✅ Successful evaluations: {successful}")
    print(f"❌ Failed evaluations: {failed}")
    print("\n🔬 KEY FINDINGS FOR SOLARFLARENET COMPARISON:")
    print("   • TSS values > 0 indicate skill above random chance")
    print("   • Brier Skill Score > 0 indicates improvement over climatology")
    print("   • ECE < 0.1 generally indicates well-calibrated probabilities")
    print("   • Focus on C-24h, M-24h, M5-24h, C-48h for direct paper comparison")
    print("=" * 100)
