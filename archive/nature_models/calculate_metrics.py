#!/usr/bin/env python3
"""
Calculate comprehensive performance metrics for SolarFlareNet test results.
Includes accuracy, precision, recall, F1, TSS, Brier score, ECE, and more.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    brier_score_loss
)
import sys
import os

def calculate_tss(y_true, y_pred):
    """Calculate True Skill Statistic (TSS)"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
    tss = sensitivity + specificity - 1
    return tss, sensitivity, specificity

def calculate_ece(y_true, y_prob, n_bins=10):
    """Calculate Expected Calibration Error (ECE)"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Determine if sample is in bin m (between bin_lower and bin_upper)
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece

def calculate_reliability_diagram_data(y_true, y_prob, n_bins=10):
    """Calculate data for reliability diagram"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    bin_centers = []
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            count_in_bin = in_bin.sum()
            
            bin_centers.append((bin_lower + bin_upper) / 2)
            bin_accuracies.append(accuracy_in_bin)
            bin_confidences.append(avg_confidence_in_bin)
            bin_counts.append(count_in_bin)
    
    return bin_centers, bin_accuracies, bin_confidences, bin_counts

def analyze_results(csv_file):
    """Analyze SolarFlareNet test results and calculate metrics"""
    print(f"📊 ANALYZING RESULTS: {csv_file}")
    print("=" * 60)
    
    # Load results
    df = pd.read_csv(csv_file)
    print(f"Total samples: {len(df)}")
    
    # Extract data
    y_true = df['FlareLabel'].values
    y_pred = df['Prediction'].values
    y_prob = df['PredictionProbability'].values
    
    # Basic statistics
    n_total = len(y_true)
    n_positive = y_true.sum()
    n_negative = n_total - n_positive
    base_rate = n_positive / n_total
    
    print(f"Positive samples: {n_positive} ({100*base_rate:.1f}%)")
    print(f"Negative samples: {n_negative} ({100*(1-base_rate):.1f}%)")
    print()
    
    # Classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    false_positive_rate = false_alarm_rate
    
    # TSS
    tss, sensitivity, _ = calculate_tss(y_true, y_pred)
    
    # Probabilistic metrics
    try:
        auc_roc = roc_auc_score(y_true, y_prob)
    except:
        auc_roc = float('nan')
    
    try:
        auc_pr = average_precision_score(y_true, y_prob)
    except:
        auc_pr = float('nan')
    
    brier_score = brier_score_loss(y_true, y_prob)
    ece = calculate_ece(y_true, y_prob)
    
    # Print results
    print("🎯 CLASSIFICATION METRICS:")
    print(f"   Accuracy:           {accuracy:.4f}")
    print(f"   Precision:          {precision:.4f}")
    print(f"   Recall (TPR):       {recall:.4f}")
    print(f"   F1-Score:           {f1:.4f}")
    print(f"   Specificity (TNR):  {specificity:.4f}")
    print(f"   False Alarm Rate:   {false_alarm_rate:.4f}")
    print()
    
    print("⚡ SPACE WEATHER METRICS:")
    print(f"   TSS (True Skill Statistic): {tss:.4f}")
    print(f"   Sensitivity (TPR):          {sensitivity:.4f}")
    print(f"   Specificity (TNR):          {specificity:.4f}")
    print()
    
    print("📈 PROBABILISTIC METRICS:")
    print(f"   AUC-ROC:            {auc_roc:.4f}")
    print(f"   AUC-PR:             {auc_pr:.4f}")
    print(f"   Brier Score:        {brier_score:.6f}")
    print(f"   ECE:                {ece:.6f}")
    print()
    
    print("🔢 CONFUSION MATRIX:")
    print(f"   True Negatives:     {tn:,}")
    print(f"   False Positives:    {fp:,}")
    print(f"   False Negatives:    {fn:,}")
    print(f"   True Positives:     {tp:,}")
    print()
    
    print("📊 PROBABILITY STATISTICS:")
    print(f"   Mean probability:   {y_prob.mean():.4f}")
    print(f"   Std probability:    {y_prob.std():.4f}")
    print(f"   Min probability:    {y_prob.min():.4f}")
    print(f"   Max probability:    {y_prob.max():.4f}")
    print()
    
    # Reliability analysis
    bin_centers, bin_accuracies, bin_confidences, bin_counts = calculate_reliability_diagram_data(y_true, y_prob)
    
    print("🎯 CALIBRATION ANALYSIS:")
    print("   Bin Center | Accuracy | Confidence | Count")
    print("   " + "-" * 42)
    for center, acc, conf, count in zip(bin_centers, bin_accuracies, bin_confidences, bin_counts):
        print(f"   {center:.2f}       | {acc:.4f}   | {conf:.4f}     | {count:,}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'tss': tss,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'brier_score': brier_score,
        'ece': ece,
        'false_alarm_rate': false_alarm_rate,
        'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
        'base_rate': base_rate,
        'n_samples': n_total
    }

if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        # Default to the C 24h results we just generated
        csv_file = "result/SolarFlareNet/C_24.csv"
    
    if not os.path.exists(csv_file):
        print(f"❌ Results file not found: {csv_file}")
        print("Available result files:")
        result_dir = "result/SolarFlareNet"
        if os.path.exists(result_dir):
            for file in os.listdir(result_dir):
                if file.endswith('.csv'):
                    print(f"   {os.path.join(result_dir, file)}")
        sys.exit(1)
    
    metrics = analyze_results(csv_file)
    print("\n✅ Analysis complete!") 