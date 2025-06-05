#!/usr/bin/env python
"""
Evaluation script for solar flare prediction using the EVEREST model.

This script evaluates the trained model on solar flare data, generating
comprehensive performance metrics and visualizations.

Key features:
- Per-flare class evaluation
- ROC curves and confusion matrices
- Precision-recall analysis
- Detailed performance metrics (accuracy, TSS, F1, etc.)
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import glob
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import f1_score, precision_score, recall_score

# Tell PyTorch to use single process DataLoader
os.environ["PYTORCH_WORKERS"] = "0"

# Import the modules
from everest import RETPlusWrapper, device
import model_tracking
import utils

def main():
    # Path to the trained model - modify as needed
    flare_class = "M"  # Options: C, M, or X
    time_window = "24"  # Options: 24, 48, 72 (hours)
    
    model_dir = f"trained_models/EVEREST-v1-{flare_class}-{time_window}h"
    model_weights = os.path.join(model_dir, "model_weights.pt")
    
    if not os.path.exists(model_weights):
        print(f"Error: Model weights not found at {model_weights}")
        return
    
    # Model configuration (must match the trained model)
    input_shape = (10, 9)  # 10 timesteps, 9 SHARP features
    embed_dim = 128
    num_heads = 4
    ff_dim = 256
    num_blocks = 6
    dropout = 0.2
    
    print(f"Evaluating EVEREST model for {flare_class}-class flares with {time_window}h forecast window")
    print(f"Using model from: {model_dir}")
    print(f"Using device: {device}")
    
    # Initialize the model with the same architecture
    model = RETPlusWrapper(
        input_shape=input_shape,
        early_stopping_patience=10,
        loss_weights={"focal": 0.80, "evid": 0.10, "evt": 0.10},
    )
    
    # Override the model with custom parameters
    model.model = model.model.__class__(
        input_shape=input_shape,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_blocks=num_blocks,
        dropout=dropout,
        use_attention_bottleneck=True,
        use_evidential=True, 
        use_evt=True,
        use_precursor=True
    ).to(device)
    
    # Load the trained weights
    model.model.load_state_dict(torch.load(model_weights, map_location=device))
    model.model.eval()
    
    # Load test data
    X_test, y_test = load_solar_test_data(flare_class, time_window)
    
    if len(X_test) == 0:
        print(f"No test data found for {flare_class}-class flares with {time_window}h forecast window. Exiting.")
        return
    
    # Evaluate on test data
    results = evaluate_model(model, X_test, y_test, f"{flare_class}_{time_window}h")
    
    # Create visualizations
    os.makedirs("solar_analysis", exist_ok=True)
    
    # Plot ROC curve
    plot_roc_curve(results, f"{flare_class}_{time_window}h")
    
    # Plot precision-recall curve
    plot_pr_curve(results, f"{flare_class}_{time_window}h")
    
    # Create confusion matrix visualization
    create_confusion_matrix(results, f"{flare_class}_{time_window}h")
    
    # Save summary to CSV
    create_summary_table(results, f"{flare_class}_{time_window}h")
    
    return results

def load_solar_test_data(flare_class, time_window):
    """Load solar flare test data for evaluation"""
    try:
        # Use utility function to load test data
        X_test, y_test = utils.get_test_data(time_window, flare_class)
        print(f"Loaded {len(X_test)} test samples for {flare_class}-class flares with {time_window}h forecast window")
        print(f"Class distribution: {sum(y_test)} positive, {len(y_test) - sum(y_test)} negative")
        return X_test, y_test
    except Exception as e:
        print(f"Error loading test data: {e}")
        return np.array([]), np.array([])

def evaluate_model(model, X_test, y_test, scenario_name):
    """Evaluate the model on test data and return metrics"""
    # Make predictions
    y_pred_proba = model.predict_proba(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics using sklearn for consistency
    accuracy = (y_pred.flatten() == y_test).mean()
    
    # Calculate precision, recall and F1 using sklearn functions
    precision = precision_score(y_test, y_pred.flatten(), zero_division=0)
    recall = recall_score(y_test, y_pred.flatten(), zero_division=0)
    f1 = f1_score(y_test, y_pred.flatten(), zero_division=0)
    
    # Calculate TSS
    pos_idx = np.where(y_test == 1)[0]
    neg_idx = np.where(y_test == 0)[0]
    
    if len(pos_idx) == 0:
        sensitivity = 1.0
    else:
        sensitivity = np.mean(y_pred.flatten()[pos_idx])
        
    if len(neg_idx) == 0:
        specificity = 1.0
    else:
        specificity = 1 - np.mean(y_pred.flatten()[neg_idx])
    
    tss = sensitivity + specificity - 1
    
    # Calculate confusion matrix elements
    TP = np.sum((y_pred.flatten() == 1) & (y_test == 1))
    TN = np.sum((y_pred.flatten() == 0) & (y_test == 0))
    FP = np.sum((y_pred.flatten() == 1) & (y_test == 0))
    FN = np.sum((y_pred.flatten() == 0) & (y_test == 1))
    
    # Calculate ROC AUC
    from sklearn.metrics import roc_auc_score
    roc_auc = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0
    
    # Calculate average precision score
    ap = average_precision_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0
    
    # Calculate Heidke Skill Score (HSS)
    expected_hits = ((TP + FN) * (TP + FP)) / (TP + TN + FP + FN)
    hss = (TP - expected_hits) / (TP + FN + FP - expected_hits) if (TP + FN + FP - expected_hits) != 0 else 0
    
    print(f"Results for {scenario_name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  TSS (True Skill Statistic): {tss:.4f}")
    print(f"  HSS (Heidke Skill Score): {hss:.4f}")
    print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    print(f"  ROC AUC: {roc_auc:.4f}, Average Precision: {ap:.4f}")
    print(f"  Confusion Matrix: TP={TP}, FP={FP}, TN={TN}, FN={FN}")
    
    # Store results
    results = {
        "scenario": scenario_name,
        "accuracy": accuracy,
        "tss": tss,
        "hss": hss,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "avg_precision": ap,
        "tp": TP,
        "tn": TN,
        "fp": FP,
        "fn": FN,
        "y_true": y_test,
        "y_pred": y_pred.flatten(),
        "y_pred_proba": y_pred_proba.flatten()
    }
    
    return results

def plot_roc_curve(results, scenario_name):
    """Plot ROC curve for solar flare prediction"""
    plt.figure(figsize=(10, 8))
    
    y_true = results["y_true"]
    y_score = results["y_pred_proba"]
    
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, lw=2, color='darkorange',
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    
    # Plot diagonal
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    # Add labels and title
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {scenario_name}')
    plt.legend(loc="lower right")
    
    # Save the figure
    plt.savefig(f'solar_analysis/roc_curve_{scenario_name}.png')
    plt.close()

def plot_pr_curve(results, scenario_name):
    """Plot precision-recall curve for solar flare prediction"""
    plt.figure(figsize=(10, 8))
    
    y_true = results["y_true"]
    y_score = results["y_pred_proba"]
    
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    
    plt.plot(recall, precision, lw=2, color='darkgreen',
             label=f'PR curve (AP = {ap:.2f})')
    
    # Add labels and title
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for {scenario_name}')
    plt.legend(loc="lower left")
    
    # Save the figure
    plt.savefig(f'solar_analysis/pr_curve_{scenario_name}.png')
    plt.close()

def create_confusion_matrix(results, scenario_name):
    """Create a visualization of the confusion matrix"""
    plt.figure(figsize=(8, 6))
    
    TP = results["tp"]
    TN = results["tn"]
    FP = results["fp"]
    FN = results["fn"]
    
    # Create a 2x2 confusion matrix
    cm = np.array([[TN, FP], [FN, TP]])
    
    # Plot the confusion matrix
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for {scenario_name}')
    plt.colorbar()
    
    # Add labels
    classes = ['Negative', 'Positive']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f'solar_analysis/confusion_matrix_{scenario_name}.png')
    plt.close()

def create_summary_table(results, scenario_name):
    """Create a summary table of results as a CSV file"""
    # Create a DataFrame with results
    summary = pd.DataFrame({
        "Metric": [
            "Accuracy", "TSS", "HSS", "Precision", "Recall", "F1",
            "ROC AUC", "Avg Precision", "TP", "TN", "FP", "FN"
        ],
        "Value": [
            results["accuracy"], results["tss"], results["hss"],
            results["precision"], results["recall"], results["f1"],
            results["roc_auc"], results["avg_precision"],
            results["tp"], results["tn"], results["fp"], results["fn"]
        ]
    })
    
    # Save to CSV
    summary.to_csv(f'solar_analysis/summary_{scenario_name}.csv', index=False)
    
    # Also print a formatted table
    print("\n=== Summary of Results ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

if __name__ == "__main__":
    # Make sure the output directory exists
    os.makedirs("solar_analysis", exist_ok=True)
    
    # Run the evaluation
    main() 