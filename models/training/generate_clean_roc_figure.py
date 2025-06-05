#!/usr/bin/env python3
"""
Generate clean, honest ROC curve analysis for EVEREST only.
No synthetic baseline curves - only real data.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import EVEREST components
from models.solarknowledge_ret_plus import RETPlusWrapper
from models.utils import get_testing_data

def calculate_tss(y_true, y_pred):
    """Calculate True Skill Statistic."""
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        if (tp + fn) == 0 or (tn + fp) == 0:
            return 0.0
        sensitivity = tp / (tp + fn)  # True Positive Rate
        specificity = tn / (tn + fp)  # True Negative Rate
        return sensitivity + specificity - 1
    return 0.0

def main():
    # Load model and data
    print("Loading EVEREST model...")
    model_path = "../../archive/saved_models/M5_72/run_001/model_weights.pt"
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        return
    
    # Get test data first to determine input shape
    print("Loading test data...")
    try:
        test_data = get_testing_data(72, 'M5')  # time_window first, then flare_class
        X_test = test_data[0]  # X_train is returned as first element
        y_test = test_data[1]  # y_train is returned as second element
        print(f"Test data shape: {X_test.shape}, Labels: {len(y_test)}")
        input_shape = (X_test.shape[1], X_test.shape[2])  # (sequence_length, features)
    except Exception as e:
        print(f"Error loading test data: {e}")
        return
    
    # Load model with correct parameters
    wrapper = RETPlusWrapper(
        input_shape=input_shape,
        early_stopping_patience=10,
        use_attention_bottleneck=True,
        use_evidential=True,
        use_evt=True,
        use_precursor=True,
        compile_model=False
    )
    wrapper.load(model_path)
    
    # Get predictions
    print("Generating predictions...")
    try:
        y_proba = wrapper.predict_proba(X_test)
        if y_proba.ndim > 1:
            y_proba = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba.ravel()
        
        print(f"Prediction range: {y_proba.min():.4f} to {y_proba.max():.4f}")
        print(f"Predictions shape: {y_proba.shape}")
    except Exception as e:
        print(f"Error generating predictions: {e}")
        return
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    # Find optimal threshold (maximize TSS)
    tss_scores = []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        tss = calculate_tss(y_test, y_pred)
        tss_scores.append(tss)
    
    optimal_idx = np.argmax(tss_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_tss = tss_scores[optimal_idx]
    optimal_fpr = fpr[optimal_idx]
    optimal_tpr = tpr[optimal_idx]
    
    # Calculate final performance metrics
    y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred_optimal)
    tn, fp, fn, tp = cm.ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nEVEREST M5-72h Performance:")
    print(f"ROC AUC: {roc_auc:.3f}")
    print(f"Optimal TSS: {optimal_tss:.3f} (threshold: {optimal_threshold:.3f})")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")
    
    # Create the main figure
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # --- Panel 1: ROC Curve with TSS Isoclines ---
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, label='Random Classifier')
    
    # Add TSS isoclines
    tss_values = [0.2, 0.4, 0.6, 0.8, 0.9]
    for tss_val in tss_values:
        # TSS = TPR + TNR - 1, so TPR = TSS + FPR for the isocline
        tpr_iso = np.maximum(0, np.minimum(1, tss_val + fpr))
        ax1.plot(fpr, tpr_iso, '--', color='gray', alpha=0.6, linewidth=0.8)
        
        # Add labels at the end of each isocline
        label_x = 0.8
        label_y = tss_val + label_x
        if label_y <= 1.0:
            ax1.text(label_x, label_y, f'TSS = {tss_val}', 
                    fontsize=8, color='gray', alpha=0.8)
    
    # Plot EVEREST ROC curve
    ax1.plot(fpr, tpr, color='#2E86AB', linewidth=2.5, 
             label=f'EVEREST (AUC = {roc_auc:.3f})')
    
    # Mark optimal operating point
    ax1.plot(optimal_fpr, optimal_tpr, 'o', color='#F24236', markersize=8,
             label=f'Optimal Point (TSS = {optimal_tss:.3f})')
    
    ax1.set_xlabel('False Positive Rate', fontsize=11)
    ax1.set_ylabel('True Positive Rate', fontsize=11)
    ax1.set_title('ROC Curve: EVEREST M5-72h', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    
    # --- Panel 2: Performance Summary ---
    ax2.axis('off')
    
    # Performance metrics box
    metrics_text = f"""
    EVEREST M5-72h Performance
    ══════════════════════════
    
    ROC AUC:        {roc_auc:.3f}
    True Skill Statistic: {optimal_tss:.3f}
    
    At Optimal Threshold ({optimal_threshold:.3f}):
    ────────────────────────────────
    Precision:      {precision:.3f}
    Recall:         {recall:.3f}
    F1-Score:       {f1:.3f}
    
    Confusion Matrix:
    ────────────────
    True Pos:   {tp:4d}
    False Pos:  {fp:4d}
    True Neg:   {tn:4d}
    False Neg:  {fn:4d}
    
    Operating Point:
    ────────────────
    TPR = {optimal_tpr:.3f}
    FPR = {optimal_fpr:.3f}
    """
    
    ax2.text(0.05, 0.95, metrics_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs('figs', exist_ok=True)
    plt.savefig('figs/everest_roc_clean.pdf', dpi=300, bbox_inches='tight')
    print(f"\nFigure saved: figs/everest_roc_clean.pdf")
    
    # Also create a simpler single-panel version for main text
    fig2, ax = plt.subplots(1, 1, figsize=(6, 5))
    
    # Simple ROC curve
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, label='Random')
    ax.plot(fpr, tpr, color='#2E86AB', linewidth=2.5, 
            label=f'EVEREST (AUC = {roc_auc:.3f})')
    ax.plot(optimal_fpr, optimal_tpr, 'o', color='#F24236', markersize=8,
            label=f'Optimal (TSS = {optimal_tss:.3f})')
    
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('EVEREST M5-72h ROC Curve', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    plt.tight_layout()
    plt.savefig('figs/everest_roc_simple.pdf', dpi=300, bbox_inches='tight')
    print(f"Simple figure saved: figs/everest_roc_simple.pdf")
    
    plt.show()

if __name__ == "__main__":
    main() 