#!/usr/bin/env python3
"""
Generate simple but insightful Precision-Recall curve for EVEREST.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import EVEREST components
from models.solarknowledge_ret_plus import RETPlusWrapper
from models.utils import get_testing_data

def main():
    """Generate precision-recall curve."""
    
    print("Loading test data...")
    try:
        test_data = get_testing_data(72, 'M5')
        X_test = test_data[0]
        y_test = test_data[1]
        print(f"Test data shape: {X_test.shape}, Labels: {len(y_test)}")
        input_shape = (X_test.shape[1], X_test.shape[2])
    except Exception as e:
        print(f"Error loading test data: {e}")
        return
    
    print("Loading model...")
    try:
        wrapper = RETPlusWrapper(
            input_shape=input_shape,
            early_stopping_patience=10,
            use_attention_bottleneck=True,
            use_evidential=True,
            use_evt=True,
            use_precursor=True,
            compile_model=False
        )
        wrapper.load('../../archive/saved_models/M5_72/run_001/model_weights.pt')
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    print("Generating predictions...")
    try:
        y_proba = wrapper.predict_proba(X_test)
        if y_proba.ndim > 1:
            y_proba = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba.ravel()
        
        print(f"Prediction range: [{y_proba.min():.4f}, {y_proba.max():.4f}]")
        print(f"Mean probability: {y_proba.mean():.4f}")
    except Exception as e:
        print(f"Error generating predictions: {e}")
        return
    
    print("Calculating precision-recall curve...")
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    ap_score = average_precision_score(y_test, y_proba)
    positive_rate = np.mean(y_test)
    
    # Find optimal F1 threshold
    f1_scores = []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        if len(np.unique(y_pred)) > 1:
            f1 = f1_score(y_test, y_pred)
        else:
            f1 = 0.0
        f1_scores.append(f1)
    
    f1_scores = np.array(f1_scores)
    optimal_f1_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_f1_idx]
    optimal_precision = precision[optimal_f1_idx]
    optimal_recall = recall[optimal_f1_idx]
    optimal_f1 = f1_scores[optimal_f1_idx]
    
    print(f"AP Score: {ap_score:.3f}")
    print(f"Baseline (Random): {positive_rate:.3f}")
    print(f"Improvement: {(ap_score/positive_rate - 1)*100:.1f}%")
    print(f"Optimal F1: {optimal_f1:.3f} at threshold {optimal_threshold:.3f}")
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Main precision-recall curve
    ax1.plot(recall, precision, color='#2E86AB', linewidth=3,
            label=f'EVEREST (AP = {ap_score:.3f})')
    
    # Plot optimal F1 point
    ax1.plot(optimal_recall, optimal_precision, 'o', 
            color='#F24236', markersize=10, markeredgecolor='white', 
            markeredgewidth=2, label=f'Optimal F1 = {optimal_f1:.3f}')
    
    # Add baseline
    ax1.axhline(y=positive_rate, color='gray', linestyle='--', alpha=0.7,
              label=f'Random (AP = {positive_rate:.3f})')
    
    ax1.set_xlabel('Recall', fontsize=12)
    ax1.set_ylabel('Precision', fontsize=12)
    ax1.set_title('Precision-Recall: EVEREST M5-72h', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower left', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    
    # F1 score vs threshold
    thresholds_plot = np.linspace(0.01, 0.99, 50)
    f1_plot = []
    for threshold in thresholds_plot:
        y_pred = (y_proba >= threshold).astype(int)
        if len(np.unique(y_pred)) > 1:
            f1 = f1_score(y_test, y_pred)
        else:
            f1 = 0.0
        f1_plot.append(f1)
    
    ax2.plot(thresholds_plot, f1_plot, color='#2E86AB', linewidth=2)
    ax2.plot(optimal_threshold, optimal_f1, 'o', color='#F24236', markersize=8)
    ax2.set_xlabel('Threshold', fontsize=12)
    ax2.set_ylabel('F1-Score', fontsize=12)
    ax2.set_title('F1-Score vs Threshold', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.0])
    
    plt.tight_layout()
    
    # Save figures
    os.makedirs('figs', exist_ok=True)
    plt.savefig('figs/precision_recall_simple.pdf', dpi=300, bbox_inches='tight')
    print("Figure saved: figs/precision_recall_simple.pdf")
    
    # Copy to main figs directory
    try:
        import shutil
        shutil.copy('figs/precision_recall_simple.pdf', '../../figs/precision_recall.pdf')
        print("Figure copied to ../../figs/precision_recall.pdf")
    except Exception as e:
        print(f"Could not copy to main figs: {e}")
    
    plt.show()

if __name__ == "__main__":
    main() 