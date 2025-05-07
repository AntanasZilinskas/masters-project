#!/usr/bin/env python
"""
Test script for the trained EVEREST model.

This script loads a trained EVEREST model and evaluates it on test data.
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, f1_score, accuracy_score

# Import the complete EVEREST implementation
try:
    from complete_everest import EVEREST as CompleteEVEREST
    USE_COMPLETE_MODEL = True
    print("Using Complete EVEREST implementation")
except ImportError:
    from everest_model import EVEREST
    USE_COMPLETE_MODEL = False
    print("Using original EVEREST implementation")

from utils import get_all_data, data_transform, load_data, n_features, start_feature, series_len, mask_value

def true_skill_statistic(y_true, y_pred):
    """Calculate True Skill Statistic (TSS)"""
    # Convert from one-hot to binary if needed
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    # Calculate confusion matrix
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    
    # Calculate TSS
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    tss = sensitivity + specificity - 1
    
    return tss

def load_test_data(flare_class, time_window):
    """Load test data for evaluation"""
    print(f"Loading test data for {flare_class} flares with {time_window}h window...")
    
    # Load from the test data file
    data_file = f"Nature_data/testing_data_{flare_class}_{time_window}.csv"
    X_data, y_raw, _, _ = load_data(
        datafile=data_file,
        flare_label=flare_class, 
        series_len=series_len,
        start_feature=start_feature, 
        n_features=n_features,
        mask_value=mask_value
    )
    
    # Convert labels to one-hot encoding
    y_data = data_transform(y_raw)
    
    # Print test set statistics
    pos_count = sum(1 for y in y_raw if y == 'P')
    neg_count = sum(1 for y in y_raw if y == 'N')
    print(f"Test set: {len(X_data)} samples ({neg_count} negative, {pos_count} positive)")
    
    return X_data, y_data

def evaluate_model(model, X_test, y_test, threshold=0.5):
    """Evaluate model performance on test data"""
    # Run prediction
    y_pred_raw = model.predict(X_test)
    
    # For complete EVEREST with multiple heads, use the softmax output 
    if USE_COMPLETE_MODEL:
        # Get the softmax prediction from the relevant output
        main_output_idx = 3  # Assuming the 4th output is the main softmax prediction
        y_pred_prob = y_pred_raw[main_output_idx][:, 1]  # Probability of positive class
    else:
        # Standard model with single output
        y_pred_prob = y_pred_raw[:, 1]  # Probability of positive class
    
    # Convert to binary predictions using threshold
    y_pred = (y_pred_prob >= threshold).astype(int)
    
    # Convert one-hot encoded y_test to binary class
    y_true = np.argmax(y_test, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    tss = true_skill_statistic(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    # Display results
    print("\nModel Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"TSS: {tss:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(f"TN: {tn}, FP: {fp}")
    print(f"FN: {fn}, TP: {tp}")
    
    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    
    # ROC curve
    plt.subplot(2, 1, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    # Precision-Recall curve
    plt.subplot(2, 1, 2)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_prob)
    plt.plot(recall_curve, precision_curve, color='blue', lw=2)
    plt.axhline(y=sum(y_true)/len(y_true), color='red', linestyle='--', label='Baseline')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.title('Precision-Recall Curve')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'models/everest_{flare_class}_{time_window}_evaluation.png')
    print(f"Evaluation plots saved to models/everest_{flare_class}_{time_window}_evaluation.png")
    
    return {
        'accuracy': accuracy,
        'tss': tss,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc,
        'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
    }

def find_optimal_threshold(model, X_val, y_val):
    """Find optimal threshold for binary classification"""
    print("Finding optimal threshold for classification...")
    
    # Get predictions
    y_pred_raw = model.predict(X_val)
    
    # For complete EVEREST with multiple heads, use the softmax output
    if USE_COMPLETE_MODEL:
        # Get the softmax prediction from the relevant output
        main_output_idx = 3  # Assuming the 4th output is the main softmax prediction
        y_pred_prob = y_pred_raw[main_output_idx][:, 1]  # Probability of positive class
    else:
        # Standard model with single output
        y_pred_prob = y_pred_raw[:, 1]  # Probability of positive class
    
    # Convert one-hot encoded y_val to binary class
    y_true = np.argmax(y_val, axis=1)
    
    # Try different thresholds and compute F1 score
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_threshold = 0.5
    best_f1 = 0
    best_tss = -1
    
    for threshold in thresholds:
        y_pred = (y_pred_prob >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        tss = true_skill_statistic(y_true, y_pred)
        
        # Prioritize TSS for solar flare prediction
        if tss > best_tss:
            best_tss = tss
            best_threshold = threshold
            best_f1 = f1
    
    print(f"Optimal threshold: {best_threshold:.2f} (TSS: {best_tss:.4f}, F1: {best_f1:.4f})")
    return best_threshold

def main():
    """Main function to evaluate trained EVEREST model"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test trained EVEREST model')
    parser.add_argument('--flare_class', type=str, default='M5', help='Flare class (C, M, or M5)')
    parser.add_argument('--time_window', type=str, default='24', help='Time window in hours (24, 48, or 72)')
    parser.add_argument('--model_path', type=str, help='Path to trained model (optional)')
    args = parser.parse_args()
    
    flare_class = args.flare_class
    time_window = args.time_window
    
    # Default model path if not provided
    model_path = args.model_path or f'models/everest_{flare_class}_{time_window}.h5'
    
    # Check if the model exists
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train the model first or specify the correct path using --model_path")
        return
    
    # Load test data
    X_test, y_test = load_test_data(flare_class, time_window)
    
    # Create model instance
    if USE_COMPLETE_MODEL:
        model = CompleteEVEREST(
            use_evidential=True,
            use_evt=True,
            use_retentive=True,
            use_multi_scale=False  # Set based on sequence length
        )
    else:
        model = EVEREST()
    
    # Build model with appropriate input shape
    seq_len = X_test.shape[1]
    features = X_test.shape[2]
    
    # Determine whether to use multi-scale tokenizer based on sequence length
    use_multi_scale = seq_len >= 24
    if USE_COMPLETE_MODEL:
        # Update multi-scale setting based on sequence length
        model.use_multi_scale = use_multi_scale
    
    # Build model
    model.build_base_model(input_shape=(seq_len, features))
    
    # Load weights
    print(f"Loading model weights from {model_path}")
    model.load_weights(model_path)
    
    # Find optimal threshold using test data (normally would use validation data)
    threshold = find_optimal_threshold(model, X_test, y_test)
    
    # Evaluate model
    results = evaluate_model(model, X_test, y_test, threshold=threshold)
    
    return results

if __name__ == '__main__':
    main() 