# -----------------------------
# File: test_everest.py
# -----------------------------
"""
Test script to verify that the modified EVEREST model with custom Performer works.
"""

import numpy as np
import tensorflow as tf
from everest_model import EVEREST
import json
from sklearn.metrics import confusion_matrix
from utils import get_testing_data
from model_tracking import load_model  # Add import for load_model

def test_everest_model():
    print("Testing EVEREST model with custom Performer implementation...")
    
    # Create dummy data
    seq_len, features = 100, 14  # Typical dimensions for SHARP data
    batch_size = 4
    
    # Random input data
    X = np.random.random((batch_size, seq_len, features)).astype("float32")
    y = tf.keras.utils.to_categorical(np.random.randint(0, 2, size=batch_size), 2)
    
    # Create the model
    print("Creating EVEREST model...")
    model = EVEREST()
    model.build_base_model((seq_len, features))
    model.compile()
    
    # Display model summary
    print("\nModel summary:")
    model.model.summary()
    
    # Run a single training epoch
    print("\nRunning a test training epoch...")
    model.fit(X, y, epochs=1)
    
    # Test Monte Carlo dropout prediction
    print("\nTesting Monte Carlo dropout prediction...")
    mean_preds, std_preds = model.mc_predict(X)
    
    print(f"MC prediction shapes: mean={mean_preds.shape}, std={std_preds.shape}")
    print(f"Average uncertainty (std): {std_preds.mean()}")
    
    print("\nTest completed successfully!")
    return True

def compute_roc_curve(y_prob, y_true, n_points=50):
    """
    Compute ROC curve data and derived metrics like TSS at different thresholds.
    
    Args:
        y_prob: Predicted probabilities (output from predict_proba)
        y_true: Ground truth labels (0 or 1)
        n_points: Number of threshold points to evaluate
        
    Returns:
        Dictionary with ROC curve data including thresholds, TSS values,
        and other performance metrics
    """
    thresholds = np.linspace(0.01, 0.99, n_points)
    tss_values = []
    precision_values = []
    recall_values = []
    f1_values = []
    
    for threshold in thresholds:
        y_pred = (y_prob > threshold).astype(int)
        
        # Compute confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        tss = (tp / (tp + fn)) + (tn / (tn + fp)) - 1
        
        # Store values
        tss_values.append(float(tss))
        precision_values.append(float(precision))
        recall_values.append(float(recall))
        f1_values.append(float(f1))
    
    # Find best threshold for TSS
    best_idx = np.argmax(tss_values)
    best_thr = float(thresholds[best_idx])
    best_tss = float(tss_values[best_idx])
    
    return {
        'thresholds': thresholds.tolist(),
        'tss': tss_values,
        'precision': precision_values,
        'recall': recall_values, 
        'f1': f1_values,
        'best_threshold': best_thr,
        'best_tss': best_tss
    }

def test(time_window, flare_class, version=None, checkpoint_path=None, savefile=None):
    """Test a previously trained model on the standard test set."""
    
    # Load testing data
    X, y_raw = get_testing_data(time_window, flare_class)
    
    # Handle different label formats
    if y_raw.dtype == np.int64 or y_raw.dtype == np.int32 or y_raw.dtype == np.float64 or y_raw.dtype == np.float32:
        y = y_raw.astype("int")
    else:
        y = np.array([1 if label == 'P' else 0 for label in y_raw]).astype("int")
    
    print(f"Test class distribution: {np.bincount(y)}")
    
    # Load model
    model, metadata, version = load_model(checkpoint_path, version, flare_class, time_window)
    
    # Get the threshold from metadata if available (or default to 0.5)
    threshold = metadata.get('val_best_thr', 0.5)
    print(f"Using threshold: {threshold:.2f}")
    
    # Get predictions
    probs = model.predict_proba(X)
    y_pred = (probs > threshold).astype(int)
    
    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    tss = (tp / (tp + fn)) + (tn / (tn + fp)) - 1
    hss = 2 * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    
    # Print results
    print(f"\nTest Results at threshold {threshold:.2f}:")
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"TSS: {tss:.4f}")
    print(f"HSS: {hss:.4f}")
    
    # Record metrics
    metrics = {
        'test_accuracy': float(accuracy),
        'test_precision': float(precision),
        'test_recall': float(recall),
        'test_f1': float(f1),
        'test_tss': float(tss),
        'test_hss': float(hss),
        'test_threshold': float(threshold),
        'test_tp': int(tp),
        'test_fp': int(fp),
        'test_tn': int(tn),
        'test_fn': int(fn)
    }
    
    # Compute TSS for different thresholds
    if savefile:
        roc_data = compute_roc_curve(probs, y)
        with open(savefile, 'w') as f:
            json.dump({'metrics': metrics, 'roc': roc_data}, f)
    
    return metrics

if __name__ == "__main__":
    test_everest_model()