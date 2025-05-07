#!/usr/bin/env python
"""
Test script for the trained EVEREST model.

This script loads the trained EVEREST model and evaluates it on 
sample data to verify that the implementation is working correctly.
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Import both implementations for compatibility
try:
    # Try to import the complete implementation first
    from complete_everest import EVEREST as CompleteEVEREST
    USE_COMPLETE_MODEL = True
    print("Using complete EVEREST implementation")
except ImportError:
    # Fall back to original implementation
    from everest_model import EVEREST
    USE_COMPLETE_MODEL = False
    print("Using original EVEREST implementation")

def generate_test_data(n_samples=100, seq_len=144, n_features=14, positive_ratio=0.05):
    """Generate synthetic test data for model evaluation."""
    # Generate features
    X = np.random.normal(0, 1, (n_samples, seq_len, n_features)).astype(np.float32)
    
    # Generate binary labels with imbalance
    y = np.zeros(n_samples)
    y[:int(n_samples * positive_ratio)] = 1
    np.random.shuffle(y)
    
    # Convert to one-hot encoding
    y_onehot = tf.keras.utils.to_categorical(y, 2)
    
    print(f"Generated {n_samples} test samples with {int(y.sum())} positive events")
    
    return X, y_onehot, y

def load_model(flare_class="M5", time_window="24", use_advanced_model=True):
    """Load the trained EVEREST model."""
    # Create a new model with the same architecture
    if USE_COMPLETE_MODEL:
        model = CompleteEVEREST(
            use_evidential=use_advanced_model,
            use_evt=use_advanced_model,
            use_retentive=True,
            use_multi_scale=True
        )
    else:
        model = EVEREST(use_advanced_heads=use_advanced_model)
    
    # Load model weights
    model_path = f"models/everest_{flare_class}_{time_window}"
    if os.path.exists(f"{model_path}.h5"):
        print(f"Loading model from {model_path}.h5")
        # Need to build the model first to match the architecture
        input_shape = (144, 14)  # Default shape
        model.build_base_model(input_shape)
        model.compile()
        
        # Load weights
        model.model.load_weights(f"{model_path}.h5")
    elif os.path.exists(f"{model_path}/model_weights.h5"):
        print(f"Loading model from {model_path}/model_weights.h5")
        # Need to build the model first to match the architecture
        input_shape = (144, 14)  # Default shape
        model.build_base_model(input_shape)
        model.compile()
        
        # Load weights
        model.model.load_weights(f"{model_path}/model_weights.h5")
    else:
        raise FileNotFoundError(f"Model weights not found at {model_path}")
        
    return model

def evaluate_model(model, X, y_true_onehot, y_true):
    """
    Evaluate the model and plot ROC curve.
    
    Args:
        model: Trained EVEREST model
        X: Test data
        y_true_onehot: One-hot encoded true labels
        y_true: Binary true labels
    """
    # Get predictions
    try:
        # For complete model, use predict_with_uncertainty
        if hasattr(model, 'predict_with_uncertainty'):
            results = model.predict_with_uncertainty(X, mc_passes=10)
            y_pred_proba = results['probabilities']
            if 'uncertainty' in results:
                uncertainty = results['uncertainty']
                print(f"Average predictive uncertainty: {uncertainty.mean():.4f}")
        elif hasattr(model, 'mc_predict'):
            # Use Monte Carlo prediction
            mc_preds, uncertainty = model.mc_predict(X, n_passes=10)
            if isinstance(mc_preds, dict) and 'softmax_dense' in mc_preds:
                y_pred_proba = mc_preds['softmax_dense'][:, 1]
            else:
                y_pred_proba = mc_preds[:, 1]
            print(f"Average predictive uncertainty: {uncertainty.mean():.4f}")
        else:
            # Standard predict with probability output
            preds = model.model.predict(X)
            if isinstance(preds, dict) and 'softmax_dense' in preds:
                y_pred_proba = preds['softmax_dense'][:, 1]
            else:
                y_pred_proba = preds[:, 1]
    except Exception as e:
        print(f"Error during prediction: {e}")
        # Try a simpler approach
        preds = model.model.predict(X)
        if isinstance(preds, dict):
            print(f"Model outputs: {list(preds.keys())}")
            y_pred_proba = preds.get('softmax_dense', list(preds.values())[0])
            if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
                y_pred_proba = y_pred_proba[:, 1]
        else:
            y_pred_proba = preds[:, 1] if preds.shape[1] > 1 else preds.flatten()
    
    # Set threshold for binary predictions
    threshold = 0.5
    y_pred = (y_pred_proba > threshold).astype(int)
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate TSS
    tpr = recall  # True Positive Rate
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
    tss = tpr + tnr - 1
    
    # Print results
    print(f"\nEvaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"TSS: {tss:.4f}")
    print(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    # Save the figure
    plt.savefig('everest_roc_curve.png')
    print("ROC curve saved to everest_roc_curve.png")

def main():
    """Main function to test the EVEREST model."""
    # Set random seed for reproducibility
    tf.keras.utils.set_random_seed(42)
    np.random.seed(42)
    
    # Check for GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Found GPU: {gpus[0]}")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except:
                pass
    
    # Generate test data
    X, y_onehot, y = generate_test_data(n_samples=200)
    
    # Load model
    model = load_model(flare_class="M5", time_window="24", use_advanced_model=True)
    
    # Evaluate model
    evaluate_model(model, X, y_onehot, y)
    
    print("Test completed successfully!")

if __name__ == "__main__":
    main() 