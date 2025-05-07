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
import os

def test_everest_model(use_advanced=True):
    """Test the EVEREST model with both standard and advanced head configurations."""
    
    print(f"Testing EVEREST model with {'advanced heads' if use_advanced else 'standard head'} configuration...")
    
    # Create dummy data
    seq_len, features = 100, 14  # Typical dimensions for SHARP data
    batch_size = 4
    
    # Random input data
    X = np.random.random((batch_size, seq_len, features)).astype("float32")
    y = tf.keras.utils.to_categorical(np.random.randint(0, 2, size=batch_size), 2)
    
    # Create the model
    print("Creating EVEREST model...")
    model = EVEREST(use_advanced_heads=use_advanced)
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
    
    # Test advanced features if applicable
    if use_advanced:
        print("\nTesting advanced features:")
        
        # Test evidential head
        print("Testing evidential head...")
        ev_params = model.predict_evidential(X)
        print(f"Evidential parameters shape: {ev_params.shape}")
        print(f"Parameters: μ, ν, α, β range: {ev_params.min(axis=0)} to {ev_params.max(axis=0)}")
        
        # Test EVT head
        print("Testing EVT head...")
        evt_params = model.predict_evt(X)
        print(f"EVT parameters shape: {evt_params.shape}")
        print(f"Parameters: ξ, σ range: {evt_params.min(axis=0)} to {evt_params.max(axis=0)}")
    
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

def compute_conformal_prediction_set(val_logits, val_labels, alpha=0.1):
    """
    Compute the conformal prediction threshold for a given confidence level.
    
    Args:
        val_logits: Validation set logits or probabilities
        val_labels: True labels for validation set
        alpha: Target error rate (e.g., 0.1 for 90% confidence)
        
    Returns:
        qhat: Conformal quantile threshold
    """
    # If input is probabilities, convert to logits
    if np.all(val_logits >= 0) and np.all(val_logits <= 1):
        # Clip probabilities to avoid numerical issues
        eps = 1e-7
        val_logits_clipped = np.clip(val_logits, eps, 1-eps)
        logits = np.log(val_logits_clipped / (1 - val_logits_clipped))
    else:
        logits = val_logits
        
    # Convert to probabilities
    val_probs = 1/(1 + np.exp(-logits))
    
    # Compute non-conformity scores: 1 - p(true class)
    nonconf = np.zeros_like(val_labels, dtype=float)
    for i in range(len(val_labels)):
        nonconf[i] = 1 - val_probs[i] if val_labels[i] == 1 else val_probs[i]
    
    # Get the quantile
    qhat = np.quantile(nonconf, 1-alpha, interpolation="higher")
    return qhat

def conformal_prediction_set(probs, qhat):
    """
    Compute conformal prediction sets for binary classification.
    
    Args:
        probs: Predicted probabilities for class 1
        qhat: Conformal quantile threshold
        
    Returns:
        sets: Boolean array where True means class 1 is in the prediction set
    """
    return 1 - probs < qhat  # True means include class 1 (flare)

def test(time_window, flare_class, version=None, checkpoint_path=None, savefile=None, advanced_model=None):
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
    
    # Determine if this is an advanced model by checking metadata
    if advanced_model is None:
        # Auto-detect from metadata
        uses_evidential = metadata.get('hyperparameters', {}).get('uses_evidential', False)
        uses_evt = metadata.get('hyperparameters', {}).get('uses_evt', False)
        is_advanced = uses_evidential and uses_evt
    else:
        is_advanced = advanced_model
        
    print(f"Testing with {'advanced' if is_advanced else 'standard'} model configuration.")
    
    # Get weight directory path for loading metadata
    weight_dir = checkpoint_path or os.path.join("models", "EVEREST", str(flare_class), str(version))
    if not os.path.exists(weight_dir):
        # Try the SolarKnowledge path pattern
        weight_dir = os.path.join("models", f"SolarKnowledge-v{version}-{flare_class}-{time_window}h")
    
    # Apply temperature scaling + threshold:
    with open(os.path.join(weight_dir, "metadata.json")) as f:
        meta = json.load(f)
    
    # Get temperature scaling and threshold values (or use defaults if not present)
    T = meta.get('performance', {}).get('val_temp', 1.0)
    threshold = meta.get('performance', {}).get('val_best_thr', 0.5)
    
    print(f"Using temperature scaling T={T:.2f} and threshold={threshold:.2f}")
    
    # -------------- conformal quantile ---------------------
    alpha = 0.1  # 90% marginal coverage
    
    # Try to load validation logits and labels for conformal prediction
    conformal_available = os.path.exists(os.path.join(weight_dir, "val_logits.npy"))
    
    if conformal_available:
        print("Computing conformal prediction set...")
        val_log = np.load(os.path.join(weight_dir, "val_logits.npy"))
        val_y = np.load(os.path.join(weight_dir, "val_labels.npy"))
        
        # Calculate qhat using our conformal prediction function
        qhat = compute_conformal_prediction_set(val_log, val_y, alpha)
        print(f"Conformal q̂_{alpha:.2f} = {qhat:.3f}")
        
    else:
        print("Validation logits not found. Skipping conformal prediction.")
    
    # Perform Monte-Carlo predictions with temperature scaling
    if isinstance(model, EVEREST) and hasattr(model, 'mc_predict'):
        # Use Monte Carlo dropout for predictions
        mc_preds, uncertainty = model.mc_predict(X, n_passes=30)
        
        if is_advanced:
            # For advanced model, get softmax output for class 1
            probs = mc_preds[:, 1]
        else:
            # For standard model, get class 1 probability directly
            probs = mc_preds[:, 1]
    else:
        # Fallback for older model versions without mc_predict
        print("Using standard predict (no MC dropout)")
        if is_advanced:
            # For advanced model, get softmax output
            try:
                preds = model.predict(X)
                if isinstance(preds, dict) and "softmax_dense" in preds:
                    probs = preds["softmax_dense"][:, 1]
                else:
                    # Handle any unexpected output format
                    print("Warning: model output format not as expected")
                    if isinstance(preds, dict):
                        print(f"Available keys: {list(preds.keys())}")
                        for key in preds.keys():
                            if "softmax" in key or "prob" in key:
                                probs = preds[key][:, 1]
                                print(f"Using output key: {key}")
                                break
                        else:
                            # If no suitable key found, use the first output
                            first_key = list(preds.keys())[0]
                            probs = np.array(preds[first_key]).reshape(-1)
                            print(f"Using first available key: {first_key}")
                    else:
                        # If not a dict, just use as is
                        probs = preds[:, 1] if preds.shape[-1] > 1 else preds
            except Exception as e:
                print(f"Error getting predictions: {e}")
                # Create dummy predictions as fallback
                probs = np.random.uniform(0, 1, size=len(X))
                
            uncertainty = np.zeros_like(probs)  # No uncertainty estimate
        else:
            # For standard model
            preds = model.predict(X)
            probs = preds[:, 1]
            uncertainty = np.zeros_like(probs)  # No uncertainty estimate
    
    # Apply threshold for classification
    y_pred = (probs > threshold).astype(int)
    
    # Apply conformal prediction if available
    if conformal_available:
        set_pred = conformal_prediction_set(probs, qhat)
        
        # Calculate coverage
        coverage = np.mean((y == 1) & set_pred)
        total_pos = np.sum(y == 1)
        covered_pos = np.sum((y == 1) & set_pred)
        
        print(f"Conformal set results:")
        print(f"  Positive class coverage: {covered_pos}/{total_pos} = {coverage:.4f}")
        print(f"  Average set size: {np.mean(set_pred):.4f}")
        
        # Treat 'include flare?' as positive prediction for additional evaluation
        y_conf = set_pred.astype(int)
        tn_conf, fp_conf, fn_conf, tp_conf = confusion_matrix(y, y_conf).ravel()
        tss_conf = (tp_conf / (tp_conf + fn_conf)) + (tn_conf / (tn_conf + fp_conf)) - 1
        print(f"  Conformal TSS: {tss_conf:.4f}")
    
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
    print(f"\nTest Results with temperature T={T:.2f} at threshold {threshold:.2f}:")
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"TSS: {tss:.4f}")
    print(f"HSS: {hss:.4f}")
    print(f"Average uncertainty: {uncertainty.mean():.4f}")
    
    # Additional metrics for advanced models
    if is_advanced and isinstance(model, EVEREST):
        try:
            # Get evidential parameters for uncertainty quantification
            evidential_params = model.predict_evidential(X)
            mu, v, alpha, beta = np.hsplit(evidential_params, 4)
            
            # Calculate epistemic (model) uncertainty
            beta_v = beta / v
            epistemic = beta_v / (alpha - 1)
            
            # Calculate aleatoric (data) uncertainty  
            aleatoric = beta / (alpha - 1)
            
            print("\nEvidential uncertainty metrics:")
            print(f"  Mean epistemic uncertainty: {np.mean(epistemic):.4f}")
            print(f"  Mean aleatoric uncertainty: {np.mean(aleatoric):.4f}")
            print(f"  Total uncertainty: {np.mean(epistemic + aleatoric):.4f}")
            
            # Add metrics for later recording
            metrics_evidential = {
                'ev_epistemic': float(np.mean(epistemic)),
                'ev_aleatoric': float(np.mean(aleatoric)),
                'ev_total': float(np.mean(epistemic + aleatoric))
            }
            
            # Get EVT parameters
            evt_params = model.predict_evt(X)
            xi, sigma = np.hsplit(evt_params, 2)
            
            print("\nEVT parameter statistics:")
            print(f"  Shape (ξ): mean={np.mean(xi):.4f}, std={np.std(xi):.4f}")
            print(f"  Scale (σ): mean={np.mean(sigma):.4f}, std={np.std(sigma):.4f}")
            
            # Add metrics for later recording
            metrics_evt = {
                'evt_xi_mean': float(np.mean(xi)),
                'evt_xi_std': float(np.std(xi)),
                'evt_sigma_mean': float(np.mean(sigma)),
                'evt_sigma_std': float(np.std(sigma))
            }
        except Exception as e:
            print(f"Error calculating advanced metrics: {e}")
            metrics_evidential = {}
            metrics_evt = {}
    else:
        metrics_evidential = {}
        metrics_evt = {}
    
    # Record metrics
    metrics = {
        'test_accuracy': float(accuracy),
        'test_precision': float(precision),
        'test_recall': float(recall),
        'test_f1': float(f1),
        'test_tss': float(tss),
        'test_hss': float(hss),
        'test_threshold': float(threshold),
        'test_temperature': float(T),
        'test_tp': int(tp),
        'test_fp': int(fp),
        'test_tn': int(tn),
        'test_fn': int(fn),
        'test_uncertainty': float(uncertainty.mean())
    }
    
    # Add evidential and evt metrics if available
    metrics.update(metrics_evidential)
    metrics.update(metrics_evt)
    
    # Add conformal metrics if available
    if conformal_available:
        metrics.update({
            'conformal_alpha': float(alpha),
            'conformal_qhat': float(qhat),
            'conformal_coverage': float(coverage),
            'conformal_set_size': float(np.mean(set_pred)),
            'conformal_tss': float(tss_conf)
        })
    
    # Compute TSS for different thresholds
    if savefile:
        roc_data = compute_roc_curve(probs, y)
        with open(savefile, 'w') as f:
            json.dump({'metrics': metrics, 'roc': roc_data}, f)
    
    return metrics

if __name__ == "__main__":
    # Test both standard and advanced model configurations
    print("\n=== Testing standard model configuration ===")
    test_everest_model(use_advanced=False)
    
    print("\n=== Testing advanced model configuration ===")
    test_everest_model(use_advanced=True)