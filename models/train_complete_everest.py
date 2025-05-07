#!/usr/bin/env python
"""
Train the complete EVEREST model with all components

This script creates and trains the complete EVEREST model with all components:
1. Performer (FAVOR+) linear attention
2. Retentive mechanism
3. Class-Balanced Focal loss
4. Multi-scale tokenization
5. Diffusion oversampling
6. Evidential uncertainty
7. EVT tail modeling
8. Conformal calibration

Usage:
    python train_complete_everest.py --flare M5 --window 24
"""

import argparse
import numpy as np
import tensorflow as tf
import os
import json
from datetime import datetime

# Import the complete EVEREST model and utils
from complete_everest import EVEREST
from utils import get_training_data, get_testing_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

def setup_gpu():
    """Configure GPU settings for better training stability."""
    # Configure memory growth to avoid OOM errors
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Found {len(gpus)} GPU(s): {gpus}")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU memory growth enabled")
        except Exception as e:
            print(f"Error configuring GPU: {e}")
    else:
        print("No GPU found, using CPU")
    
    # Set random seed for reproducibility
    tf.keras.utils.set_random_seed(42)
    np.random.seed(42)

def train_everest(flare_class, time_window, use_all_components=True, epochs=50, batch_size=64):
    """
    Train the complete EVEREST model.
    
    Args:
        flare_class: Flare class to predict (e.g., "M5", "M", "C")
        time_window: Time window in hours (e.g., 24, 48, 72)
        use_all_components: Whether to use all components or a simpler model
        epochs: Number of epochs to train
        batch_size: Batch size for training
        
    Returns:
        Trained model and evaluation metrics
    """
    # Setup GPU
    setup_gpu()
    
    # Load training data
    print(f"Loading training data for {flare_class} flares with {time_window}h window...")
    X, y_raw, df = get_training_data(time_window, flare_class, return_df=True)
    
    # Convert labels to binary format if needed
    if isinstance(y_raw, list) or (hasattr(y_raw, 'dtype') and y_raw.dtype.kind in 'OSU'):
        # String labels: 'P' = positive, others = negative
        y = np.array([1 if label == 'P' else 0 for label in y_raw], dtype=np.int32)
    else:
        # Already numerical
        y = y_raw.astype(np.int32)
    
    # Create one-hot encoded labels
    y_onehot = tf.keras.utils.to_categorical(y, 2)
    
    # Split data chronologically (by index) for validation
    # Use the last 10% for validation to respect temporal order
    train_size = int(0.9 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y_onehot[:train_size], y_onehot[train_size:]
    y_train_raw, y_val_raw = y[:train_size], y[train_size:]
    
    # Count class instances
    pos_train = np.sum(y_train_raw)
    neg_train = len(y_train_raw) - pos_train
    pos_val = np.sum(y_val_raw)
    neg_val = len(y_val_raw) - pos_val
    
    print(f"Train set: {len(X_train)} samples ({neg_train} negative, {pos_train} positive)")
    print(f"Validation set: {len(X_val)} samples ({neg_val} negative, {pos_val} positive)")
    print(f"Class imbalance ratio: {neg_train/pos_train:.2f}")
    
    # Create class counts for focal loss
    class_counts = [neg_train, pos_train]
    
    # Create and configure the model
    model = EVEREST(
        use_evidential=use_all_components,
        use_evt=use_all_components,
        use_retentive=use_all_components,
        use_multi_scale=use_all_components,
        early_stopping_patience=10
    )
    
    # Build model
    input_shape = X_train.shape[1:]
    model.build_base_model(
        input_shape=input_shape,
        embed_dim=128,
        num_heads=4,
        ff_dim=256,
        n_blocks=4,  # Use 4 transformer blocks (smaller but effective)
        dropout=0.4,  # Higher dropout for better regularization
        causal=False  # Non-causal attention for this task
    )
    
    # Compile with class-balanced focal loss
    model.compile(lr=2e-4, class_counts=class_counts)
    
    # Calculate class weights
    weight_ratio = min(5.0, np.sqrt(neg_train / pos_train))
    class_weight = {
        0: 1.0,
        1: weight_ratio
    }
    print(f"Using class weights: {class_weight}")
    
    # Create callbacks
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=f"logs/everest/{flare_class}_{time_window}",
        histogram_freq=1,
        update_freq='epoch'
    )
    
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        f"models/everest_{flare_class}_{time_window}.h5",
        monitor='val_softmax_dense_tss' if use_all_components else 'val_tss',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    
    # Train model with diffusion oversampling
    print(f"Training EVEREST model for {epochs} epochs...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=[tensorboard_callback, checkpoint_callback],
        verbose=2,
        use_diffusion=use_all_components,  # Use diffusion oversampling if all components are enabled
        diffusion_ratio=0.25  # Generate 25% synthetic samples
    )
    
    # Calibrate using conformal prediction
    if use_all_components:
        print("Calibrating model with conformal prediction...")
        calibration_threshold = model.calibrate(X_val, y_val, alpha=0.1, mc_samples=20)
        print(f"Calibration threshold: {calibration_threshold:.4f}")
    
    # Evaluate on validation set
    print("Evaluating on validation set...")
    val_results = evaluate_model(model, X_val, y_val_raw, use_all_components)
    
    # Save model
    model_path = f"models/everest_{flare_class}_{time_window}"
    model.save_weights(model_path, flare_class=flare_class)
    
    # Save evaluation results
    results = {
        "model": "EVEREST",
        "flare_class": flare_class,
        "time_window": time_window,
        "timestamp": datetime.now().isoformat(),
        "validation": val_results,
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "class_weight": class_weight,
            "use_all_components": use_all_components,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "class_counts": class_counts
        }
    }
    
    # Save results as JSON
    results_path = os.path.join(model_path, "training_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Training completed. Model saved to {model_path}")
    return model, results

def evaluate_model(model, X, y_true, use_all_components=True):
    """
    Evaluate the model on the given data.
    
    Args:
        model: Trained EVEREST model
        X: Input data
        y_true: True labels (binary format)
        use_all_components: Whether the model uses all components
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Make predictions with uncertainty
    if use_all_components:
        results = model.predict_with_uncertainty(X, mc_passes=20)
        
        # Extract probabilities and predictions
        probs = results['probabilities']
        uncertainty = results['mc_uncertainty']
        
        # Get evidential and EVT metrics if available
        evidential_metrics = {}
        if 'evidential' in results and results['evidential'] is not None:
            ev = results['evidential']
            evidential_metrics = {
                "epistemic_uncertainty_mean": float(np.mean(ev['epistemic'])),
                "aleatoric_uncertainty_mean": float(np.mean(ev['aleatoric'])),
                "total_uncertainty_mean": float(np.mean(ev['total']))
            }
            
        evt_metrics = {}
        if 'evt' in results and results['evt'] is not None:
            evt = results['evt']
            evt_metrics = {
                "shape_mean": float(np.mean(evt['shape'])),
                "shape_std": float(np.std(evt['shape'])),
                "scale_mean": float(np.mean(evt['scale'])),
                "scale_std": float(np.std(evt['scale']))
            }
            
        conformal_metrics = {}
        if 'conformal' in results and results['conformal'] is not None:
            conf = results['conformal']
            # Calculate conformal metrics
            conf_preds = conf['sets'].astype(int)
            tn_conf, fp_conf, fn_conf, tp_conf = confusion_matrix(y_true, conf_preds).ravel()
            
            conformal_metrics = {
                "conformal_precision": float(precision_score(y_true, conf_preds, zero_division=0)),
                "conformal_recall": float(recall_score(y_true, conf_preds, zero_division=0)),
                "conformal_f1": float(f1_score(y_true, conf_preds, zero_division=0)),
                "conformal_tss": float((tp_conf/(tp_conf+fn_conf+1e-8)) + (tn_conf/(tn_conf+fp_conf+1e-8)) - 1),
                "conformal_set_size": float(np.mean(conf['sets'])),
            }
    else:
        # Standard prediction without uncertainty
        probs = model.predict_proba(X)
        uncertainty = None
    
    # Find best threshold using TSS
    thresholds = np.linspace(0.1, 0.9, 17)
    best_tss = -1
    best_threshold = 0.5
    
    # Calculate metrics for different thresholds
    threshold_metrics = []
    
    for threshold in thresholds:
        preds = (probs > threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        tss = (tp/(tp+fn+1e-8)) + (tn/(tn+fp+1e-8)) - 1
        
        if tss > best_tss:
            best_tss = tss
            best_threshold = threshold
            
        threshold_metrics.append({
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(accuracy),
            "tss": float(tss),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn)
        })
    
    # Calculate metrics at best threshold
    best_preds = (probs > best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, best_preds).ravel()
    
    best_metrics = {
        "threshold": float(best_threshold),
        "precision": float(tp / (tp + fp + 1e-8)),
        "recall": float(tp / (tp + fn + 1e-8)),
        "f1": float(2 * tp / (2 * tp + fp + fn + 1e-8)),
        "accuracy": float((tp + tn) / (tp + tn + fp + fn)),
        "tss": float(best_tss),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn)
    }
    
    # Build full results
    results = {
        "best_metrics": best_metrics,
        "threshold_metrics": threshold_metrics
    }
    
    # Add uncertainty metrics if available
    if uncertainty is not None:
        results["uncertainty"] = {
            "mean": float(np.mean(uncertainty)),
            "std": float(np.std(uncertainty)),
            "min": float(np.min(uncertainty)),
            "max": float(np.max(uncertainty))
        }
        
    # Add evidential and EVT metrics if available
    if use_all_components:
        if evidential_metrics:
            results["evidential"] = evidential_metrics
        if evt_metrics:
            results["evt"] = evt_metrics
        if conformal_metrics:
            results["conformal"] = conformal_metrics
    
    # Print some key metrics
    print(f"Best TSS: {best_metrics['tss']:.4f} at threshold {best_threshold:.2f}")
    print(f"Precision: {best_metrics['precision']:.4f}, Recall: {best_metrics['recall']:.4f}")
    print(f"F1 Score: {best_metrics['f1']:.4f}, Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    
    if uncertainty is not None:
        print(f"Mean prediction uncertainty: {np.mean(uncertainty):.4f}")
    
    return results

def test_model(model, flare_class, time_window, use_all_components=True):
    """
    Test the model on the test set.
    
    Args:
        model: Trained EVEREST model
        flare_class: Flare class to predict
        time_window: Time window in hours
        use_all_components: Whether the model uses all components
        
    Returns:
        Test evaluation metrics
    """
    # Load test data
    print(f"Loading test data for {flare_class} flares with {time_window}h window...")
    X_test, y_test_raw = get_testing_data(time_window, flare_class)
    
    # Convert labels to binary format if needed
    if isinstance(y_test_raw, list) or (hasattr(y_test_raw, 'dtype') and y_test_raw.dtype.kind in 'OSU'):
        # String labels: 'P' = positive, others = negative
        y_test = np.array([1 if label == 'P' else 0 for label in y_test_raw], dtype=np.int32)
    else:
        # Already numerical
        y_test = y_test_raw.astype(np.int32)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = evaluate_model(model, X_test, y_test, use_all_components)
    
    # Save test results
    model_path = f"models/everest_{flare_class}_{time_window}"
    results_path = os.path.join(model_path, "test_results.json")
    
    with open(results_path, "w") as f:
        json.dump(test_results, f, indent=2)
    
    print(f"Test evaluation completed. Results saved to {results_path}")
    return test_results

def load_model(flare_class, time_window, use_all_components=True):
    """
    Load a previously trained EVEREST model.
    
    Args:
        flare_class: Flare class
        time_window: Time window in hours
        use_all_components: Whether to load a model with all components
        
    Returns:
        Loaded model
    """
    # Get sample data to determine input shape
    X, _ = get_testing_data(time_window, flare_class)
    input_shape = X.shape[1:]
    
    # Create model with same configuration
    model = EVEREST(
        use_evidential=use_all_components,
        use_evt=use_all_components,
        use_retentive=use_all_components,
        use_multi_scale=use_all_components
    )
    
    # Build model with same architecture
    model.build_base_model(input_shape)
    model.compile()
    
    # Load weights
    model_path = f"models/everest_{flare_class}_{time_window}"
    model.load_weights(model_path, flare_class=flare_class)
    
    return model

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train and evaluate the complete EVEREST model")
    parser.add_argument("--flare", default="M5", help="Flare class (C, M, M5)")
    parser.add_argument("--window", default="24", help="Time window in hours (24, 48, 72)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--simple", action="store_true", help="Use simpler model without all components")
    parser.add_argument("--test-only", action="store_true", help="Only run testing on a pre-trained model")
    
    args = parser.parse_args()
    
    # Process arguments
    flare_class = args.flare
    time_window = args.window
    epochs = args.epochs
    batch_size = args.batch_size
    use_all_components = not args.simple
    test_only = args.test_only
    
    if test_only:
        # Load model and run testing only
        print(f"Loading pre-trained model for {flare_class} flares with {time_window}h window...")
        model = load_model(flare_class, time_window, use_all_components)
        test_results = test_model(model, flare_class, time_window, use_all_components)
    else:
        # Train model and run testing
        print(f"Training and testing model for {flare_class} flares with {time_window}h window...")
        model, train_results = train_everest(
            flare_class, 
            time_window, 
            use_all_components, 
            epochs, 
            batch_size
        )
        test_results = test_model(model, flare_class, time_window, use_all_components)
    
    print("Done!") 