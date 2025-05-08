#!/usr/bin/env python
"""
Standalone training script for EVEREST model with TSS optimization.
This script simplifies the training process and ensures proper contribution
from all heads for better TSS performance.
"""

import os
import argparse
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from utils import get_training_data, data_transform
from everest_model import EVEREST
from model_tracking import save_model_with_metadata, get_next_version

def train_model(flare_class, time_window, epochs=50, batch_size=256, use_advanced=True):
    """
    Train EVEREST model with direct optimization for TSS.
    
    Args:
        flare_class: Flare class (M5, M, C)
        time_window: Time window in hours (24, 48, 72)
        epochs: Number of training epochs
        batch_size: Batch size for training
        use_advanced: Whether to use advanced heads (Evidential, EVT)
    
    Returns:
        Trained model
    """
    print(f"Training EVEREST model for {flare_class} flares with {time_window}h window")
    
    # Load data
    print(f"Loading data for {flare_class}-class flares with {time_window}h window...")
    X, y_raw = get_training_data(time_window, flare_class)
    y = data_transform(y_raw)
    
    # Create validation set (20% of data, last portion for time series)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"Training set: {len(X_train)} samples, Validation set: {len(X_val)} samples")
    
    # Calculate class weights based on class distribution
    class_counts = np.sum(y_train, axis=0)
    print(f"Class distribution in training set: {class_counts}")
    
    # Calculate weight ratio (more aggressive for M5)
    if flare_class == 'M5':
        weight_ratio = min(10.0, class_counts[0] / class_counts[1] * 2.0)
    elif flare_class == 'M':
        weight_ratio = min(7.0, class_counts[0] / class_counts[1] * 1.5)
    else:
        weight_ratio = min(5.0, class_counts[0] / class_counts[1] * 1.0)
    
    class_weight = {
        0: 1.0,
        1: weight_ratio
    }
    print(f"Using class weights: {class_weight}")
    
    # Create model
    model = EVEREST(use_advanced_heads=use_advanced)
    
    # Build model with stronger regularization
    model.build_base_model(
        input_shape=X_train.shape[1:],
        embed_dim=128,
        num_heads=4,
        ff_dim=256,
        dropout=0.4  # Increased dropout for better generalization
    )
    
    # Compile model with non-zero weights from the start
    model.compile(lr=2e-4)  # Lower learning rate for stability
    
    # Add callbacks
    callbacks = [
        # Early stopping based on validation TSS
        tf.keras.callbacks.EarlyStopping(
            monitor='val_softmax_dense_tss' if use_advanced else 'val_tss',
            mode='max',
            patience=10,
            restore_best_weights=True,
            min_delta=0.005  # Require at least 0.5% improvement
        ),
        # Learning rate reduction
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=5,
            verbose=1,
            min_lr=1e-6
        ),
        # Checkpoint saving
        tf.keras.callbacks.ModelCheckpoint(
            f'models/everest_{flare_class}_{time_window}_best.h5',
            monitor='val_softmax_dense_tss' if use_advanced else 'val_tss',
            mode='max',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Add any callbacks from the model
    if hasattr(model, 'callbacks'):
        callbacks.extend(model.callbacks)
    
    # Create time-jittered augmentation for positive samples
    # This helps with the class imbalance
    aug_X, aug_y = [], []
    for i, label in enumerate(y_train):
        if label[1] == 1:  # Only augment positive samples
            # Add time shifts
            for shift in [-2, -1, 1, 2]:
                rolled = np.roll(X_train[i], shift, axis=0)
                # Add noise to prevent memorization
                rolled = rolled + np.random.normal(0, 0.05, rolled.shape)
                aug_X.append(rolled)
                aug_y.append(label)
    
    if len(aug_X) > 0:
        print(f"Adding {len(aug_X)} augmented positive samples")
        X_train = np.concatenate([X_train, np.array(aug_X)], axis=0)
        y_train = np.concatenate([y_train, np.array(aug_y)], axis=0)
    
    # Prepare targets for model
    if use_advanced:
        print("Preparing targets for multi-output model...")
        y_train_dict = {
            "softmax_dense": y_train,
            "logits_dense": y_train,
            "evidential_head": y_train,
            "evt_head": y_train
        }
        
        y_val_dict = {
            "softmax_dense": y_val,
            "logits_dense": y_val,
            "evidential_head": y_val,
            "evt_head": y_val
        }
        
        # Create sample weights from class weights
        sample_weights = np.ones(len(X_train))
        for i, label in enumerate(y_train):
            # Get class with highest probability
            cls = np.argmax(label)
            # Apply corresponding weight
            sample_weights[i] = class_weight[cls]
        
        print(f"Training multi-output model with {len(X_train)} samples...")
        history = model.fit(
            X_train, 
            y_train_dict,
            validation_data=(X_val, y_val_dict),
            epochs=epochs,
            batch_size=batch_size,
            sample_weight=sample_weights,  # Use sample weights instead of class weights
            callbacks=callbacks,
            verbose=2
        )
    else:
        print(f"Training standard model with {len(X_train)} samples...")
        history = model.fit(
            X_train, 
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=2
        )
    
    # Evaluate model
    print("\nEvaluating model performance...")
    if use_advanced:
        preds = model.model.predict(X_val)
        probs = preds["softmax_dense"][:, 1]
    else:
        preds = model.model.predict(X_val)
        probs = preds[:, 1]
    
    # Find optimal threshold for TSS
    thresholds = np.linspace(0.1, 0.9, 17)
    best_thr, best_tss = 0.5, -1
    best_f1, best_f1_thr = 0, 0.5
    best_precision, best_recall = 0, 0
    
    for thr in thresholds:
        # Apply threshold
        y_pred = (probs > thr).astype(int)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_val.argmax(1), y_pred)
        if cm.size == 4:  # Both classes present
            tn, fp, fn, tp = cm.ravel()
            
            # Calculate metrics
            precision = tp / (tp + fp + 1e-9)
            recall = tp / (tp + fn + 1e-9)
            f1 = 2 * precision * recall / (precision + recall + 1e-9)
            tss = tp/(tp+fn+1e-9) + tn/(tn+fp+1e-9) - 1
            
            # Track best TSS
            if tss > best_tss:
                best_tss = tss
                best_thr = thr
                best_precision = precision
                best_recall = recall
            
            # Track best F1
            if f1 > best_f1:
                best_f1 = f1
                best_f1_thr = thr
    
    print(f"Best TSS: {best_tss:.4f} at threshold {best_thr:.2f}")
    print(f"Precision: {best_precision:.4f}, Recall: {best_recall:.4f}")
    print(f"Best F1: {best_f1:.4f} at threshold {best_f1_thr:.2f}")
    
    # Save model with optimal threshold
    metrics = {
        'val_best_tss': float(best_tss),
        'val_best_thr': float(best_thr),
        'val_precision': float(best_precision),
        'val_recall': float(best_recall),
        'val_best_f1': float(best_f1),
        'val_best_f1_thr': float(best_f1_thr),
    }
    
    # Add metrics from history
    for metric in ['loss', 'softmax_dense_tss', 'softmax_dense_prec', 'softmax_dense_rec']:
        if metric in history.history:
            metrics[f'final_{metric}'] = history.history[metric][-1]
        val_metric = f'val_{metric}'
        if val_metric in history.history:
            metrics[f'final_{val_metric}'] = history.history[val_metric][-1]
    
    # Hyperparameters for metadata
    hyperparams = {
        'learning_rate': 2e-4,
        'batch_size': batch_size,
        'epochs': epochs,
        'dropout': 0.4,
        'embed_dim': 128,
        'ff_dim': 256,
        'num_heads': 4,
        'uses_evidential': use_advanced,
        'uses_evt': use_advanced,
        'uses_focal_loss': False,
        'class_weights': class_weight
    }
    
    # Get version for model tracking
    version = get_next_version(flare_class, time_window)
    
    # Save model with metadata
    description = f"EVEREST model with optimized TSS for {flare_class} flares ({time_window}h window)"
    if use_advanced:
        description = f"EVEREST model with evidential and EVT heads, optimized for TSS, for {flare_class} flares ({time_window}h window)"
    
    print(f"Saving model with version v{version}...")
    save_model_with_metadata(
        model=model,
        metrics=metrics,
        hyperparams=hyperparams,
        history=history,
        version=version,
        flare_class=flare_class,
        time_window=time_window,
        description=description
    )
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EVEREST model with TSS optimization")
    parser.add_argument("--flare", default="M5", help="Flare class (C, M, M5)")
    parser.add_argument("--window", default="24", help="Forecast window in hours (24, 48, 72)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--simple", action="store_true", help="Use simple model without advanced heads")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    tf.keras.utils.set_random_seed(42)
    np.random.seed(42)
    
    # Train model
    train_model(
        args.flare,
        args.window,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_advanced=not args.simple
    ) 