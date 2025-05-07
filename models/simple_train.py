#!/usr/bin/env python
"""
Simplified training script for EVEREST model with proper implementation

This script provides backward compatibility with the original simple_train.py script,
but uses the complete EVEREST implementation with all required components.
"""

import os
import argparse
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import datetime
import time

# Import both implementations to provide compatibility
try:
    # Try to import the complete implementation first
    from complete_everest import EVEREST as CompleteEVEREST
    from metrics import CategoricalTSSMetric, ThresholdTuner
    USE_COMPLETE_MODEL = True
    print("Using complete EVEREST implementation")
except ImportError:
    # Fall back to original implementation
    from everest_model import EVEREST
    from simple_fix import fixed_evidential_nll, fixed_evt_loss, patch_everest_model
    # Apply patches to the original model
    patch_everest_model()
    # Import original modules with patched functions
    import evidential_head
    import evt_head
    # Apply fixed implementations
    evidential_head.evidential_nll = fixed_evidential_nll
    evt_head.evt_loss = fixed_evt_loss
    USE_COMPLETE_MODEL = False
    print("Using patched original EVEREST implementation")

# Custom callback for better terminal logging
class DetailedLoggingCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, metrics_to_print=None, print_interval=5):
        super().__init__()
        self.validation_data = validation_data
        self.metrics_to_print = metrics_to_print or ["loss", "softmax_dense_tss", "softmax_dense_prec", "softmax_dense_rec"]
        self.print_interval = print_interval  # Batches between prints
        self.batch_times = []
        self.epoch_start_time = None
        self.batch_start_time = None
        
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        print(f"\n{'='*20} Epoch {epoch+1} {'='*20}")
        print(f"Started at: {datetime.datetime.now().strftime('%H:%M:%S')}")
        
    def on_batch_begin(self, batch, logs=None):
        self.batch_start_time = time.time()
        
    def on_batch_end(self, batch, logs=None):
        if batch % self.print_interval == 0:
            batch_time = time.time() - self.batch_start_time
            self.batch_times.append(batch_time)
            metrics_str = " - ".join([f"{k}: {logs.get(k, 0):.4f}" for k in self.metrics_to_print if k in logs])
            
            # Calculate ETA for epoch
            if len(self.batch_times) > 1:
                avg_batch_time = sum(self.batch_times[-10:]) / min(len(self.batch_times), 10)
                batches_remaining = self.params['steps'] - batch
                eta_seconds = avg_batch_time * batches_remaining
                eta = datetime.timedelta(seconds=int(eta_seconds))
                print(f"Batch {batch}/{self.params['steps']} - {metrics_str} - batch time: {batch_time:.2f}s - ETA: {eta}")
            else:
                print(f"Batch {batch}/{self.params['steps']} - {metrics_str} - batch time: {batch_time:.2f}s")
    
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        metrics_str = " - ".join([f"{k}: {logs.get(k, 0):.4f}" for k in logs.keys()])
        print(f"\nEpoch {epoch+1} completed in {epoch_time:.2f}s")
        print(f"Metrics: {metrics_str}")
        
        # Reset batch timings for next epoch
        self.batch_times = []

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train EVEREST model for solar flare prediction")
    parser.add_argument("--flare", default="M5", help="Flare class (e.g., M5, M, C)")
    parser.add_argument("--window", default="24", help="Time window in hours")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--simple", action="store_true", help="Use simplified model without advanced heads")
    parser.add_argument("--complete", action="store_true", help="Use full implementation with all components")
    parser.add_argument("--log-interval", type=int, default=10, help="Interval (in batches) for printing progress")
    parser.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging")
    
    return parser.parse_args()

def load_data(time_window, flare_class, test_size=0.2, random_state=42):
    """
    Load and preprocess training data exactly matching SolarKnowledge's approach.
    
    Args:
        time_window: Time window in hours (e.g., "24", "48")
        flare_class: Flare class (e.g., "M5", "M", "C")
        test_size: Fraction of data to use for validation
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_val, y_train, y_val
    """
    # Import the same functions as SolarKnowledge
    from utils import get_training_data, data_transform
    from sklearn.model_selection import TimeSeriesSplit
    import numpy as np
    
    # Use the same data loading approach
    print(f"Loading real solar flare data for {flare_class} flares with {time_window}h window...")
    
    # Load raw data with DataFrame - exactly as SolarKnowledge does
    X_train_data, y_train_raw, train_df = get_training_data(time_window, flare_class, return_df=True)
    
    # Transform labels to one-hot encoding - same as SolarKnowledge
    y_train_oh = data_transform(y_train_raw)
    
    # Print data information - mimics SolarKnowledge's reporting
    class_counts = np.sum(y_train_oh, axis=0)
    print(f"Loaded {len(X_train_data)} samples with class distribution: {class_counts}")
    if class_counts[0] > 0 and class_counts[1] > 0:
        print(f"Class imbalance ratio: {class_counts[0]/class_counts[1]:.2f}")
    
    # Use TimeSeriesSplit like SolarKnowledge (with gap=72)
    # This respects the chronological nature of the data
    tscv = TimeSeriesSplit(gap=72, n_splits=5, test_size=int(0.1*len(X_train_data)))
    train_idx, val_idx = list(tscv.split(X_train_data))[-1]  # Use the last fold
    
    X_train, X_val = X_train_data[train_idx], X_train_data[val_idx]
    y_train, y_val = y_train_oh[train_idx], y_train_oh[val_idx]
    
    # Print statistics
    pos_train = np.sum(y_train[:, 1])
    neg_train = len(y_train) - pos_train
    pos_val = np.sum(y_val[:, 1])
    neg_val = len(y_val) - pos_val
    
    print(f"Train set: {len(X_train)} samples ({neg_train} negative, {pos_train} positive)")
    print(f"Validation set: {len(X_val)} samples ({neg_val} negative, {pos_val} positive)")
    
    return X_train, X_val, y_train, y_val

def train_complete_model(X_train, X_val, y_train, y_val, flare_class, time_window, 
                        epochs=50, batch_size=64, use_advanced_model=True, use_calibration=False,
                        log_interval=10, enable_tensorboard=False):
    """
    Train using the complete EVEREST implementation with the same approach as SolarKnowledge.
    
    Args:
        X_train, X_val, y_train, y_val: Training and validation data
        flare_class: Flare class
        time_window: Time window in hours
        epochs: Number of training epochs
        batch_size: Batch size
        use_advanced_model: Whether to use advanced heads
        use_calibration: Whether to use conformal calibration
        log_interval: Interval (in batches) for printing progress
        enable_tensorboard: Whether to enable TensorBoard logging
        
    Returns:
        Trained model
    """
    # Check sequence length and decide whether to use multi-scale tokenizer
    seq_len = X_train.shape[1]
    print(f"Input shape: {X_train.shape}, features: {X_train.shape[-1]}")
    
    # If sequence length is too small, disable multi-scale tokenizer
    use_multi_scale = seq_len >= 24  # Need at least 24 tokens for 3h pooling (18 tokens)
    
    if not use_multi_scale:
        print(f"Sequence length {seq_len} is too small for multi-scale tokenizer. Disabling it.")
    
    # Create model with proper configuration
    model = CompleteEVEREST(
        use_evidential=use_advanced_model,
        use_evt=use_advanced_model,
        use_retentive=True,  # Always use retentive memory
        use_multi_scale=use_multi_scale  # Only use if sequence length is sufficient
    )
    
    # Build and compile model
    model.build_base_model(
        input_shape=X_train.shape[1:],
        embed_dim=128,
        num_heads=4,
        ff_dim=256,
        n_blocks=4,
        dropout=0.4
    )
    
    # Calculate class weights exactly like SolarKnowledge
    class_counts = np.sum(y_train, axis=0)
    n_samples = len(y_train)
    n_classes = len(class_counts)
    
    # Calculate weights inversely proportional to class frequencies
    class_weight = {}
    for i in range(n_classes):
        # More aggressive weighting for very rare classes like M5
        if flare_class == 'M5':
            # Higher weight for the positive class in M5 flares (very rare)
            class_weight[i] = n_samples / (n_classes * class_counts[i]) if i == 1 else 1.0
        elif flare_class == 'M':
            # Moderate weight for M-class flares (rare)
            class_weight[i] = n_samples / (n_classes * class_counts[i]) * 0.8 if i == 1 else 1.0
        else:
            # Lower weight for C-class flares (more common)
            class_weight[i] = n_samples / (n_classes * class_counts[i]) * 0.6 if i == 1 else 1.0
    
    print(f"Using class weights: {class_weight}")
    
    # Convert class weights to sample weights for multi-output compatibility
    sample_weights = np.ones(len(X_train))
    for idx in range(len(X_train)):
        # Get the class index with the highest value
        class_idx = np.argmax(y_train[idx])
        # Apply the corresponding weight
        sample_weights[idx] = class_weight[class_idx]
    
    # Check that positive examples have higher weights
    pos_indices = np.where(y_train[:, 1] > 0.5)[0]
    pos_weight = np.mean(sample_weights[pos_indices]) if len(pos_indices) > 0 else 0
    print(f"Created sample weights with average positive weight: {pos_weight:.2f}")
    
    # Compile model with class counts for focal loss
    model.compile(lr=2e-4, class_counts=[class_counts[0], class_counts[1]])
    
    # Create a list for all callbacks
    callbacks = []
    
    # Create checkpoint callback
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        f'models/everest_{flare_class}_{time_window}.h5',
        monitor='val_softmax_dense_tss' if use_advanced_model else 'val_tss',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    callbacks.append(checkpoint)
    
    # Add early stopping - 5 epochs patience like SolarKnowledge
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_softmax_dense_tss' if use_advanced_model else 'val_tss',
        mode='max',
        patience=5,
        restore_best_weights=True
    )
    callbacks.append(early_stopping)
    
    # Add ReduceLROnPlateau callback exactly like SolarKnowledge
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=3,
        verbose=1,
        min_lr=1e-6
    )
    callbacks.append(reduce_lr)
    
    # Add custom detailed logging callback
    detailed_logging = DetailedLoggingCallback(
        validation_data=(X_val, y_val),
        print_interval=log_interval
    )
    callbacks.append(detailed_logging)
    
    # Add TensorBoard callback if enabled
    if enable_tensorboard:
        log_dir = f"logs/everest_{flare_class}_{time_window}/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_images=True,
            update_freq='epoch'
        )
        callbacks.append(tensorboard_callback)
        print(f"TensorBoard logs will be saved to {log_dir}")
        print("Run 'tensorboard --logdir logs/' to view")
    
    # Add time-jitter augmentation exactly like SolarKnowledge
    aug_X, aug_y = [], []
    for i, label in enumerate(y_train):
        if label[1] == 1:  # Only augment positive (flare) samples
            # Add more shift variations and more noise
            for shift in [-1, 1]:          # ±10‑min time shifts
                rolled = np.roll(X_train[i], shift, axis=0)
                # Add noise to prevent memorization (2% noise)
                rolled += np.random.normal(0, 0.02, rolled.shape)
                aug_X.append(rolled)
                aug_y.append(label)  # Keep the same label
    
    # Add the augmented samples if any were created
    if len(aug_X) > 0:
        print(f"Added {len(aug_X)} time-jittered positive samples")
        X_train_aug = np.concatenate([X_train, np.array(aug_X)], axis=0)
        y_train_aug = np.concatenate([y_train, np.array(aug_y)], axis=0)
        
        # Update sample weights to include augmented samples
        aug_weights = np.ones(len(aug_X)) * class_weight[1]  # All augmented samples are positive
        sample_weights = np.concatenate([sample_weights, aug_weights])
    else:
        X_train_aug = X_train
        y_train_aug = y_train
    
    # Add Gaussian noise augmentation for M and M5 classes like SolarKnowledge
    if flare_class in ['M', 'M5']:
        # Get all positive samples
        pos_indices = np.where(y_train_aug[:, 1] > 0.5)[0]
        pos_samples = X_train_aug[pos_indices]
        
        if len(pos_samples) > 0:
            # Calculate noise scale as 5% of standard deviation (SolarKnowledge uses 5%)
            noise_std = 0.05 * np.std(pos_samples)
            
            # Create noisy copies
            noise = np.random.normal(0, noise_std, pos_samples.shape)
            noisy_samples = pos_samples + noise
            
            # Create labels for noisy samples (all positive)
            noisy_labels = np.array([y_train_aug[idx] for idx in pos_indices])
            
            # Add to training data
            print(f"Added {len(noisy_samples)} Gaussian-noise augmented positive samples")
            X_train_aug = np.concatenate([X_train_aug, noisy_samples], axis=0)
            y_train_aug = np.concatenate([y_train_aug, noisy_labels], axis=0)
            
            # Update sample weights for noisy samples
            noisy_weights = np.ones(len(noisy_samples)) * class_weight[1]
            sample_weights = np.concatenate([sample_weights, noisy_weights])
    
    # Print training summary before starting
    print(f"\n{'='*30}")
    print(f"TRAINING SUMMARY:")
    print(f"{'='*30}")
    print(f"Flare class: {flare_class}, Window: {time_window}h")
    print(f"Model type: {'Advanced' if use_advanced_model else 'Standard'}")
    print(f"Training samples: {len(X_train_aug)} (after augmentation)")
    print(f"Validation samples: {len(X_val)}")
    print(f"Batch size: {batch_size}, Epochs: {epochs}")
    print(f"Class weights: Negative={class_weight[0]}, Positive={class_weight[1]}")
    print(f"{'='*30}\n")
    
    # Train the model with all the SolarKnowledge-matching configurations
    print(f"Training with {len(X_train_aug)} samples after augmentation")
    history = model.fit(
        X_train_aug, y_train_aug,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        sample_weight=sample_weights,  # Use sample weights for multi-output compatibility
        callbacks=callbacks,
        verbose=0,  # Disable default progress bar since we're using our custom callback
        use_diffusion=False  # Disable diffusion to avoid errors
    )
    
    # Save history attribute for later saving with metadata
    model.history = history
    
    # Calibrate model if using advanced heads
    if use_advanced_model and use_calibration:
        try:
            print("Calibrating model with conformal prediction...")
            model.calibrate(X_val, y_val, alpha=0.1, mc_samples=20)
        except Exception as e:
            print(f"Error during calibration: {e}")
            print("Skipping calibration step")
    
    return model

def train_original_model(X_train, X_val, y_train, y_val, flare_class, time_window, 
                        epochs=50, batch_size=64, use_advanced_model=True):
    """
    Train using the patched original EVEREST implementation.
    
    Args:
        X_train, X_val, y_train, y_val: Training and validation data
        flare_class: Flare class
        time_window: Time window in hours
        epochs: Number of training epochs
        batch_size: Batch size
        use_advanced_model: Whether to use advanced heads
        
    Returns:
        Trained model
    """
    # Create model
    input_shape = X_train.shape[1:]
    model = EVEREST(use_advanced_heads=use_advanced_model)
    model.build_base_model(input_shape, dropout=0.5)
    
    # Create callback to tune threshold
    class ThresholdTuningCallback(tf.keras.callbacks.Callback):
        def __init__(self, validation_data, patience=10):
            super().__init__()
            self.validation_data = validation_data
            self.patience = patience
            self.best_threshold = 0.5
            self.best_f1 = 0.0
            self.best_precision = 0.0
            self.best_recall = 0.0
        
        def on_epoch_end(self, epoch, logs=None):
            x_val, y_val = self.validation_data
            preds = self.model.predict(x_val)['softmax_dense'][:, 1]
            
            thresholds = np.linspace(0.2, 0.8, 13)
            best_f1 = 0
            best_threshold = 0.5
            best_precision = 0
            best_recall = 0
            
            for t in thresholds:
                y_pred = (preds >= t).astype(int)
                y_true = np.argmax(y_val, axis=1)
                
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                precision = tp / (tp + fp + 1e-10)
                recall = tp / (tp + fn + 1e-10)
                f1 = 2 * precision * recall / (precision + recall + 1e-10)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = t
                    best_precision = precision
                    best_recall = recall
            
            if best_f1 > self.best_f1:
                self.best_f1 = best_f1
                self.best_threshold = best_threshold
                self.best_precision = best_precision
                self.best_recall = best_recall
                
            logs['best_threshold'] = self.best_threshold
            logs['best_f1'] = self.best_f1
            logs['best_precision'] = self.best_precision
            logs['best_recall'] = self.best_recall
            
            print(f"\nNew best threshold: {self.best_threshold:.4f}, F1: {self.best_f1:.4f}, Precision: {self.best_precision:.4f}, Recall: {self.best_recall:.4f}")
    
    # Add our own simple head linking for the EVT head
    def modified_compile(self, lr=1e-3):
        """Compile with simpler loss functions without the complex capture mechanism."""
        # Define specialized loss functions for each head type
        def softmax_loss(y_true, y_pred):
            # Both y_true and y_pred are (batch_size, 2)
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)
            
            # Standard categorical cross entropy with L2 regularization
            cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
            return tf.reduce_mean(cce)
        
        def logits_loss(y_true, y_pred):
            # y_true is (batch_size, 2), y_pred is (batch_size, 1)
            # Extract just the positive class probability from y_true
            y_true_binary = tf.cast(y_true[:, 1:2], tf.float32)
            # Use binary cross-entropy for logits
            return tf.reduce_mean(tf.keras.losses.binary_crossentropy(
                y_true_binary, y_pred, from_logits=True))
        
        def evidential_loss(y_true, y_pred):
            # y_true is (batch_size, 2), y_pred is (batch_size, 4)
            # Extract just the positive class probability from y_true
            y_true_binary = tf.cast(y_true[:, 1:2], tf.float32)
            # Use the fixed evidential NLL function
            return fixed_evidential_nll(y_true_binary, y_pred)
        
        def evt_loss_fn(y_true, y_pred):
            # y_true is (batch_size, 2), y_pred is (batch_size, 2)
            # Create synthetic extreme values for positive class
            y_true_binary = tf.cast(y_true[:, 1:2], tf.float32)
            synthetic_extremes = y_true_binary * 5.0  # Strong positive signal for positive class
            
            # Use fixed EVT loss with higher threshold
            return fixed_evt_loss(synthetic_extremes, y_pred)
        
        # Compile the model with all loss functions
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss={
                'softmax_dense': softmax_loss,
                'logits_dense': logits_loss,
                **({'evidential_head': evidential_loss} if self.use_advanced_heads else {}),
                **({'evt_head': evt_loss_fn} if self.use_advanced_heads else {})
            },
            loss_weights={
                'softmax_dense': 1.0,
                'logits_dense': 0.2,
                **({'evidential_head': 0.2} if self.use_advanced_heads else {}),
                **({'evt_head': 0.2} if self.use_advanced_heads else {})
            },
            metrics={
                "softmax_dense": ["accuracy", "Precision", "Recall", "AUC"]
            }
        )
    
    # Override the compile method
    model.compile = modified_compile.__get__(model)
    model.compile(lr=2e-4)
    
    # Calculate class weights
    pos = np.sum(y_train[:, 1])
    neg = len(y_train) - pos
    weight_ratio = min(3.0, np.sqrt(neg / pos))
    class_weight = {
        0: 1.0,
        1: weight_ratio
    }
    print(f"Using class weights: {class_weight}")
    
    # Prepare callbacks
    threshold_tuner = ThresholdTuningCallback(validation_data=(X_val, y_val))
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        f'models/everest_{flare_class}_{time_window}.h5',
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=10,
        restore_best_weights=True
    )
    
    # Train model
    print("Starting training with sample weights...")
    history = model.model.fit(
        X_train, 
        prepare_targets(y_train, use_advanced_model),
        validation_data=(X_val, prepare_targets(y_val, use_advanced_model)),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=[early_stopping, checkpoint, threshold_tuner],
        verbose=2
    )
    
    return model

def prepare_targets(y, use_advanced_model):
    """
    Prepare targets for multi-output model.
    
    Args:
        y: One-hot encoded labels
        use_advanced_model: Whether using advanced heads
        
    Returns:
        Prepared targets
    """
    if not use_advanced_model:
        return y
    
    # Create targets for multi-output model
    targets = {
        "softmax_dense": y,
        "logits_dense": y
    }
    
    # Add targets for advanced heads
    if use_advanced_model:
        targets["evidential_head"] = y
        targets["evt_head"] = y
    
    return targets

def evaluate_model(model, X_val, y_val, is_complete_model=True):
    """
    Evaluate the trained model.
    
    Args:
        model: Trained model
        X_val: Validation features
        y_val: Validation labels
        is_complete_model: Whether model is the complete implementation
        
    Returns:
        Dictionary of evaluation metrics
    """
    print("Evaluating model...")
    
    # Get predictions
    if is_complete_model:
        # For complete model, use predict_with_uncertainty
        results = model.predict_with_uncertainty(X_val, mc_passes=10)
        probs = results['probabilities']
    else:
        # For original model, get softmax predictions
        preds = model.model.predict(X_val)
        probs = preds['softmax_dense'][:, 1] if isinstance(preds, dict) else preds[:, 1]
    
    # Convert y_val to binary if one-hot encoded
    y_true = np.argmax(y_val, axis=1) if y_val.shape[-1] > 1 else y_val
    
    # Try different thresholds to find the best one
    thresholds = np.linspace(0.1, 0.9, 17)
    best_metrics = {}
    best_tss = -1
    
    for threshold in thresholds:
        # Get predictions using this threshold
        y_pred = (probs >= threshold).astype(int)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate metrics
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        # Calculate TSS
        sensitivity = tp / (tp + fn + 1e-10)
        specificity = tn / (tn + fp + 1e-10)
        tss = sensitivity + specificity - 1
        
        # Check if this is the best TSS
        if tss > best_tss:
            best_tss = tss
            best_metrics = {
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy,
                'tss': tss,
                'tp': tp,
                'fp': fp,
                'tn': tn,
                'fn': fn
            }
    
    # Print results
    print(f"Best TSS: {best_metrics['tss']:.4f} at threshold {best_metrics['threshold']:.2f}")
    print(f"Precision: {best_metrics['precision']:.4f}, Recall: {best_metrics['recall']:.4f}")
    print(f"F1 Score: {best_metrics['f1']:.4f}, Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"Confusion Matrix: TP={best_metrics['tp']}, FP={best_metrics['fp']}, TN={best_metrics['tn']}, FN={best_metrics['fn']}")
    
    return best_metrics

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    tf.keras.utils.set_random_seed(42)
    np.random.seed(42)
    
    # Check for available GPUs
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"SUCCESS: Found GPU: {gpus[0]}")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except:
                pass
        print(f"SUCCESS: Found GPU: {tf.test.gpu_device_name()}")
    
    # Print TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Python version: {os.sys.version.split(' ')[0]}")
    print(f"Tensorflow bakcend version: {tf.keras.backend.backend()}")
    
    # Import model_tracking functions (exactly as SolarKnowledge does)
    # Handle both possible import styles
    try:
        from model_tracking import save_model_with_metadata, get_next_version
        model_tracking_imported = True
    except ImportError:
        try:
            # Try relative import
            from models.model_tracking import save_model_with_metadata, get_next_version
            model_tracking_imported = True
        except ImportError:
            print("Warning: model_tracking.py not found. Model will be saved without metadata.")
            model_tracking_imported = False
    
    # Determine model type
    # Override USE_COMPLETE_MODEL if specified in args
    use_complete_model = USE_COMPLETE_MODEL or args.complete
    use_advanced_model = not args.simple
    
    # Get next version (like SolarKnowledge does)
    if model_tracking_imported:
        version = get_next_version(args.flare, args.window)
        print(f"Using version v{version} for this model")
    
    # Load and preprocess data
    X_train, X_val, y_train, y_val = load_data(
        time_window=args.window,
        flare_class=args.flare
    )
    
    # Train model
    if use_complete_model:
        model = train_complete_model(
            X_train, X_val, y_train, y_val,
            flare_class=args.flare,
            time_window=args.window,
            epochs=args.epochs,
            batch_size=args.batch_size,
            use_advanced_model=use_advanced_model,
            use_calibration=False,  # Disable calibration
            log_interval=args.log_interval,
            enable_tensorboard=args.tensorboard
        )
    else:
        model = train_original_model(
            X_train, X_val, y_train, y_val,
            flare_class=args.flare,
            time_window=args.window,
            epochs=args.epochs,
            batch_size=args.batch_size,
            use_advanced_model=use_advanced_model
        )
    
    # Evaluate model
    metrics = evaluate_model(
        model, X_val, y_val,
        is_complete_model=use_complete_model
    )
    
    # Save model with metadata
    if model_tracking_imported:
        # Create hyperparameters dictionary
        hyperparams = {
            'learning_rate': 2e-4,
            'batch_size': args.batch_size,
            'early_stopping_patience': 5,  # Match SolarKnowledge
            'epochs': args.epochs,
            'num_transformer_blocks': 4,
            'embed_dim': 128, 
            'num_heads': 4,
            'ff_dim': 256,
            'dropout_rate': 0.4,
            'focal_loss': True,
            'focal_loss_alpha': 0.25,  # Add specific parameters like SolarKnowledge
            'focal_loss_gamma': 2.0,
            'use_retentive': True,
            'use_multi_scale': X_train.shape[1] >= 24,
            'use_evidential': use_advanced_model,
            'use_evt': use_advanced_model
        }
        
        # Extract training history (exactly as SolarKnowledge does)
        if hasattr(model, 'history'):
            history = model.history
        else:
            # Create a simple history dict if not available
            history = {
                'loss': [metrics.get('final_loss', 0.0)],
                'accuracy': [metrics.get('accuracy', 0.0)],
                'tss': [metrics.get('tss', 0.0)]
            }
        
        # Create description
        if use_advanced_model:
            description = f"EVEREST model with evidential/EVT heads for {args.flare} flares with {args.window}h forecast window"
        else:
            description = f"EVEREST model for {args.flare} flares with {args.window}h forecast window"
        
        # Save model with metadata - exactly as SolarKnowledge does
        model_dir = save_model_with_metadata(
            model=model,
            metrics=metrics,
            hyperparams=hyperparams,
            history=history,
            version=version,
            flare_class=args.flare,
            time_window=args.window,
            description=description
        )
        
        print(f"\nModel saved to {model_dir}")
    else:
        # Fallback to direct saving if model_tracking not available
        if use_complete_model:
            weights_path = f"models/everest_{args.flare}_{args.window}"
            model.save_weights(weights_path, flare_class=args.flare)
            print(f"\nModel weights saved to {weights_path}")
    
    # Print final metrics
    print(f"\nFinal evaluation metrics:")
    print(f"TSS: {metrics['tss']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    
    return metrics

if __name__ == "__main__":
    main() 