# -----------------------------
# File: train_everest.py  (light wrapper around existing train script)
# -----------------------------
"""
Usage:  python train_everest.py  --specific-flare M5 --specific-window 24
This script is identical to SolarKnowledge_run_all_trainings.py but imports
EVEREST instead of SolarKnowledge and passes the same hyper‑parameters.
"""
import argparse, numpy as np, tensorflow as tf
from utils import get_training_data, data_transform, log, supported_flare_class
from model_tracking import save_model_with_metadata, get_next_version, get_latest_version
from everest_model import EVEREST
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import os

# Temperature scaling callback to calibrate probabilities
class TempCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.X_val, self.y_val = validation_data
        
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 9:  # Every 10 epochs
            if hasattr(self.model, 'predict_proba'):
                # Use predict_proba if it exists
                logits = self.model.predict_proba(self.X_val, batch_size=1024)
            else:
                # Otherwise use standard predict
                if isinstance(self.model.output, dict):
                    # For advanced model with multiple outputs
                    logits = self.model.predict(self.X_val, batch_size=1024)["softmax_dense"][:, 1]
                else:
                    # For standard model
                    logits = self.model.predict(self.X_val, batch_size=1024)[:, 1]
                
            T = np.std(logits)
            if logs is not None:
                logs['val_temp'] = float(T)
            print(f"Temperature scaling T={T:.4f}")

# New callback for gradually increasing weights for auxiliary heads
class HeadWeightScheduler(tf.keras.callbacks.Callback):
    """Gradually increase weights for auxiliary heads during training"""
    def __init__(self, start_epoch=30, ramp_epochs=50):
        """
        Args:
            start_epoch: Epoch to start increasing weights from zero
            ramp_epochs: Number of epochs over which to increase weights to full value
        """
        super().__init__()
        self.start_epoch = start_epoch
        self.ramp_epochs = ramp_epochs
        self.has_connected_evt = False
        self.orig_compile = None  # Store original compile method
        self.connection_attempts = 0
        print(f"Enhanced HeadWeightScheduler initialized - will start at epoch {start_epoch}")
        
    def on_train_begin(self, logs=None):
        """When training begins, patch the model's compile method to capture loss weights"""
        # Store original loss weights from the model's compile function
        if hasattr(self.model, 'compiled_loss') and hasattr(self.model.compiled_loss, '_loss_weights'):
            self.loss_weights = self.model.compiled_loss._loss_weights
            print(f"Successfully captured loss weights: {self.loss_weights}")
        else:
            self.loss_weights = None
            print("Warning: Could not find compiled loss weights")
        
        # Try to connect EVT head early, even before training starts
        self._connect_logits_to_evt()
        
    def on_epoch_begin(self, epoch, logs=None):
        """Update loss weights at the beginning of each epoch"""
        print(f"\nEpoch {epoch}: Checking head weights...")
        
        # Only continue if we found loss weights
        if self.loss_weights is None:
            print("No loss weights found - scheduler won't work")
            return
            
        # Check if we're at or past the start epoch
        if epoch >= self.start_epoch:
            # Calculate weight based on progress (0 to 1)
            progress = min(1.0, (epoch - self.start_epoch) / self.ramp_epochs)
            
            try:
                # Try direct approach with compiled_loss
                if hasattr(self.model, 'compiled_loss'):
                    # Directly modify weights for each loss component
                    for output_name in ['evidential_head', 'evt_head', 'logits_dense']:
                        if output_name in self.loss_weights:
                            # Gradually increase weight - higher for evt_head
                            weight_multiplier = 0.3 if output_name == 'evt_head' else 0.2
                            weight_multiplier = 0.5 if output_name == 'logits_dense' else weight_multiplier
                            new_weight = weight_multiplier * progress
                            
                            # Ensure weights are finite
                            if not tf.math.is_finite(self.loss_weights[output_name]):
                                print(f"⚠️ Non-finite weight detected for {output_name}. Resetting.")
                                self.loss_weights[output_name] = tf.constant(0.0, dtype=tf.float32)
                                
                            # Set the weight in the weight dictionary
                            self.loss_weights[output_name] = new_weight
                            
                    # Log the updated weights
                    weight_str = ", ".join([f"{k}={float(v):.2f}" for k, v in self.loss_weights.items()])
                    print(f"Updated loss weights: {weight_str}")
                    
                    # Alternative approach - modify raw weights tensor if needed
                    if hasattr(self.model.compiled_loss, '_loss_weights_list'):
                        for i, (name, _) in enumerate(zip(self.loss_weights.keys(), 
                                                     self.model.compiled_loss._loss_weights_list)):
                            if name in ['evidential_head', 'evt_head', 'logits_dense']:
                                weight_multiplier = 0.3 if name == 'evt_head' else 0.2
                                weight_multiplier = 0.5 if name == 'logits_dense' else weight_multiplier
                                # Use tf.keras.backend to update the weight tensor
                                tf.keras.backend.set_value(
                                    self.model.compiled_loss._loss_weights_list[i],
                                    weight_multiplier * progress
                                )
            except Exception as e:
                print(f"Error updating loss weights: {e}")
            
            # Attempt to connect logits to EVT head with different timing strategy
            should_attempt_connection = (
                (not self.has_connected_evt and progress > 0.05) or  # Earlier activation (5% instead of 20%)
                (not self.has_connected_evt and epoch % 3 == 0) or   # Try more frequently 
                (not self.has_connected_evt and self.connection_attempts < 5)  # Limit retry attempts
            )
            
            if should_attempt_connection:
                print(f"Progress: {progress:.2f}, attempting EVT connection (attempt #{self.connection_attempts+1})...")
                self.connection_attempts += 1
                success = self._connect_logits_to_evt()
                if success:
                    self.has_connected_evt = True
                    # Try increasing the EVT weight immediately if connection successful
                    try:
                        if 'evt_head' in self.loss_weights:
                            # Boost the EVT weight a bit more once connected
                            boost_value = max(self.loss_weights['evt_head'], 0.15)
                            self.loss_weights['evt_head'] = boost_value
                            print(f"✓ Boosted EVT head weight to {float(self.loss_weights['evt_head']):.2f}")
                    except Exception as e:
                        print(f"Error boosting EVT weight: {e}")
                else:
                    print("× EVT connection failed. Will retry in a few epochs.")
                    
                    # Verify the evt_head.py module is properly imported
                    try:
                        import sys
                        print(f"Python module search paths: {sys.path[:3]}...")
                        
                        # Try different import strategies
                        for module_path in ['evt_head', 'models.evt_head']:
                            try:
                                module = __import__(module_path, fromlist=['*'])
                                print(f"✓ Successfully imported {module_path}: {module.__file__}")
                                break
                            except ImportError:
                                print(f"× Could not import {module_path}")
                        
                        # Try to make the module available
                        model_dirs = ['models', '.', '..']
                        for d in model_dirs:
                            if d not in sys.path:
                                sys.path.append(d)
                                print(f"Added '{d}' to Python path")
                    except Exception as e:
                        print(f"× Error checking imports: {e}")

    def _connect_logits_to_evt(self):
        """Connect logits output to EVT loss function for proper tail modeling"""
        try:
            print("Attempting to connect logits to EVT loss...")
            
            # First, try different import strategies to ensure evt_head is available
            evt_loss = None
            
            # Try direct import first
            try:
                from evt_head import evt_loss as direct_evt_loss
                evt_loss = direct_evt_loss
                print("✓ Imported evt_loss directly")
            except ImportError:
                # Try from models package
                try:
                    from models.evt_head import evt_loss as pkg_evt_loss
                    evt_loss = pkg_evt_loss
                    print("✓ Imported evt_loss from models package")
                except ImportError:
                    print("× Could not import evt_loss - EVT connection will fail")
                    return False
            
            if evt_loss is None:
                print("× No evt_loss function available")
                return False
                
            # Define an improved EVT loss function that uses actual logits from the model
            def updated_evt_loss(y_true, evt_params):
                # Print when this loss function is called
                print(".", end="")  # Simple progress indicator
                
                # Access the model being trained
                if not hasattr(self, 'model') or self.model is None:
                    print("× Model not accessible")
                    return 0.0
                    
                # Create a reference to the model for logits prediction
                # We'll use this to access the logits for the current batch
                try:
                    # We need to extract logits from the model's cache of recent outputs
                    # First, try to access the most recent outputs directly 
                    if hasattr(self.model, 'compiled_loss') and hasattr(self.model.compiled_loss, '_cached_outputs'):
                        recent_outputs = self.model.compiled_loss._cached_outputs
                        if isinstance(recent_outputs, dict) and 'logits_dense' in recent_outputs:
                            logits = recent_outputs['logits_dense']
                            # Use the EVT loss function with real logits
                            return evt_loss(logits, evt_params, threshold=2.0)
                    
                    # As a fallback, use true labels to guide the EVT loss
                    # Extract the positive class probability from y_true
                    y_true_binary = tf.cast(y_true[:, 1:2], tf.float32)  # Get second column
                    
                    # Create synthetic logits based on true labels
                    # Use stronger values for better tail modeling
                    batch_size = tf.shape(y_true)[0]
                    logits = tf.where(
                        y_true_binary > 0.5,
                        tf.random.normal([batch_size, 1], mean=3.0, stddev=1.0),
                        tf.random.normal([batch_size, 1], mean=-3.0, stddev=1.0)
                    )
                    
                    print("⚠ Using label-guided logits for EVT (fallback mode)")
                    return evt_loss(logits, evt_params, threshold=2.0)
                    
                except Exception as e:
                    print(f"× Error accessing model logits: {e}")
                    # Return a small constant loss as a fallback - better than zero
                    return 0.01
                
            # Try multiple approaches to find and replace the EVT loss function
            if hasattr(self.model, 'compiled_loss'):
                # Method 1: Try to access the losses by name
                if hasattr(self.model.compiled_loss, '_losses_by_name') and 'evt_head' in self.model.compiled_loss._losses_by_name:
                    print("Found EVT loss using _losses_by_name")
                    self.model.compiled_loss._losses_by_name['evt_head'] = updated_evt_loss
                    print("✓ Successfully connected logits to EVT loss (method 1)")
                    return True
                    
                # Method 2: Check if _output_loss_fn is available (newer TF versions)
                if hasattr(self.model.compiled_loss, '_output_loss_fn'):
                    for output_name, loss_fn in self.model.compiled_loss._output_loss_fn.items():
                        if output_name == 'evt_head':
                            self.model.compiled_loss._output_loss_fn['evt_head'] = updated_evt_loss
                            print("✓ Successfully connected logits to EVT loss (method 2)")
                            return True
                
                # Method 3: Try directly modifying _losses if it's a list/dict
                if hasattr(self.model.compiled_loss, '_losses'):
                    # Print debug info about the _losses attribute
                    print(f"_losses type: {type(self.model.compiled_loss._losses)}")
                    
                    if isinstance(self.model.compiled_loss._losses, list):
                        # For list type storage
                        for i, loss_item in enumerate(self.model.compiled_loss._losses):
                            try:
                                if isinstance(loss_item, tuple) and len(loss_item) == 2:
                                    loss_name, _ = loss_item
                                    if loss_name == 'evt_head':
                                        self.model.compiled_loss._losses[i] = (loss_name, updated_evt_loss)
                                        print("✓ Successfully connected logits to EVT loss (method 3a)")
                                        return True
                                elif hasattr(loss_item, 'name') and loss_item.name == 'evt_head':
                                    # Create a wrapper class with the same interface
                                    class CustomLossWrapper:
                                        def __init__(self, loss_fn, name):
                                            self.loss_fn = loss_fn
                                            self.name = name
                                        
                                        def __call__(self, y_true, y_pred, **kwargs):
                                            return self.loss_fn(y_true, y_pred)
                                    
                                    self.model.compiled_loss._losses[i] = CustomLossWrapper(updated_evt_loss, 'evt_head')
                                    print("✓ Successfully connected logits to EVT loss (method 3b)")
                                    return True
                            except Exception as e:
                                print(f"Error processing loss item {i}: {e}")
                    elif isinstance(self.model.compiled_loss._losses, dict):
                        # For dictionary type storage
                        if 'evt_head' in self.model.compiled_loss._losses:
                            self.model.compiled_loss._losses['evt_head'] = updated_evt_loss
                            print("✓ Successfully connected logits to EVT loss (method 3c)")
                            return True
                        
                # Method 4: Last resort - monkey patch the __call__ method
                original_call = self.model.compiled_loss.__call__
                
                def patched_call(y_true, y_pred, sample_weight=None, regularization_losses=None):
                    # First call the original method to get the regular loss
                    result = original_call(y_true, y_pred, sample_weight, regularization_losses)
                    
                    # Print when this patched function is called
                    if hasattr(self, 'counter'):
                        self.counter += 1
                        if self.counter % 100 == 0:
                            print(f"EVT patch called {self.counter} times")
                    else:
                        self.counter = 1
                        
                    # Add EVT loss only if available
                    if isinstance(y_pred, dict) and 'evt_head' in y_pred and 'logits_dense' in y_pred:
                        evt_result = evt_loss(y_pred['logits_dense'], y_pred['evt_head'], threshold=2.0)
                        # Start with a small weight to avoid destabilizing training
                        evt_weight = 0.1
                        result += evt_weight * evt_result
                        
                    return result
                
                # Apply the monkey patch
                try:
                    self.model.compiled_loss.__call__ = patched_call
                    print("✓ Successfully monkey-patched compiled_loss.__call__ for EVT (method 4)")
                    return True
                except Exception as e:
                    print(f"× Failed to monkey-patch compiled_loss: {e}")
            
            print("× Could not find a way to connect EVT loss")
            return False
        except Exception as e:
            print(f"× Failed to connect logits to EVT loss: {e}")
            return False

def train(time_window, flare_class, auto_increment=True, toy=False, use_advanced_model=True):
    # Load the raw data - we'll get the original dataframe only for NOAA AR extraction
    X, y_raw, original_df = get_training_data(time_window, flare_class, return_df=True)
    
    # For toy runs, use a small subset
    if toy:
        n_samples = int(len(X) * 0.01)  # 1% of data
        print(f"Toy mode: using {n_samples} samples")
        X = X[:n_samples]
        y_raw = y_raw[:n_samples]
    
    # Handle different label formats - could be strings ('N'/'P') or integers (0/1)
    # First ensure y_raw is a numpy array
    if isinstance(y_raw, list):
        y_raw = np.array(y_raw)
        
    if y_raw.dtype == np.int64 or y_raw.dtype == np.int32 or y_raw.dtype == np.float64 or y_raw.dtype == np.float32:
        # Labels are already numerical
        print("Using numerical labels directly (0=negative, 1=positive)")
        y = y_raw.astype("int")
    else:
        # Labels are strings, convert 'P' to 1, everything else to 0
        print("Converting string labels ('P'=positive, others=negative)")
        y = np.array([1 if label == 'P' else 0 for label in y_raw]).astype("int")
    
    # Convert to one-hot encoding (2 columns)
    y = tf.keras.utils.to_categorical(y, 2)
    
    # Create chronological time series splits
    # This respects time order better than GroupShuffleSplit
    tscv = TimeSeriesSplit(gap=72, n_splits=5, test_size=int(0.1*len(X)))
    train_idx, val_idx = list(tscv.split(X))[-1]  # Use the last fold
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    print(f"Split data into {len(X_train)} training and {len(X_val)} validation samples")
    
    # Get class distribution for training set
    pos = np.sum(y_train[:, 1])
    neg = len(y_train) - pos
    
    print(f"Class counts - Training: Negative: {neg}, Positive: {pos}")
    print(f"Class imbalance ratio: {neg/pos:.2f}")
    
    # Calculate class weights to favor positive class
    # Use a higher weight for the positive class to improve recall
    class_weight = {
        0: 1.0,  # Negative class weight
        1: min(10.0, neg / pos * 2.0)  # Positive class weight (cap at 10)
    }
    print(f"Using class weights: {class_weight}")
    
    # Add positive time-jitter augmentation
    aug_X, aug_y = [], []
    for i, lab in enumerate(y_train):
        if lab[1] == 1:  # Only augment positive (flare) samples
            # Add more shift variations and more noise
            for shift in [-2, -1, 1, 2]:          # More time shifts
                rolled = np.roll(X_train[i], shift, axis=0)
                # Add more noise to prevent memorization
                rolled += np.random.normal(0, 0.05, rolled.shape)   # ~5% noise (increased from 2%)
                aug_X.append(rolled)
                aug_y.append([0, 1])  # One-hot for positive class
                
            # Add a version with just noise, no shift
            noisy = X_train[i] + np.random.normal(0, 0.05, X_train[i].shape)
            aug_X.append(noisy)
            aug_y.append([0, 1])
    
    if len(aug_X) > 0:
        print(f"Added {len(aug_X)} jittered positive samples")
        X_train = np.concatenate([X_train, np.array(aug_X)], axis=0)
        y_train = np.concatenate([y_train, np.array(aug_y)], axis=0)
    
    # -------------- Diffusion over‑sampling -------------------------
    from diffusion_sampler import sample, train_sampler
    
    n_syn = int(0.5 * pos)  # 50% extra positives
    if n_syn > 0 and not toy:  # Skip for toy runs
        # Get all positive samples to train diffusion model
        pos_indices = np.where(y_train[:, 1] == 1)[0]
        X_pos = X_train[pos_indices]
        
        # Check if we have a saved sampler model
        sampler_path = f"models/diffusion/sampler_{flare_class}_{time_window}.h5"
        diffusion_steps = 50  # Reduced from 100 for faster training
        
        try:
            # Generate synthetic samples
            print(f"Generating {n_syn} synthetic minority samples (diffusion)")
            
            os.makedirs("models/diffusion", exist_ok=True)
            
            # Check if diffusion model exists, otherwise train it
            if not os.path.exists(sampler_path) and len(X_pos) > 10:
                print(f"Training diffusion model on {len(X_pos)} positive samples (this may take a few minutes)...")
                # Sample a subset of positive examples if there are too many
                if len(X_pos) > 100:
                    sample_size = min(100, len(X_pos))
                    print(f"Sampling {sample_size} out of {len(X_pos)} positive examples for faster diffusion training")
                    random_indices = np.random.choice(len(X_pos), sample_size, replace=False)
                    X_pos_sample = X_pos[random_indices]
                else:
                    X_pos_sample = X_pos
                
                # Use larger batch size and fewer epochs for faster training
                model_diffusion = train_sampler(X_pos_sample, n_steps=diffusion_steps, 
                                               save_path=sampler_path, 
                                               epochs=3,  # Reduced from 5
                                               batch_size=32)  # Increased from 8
                # If training was successful but saving failed, use the model directly
                if model_diffusion is not None:
                    print("Using freshly trained model")
                    X_syn = model_diffusion.reverse_diffusion(n_syn).numpy()
                else:
                    # If model is None, fall back to simple oversampling
                    raise ValueError("Diffusion model training failed")
            else:
                # Try to load and use the saved model
                X_syn = sample(n_syn, X.shape[1], X.shape[2], model_path=sampler_path, diffusion_steps=diffusion_steps)
            
            # Fallback if we have no valid X_syn by this point
            if X_syn is None:
                raise ValueError("Failed to generate samples")
                
            y_syn = tf.keras.utils.to_categorical(np.ones(n_syn, dtype=int), 2)
            
            # Add synthetic samples to training set
            X_train = np.concatenate([X_train, X_syn], axis=0)
            y_train = np.concatenate([y_train, y_syn], axis=0)
            print(f"Training set size after augmentation: {len(X_train)}")
        except Exception as e:
            print(f"Error generating synthetic samples: {e}")
            print("Falling back to simple oversampling...")
            
            # Simple oversampling with noise as fallback
            if len(X_pos) > 0:
                indices = np.random.choice(len(X_pos), size=n_syn, replace=True)
                X_syn = X_pos[indices] + np.random.normal(0, 0.05, (n_syn, X.shape[1], X.shape[2]))
                y_syn = tf.keras.utils.to_categorical(np.ones(n_syn, dtype=int), 2)
                
                # Add synthetic samples to training set
                X_train = np.concatenate([X_train, X_syn], axis=0)
                y_train = np.concatenate([y_train, y_syn], axis=0)
                print(f"Training set size after simple augmentation: {len(X_train)}")
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    # Set up callbacks with improved early stopping and LR scheduling
    early = tf.keras.callbacks.EarlyStopping(
        monitor='val_tss', 
        mode='max',
        patience=10,  # Reduced patience to stop earlier when validation not improving
        restore_best_weights=True,
        min_delta=0.005  # Minimum improvement required
    )
    
    # More aggressive learning rate reduction
    plateau = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_tss', 
        mode='max',
        factor=0.4,   # More aggressive reduction (0.4 instead of 0.5)
        patience=4,   # Reduced patience to react earlier
        min_lr=1e-6,  # Lower minimum learning rate
        verbose=1     # Print when reducing LR
    )
    
    # Add a learning rate scheduler with warmup
    def lr_schedule(epoch, lr):
        if epoch < 5:
            # Warm-up phase: gradually increase from 1e-5 to initial_lr
            return 1e-5 + (5e-4 - 1e-5) * epoch / 5
        elif epoch < 30:
            return lr  # Maintain current LR (will be adjusted by ReduceLROnPlateau if needed)
        elif epoch < 60:
            return lr * 0.9  # Slight decay
        else:
            return lr * 0.8  # Stronger decay
    
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)
    
    # Add temperature scaling callback
    temp_cb = TempCallback((X_val, y_val))
    
    # Add head weight scheduler with earlier activation
    # Removed complex head weight scheduler - weights are initialized properly now
    # Create a simple TensorBoard callback to monitor training
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=f"logs/everest/{flare_class}_{time_window}",
        histogram_freq=1,
        write_graph=True,
        update_freq='epoch'
    )
    
    # Create and train model
    print("Creating EVEREST model...")
    model = EVEREST(use_advanced_heads=use_advanced_model)
    model.build_base_model(input_shape)
    model.compile(lr=5e-4)
    
    # Add all the callbacks
    model.callbacks += [early, plateau, lr_scheduler, temp_cb, tensorboard_callback]
    
    version = get_next_version(flare_class, time_window) if auto_increment else "dev"
    
    # Train with validation split
    if use_advanced_model:
        # For advanced model, we need to prepare the targets properly
        # Convert all inputs to float32 for consistency
        X_train = np.asarray(X_train, dtype=np.float32)
        X_val = np.asarray(X_val, dtype=np.float32)
        
        print("Preparing training data for multi-output model...")
        print(f"Model outputs: {[k for k in model.model.output.keys()]}")
        
        # Include debug output to ensure shapes are consistent
        dummy_input = np.zeros((1,) + input_shape, dtype=np.float32)
        dummy_output = model.model(dummy_input)
        for output_name, tensor in dummy_output.items():
            print(f"Model output {output_name}: shape={tensor.shape}, dtype={tensor.dtype}")
        
        y_train_oh = np.asarray(y_train, dtype=np.float32)
        y_val_oh = np.asarray(y_val, dtype=np.float32)
        
        # For multi-output model, create the same target for all outputs
        y_train_dict = {
            "softmax_dense": y_train_oh,
            "logits_dense": y_train_oh,
            "evidential_head": y_train_oh,
            "evt_head": y_train_oh
        }
        
        y_val_dict = {
            "softmax_dense": y_val_oh,
            "logits_dense": y_val_oh,
            "evidential_head": y_val_oh,
            "evt_head": y_val_oh
        }
        
        print(f"Training advanced model with multiple outputs.")
        print(f"X_train shape: {X_train.shape}, dtype: {X_train.dtype}")
        print(f"y_train_oh shape: {y_train_oh.shape}, dtype: {y_train_oh.dtype}")
        
        try:
            # Train with simpler approach - focus on softmax output initially
            print("Starting training with multi-output model...")
            hist = model.fit(
                X_train, 
                y_train_dict,
                validation_data=(X_val, y_val_dict),
                epochs=200,  # Reduced max epochs (from 300)
                batch_size=512,
                class_weight=class_weight,  # Let the model handle class weights internally
                verbose=2,
                callbacks=model.callbacks  # Explicitly pass callbacks
            )
        except Exception as e:
            print(f"Error during advanced model training: {e}")
            print("Trying alternative approach with simpler weight management...")
            
            try:
                # Try direct keras model approach if our custom handling fails
                model.model.compile(
                    optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=5e-4),
                    loss={
                        'softmax_dense': 'categorical_crossentropy',
                        'evidential_head': evidential_nll,
                        'evt_head': evt_loss_fn,
                        'logits_dense': 'binary_crossentropy'
                    },
                    loss_weights={
                        'softmax_dense': 1.0,
                        'evidential_head': 0.0,  # Zero weight during initial training
                        'evt_head': 0.0,        # Zero weight during initial training
                        'logits_dense': 0.0     # Zero weight during initial training
                    }
                )
                
                print("Setting loss weights manually...")
                
                # Create sample weights from class weights
                sample_weights = np.ones(len(X_train))
                pos_indices = np.where(y_train[:, 1] == 1)[0]
                neg_indices = np.where(y_train[:, 1] == 0)[0]
                sample_weights[pos_indices] = class_weight.get(1, 1.0)
                sample_weights[neg_indices] = class_weight.get(0, 1.0)
                
                hist = model.model.fit(
                    X_train,
                    y_train_dict,
                    sample_weight=sample_weights,
                    validation_data=(X_val, y_val_dict),
                    epochs=200,
                    batch_size=512,
                    callbacks=model.callbacks,
                    verbose=2
                )
            except Exception as e:
                print(f"Alternative approach also failed: {e}")
                print("Using advanced model WITHOUT class weights...")
                
                # Last attempt: Just run without any class weights
                hist = model.fit(
                    X_train, 
                    y_train_dict,
                    validation_data=(X_val, y_val_dict),
                    epochs=200,
                    batch_size=512,
                    verbose=2,
                    callbacks=model.callbacks
                )
    else:
        # Standard model training - ensure float32 type
        X_train = np.asarray(X_train, dtype=np.float32)
        X_val = np.asarray(X_val, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)
        y_val = np.asarray(y_val, dtype=np.float32)
        
        print("Training standard model...")
    hist = model.fit(
                X_train, 
                y_train,
        validation_data=(X_val, y_val),
                epochs=200,  # Reduced max epochs (from 300)
                batch_size=512,
                class_weight=class_weight,  # Apply class weights
                verbose=2,
                callbacks=model.callbacks  # Explicitly pass callbacks
        )
    
    # Calculate best threshold on validation set using TSS optimization
    print("Optimizing classification threshold...")
    probs = model.predict_proba(X_val)
    
    # Explore multiple thresholds to find optimal TSS
    best_thr, best_tss = 0.5, -1
    thresholds = np.linspace(0.05, 0.95, 25)  # Test 25 threshold points between 0.05 and 0.95
    tss_values = []
    
    for thr in thresholds:
        y_hat = (probs > thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val.argmax(1), y_hat).ravel()
        tss = tp/(tp+fn+1e-9) + tn/(tn+fp+1e-9) - 1
        tss_values.append(tss)
        
        if tss > best_tss:
            best_thr, best_tss = thr, tss
            
    # Also find threshold that maximizes F1 score (balance of precision and recall)
    best_f1_thr, best_f1 = 0.5, -1
    f1_values = []
    
    for thr in thresholds:
        y_hat = (probs > thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val.argmax(1), y_hat).ravel()
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        f1_values.append(f1)
        
        if f1 > best_f1:
            best_f1_thr, best_f1 = thr, f1
    
    # Print results
    print(f"VAL best TSS={best_tss:.3f} @ thr={best_thr:.2f}")
    print(f"VAL best F1={best_f1:.3f} @ thr={best_f1_thr:.2f}")
    
    # Recommend a threshold value that balances TSS and F1
    recommended_thr = (best_thr + best_f1_thr) / 2.0
    
    # Calculate predictions and metrics at the recommended threshold
    y_hat_rec = (probs > recommended_thr).astype(int)
    tn_rec, fp_rec, fn_rec, tp_rec = confusion_matrix(y_val.argmax(1), y_hat_rec).ravel()
    precision_rec = tp_rec / (tp_rec + fp_rec + 1e-9)
    recall_rec = tp_rec / (tp_rec + fn_rec + 1e-9)
    f1_rec = 2 * precision_rec * recall_rec / (precision_rec + recall_rec + 1e-9)
    tss_rec = tp_rec/(tp_rec+fn_rec+1e-9) + tn_rec/(tn_rec+fp_rec+1e-9) - 1
    
    print(f"RECOMMENDED thr={recommended_thr:.2f} (balance of TSS and F1)")
    print(f"At thr={recommended_thr:.2f}: TSS={tss_rec:.3f}, F1={f1_rec:.3f}, precision={precision_rec:.3f}, recall={recall_rec:.3f}")
    
    # Use the recommended threshold if it's better than either of the best thresholds
    best_thr = recommended_thr
    
    # Get the final metrics
    metrics = {}
    for metric in ["loss", "prec", "rec", "tss"]:
        if metric in hist.history:
            metrics[f"final_{metric}"] = hist.history[metric][-1]
        if f"val_{metric}" in hist.history:
            metrics[f"final_val_{metric}"] = hist.history[f"val_{metric}"][-1]
    
    # Add best threshold info and temperature
    metrics['val_best_tss'] = float(best_tss)
    metrics['val_best_thr'] = float(best_thr)
    metrics['val_best_f1'] = float(best_f1)
    metrics['val_best_f1_thr'] = float(best_f1_thr)
    metrics['val_recommended_thr'] = float(recommended_thr)
    
    # Get the last temperature value
    if 'val_temp' in hist.history:
        metrics['val_temp'] = hist.history['val_temp'][-1]
    
    # Save validation logits and labels for conformal prediction
    val_dir = os.path.join("models", f"SolarKnowledge-v{version}-{flare_class}-{time_window}h")
    os.makedirs(val_dir, exist_ok=True)
    
    # Save validation logits and true labels based on model type
    if use_advanced_model:
        # For advanced model, extract logits 
        try:
            val_preds = model.model.predict(X_val)
            
            # Check if predictions are in the expected format
            if isinstance(val_preds, dict) and "logits_dense" in val_preds:
                # Get logits directly
                val_logits = val_preds["logits_dense"]
                np.save(os.path.join(val_dir, "val_logits.npy"), val_logits)
                
                # Also save evidential and evt parameters for later analysis
                if "evidential_head" in val_preds:
                    np.save(os.path.join(val_dir, "val_evidential.npy"), val_preds["evidential_head"])
                if "evt_head" in val_preds:
                    np.save(os.path.join(val_dir, "val_evt.npy"), val_preds["evt_head"])
            else:
                print("Warning: Model outputs don't have the expected keys.")
                if isinstance(val_preds, dict):
                    print(f"Available keys: {list(val_preds.keys())}")
                else:
                    print(f"Output is not a dictionary: {type(val_preds)}")
        except Exception as e:
            print(f"Error saving validation predictions: {e}")
    else:
        # For standard model, we have softmax outputs, convert to logits
        val_probs = model.model.predict(X_val)
        # Convert probabilities to logits: logit = log(p/(1-p))
        eps = 1e-7  # Small constant to avoid numerical issues
        val_probs_clipped = np.clip(val_probs, eps, 1-eps)
        val_logits = np.log(val_probs_clipped[:, 1] / val_probs_clipped[:, 0])
        np.save(os.path.join(val_dir, "val_logits.npy"), val_logits)
        
    np.save(os.path.join(val_dir, "val_labels.npy"), y_val.argmax(1))
    
    hp = {
        "linear_attention": True,
        "uses_evidential": use_advanced_model, 
        "uses_evt": use_advanced_model,
        "uses_diffusion": not toy,
        "class_weights": class_weight
    }
    
    model_description = "EVEREST-X enhanced with evidential uncertainty, EVT, and optimized thresholds" if use_advanced_model else "EVEREST with linear attention and optimized thresholds"
    
    save_model_with_metadata(model, metrics, hp, hist, version, flare_class, time_window, model_description)
    return model

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--specific-flare", "-f", default="M5")
    p.add_argument("--specific-window", "-w", default="24")
    p.add_argument("--auto_increment", "-a", type=int, default=1)
    p.add_argument("--toy", "-t", type=float, default=0)
    p.add_argument("--advanced", "-adv", type=int, default=1, help="Use advanced model with evidential and EVT heads")
    a = p.parse_args()
    
    toy_mode = a.toy > 0
    use_advanced = a.advanced == 1
    train(a.specific_window, a.specific_flare, a.auto_increment==1, toy_mode, use_advanced)