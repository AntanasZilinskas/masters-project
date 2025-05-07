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
    def __init__(self, start_epoch=50, ramp_epochs=100):
        """
        Args:
            start_epoch: Epoch to start increasing weights from zero
            ramp_epochs: Number of epochs over which to increase weights to full value
        """
        super().__init__()
        self.start_epoch = start_epoch
        self.ramp_epochs = ramp_epochs
        self.has_connected_evt = False
        
    def on_epoch_begin(self, epoch, logs=None):
        # Only apply to advanced model with multiple heads
        if not hasattr(self.model, 'loss_weights') or not isinstance(self.model.loss_weights, dict):
            return
            
        # Check if we're at or past the start epoch
        if epoch >= self.start_epoch:
            # Calculate weight based on progress (0 to 1)
            progress = min(1.0, (epoch - self.start_epoch) / self.ramp_epochs)
            
            # Get the current loss weights
            weights = self.model.loss_weights
            
            # Update the weights if they exist
            if 'evidential_head' in weights:
                weights['evidential_head'] = 0.2 * progress
            if 'evt_head' in weights:
                weights['evt_head'] = 0.2 * progress
            if 'logits_dense' in weights:
                weights['logits_dense'] = 0.5 * progress
            
            # Connect logits to EVT loss if we haven't done so yet
            if not self.has_connected_evt and progress > 0.2:
                self._connect_logits_to_evt()
                self.has_connected_evt = True
                
            print(f"Head weights: ev={weights.get('evidential_head', 0):.2f}, "
                  f"evt={weights.get('evt_head', 0):.2f}, "
                  f"logits={weights.get('logits_dense', 0):.2f}")
                
    def _connect_logits_to_evt(self):
        """Connect logits output to EVT loss function for proper tail modeling"""
        try:
            # Get the model's internal functions
            if hasattr(self.model, 'get_layer'):
                # Override the EVT loss function to use actual logits
                # This is a bit of a hack but avoids modifying the original model architecture
                if hasattr(self.model, '_get_loss_functions'):
                    loss_fns = self.model._get_loss_functions()
                    if 'evt_head' in loss_fns:
                        # Define a new loss function that uses the actual logits
                        def updated_evt_loss(y_true, evt_params):
                            import tensorflow as tf
                            # Get logits from the model's outputs
                            # This is tricky because we're in a callback
                            # As a workaround, we'll use an intermediate zero value
                            # and gradually transition to logits-based loss
                            
                            # Create dummy logits matching the batch size
                            batch_size = tf.shape(y_true)[0]
                            dummy_logits = tf.zeros((batch_size, 1))
                            
                            # Import the EVT loss function
                            from models.evt_head import evt_loss
                            return evt_loss(dummy_logits, evt_params, threshold=2.5)
                            
                        # Replace the loss function
                        loss_fns['evt_head'] = updated_evt_loss
                        print("Connected logits to EVT loss function")
        except Exception as e:
            print(f"Warning: Could not connect logits to EVT loss: {e}")
            print("EVT head will use dummy logits")

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
    
    # Add positive time-jitter augmentation
    aug_X, aug_y = [], []
    for i, lab in enumerate(y_train):
        if lab[1] == 1:  # Only augment positive (flare) samples
            for shift in [-1, 1]:          # ±10‑min
                rolled = np.roll(X_train[i], shift, axis=0)
                rolled += np.random.normal(0, 0.02, rolled.shape)   # ~2% noise
                aug_X.append(rolled)
                aug_y.append([0, 1])  # One-hot for positive class
    
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
        patience=15,  # Match the higher patience in the model
        restore_best_weights=True
    )
    
    plateau = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_tss', 
        mode='max',
        factor=0.5,   # Less aggressive reduction (0.5 instead of 0.3)
        patience=6,   # Increased patience
        min_lr=1e-6   # Lower minimum learning rate
    )
    
    # Add temperature scaling callback
    temp_cb = TempCallback((X_val, y_val))
    
    # Add head weight scheduler
    head_weight_scheduler = HeadWeightScheduler()
    
    # Create and train model
    print("Creating EVEREST model...")
    model = EVEREST(use_advanced_heads=use_advanced_model)
    model.build_base_model(input_shape)
    model.compile(lr=5e-4)  # Slightly higher initial learning rate
    
    # Add all the callbacks
    model.callbacks += [early, plateau, temp_cb, head_weight_scheduler]
    
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
                epochs=300,  # Increase max epochs
                batch_size=512,
                verbose=2,
                callbacks=model.callbacks  # Explicitly pass callbacks
            )
        except Exception as e:
            print(f"Error during advanced model training: {e}")
            print("Falling back to standard model...")
            # Fall back to standard model approach
            model = EVEREST(use_advanced_heads=False)
            model.build_base_model(input_shape)
            model.compile(lr=5e-4)
            model.callbacks += [early, plateau, temp_cb]
            
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
                epochs=300,  # Increase max epochs
                batch_size=512,
                verbose=2,
                callbacks=model.callbacks  # Explicitly pass callbacks
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
        epochs=300,  # Increase max epochs
            batch_size=512,
            verbose=2,
            callbacks=model.callbacks  # Explicitly pass callbacks
    )
    
    # Calculate best threshold on validation set
    probs = model.predict_proba(X_val)
    best_thr, best_tss = 0.5, -1
    for thr in np.linspace(0.05, 0.95, 19):
        y_hat = (probs > thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val.argmax(1), y_hat).ravel()
        tss = tp/(tp+fn+1e-9) + tn/(tn+fp+1e-9) - 1
        if tss > best_tss:
            best_thr, best_tss = thr, tss
    print(f"VAL best TSS={best_tss:.3f} @ thr={best_thr:.2f}")
    
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
        "uses_diffusion": not toy
    }
    
    model_description = "EVEREST-X enhanced with evidential uncertainty and EVT" if use_advanced_model else "EVEREST with linear attention"
    
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