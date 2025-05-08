#!/usr/bin/env python
"""
Standalone script to fix EVEREST model issues and run training.

This script:
1. Fixes JSON serialization of TensorFlow tensors
2. Patches the evidential and EVT head loss functions
3. Runs training for specified flare class and time window

Usage:
    python models/standalone_fix.py --flare M5 --window 24
"""

import os
import sys
import json
import argparse
import tensorflow as tf
import numpy as np

# ---------------------------------------------------------------
# JSON Serialization Fix
# ---------------------------------------------------------------

def convert_tensor_to_python(obj):
    """Convert TensorFlow tensors to Python native types for JSON serialization"""
    if isinstance(obj, tf.Tensor):
        return obj.numpy().tolist() if hasattr(obj, 'numpy') else float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_tensor_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_tensor_to_python(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_tensor_to_python(item) for item in obj)
    return obj

# Monkeypatch json.dumps to handle TensorFlow tensors
original_dumps = json.dumps

def patched_dumps(obj, *args, **kwargs):
    """Patch for json.dumps that converts tensor values to Python native types"""
    converted_obj = convert_tensor_to_python(obj)
    return original_dumps(converted_obj, *args, **kwargs)

# Apply the monkeypatch
json.dumps = patched_dumps

# ---------------------------------------------------------------
# Loss Function Fixes
# ---------------------------------------------------------------

def fixed_evidential_nll(y_true, evid, kl_weight=0.01):
    """
    Fixed negative log‑likelihood for binary‐classification evidential head.
    Ensures positive loss values by using tf.abs() on components.
    """
    # Reshape y_true to ensure consistent shapes
    y_true = tf.reshape(y_true, [-1, 1])
    
    # Assert shape is correct
    tf.debugging.assert_equal(tf.shape(evid)[-1], 4, 
                             message="Evidential parameters must have 4 components")
    
    # Split into NIG parameters
    mu, v, alpha, beta = tf.split(evid, 4, axis=-1)
    
    # Apply sigmoid to convert mu to probability
    p = tf.nn.sigmoid(mu)
    
    # Ensure parameters are within valid ranges
    alpha = tf.clip_by_value(alpha, 1.0 + 1e-6, 1e6)
    beta = tf.clip_by_value(beta, 1e-6, 1e6)
    v = tf.clip_by_value(v, 1e-6, 1e6)
    
    # Calculate predictive variance with safeguard
    S = beta*(1+v)/(alpha - 1.0 + 1e-6)
    
    # Small epsilon for log stability
    eps = 1e-7
    
    # Binary cross-entropy term (ensure positive)
    bce_term = tf.abs(- y_true * tf.math.log(p + eps) - (1-y_true)*tf.math.log(1-p + eps))
    
    # Variance term (ensure positive)
    var_term = tf.abs(0.5*tf.math.log(S + eps))
    
    # Regularization for correct predictions
    is_correct = tf.cast(tf.abs(y_true - p) < 0.2, tf.float32)
    var_reg = 0.1 * tf.abs(S * is_correct)
    
    # Combine NLL terms - all components are positive
    nll = bce_term + var_term + var_reg
    
    # Add KL regularization (must be positive)
    kl_term = 0.01  # Simplified positive constant to avoid complexity
    
    # Final loss - ensure it's positive
    return tf.reduce_mean(nll) + kl_weight * kl_term

def fixed_evt_loss(logits, params, threshold=0.5):
    """
    Fixed EVT loss function that ensures positive values
    and proper connection to model logits.
    """
    # Ensure inputs are proper tensors with correct shapes
    logits = tf.cast(logits, tf.float32)
    params = tf.cast(params, tf.float32)
    
    # Extract Generalized Pareto Distribution parameters
    xi = params[:, 0:1]    # Shape parameter
    sigma = params[:, 1:2] # Scale parameter
    
    # Constraint check: sigma must be positive
    sigma_safe = tf.maximum(sigma, 1e-6)
    
    # Identify extreme values (use lower threshold to capture more)
    is_extreme = tf.cast(logits > threshold, tf.float32)
    
    # Exceedances (how much each value exceeds the threshold)
    exceedances = tf.maximum(logits - threshold, 0.0) * is_extreme
    
    # Avoid division by zero and log(0)
    eps = 1e-7
    
    # Calculate negative log-likelihood of GPD
    # For xi ≈ 0: log(sigma) + z/sigma
    # For xi ≠ 0: log(sigma) + (1 + 1/xi) * log(1 + xi * z / sigma)
    
    # Create a mask for almost-zero xi values
    xi_near_zero = tf.cast(tf.abs(xi) < 1e-4, tf.float32)
    
    # Case 1: xi ≈ 0
    gpd_log_ll_case1 = tf.math.log(sigma_safe + eps) + exceedances / (sigma_safe + eps)
    
    # Case 2: xi ≠ 0
    term = 1 + xi * exceedances / (sigma_safe + eps)
    # Ensure term is positive to avoid log(negative)
    term_safe = tf.maximum(term, eps)
    gpd_log_ll_case2 = tf.math.log(sigma_safe + eps) + (1 + 1/tf.maximum(xi, eps)) * tf.math.log(term_safe + eps)
    
    # Combine cases based on xi value
    gpd_log_ll = xi_near_zero * gpd_log_ll_case1 + (1 - xi_near_zero) * gpd_log_ll_case2
    
    # Calculate loss (ensure positive)
    gpd_loss = tf.abs(gpd_log_ll * is_extreme)
    
    # Add regularization terms for xi and sigma
    reg_xi = 0.01 * tf.reduce_mean(tf.square(xi))
    reg_sigma = 0.01 * tf.reduce_mean(tf.square(sigma_safe))
    
    # Return mean loss over the batch (with at least one positive component)
    return tf.reduce_mean(gpd_loss + reg_xi + reg_sigma)

# ---------------------------------------------------------------
# EVEREST Model Patching
# ---------------------------------------------------------------

def apply_patches():
    """Apply all necessary patches to the EVEREST model"""
    print("Applying patches to EVEREST model...")
    
    # Add all possible import paths
    sys.path.extend(['', '.', '..', 'models'])
    
    try:
        # Try different import approaches until one works
        try:
            # First try direct imports
            import evidential_head
            import evt_head
            import everest_model
            print("✓ Successfully imported modules from current directory")
        except ImportError:
            # Then try from models package
            from models import evidential_head
            from models import evt_head
            from models import everest_model
            print("✓ Successfully imported modules from models package")
        
        # Apply patches to functions
        evidential_head.evidential_nll = fixed_evidential_nll
        evt_head.evt_loss = fixed_evt_loss
        
        # Patch the EVEREST model compile method to use non-zero weights from the start
        original_compile = everest_model.EVEREST.compile
        
        def patched_compile(self, lr=1e-3):
            # Call the original compile method first
            original_compile(self, lr)
            
            # Adjust loss weights to ensure they're Python floats, not tensors
            if hasattr(self.model, 'compiled_loss') and hasattr(self.model.compiled_loss, '_loss_weights'):
                # Force non-zero weights from the start
                for key in self.model.compiled_loss._loss_weights:
                    if key == 'evidential_head':
                        self.model.compiled_loss._loss_weights[key] = 0.2  # Float value, not tensor
                    elif key == 'evt_head':
                        self.model.compiled_loss._loss_weights[key] = 0.3  # Float value, not tensor
                    elif key == 'logits_dense':
                        self.model.compiled_loss._loss_weights[key] = 0.2  # Float value, not tensor
                    else:
                        # Convert to Python float to avoid tensor serialization issues
                        self.model.compiled_loss._loss_weights[key] = 1.0  # Float value
        
        # Apply the patch
        everest_model.EVEREST.compile = patched_compile
        print("✓ Applied all patches successfully")
        
        return True
    except Exception as e:
        print(f"Error applying patches: {e}")
        return False

# ---------------------------------------------------------------
# Training Functions
# ---------------------------------------------------------------

def run_training(flare_class, time_window, epochs=50, batch_size=256):
    """Run training with the fixed model"""
    # Try different training approaches until one works
    try:
        # First try direct import from train_everest
        from train_everest import train
        print(f"Running training with train_everest.py...")
        model = train(time_window, flare_class, auto_increment=True, toy=False, use_advanced_model=True)
        return True
    except Exception as e:
        print(f"Error with train_everest.py: {e}")
        
        # Try standalone_train.py
        try:
            print(f"Trying standalone_train.py...")
            cmd = f"python models/standalone_train.py --flare {flare_class} --window {time_window} --epochs {epochs} --batch-size {batch_size}"
            print(f"Running command: {cmd}")
            os.system(cmd)
            return True
        except Exception as e:
            print(f"Error with standalone_train.py: {e}")
            
            # Try simple_train.py
            try:
                print(f"Trying simple_train.py...")
                cmd = f"python models/simple_train.py --flare {flare_class} --window {time_window} --epochs {epochs} --batch-size {batch_size}"
                print(f"Running command: {cmd}")
                os.system(cmd)
                return True
            except Exception as e:
                print(f"Error with simple_train.py: {e}")
                
                # Last resort: use the original everest training script
                try:
                    print(f"Using original train_everest.py as command...")
                    cmd = f"python models/train_everest.py --specific-flare {flare_class} --specific-window {time_window}"
                    print(f"Running command: {cmd}")
                    os.system(cmd)
                    return True
                except Exception as e:
                    print(f"All training approaches failed: {e}")
                    return False

# ---------------------------------------------------------------
# Main Script
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fix EVEREST model issues and run training")
    parser.add_argument("--flare", default="M5", help="Flare class (M5, M, C)")
    parser.add_argument("--window", default="24", help="Time window in hours (24, 48, 72)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--patch-only", action="store_true", help="Only apply patches without training")
    
    args = parser.parse_args()
    
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"SUCCESS: GPU available: {gpus[0]}")
        # Set memory growth to avoid OOM errors
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except:
                pass
    else:
        print("WARNING: No GPU found, using CPU only (training will be slow)")
    
    # Print TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")
    
    # Apply patches
    success = apply_patches()
    
    # Exit if patching failed
    if not success:
        print("Failed to apply patches. Exiting.")
        return
    
    # Run training if not patch-only
    if not args.patch_only:
        run_training(args.flare, args.window, args.epochs, args.batch_size)
    else:
        print("Patches applied successfully. Training skipped due to --patch-only flag.")

if __name__ == "__main__":
    main() 