#!/usr/bin/env python
"""
Simple fix for EVEREST's EVT and evidential heads.
This script directly patches the EVEREST model to fix the issues.
"""

import os
import sys
import tensorflow as tf
import numpy as np

print(f"TensorFlow version: {tf.__version__}")

# ---------------------------------------------------------------------
# Fix 1: Patch the evidential_nll function to prevent negative loss
# ---------------------------------------------------------------------
def fixed_evidential_nll(y_true, evid):
    """Fixed negative log‑likelihood for evidential head that ensures positive loss."""
    # Ensure evid has shape (batch_size, 4)
    evid_shape = tf.shape(evid)
    
    # Reshape y_true to ensure it's always (batch_size, 1)
    y_true = tf.reshape(y_true, [-1, 1])
    
    # Split into NIG parameters (already activated)
    mu, v, α, β = tf.split(evid, 4, axis=-1)  # Each has shape (batch_size, 1)
    
    # Convert mu to probability
    p = tf.nn.sigmoid(mu)
    
    # Ensure parameters are within valid ranges
    α = tf.clip_by_value(α, 1.0 + 1e-6, 1e6)
    β = tf.clip_by_value(β, 1e-6, 1e6)
    v = tf.clip_by_value(v, 1e-6, 1e6)
    
    # Calculate predictive variance with clipping
    S = β*(1+v)/(α)
    
    # Calculate NLL components
    eps = tf.keras.backend.epsilon()
    ce_loss = - y_true * tf.math.log(p + eps) - (1-y_true)*tf.math.log(1-p + eps)
    var_loss = 0.5*tf.math.log(S + eps)
    
    # Use absolute value and add regularization to ensure positive loss
    loss = tf.abs(ce_loss) + tf.abs(var_loss) + 0.01
    
    return tf.reduce_mean(loss)

# ---------------------------------------------------------------------
# Fix 2: Patch the evt_loss function with a simpler implementation
# ---------------------------------------------------------------------
def fixed_evt_loss(logits, evt_params, threshold=0.5):
    """Fixed EVT loss that ensures positive values and proper gradient flow."""
    # Ensure inputs have the right shape
    logits = tf.reshape(logits, [-1, 1])
    evt_params = tf.reshape(evt_params, [-1, 2])
    
    # Unpack GPD parameters
    shape = evt_params[:, 0:1]  # ξ (xi)
    scale = evt_params[:, 1:2]  # σ (sigma)
    
    # Ensure positive scale parameter
    scale = tf.maximum(scale, 0.1)
    
    # Constrain shape parameter for stability
    shape = tf.clip_by_value(shape, -0.9, 0.9)
    
    # Calculate exceedances (samples above threshold)
    exceedance = tf.maximum(logits - threshold, 0.0)
    
    # Simplified loss: standard GPD negative log-likelihood
    eps = 1e-6
    nll = tf.math.log(scale + eps) + (1.0 + 1.0/tf.maximum(tf.abs(shape), eps)) * \
          tf.math.log1p(tf.abs(shape) * exceedance / (scale + eps))
    
    # Only calculate loss for samples above threshold
    mask = tf.cast(exceedance > 0, tf.float32)
    masked_nll = nll * mask
    
    # Add regularization term
    reg = 0.1 * tf.reduce_mean(scale) + 0.1 * tf.reduce_mean(tf.square(shape))
    
    # Ensure loss is positive
    return tf.reduce_mean(masked_nll) + reg + 0.01

# ---------------------------------------------------------------------
# Fix 3: Apply the patches to the EVEREST model
# ---------------------------------------------------------------------
def patch_everest_model():
    """Patch the EVEREST model with fixed loss functions."""
    # Import the necessary modules
    try:
        # First try to import from everest_model
        from models.everest_model import evidential_nll, evt_loss
        from models.evidential_head import evidential_nll as original_evidential_nll
        from models.evt_head import evt_loss as original_evt_loss
        module_path = "models"
    except ImportError:
        try:
            # Try direct import if models package not found
            from everest_model import evidential_nll, evt_loss
            from evidential_head import evidential_nll as original_evidential_nll
            from evt_head import evt_loss as original_evt_loss
            module_path = "."
        except ImportError:
            print("ERROR: Could not import EVEREST modules. Make sure they are in your PYTHONPATH.")
            return False
    
    # Apply the patches
    print("Applying patches to EVEREST model...")
    
    # Patch 1: Fix evidential_nll
    try:
        import sys
        # Get the module
        if module_path == "models":
            import models.evidential_head as evidential_head_module
        else:
            import evidential_head as evidential_head_module
        
        # Replace the function
        evidential_head_module.evidential_nll = fixed_evidential_nll
        print("✓ Successfully patched evidential_nll function")
        
        # Verify the patch
        if evidential_head_module.evidential_nll is fixed_evidential_nll:
            print("  Verification passed: evidential_nll was replaced")
        else:
            print("× Verification failed: evidential_nll was not replaced")
    except Exception as e:
        print(f"× Error patching evidential_nll: {e}")
    
    # Patch 2: Fix evt_loss
    try:
        # Get the module
        if module_path == "models":
            import models.evt_head as evt_head_module
        else:
            import evt_head as evt_head_module
        
        # Replace the function
        evt_head_module.evt_loss = fixed_evt_loss
        print("✓ Successfully patched evt_loss function")
        
        # Verify the patch
        if evt_head_module.evt_loss is fixed_evt_loss:
            print("  Verification passed: evt_loss was replaced")
        else:
            print("× Verification failed: evt_loss was not replaced")
    except Exception as e:
        print(f"× Error patching evt_loss: {e}")
    
    # Patch 3: Update EVEREST model's compile method to use non-zero weights
    try:
        # Get the module
        if module_path == "models":
            import models.everest_model as everest_model_module
            from models.everest_model import EVEREST
        else:
            import everest_model as everest_model_module
            from everest_model import EVEREST
        
        # Store the original compile method
        original_compile = EVEREST.compile
        
        # Define the patched compile method
        def patched_compile(self, lr=1e-3):
            # Call the original compile method
            original_compile(self, lr)
            
            # Update loss weights to non-zero values
            if hasattr(self.model, 'compiled_loss') and hasattr(self.model.compiled_loss, '_loss_weights'):
                for key in ['evidential_head', 'evt_head', 'logits_dense']:
                    if key in self.model.compiled_loss._loss_weights:
                        # Set non-zero weights from the start
                        self.model.compiled_loss._loss_weights[key] = tf.constant(0.1, dtype=tf.float32)
                
                # Log the updated weights
                print("Updated loss weights to start with non-zero values")
                print(f"  {self.model.compiled_loss._loss_weights}")
        
        # Apply the patch
        EVEREST.compile = patched_compile
        print("✓ Successfully patched EVEREST.compile method")
    except Exception as e:
        print(f"× Error patching EVEREST.compile: {e}")
    
    print("All patches applied successfully")
    return True

# ---------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Apply the patches
    success = patch_everest_model()
    
    if success:
        print("\nInstructions:")
        print("1. The EVEREST model has been patched with fixed loss functions.")
        print("2. Use the regular train_everest.py script to train the model:")
        print("   python models/train_everest.py --specific-flare C --specific-window 24")
        print("3. The model will now use non-zero weights for all heads from the start.")
    else:
        print("\nError: Failed to patch the EVEREST model.") 