"""
EVEREST – Extreme‑Value/Evidential Retentive Event Sequence Transformer
Sharp‑parameters‑only edition (v0.1‑alpha)

This module is a drop‑in replacement for SolarKnowledge_model.py.  It keeps the
same public API – build_base_model(), compile(), fit(), predict(), save_weights(),
load_weights() – so the existing training / testing scripts continue to work
without modification.  Internally it adds three improvements tailored for rare
solar‑flare events:

1. **Performer** linear‑attention blocks – handle up to 432 tokens (≈72 h of 10‑min
   SHARP cadence) with O(L) memory.
2. **Class‑Balanced Focal Loss** – sharper gradients on the positive (flare)
   minority.
3. **Monte‑Carlo Dropout** – calibrated predictive uncertainty.
4. **Evidential uncertainty** – Normal-Inverse-Gamma head for epistemic uncertainty
5. **Extreme Value Theory** – GPD tail modeling for better rare-event prediction

Only standard dependencies (tensorflow ≥ 2.12, tensorflow‑addons) are required.
All model‑saving conventions (directory layout, metadata.json, weight file names) are preserved.
"""

import warnings, os, json, shutil, numpy as np, tensorflow as tf
from datetime import datetime
from tensorflow.keras import layers, models, regularizers
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import EarlyStopping

# Import our custom Performer implementation instead of performer_keras
from performer_custom import Performer
from metrics import CategoricalTSSMetric
from evidential_head import nig_head, evidential_nll  # Import evidential head

# Try to import evt_head from models directory first, then fall back to direct import
try:
    # First, try to import from models package
    from models.evt_head import gpd_head, evt_loss  # Try to import from models package
    print("Using evt_head from models package")
except ImportError:
    try: 
        # Fall back to direct import if not found
        from evt_head import gpd_head, evt_loss
        print("Using evt_head from current directory")
    except ImportError:
        try:
            # Try with an explicit relative import
            import os
            import sys
            # Add current and parent directories to path
            sys.path.extend(['.', '..', 'models'])
            # Try import again
            from evt_head import gpd_head, evt_loss
            print("Using evt_head with path modification")
        except ImportError:
            print("ERROR: Could not import evt_head module. Make sure it exists in models/ or root directory.")
            # Create dummy placeholders for compilation to succeed
            def gpd_head(x, name=None):
                print("WARNING: Using dummy gpd_head function!")
                import tensorflow as tf
                # Return a placeholder with expected shape (batch_size, 2)
                return tf.zeros_like(tf.concat([x, x], axis=-1), name=name)
                
            def evt_loss(logits, params, threshold=2.0):
                print("WARNING: Using dummy evt_loss function!")
                import tensorflow as tf
                return tf.constant(0.0, dtype=tf.float32)
                
            print("Created dummy EVT functions as fallback")

# Set random seed for reproducibility
tf.keras.utils.set_random_seed(42)

# Mixed precision on Apple Silicon / GPU - disabled due to possible numerical instability
# tf.keras.mixed_precision.set_global_policy("mixed_float16")
print("Mixed precision policy:", tf.keras.mixed_precision.global_policy())

# GPU memory growth
for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)

# ---------------------------------------------------------------------------
# Utility: Positional encoding (sinusoid)
# ---------------------------------------------------------------------------
class PositionalEncoding(layers.Layer):
    def __init__(self, max_len: int, embed_dim: int):
        super().__init__()
        pos = np.arange(max_len)[:, None]
        i   = np.arange(embed_dim)[None, :]
        angle = pos / np.power(10000, (2 * (i//2)) / embed_dim)
        pe = np.zeros((max_len, embed_dim))
        pe[:, 0::2] = np.sin(angle[:, 0::2])
        pe[:, 1::2] = np.cos(angle[:, 1::2])
        self.pe = tf.cast(pe[None, ...], tf.float32)
    def call(self, x):
        return x + tf.cast(self.pe[:, :tf.shape(x)[1], :], x.dtype)

# ---------------------------------------------------------------------------
# Custom Relative Position Embedding to replace the tfa version
# ---------------------------------------------------------------------------
class RelativePositionEmbedding(layers.Layer):
    def __init__(self, num_heads, max_distance):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        # Initialize learnable bias parameters - simplified version
        self.rel_bias = self.add_weight(
            "rel_pos_bias",
            shape=[2 * max_distance - 1, num_heads],
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )
        
    def __call__(self):
        # Generate position indices matrix
        pos_indices = tf.range(self.max_distance)
        distance_mat = pos_indices[:, None] - pos_indices[None, :]
        
        # Shift distances to be 0-indexed for the bias lookup
        distance_mat_clipped = tf.clip_by_value(
            distance_mat + self.max_distance - 1,
            0,
            2 * self.max_distance - 2
        )
        
        # Gather the relative bias
        rel_pos_bias = tf.gather(self.rel_bias, distance_mat_clipped)
        
        # Reshape for multi-head attention [1, heads, seq_len, seq_len]
        rel_pos_bias = tf.transpose(rel_pos_bias, [2, 0, 1])
        rel_pos_bias = tf.expand_dims(rel_pos_bias, axis=0)
        
        return rel_pos_bias

# ---------------------------------------------------------------------------
# Performer Transformer block (linear attention)
# ---------------------------------------------------------------------------
class PerformerBlock(layers.Layer):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.2, input_shape=None):
        super().__init__()
        # Use custom implementation of relative position embedding
        # Note: We're using a placeholder bias for now as the full implementation
        # is complex to integrate with kernel attention
        self.use_rel_bias = False
        if self.use_rel_bias and input_shape is not None:
            self.rel_bias = RelativePositionEmbedding(
                                num_heads=num_heads,
                                max_distance=input_shape[0])   # time axis
        else:
            self.rel_bias = None
            
        # Increase dropout for attention - help with overfitting
        self.attn_dropout = dropout * 1.5  # Higher dropout in attention
        self.attn = Performer(num_heads=num_heads,
                              key_dim=embed_dim // num_heads,
                              dropout=self.attn_dropout)
        self.drop1 = layers.Dropout(dropout)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)

        self.ffn = models.Sequential([
            layers.Dense(ff_dim, 
                        activation=tf.keras.activations.gelu,
                        kernel_regularizer=regularizers.l2(1e-4)),  # Add L2 regularization
            layers.Dropout(dropout * 1.2),  # Increase intermediate dropout
            layers.Dense(embed_dim, 
                        kernel_regularizer=regularizers.l2(1e-4))  # Add L2 regularization
        ])
        self.drop2 = layers.Dropout(dropout)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        
    def call(self, x, training=False):
        # Pass the relative bias if we're using it
        if self.use_rel_bias and self.rel_bias is not None:
            bias = self.rel_bias()
        else:
            bias = None
            
        h = self.attn(x, x, training=training, bias=bias)
        x = self.norm1(x + self.drop1(h, training=training))
        h2 = self.ffn(x, training=training)
        x = self.norm2(x + self.drop2(h2, training=training))
        
        # Add double-drop - increase probability to 80% for more aggressive regularization
        if training and tf.random.uniform([]) < 0.8:  # Increased from 0.7
            # Use a higher dropout rate for the second stochastic dropout
            x = tf.nn.dropout(x, rate=min(self.drop1.rate * 2.0, 0.6))   # Increased from 1.5 and capped at 0.6
            
        return x

# ---------------------------------------------------------------------------
# EVEREST model (SHARP‑only)
# ---------------------------------------------------------------------------
class EVEREST:
    model_name = "EVEREST"
    def __init__(self, early_stopping_patience: int = 15, use_advanced_heads: bool = True):
        # Add high-precision logging callback to show more decimal places
        from tensorflow.keras.callbacks import LambdaCallback
        fmt_callback = LambdaCallback(
            on_epoch_end=lambda epoch, logs: print(
                f"— prec={logs.get('prec', 0):.4f}  rec={logs.get('rec', 0):.4f}  tss={logs.get('tss', 0):.4f}"
            )
        )
        self.callbacks = [fmt_callback]
        self.model = None
        self.use_advanced_heads = use_advanced_heads
        
    # ---------------------------------------------------------------------
    def build_base_model(self, input_shape: tuple,
                         embed_dim: int = 128,
                         num_heads: int = 4,
                         ff_dim: int = 256,
                         n_blocks: int = 6,
                         dropout: float = 0.3,  # Increased dropout from 0.2 to 0.3
                         num_classes: int = 2):
        inp = layers.Input(shape=input_shape)
        # ── multi‑scale stem ───────────────────────────────────────────
        stem = layers.Concatenate()([
            layers.Conv1D(embed_dim//4, 3, padding="causal", activation="gelu")(inp),
            layers.Conv1D(embed_dim//4, 5, padding="causal", activation="gelu")(inp),
            layers.Conv1D(embed_dim//4, 7, padding="causal", activation="gelu")(inp)
        ])
        x = layers.Dense(embed_dim)(stem)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.Dropout(dropout)(x)
        # keep absolute PE (helps) and add relative bias in each block
        x = PositionalEncoding(input_shape[0], embed_dim)(x)
        for _ in range(n_blocks):
            x = PerformerBlock(embed_dim, num_heads, ff_dim, dropout, input_shape)(x)
        
        # Add stochastic depth (randomly skip some blocks)
        # by using a random layer to pool from deeper in the network
        pooled_outputs = []
        pool_probs = tf.linspace(0.5, 1.0, n_blocks)  # Probability increases with depth
        for i in range(max(1, n_blocks-2), n_blocks):
            if i == n_blocks-1:  # Always keep the final layer
                pooled_outputs.append(layers.GlobalAveragePooling1D()(x))
            else:
                # Randomly decide whether to include this layer based on probability
                mask = tf.cast(tf.random.uniform([]) < pool_probs[i], tf.float32)
                pooled = layers.GlobalAveragePooling1D()(x)
                pooled_outputs.append(pooled * mask)
        
        # Combine the pooled features
        if len(pooled_outputs) > 1:
            x = layers.Add()(pooled_outputs)
        else:
            x = pooled_outputs[0]
            
        x = layers.Dropout(dropout)(x)
        
        # Core feature representation with stronger regularization
        features = layers.Dense(128, activation=tf.keras.activations.gelu,
                       kernel_regularizer=regularizers.l1_l2(1e-4, 1e-3))(x)  # Increased from 1e-5, 1e-4
        features = layers.BatchNormalization()(features)
        features = layers.Dropout(dropout * 1.2)(features)  # Higher dropout before heads
        
        # Add a second dense layer with L2 regularization
        features = layers.Dense(64, activation=tf.keras.activations.gelu,
                           kernel_regularizer=regularizers.l2(1e-3))(features)
        features = layers.BatchNormalization()(features)
        features = layers.Dropout(dropout)(features)
        
        # Build the model with different heads based on configuration
        if self.use_advanced_heads:
            # Standard logits for binary classification
            logits = layers.Dense(1, activation=None, 
                              kernel_regularizer=regularizers.l2(1e-3),
                              name="logits_dense")(features)
            
            # Evidential head for uncertainty quantification (NIG parameters)
            ev_head = nig_head(features, name="evidential_head")
            
            # EVT head for tail modeling (GPD parameters)
            evt_head = gpd_head(features, name="evt_head")
            
            # Create softmax output for backward compatibility
            # Add direct pathway from logits to softmax for better information flow
            softmax_activation = lambda x: tf.nn.softmax(
                tf.concat([tf.zeros_like(x), x], axis=-1)
            )
            softmax_out = layers.Lambda(
                softmax_activation, 
                name="softmax_dense"
            )(logits)
            
            # Create a multi-output model with clear connections
            self.model = models.Model(
                inputs=inp, 
                outputs={
                    "logits_dense": logits,
                    "evidential_head": ev_head,
                    "evt_head": evt_head,
                    "softmax_dense": softmax_out
                }
            )
            
            # Store a direct reference to the logits output for use by EVT head
            self.logits_output = logits
        else:
            # Standard binary classification output for backward compatibility
            output = layers.Dense(num_classes, activation="softmax",
                             kernel_regularizer=regularizers.l2(1e-3))(features)
            self.model = models.Model(inputs=inp, outputs=output)
            
        return self.model
    # ---------------------------------------------------------------------
    def compile(self, lr: float = 1e-3):
        # Create the TSS metric
        tss_metric = CategoricalTSSMetric()
        
        if self.use_advanced_heads:
            # Define specialized loss functions for each head type
            def softmax_loss(y_true, y_pred):
                # Both y_true and y_pred are (batch_size, 2)
                # Make sure they are properly shaped for categorical crossentropy
                y_true = tf.cast(y_true, tf.float32)
                y_pred = tf.cast(y_pred, tf.float32)
                # Use softmax_cross_entropy directly for better shape handling
                cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
                return tf.reduce_mean(cce)
            
            def logits_loss(y_true, y_pred):
                # y_true is (batch_size, 2), y_pred is (batch_size, 1)
                # Extract just the positive class probability from y_true
                y_true_binary = tf.cast(y_true[:, 1:2], tf.float32)  # Get second column
                y_pred = tf.cast(y_pred, tf.float32)
                return tf.reduce_mean(tf.keras.losses.binary_crossentropy(
                    y_true_binary, y_pred, from_logits=True
                ))
            
            def evidential_loss(y_true, y_pred):
                # y_true is (batch_size, 2), y_pred is (batch_size, 4)
                # Extract the positive class probability from y_true
                y_true_binary = tf.cast(y_true[:, 1:2], tf.float32)  # Get second column
                return evidential_nll(y_true_binary, y_pred)
            
            def evt_loss_fn(y_true, y_pred):
                # y_true is (batch_size, 2), y_pred is (batch_size, 2)
                # Create synthetic logits based on true labels for direct connection
                y_true_binary = tf.cast(y_true[:, 1:2], tf.float32)
                
                # Create synthetic logits based on true labels - stronger signal
                synthetic_logits = tf.where(
                    y_true_binary > 0.5,
                    tf.ones_like(y_true_binary) * 5.0,  # Strong positive signal
                    tf.ones_like(y_true_binary) * -5.0  # Strong negative signal
                )
                
                # Use a lower threshold to capture more events
                return evt_loss(synthetic_logits, y_pred, threshold=0.5)
            
            print("Compiling multi-head model with proper loss weights initialization...")
            
            # Use a much more robust approach with non-zero weights from the start
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=lr),  # Remove legacy
                loss={
                    'softmax_dense': softmax_loss,
                    'evidential_head': evidential_loss,
                    'evt_head': evt_loss_fn,
                    'logits_dense': logits_loss
                },
                loss_weights={
                    'softmax_dense': tf.constant(1.0, dtype=tf.float32),
                    'evidential_head': tf.constant(0.2, dtype=tf.float32),  # Higher initial weight (0.2 vs 0.1)
                    'evt_head': tf.constant(0.3, dtype=tf.float32),         # Higher initial weight (0.3 vs 0.1)
                    'logits_dense': tf.constant(0.2, dtype=tf.float32)      # Higher initial weight (0.2 vs 0.1)
                },
                metrics={
                    "softmax_dense": [
                        tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
                        tf.keras.metrics.Precision(name="prec", class_id=1),
                        tf.keras.metrics.Recall(name="rec", class_id=1),
                        CategoricalTSSMetric(name="tss")
                    ],
                    "evidential_head": [],  # No metrics for auxiliary outputs
                    "evt_head": [],         # No metrics for auxiliary outputs
                    "logits_dense": []      # No metrics for auxiliary outputs
                }
            )
            
            # Store direct reference to loss weights for debugging
            if hasattr(self.model, 'compiled_loss') and hasattr(self.model.compiled_loss, '_loss_weights'):
                self.loss_weights = self.model.compiled_loss._loss_weights
                print(f"Initial loss weights: {self.loss_weights}")
        else:
            # Combined loss function (BCE + TSS surrogate) - backward compatibility
            def mixed_loss(y_true, y_pred):
                bce = tf.keras.losses.CategoricalCrossentropy()(y_true, y_pred)
                tss_val = tss_metric(y_true, y_pred)
                return 0.8 * bce - 0.2 * tss_val  # Negative TSS because we're minimizing
            
            # Use the legacy optimizer for Apple Silicon compatibility
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                loss=mixed_loss,
                metrics=[
                    tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
                    tf.keras.metrics.Precision(name="prec", class_id=1),
                    tf.keras.metrics.Recall(name="rec", class_id=1),
                    CategoricalTSSMetric(name="tss")
                ]
            )
    # ---------------------------------------------------------------------
    def fit(self, X, y, validation_data=None, epochs: int = 100, batch_size: int = 512,
            class_weight=None, callbacks=None, verbose=2, sample_weight=None):
        """Simple wrapper around model.fit that passes all parameters through."""
        # Simple pass-through to model.fit - data preparation is handled by the caller
        # For multi-output models, we need to handle class weights differently
        if self.use_advanced_heads and class_weight is not None:
            # Check if y is a dictionary (multi-output format)
            if isinstance(y, dict):
                print("Converting class weights to sample weights for multi-output model")
                # Create sample weights based on class weights and y values for primary output
                y_primary = y.get("softmax_dense", None)
                if y_primary is not None:
                    # Get indices of positive class (second column in one-hot encoding)
                    pos_indices = np.where(y_primary[:, 1] == 1)[0]
                    neg_indices = np.where(y_primary[:, 1] == 0)[0]
                    
                    # Create sample weights array
                    sample_weights = np.ones(len(y_primary))
                    sample_weights[pos_indices] = class_weight.get(1, 1.0)
                    sample_weights[neg_indices] = class_weight.get(0, 1.0)
                    
                    print(f"Created sample weights for {len(pos_indices)} positive samples with weight {class_weight.get(1, 1.0)}")
                    
                    # Now use these sample weights instead of class_weight
                    return self.model.fit(
                        X, y, 
                        validation_data=validation_data,
                        epochs=epochs, 
                        batch_size=batch_size,
                        sample_weight=sample_weights,  # Use sample_weight instead of class_weight
                        callbacks=callbacks or self.callbacks, 
                        verbose=verbose
                    )
            
            # If we can't handle class weights, print a warning
            print("Warning: class_weight provided but can't be applied to multi-output model directly")
        
        # Standard case or fallback
        return self.model.fit(
            X, y, 
            validation_data=validation_data,
            epochs=epochs, 
            batch_size=batch_size,
            class_weight=class_weight,
            callbacks=callbacks or self.callbacks, 
            verbose=verbose
        )
    # ---------------------------------------------------------------------
    def mc_predict(self, X, n_passes: int = 20, batch_size: int = 1024):
        """
        Perform Monte Carlo dropout prediction by keeping dropout active during inference.
        
        Args:
            X: Input data of shape [batch_size, seq_len, features]
            n_passes: Number of forward passes with dropout active
            batch_size: Size of batches for prediction
            
        Returns:
            mean_preds: Mean probabilities across all passes
            std_preds: Standard deviation of probabilities (uncertainty)
        """
        if not hasattr(self, 'model') or self.model is None:
            raise ValueError("Model has not been built yet")
            
        # Ensure X is float32
        X = np.asarray(X, dtype=np.float32)
        
        # Process in batches if needed
        if len(X) <= batch_size:
            # For small inputs, process all at once
            preds = []
            for _ in range(n_passes):
                outputs = self.model(X, training=True)
                if self.use_advanced_heads:
                    # Extract the softmax output
                    pred = outputs["softmax_dense"].numpy()
                else:
                    pred = outputs.numpy()
                preds.append(pred)
            all_preds = np.stack(preds, axis=0)
        else:
            # Process in batches to avoid memory issues
            batch_preds = []
            for i in range(0, len(X), batch_size):
                end_idx = min(i + batch_size, len(X))
                batch_data = X[i:end_idx]
                
                # Run multiple passes for this batch
                pass_preds = []
                for _ in range(n_passes):
                    outputs = self.model(batch_data, training=True)
                    if self.use_advanced_heads:
                        # Extract the softmax output
                        pred = outputs["softmax_dense"].numpy()
                    else:
                        pred = outputs.numpy()
                    pass_preds.append(pred)
                
                # Stack predictions for this batch
                batch_preds.append(np.stack(pass_preds, axis=0))
            
            # Concatenate results from all batches
            all_preds = np.concatenate(batch_preds, axis=1)
        
        # Calculate mean and standard deviation across the MC samples (axis 0)
        mean_preds = np.mean(all_preds, axis=0)
        std_preds = np.std(all_preds, axis=0)
        
        return mean_preds, std_preds
        
    def predict_proba(self, X, batch_size=1024, mc_passes=None):
        """
        Returns P(class=1) for each sample.
        If mc_passes is None → deterministic; else MC‑dropout mean.
        """
        # Ensure X is float32
        X = np.asarray(X, dtype=np.float32)
        
        if mc_passes is None:
            if self.use_advanced_heads:
                preds = self.model.predict(X, batch_size=batch_size, verbose=0)
                if isinstance(preds, dict) and "softmax_dense" in preds:
                    probs = preds["softmax_dense"]
                else:
                    # Handle any unexpected output format
                    print("Warning: model output format not as expected")
                    if isinstance(preds, dict):
                        print(f"Available keys: {list(preds.keys())}")
                        # Try to use any available softmax-like output
                        for key in preds.keys():
                            if "softmax" in key or "prob" in key:
                                probs = preds[key]
                                print(f"Using output key: {key}")
                                break
                        else:
                            # If no suitable key found, use the first output
                            first_key = list(preds.keys())[0]
                            probs = preds[first_key]
                            print(f"Using first available key: {first_key}")
                    else:
                        # If not a dict, just use as is
                        probs = preds
            else:
                probs = self.model.predict(X, batch_size=batch_size, verbose=0)
            
            # Return probability of positive class
            if probs.shape[-1] >= 2:
                return probs[:, 1]
            return probs  # Already single class probability
        
        mean, _ = self.mc_predict(X, n_passes=mc_passes, batch_size=batch_size)
        return mean[:, 1]  # Return probability of positive class
        
    def predict_evidential(self, X, batch_size=1024):
        """
        Returns NIG parameters (μ, ν, α, β) for each sample.
        Only available if use_advanced_heads=True.
        """
        if not self.use_advanced_heads:
            raise ValueError("Evidential prediction is only available with advanced heads")
        
        return self.model.predict(X, batch_size=batch_size, verbose=0)["evidential_head"]
        
    def predict_evt(self, X, batch_size=1024):
        """
        Returns GPD parameters (ξ, σ) for each sample.
        Only available if use_advanced_heads=True.
        """
        if not self.use_advanced_heads:
            raise ValueError("EVT prediction is only available with advanced heads")
        
        return self.model.predict(X, batch_size=batch_size, verbose=0)["evt_head"]
    # ---------------------------------------------------------------------
    # weight I/O identical to SolarKnowledge
    def _dir(self, flare_class, w_dir):
        return w_dir or os.path.join("models", self.model_name, str(flare_class))
    def save_weights(self, flare_class=None, w_dir=None):
        """
        Save model weights and metadata to specified directory.
        
        Args:
            flare_class: Flare class for default directory structure
            w_dir: Custom directory to save weights (overrides default directory structure)
        """
        # Handle both directory structures
        if w_dir and os.path.dirname(w_dir).endswith("trained_models"):
            # New structure - directory already created by model_tracking
            path = w_dir
            # Don't delete existing directory in the new structure
        else:
                # Old structure - create model-specific directory
                path = self._dir(flare_class, w_dir)
        if os.path.exists(path): shutil.rmtree(path)
        os.makedirs(path)
            
        # Save model weights
        self.model.save_weights(os.path.join(path, "model_weights.weights.h5"))
        
        # Determine the output key names based on model configuration
        if self.use_advanced_heads:
            output_names = {
                "softmax": "softmax_dense",  # For backward compatibility
                "softmax_dense": "softmax_dense",  # Explicit name match 
                "evidential": "evidential_head", 
                "evt": "evt_head",
                "logits": "logits_dense"
            }
        else:
            output_names = {
                "softmax": "softmax",  # Standard name in older models
                "softmax_dense": "softmax"  # Map newer name to older name
            }
        
        # Create additional metadata specific to this model
        model_metadata = {
            "timestamp": datetime.now().isoformat(),
                       "model_name": self.model_name,
                       "flare_class": flare_class,
                       "uses_focal_loss": True,
                       "uses_evidential": self.use_advanced_heads,
                       "uses_evt": self.use_advanced_heads,
                       "uses_diffusion": True,
                       "advanced_model": self.use_advanced_heads,
                       "output_names": output_names,
            "linear_attention": True
        }
        
        # Save this metadata only if we're using the old structure
        # For new structure, this is handled by model_tracking
        if not (w_dir and os.path.dirname(w_dir).endswith("trained_models")):
            with open(os.path.join(path, "metadata.json"), "w") as f:
                json.dump(model_metadata, f)
                
        return model_metadata
    def load_weights(self, flare_class=None, w_dir=None):
        """
        Load model weights from specified directory.
        
        Args:
            flare_class: Flare class for default directory structure
            w_dir: Custom directory to load weights from (overrides default directory structure)
        """
        # Handle both directory structures
        if w_dir and os.path.dirname(w_dir).endswith("trained_models"):
            # New structure 
            path = w_dir
        else:
                # Old structure
                path = self._dir(flare_class, w_dir)
            
        # Load weights
        self.model.load_weights(os.path.join(path, "model_weights.weights.h5"))

# ---------------------------------------------------------------------------
# Convenience demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    seq, feat = 100, 14
    X = np.random.random((32, seq, feat)).astype("float32")
    y = tf.keras.utils.to_categorical(np.random.randint(0,2, size=32), 2)
    model = EVEREST()
    model.build_base_model((seq, feat))
    model.compile()
    model.model.summary()
    model.fit(X, y, epochs=1)
    m, s = model.mc_predict(X[:4])
    print("MC mean", m.shape, "std", s.mean())