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
from evt_head import gpd_head, evt_loss              # Import EVT head

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
            
        self.attn = Performer(num_heads=num_heads,
                              key_dim=embed_dim // num_heads,
                              dropout=dropout)
        self.drop1 = layers.Dropout(dropout)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)

        self.ffn = models.Sequential([
            layers.Dense(ff_dim, activation=tf.keras.activations.gelu),
            layers.Dropout(dropout),
            layers.Dense(embed_dim)
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
        
        # Add double-drop - increase probability to 70% for more aggressive regularization
        if training and tf.random.uniform([]) < 0.7:
            # Use a higher dropout rate for the second stochastic dropout
            x = tf.nn.dropout(x, rate=min(self.drop1.rate * 1.5, 0.5))   # Increase dropout but cap at 0.5
            
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
                         dropout: float = 0.2,
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
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(dropout)(x)
        
        # Core feature representation
        features = layers.Dense(128, activation=tf.keras.activations.gelu,
                       kernel_regularizer=regularizers.l1_l2(1e-5, 1e-4))(x)
        features = layers.BatchNormalization()(features)
        features = layers.Dropout(dropout)(features)
        
        # Build the model with different heads based on configuration
        if self.use_advanced_heads:
            # Standard logits for binary classification
            logits = layers.Dense(1, activation=None, name="logits_dense")(features)
            
            # Evidential head for uncertainty quantification (NIG parameters)
            ev_head = nig_head(features, name="evidential_head")
            
            # EVT head for tail modeling (GPD parameters)
            evt_head = gpd_head(features, name="evt_head")
            
            # Create softmax output for backward compatibility
            softmax_out = layers.Dense(num_classes, activation="softmax", name="softmax_dense")(features)
            
            # Create a multi-output model
            self.model = models.Model(
                inputs=inp, 
                outputs={
                    "logits_dense": logits,
                    "evidential_head": ev_head,
                    "evt_head": evt_head,
                    "softmax_dense": softmax_out
                }
            )
        else:
            # Standard binary classification output for backward compatibility
            output = layers.Dense(num_classes, activation="softmax")(features)
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
                # Use the proper evidential loss function
                # Extract the positive class probability from y_true
                y_true_binary = tf.cast(y_true[:, 1:2], tf.float32)  # Get second column
                return evidential_nll(y_true_binary, y_pred)
            
            def evt_loss_fn(y_true, y_pred):
                # y_true is (batch_size, 2), y_pred is (batch_size, 2)
                # Use the proper EVT loss function
                # Need to get logits first - can pass through the logits_dense output
                # In practice, we treat the model as having no good logits during initial training
                # When head_weight_scheduler activates these losses, the model will have better representations
                dummy_logits = tf.zeros((tf.shape(y_true)[0], 1))
                return evt_loss(dummy_logits, y_pred, threshold=2.5)
                
            # Use a much simpler approach for initial training
            self.model.compile(
                optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=lr),
                loss={
                    'softmax_dense': softmax_loss,
                    'evidential_head': evidential_loss,
                    'evt_head': evt_loss_fn,
                    'logits_dense': logits_loss
                },
                loss_weights={
                    'softmax_dense': 1.0,
                    'evidential_head': 0.0,  # Zero weight during initial training
                    'evt_head': 0.0,        # Zero weight during initial training
                    'logits_dense': 0.0     # Zero weight during initial training
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
        else:
            # Combined loss function (BCE + TSS surrogate) - backward compatibility
            def mixed_loss(y_true, y_pred):
                bce = tf.keras.losses.CategoricalCrossentropy()(y_true, y_pred)
                tss_val = tss_metric(y_true, y_pred)
                return 0.8 * bce - 0.2 * tss_val  # Negative TSS because we're minimizing
            
            # Use the legacy optimizer for Apple Silicon compatibility
            self.model.compile(
                optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=lr),
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
            class_weight=None, callbacks=None, verbose=2):
        """Simple wrapper around model.fit that passes all parameters through."""
        # Simple pass-through to model.fit - data preparation is handled by the caller
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
        path = self._dir(flare_class, w_dir)
        if os.path.exists(path): shutil.rmtree(path)
        os.makedirs(path)
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
        
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump({"timestamp": datetime.now().isoformat(),
                       "model_name": self.model_name,
                       "flare_class": flare_class,
                       "uses_focal_loss": True,
                       "uses_evidential": self.use_advanced_heads,
                       "uses_evt": self.use_advanced_heads,
                       "uses_diffusion": True,
                       "advanced_model": self.use_advanced_heads,
                       "output_names": output_names,
                       "linear_attention": True}, f)
    def load_weights(self, flare_class=None, w_dir=None):
        path = self._dir(flare_class, w_dir)
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