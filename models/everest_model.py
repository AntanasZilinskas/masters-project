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

Only standard dependencies (tensorflow ≥ 2.12, tensorflow‑addons) are required.
All model‑saving conventions (directory layout, metadata.json, weight file names) are preserved.
"""

import warnings, os, json, shutil, numpy as np, tensorflow as tf
from datetime import datetime
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping

# Import our custom Performer implementation instead of performer_keras
from performer_custom import Performer

# Set random seed for reproducibility
tf.keras.utils.set_random_seed(42)

# Mixed precision on Apple Silicon / GPU
tf.keras.mixed_precision.set_global_policy("mixed_float16")
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
# Performer Transformer block (linear attention)
# ---------------------------------------------------------------------------
class PerformerBlock(layers.Layer):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.2):
        super().__init__()
        # Use the custom Performer implementation for linear attention
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
        h = self.attn(x, x, training=training)
        x = self.norm1(x + self.drop1(h, training=training))
        h2 = self.ffn(x, training=training)
        return self.norm2(x + self.drop2(h2, training=training))

# ---------------------------------------------------------------------------
# EVEREST model (SHARP‑only)
# ---------------------------------------------------------------------------
class EVEREST:
    model_name = "EVEREST"
    def __init__(self, early_stopping_patience: int = 5):
        self.callbacks = [EarlyStopping(monitor="loss", patience=early_stopping_patience,
                                        restore_best_weights=True)]
        self.model = None
    # ---------------------------------------------------------------------
    def build_base_model(self, input_shape: tuple,
                         embed_dim: int = 128,
                         num_heads: int = 4,
                         ff_dim: int = 256,
                         n_blocks: int = 6,
                         dropout: float = 0.2,
                         num_classes: int = 2):
        inp = layers.Input(shape=input_shape)
        x = layers.Dense(embed_dim)(inp)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.Dropout(dropout)(x)
        x = PositionalEncoding(input_shape[0], embed_dim)(x)
        for _ in range(n_blocks):
            x = PerformerBlock(embed_dim, num_heads, ff_dim, dropout)(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Dense(128, activation=tf.keras.activations.gelu,
                         kernel_regularizer=regularizers.l1_l2(1e-5, 1e-4))(x)
        x = layers.Dropout(dropout)(x)
        out = layers.Dense(num_classes, activation="softmax",
                           activity_regularizer=regularizers.l2(1e-5))(x)
        self.model = models.Model(inp, out)
        return self.model
    # ---------------------------------------------------------------------
    def compile(self, lr: float = 1e-4, alpha: float = 0.25, gamma: float = 2.0):
        from tensorflow.keras.metrics import Accuracy

        # Try to import tensorflow_addons for focal loss, fall back to standard loss if not available
        try:
            import tensorflow_addons as tfa
            # Use TFA's SigmoidFocalCrossEntropy directly
            loss = tfa.losses.SigmoidFocalCrossEntropy(alpha=alpha, gamma=gamma)
            print("Using TensorFlow Addons SigmoidFocalCrossEntropy")
        except ImportError:
            # Fall back to categorical crossentropy
            loss = tf.keras.losses.CategoricalCrossentropy()
            print("TensorFlow Addons not available, using standard Categorical Crossentropy loss")
        
        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                           loss=loss,
                           metrics=[Accuracy(name="acc"), TrueSkillStatisticMetric()])
    # ---------------------------------------------------------------------
    def fit(self, X, y, validation_data=None, epochs: int = 100, batch_size: int = 512,
            class_weight=None):
        if class_weight is None:
            # Calculate class weight with safeguard against missing class
            class_counts = np.sum(y, 0)
            pos = max(1, class_counts[1])  # Ensure non-zero denominator
            weight = {0: 1.0, 1: max(5, len(y)/pos)}
        else:
            weight = class_weight
            
        return self.model.fit(X, y, epochs=epochs, batch_size=batch_size,
                              validation_data=validation_data,
                              class_weight=weight,
                              callbacks=self.callbacks, verbose=2)
    # ---------------------------------------------------------------------
    def mc_predict(self, X, n_passes: int = 20, batch_size: int = 1024):
        """
        Perform Monte Carlo dropout prediction by keeping dropout active during inference.
        
        Args:
            X: Input data of shape [batch_size, seq_len, features]
            n_passes: Number of forward passes with dropout active
            batch_size: Size of batches for prediction
            
        Returns:
            mean_preds: Mean predictions across all passes
            std_preds: Standard deviation of predictions (uncertainty)
        """
        # Process in batches if needed
        if len(X) <= batch_size:
            # For small inputs, process all at once
            preds = []
            for _ in range(n_passes):
                preds.append(self.model(X, training=True).numpy())
            all_preds = np.stack(preds, 0)
        else:
            # Process in batches to avoid memory issues
            batch_preds = []
            for i in range(0, len(X), batch_size):
                end_idx = min(i + batch_size, len(X))
                batch_data = X[i:end_idx]
                
                # Run multiple passes for this batch
                pass_preds = []
                for _ in range(n_passes):
                    pass_preds.append(self.model(batch_data, training=True).numpy())
                # Stack along the pass dimension first (axis=0)
                batch_preds.append(np.stack(pass_preds, 0))
            
            # Concatenate batches along the batch dimension (axis=1)
            all_preds = np.concatenate(batch_preds, axis=1)
        
        # Calculate mean and standard deviation across passes
        return all_preds.mean(0), all_preds.std(0)
    # ---------------------------------------------------------------------
    # weight I/O identical to SolarKnowledge
    def _dir(self, flare_class, w_dir):
        return w_dir or os.path.join("models", self.model_name, str(flare_class))
    def save_weights(self, flare_class=None, w_dir=None):
        path = self._dir(flare_class, w_dir)
        if os.path.exists(path): shutil.rmtree(path)
        os.makedirs(path)
        self.model.save_weights(os.path.join(path, "model_weights.weights.h5"))
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump({"timestamp": datetime.now().isoformat(),
                       "model_name": self.model_name,
                       "flare_class": flare_class,
                       "uses_focal_loss": True,
                       "linear_attention": True}, f)
    def load_weights(self, flare_class=None, w_dir=None):
        path = self._dir(flare_class, w_dir)
        self.model.load_weights(os.path.join(path, "model_weights.weights.h5"))

# ---------------------------------------------------------------------------
# True Skill Statistic metric (unchanged)
# ---------------------------------------------------------------------------
class TrueSkillStatisticMetric(tf.keras.metrics.Metric):
    def __init__(self, name="tss", **kw):
        super().__init__(name=name, **kw)
        self.tp = self.add_weight("tp", initializer="zeros")
        self.tn = self.add_weight("tn", initializer="zeros")
        self.fp = self.add_weight("fp", initializer="zeros")
        self.fn = self.add_weight("fn", initializer="zeros")
    def update_state(self, y_true, y_pred, sample_weight=None):
        yt = tf.cast(tf.argmax(y_true, 1), tf.bool)
        yp = tf.cast(tf.argmax(y_pred, 1), tf.bool)
        self.tp.assign_add(tf.reduce_sum(tf.cast(tf.logical_and( yt,  yp), tf.float32)))
        self.tn.assign_add(tf.reduce_sum(tf.cast(tf.logical_and(~yt, ~yp), tf.float32)))
        self.fp.assign_add(tf.reduce_sum(tf.cast(tf.logical_and(~yt,  yp), tf.float32)))
        self.fn.assign_add(tf.reduce_sum(tf.cast(tf.logical_and( yt, ~yp), tf.float32)))
    def result(self):
        sens = self.tp / (self.tp + self.fn + tf.keras.backend.epsilon())
        spec = self.tn / (self.tn + self.fp + tf.keras.backend.epsilon())
        return sens + spec - 1
    def reset_state(self):
        for v in self.variables: v.assign(0.)

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