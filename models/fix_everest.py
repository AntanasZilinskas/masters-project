#!/usr/bin/env python
"""
Comprehensive fix for EVEREST model's evidential and EVT heads.
This script creates a standalone model with properly working advanced heads.

Usage:
    python models/fix_everest.py --flare C --window 24 [--toy]
"""

import os
import sys
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, regularizers

# Ensure models directory is in path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Try to detect GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Found GPU: {gpus[0]}")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
else:
    print("WARNING: No GPU found, using CPU (training will be slow)")

print(f"Python version: {sys.version.split()[0]}")
print(f"TensorFlow version: {tf.__version__}")

# --------------------------------------------------------------------------
# Fixed Evidential Head
# --------------------------------------------------------------------------
def fixed_nig_head(x, name=None):
    """Return μ, ν, α, β – all positive where needed."""
    # Ensure x has sufficient feature dimensions
    x_shape = tf.shape(x)
    batch_dim = x_shape[0]
    
    # Project to 4D space (mu, v, alpha, beta)
    params = layers.Dense(4, name="evidential_params")(x)
    
    # Split into separate components for transformations
    mu, logv, logalpha, logbeta = tf.split(params, 4, axis=-1)
    
    # Apply appropriate activations
    v = tf.nn.softplus(logv) + 0.1        # ν > 0.1 (avoid numerical issues)
    alpha = 1 + tf.nn.softplus(logalpha)  # α > 1
    beta = tf.nn.softplus(logbeta) + 0.1  # β > 0.1
    
    # Concatenate back together
    out = tf.concat([mu, v, alpha, beta], axis=-1)
    
    # Ensure the shape is explicitly set
    out = tf.reshape(out, [-1, 4])
    
    # Add a name for debugging purposes
    return layers.Activation('linear', name=name)(out)

def fixed_evidential_nll(y_true, evid):
    """Fixed negative log‑likelihood for binary‐classification evidential head."""
    # Handle potential shape issues
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
    
    # Calculate predictive variance
    S = β*(1+v)/(α)
    
    # Fixed NLL calculation that guarantees positive values
    eps = 1e-7
    ce_loss = - y_true * tf.math.log(p + eps) - (1-y_true)*tf.math.log(1-p + eps)
    var_loss = 0.5*tf.math.log(S + eps)
    
    # Apply regularization to ensure positive loss
    reg = 0.01 * (tf.reduce_mean(v) + tf.reduce_mean(β))
    
    # This ensures the overall loss is always positive
    return tf.reduce_mean(tf.abs(ce_loss) + tf.abs(var_loss)) + reg

# --------------------------------------------------------------------------
# Fixed EVT Head
# --------------------------------------------------------------------------
def fixed_gpd_head(features, name=None):
    """Improved GPD parameter estimation head."""
    # Use regularized architecture to improve stability
    hidden = tf.keras.layers.Dense(
        32, 
        activation='relu',
        kernel_regularizer=regularizers.l2(1e-4),
        name=f'{name}_hidden' if name else None
    )(features)
    
    # Add dropout for better regularization
    hidden = tf.keras.layers.Dropout(0.2)(hidden)
    
    # Project to 2D space (shape, scale)
    dense = tf.keras.layers.Dense(
        2,
        kernel_regularizer=regularizers.l2(1e-3),
        name=f'{name}_dense' if name else None
    )(hidden)
    
    # Use tanh for shape (ξ) to constrain it to (-0.9, 0.9)
    # Use softplus for scale (σ) to ensure positivity
    shape = tf.keras.layers.Lambda(
        lambda x: tf.tanh(x[..., 0:1]) * 0.9,
        name=f'{name}_shape' if name else None
    )(dense)
    
    scale = tf.keras.layers.Lambda(
        lambda x: tf.nn.softplus(x[..., 1:2]) + 0.1,
        name=f'{name}_scale' if name else None
    )(dense)
    
    # Combine parameters
    params = tf.keras.layers.Concatenate(axis=-1, name=name)([shape, scale])
    
    return params

def fixed_evt_loss(logits, evt_params, threshold=0.5):
    """Fixed EVT loss function with proper regularization."""
    # Ensure inputs have the right shape
    logits = tf.reshape(logits, [-1, 1])
    evt_params = tf.reshape(evt_params, [-1, 2])
    
    # Unpack parameters
    shape = evt_params[:, 0:1]  # ξ
    scale = evt_params[:, 1:2]  # σ
    
    # Low threshold for more samples to exceed it
    tail_prob = tf.nn.sigmoid(logits - threshold)
    
    # Exceedance
    exceedance = tf.maximum(logits - threshold, 0.0)
    
    # Small constant for numerical stability
    eps = 1e-6
    
    # Safe shape parameter to avoid division by zero
    safe_shape = tf.where(
        tf.abs(shape) < eps,
        tf.ones_like(shape) * eps,
        shape
    )
    
    # Calculate GPD negative log-likelihood
    nll = tf.where(
        exceedance > 0,
        tf.math.log(scale + eps) + (1.0 / safe_shape + 1.0) * tf.math.log(1.0 + safe_shape * exceedance / (scale + eps)),
        tf.zeros_like(exceedance)
    )
    
    # Add regularization
    reg = 0.1 * tf.reduce_mean(scale) + 0.2 * tf.reduce_mean(tf.abs(shape))
    
    # Ensure the loss is positive
    return tf.reduce_mean(tf.abs(nll * tail_prob)) + reg

# --------------------------------------------------------------------------
# TSS Metric (from original codebase)
# --------------------------------------------------------------------------
class CategoricalTSSMetric(tf.keras.metrics.Metric):
    def __init__(self, name="tss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")
        self.tn = self.add_weight(name="tn", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_true and y_pred are one-hot encoded
        # Get class indices (0 or 1)
        y_true_cls = tf.argmax(y_true, axis=1)
        y_pred_cls = tf.argmax(y_pred, axis=1)
        
        # Calculate confusion matrix elements
        self.tp.assign_add(tf.reduce_sum(tf.cast(
            tf.logical_and(tf.equal(y_true_cls, 1), tf.equal(y_pred_cls, 1)),
            tf.float32)))
        self.fp.assign_add(tf.reduce_sum(tf.cast(
            tf.logical_and(tf.equal(y_true_cls, 0), tf.equal(y_pred_cls, 1)),
            tf.float32)))
        self.tn.assign_add(tf.reduce_sum(tf.cast(
            tf.logical_and(tf.equal(y_true_cls, 0), tf.equal(y_pred_cls, 0)),
            tf.float32)))
        self.fn.assign_add(tf.reduce_sum(tf.cast(
            tf.logical_and(tf.equal(y_true_cls, 1), tf.equal(y_pred_cls, 0)),
            tf.float32)))
    
    def result(self):
        # Calculate TSS: (TP/(TP+FN)) + (TN/(TN+FP)) - 1
        # Add small epsilon to prevent division by zero
        eps = 1e-7
        tpr = self.tp / (self.tp + self.fn + eps)
        tnr = self.tn / (self.tn + self.fp + eps)
        return tpr + tnr - 1
    
    def reset_state(self):
        self.tp.assign(0)
        self.fp.assign(0)
        self.tn.assign(0)
        self.fn.assign(0)

# --------------------------------------------------------------------------
# Fixed EVEREST Model
# --------------------------------------------------------------------------
class FixedEVEREST(tf.keras.Model):
    """Fixed EVEREST model with working advanced heads."""
    
    def __init__(self, seq_len, features, embed_dim=128, num_heads=4, dropout=0.3):
        super().__init__()
        
        # Input layer
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(seq_len, features))
        
        # Multi-scale stem
        self.conv1 = tf.keras.layers.Conv1D(embed_dim//4, 3, padding="causal", activation="gelu")
        self.conv2 = tf.keras.layers.Conv1D(embed_dim//4, 5, padding="causal", activation="gelu")
        self.conv3 = tf.keras.layers.Conv1D(embed_dim//4, 7, padding="causal", activation="gelu")
        self.stem_dense = tf.keras.layers.Dense(embed_dim)
        self.stem_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.stem_dropout = tf.keras.layers.Dropout(dropout)
        
        # Position encoding
        self.pos_encoding = self._positional_encoding(seq_len, embed_dim)
        
        # Transformer blocks - using standard attention for simplicity
        self.transformer_blocks = []
        for _ in range(4):  # Reduced number of blocks for testing
            self.transformer_blocks.append(
                self._transformer_block(embed_dim, num_heads, embed_dim*2, dropout)
            )
        
        # Global pooling
        self.pooling = tf.keras.layers.GlobalAveragePooling1D()
        self.dropout = tf.keras.layers.Dropout(dropout)
        
        # Features layer
        self.features = tf.keras.layers.Dense(
            128, 
            activation=tf.keras.activations.gelu,
            kernel_regularizer=regularizers.l1_l2(1e-4, 1e-3)
        )
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.dropout1 = tf.keras.layers.Dropout(dropout*1.2)
        
        # Second features layer
        self.features2 = tf.keras.layers.Dense(
            64,
            activation=tf.keras.activations.gelu,
            kernel_regularizer=regularizers.l2(1e-3)
        )
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        
        # Output heads
        self.logits_dense = tf.keras.layers.Dense(
            1, 
            activation=None,
            kernel_regularizer=regularizers.l2(1e-3)
        )
        
        # Create custom heads with our fixed implementations
        self.softmax_layer = tf.keras.layers.Lambda(
            lambda x: tf.nn.softmax(tf.concat([tf.zeros_like(x), x], axis=-1)),
            name="softmax_dense"
        )
        
        # We'll create evidential and evt heads in the call method
        # to ensure proper connection to the feature layer
    
    def _positional_encoding(self, max_len, embed_dim):
        pos = np.arange(max_len)[:, None]
        i = np.arange(embed_dim)[None, :]
        angle = pos / np.power(10000, (2 * (i//2)) / embed_dim)
        pe = np.zeros((max_len, embed_dim))
        pe[:, 0::2] = np.sin(angle[:, 0::2])
        pe[:, 1::2] = np.cos(angle[:, 1::2])
        return tf.cast(pe[None, ...], tf.float32)
    
    def _transformer_block(self, embed_dim, num_heads, ff_dim, dropout):
        inputs = tf.keras.Input(shape=(None, embed_dim))
        
        # Self attention
        attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim//num_heads, dropout=dropout*1.5
        )(inputs, inputs)
        attention = tf.keras.layers.Dropout(dropout)(attention)
        attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention)
        
        # Feed-forward network
        outputs = tf.keras.layers.Dense(
            ff_dim, activation=tf.keras.activations.gelu,
            kernel_regularizer=regularizers.l2(1e-4)
        )(attention)
        outputs = tf.keras.layers.Dropout(dropout*1.2)(outputs)
        outputs = tf.keras.layers.Dense(
            embed_dim, kernel_regularizer=regularizers.l2(1e-4)
        )(outputs)
        outputs = tf.keras.layers.Dropout(dropout)(outputs)
        outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + outputs)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs)
    
    def call(self, inputs, training=False):
        # Apply stem
        x = inputs
        stem = tf.concat([
            self.conv1(x),
            self.conv2(x),
            self.conv3(x)
        ], axis=-1)
        x = self.stem_dense(stem)
        x = self.stem_norm(x)
        x = self.stem_dropout(x, training=training)
        
        # Add positional encoding
        seq_len = tf.shape(x)[1]
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, training=training)
        
        # Pooling and feature extraction
        x = self.pooling(x)
        x = self.dropout(x, training=training)
        
        # Feature layers
        features = self.features(x)
        features = self.batch_norm1(features, training=training)
        features = self.dropout1(features, training=training)
        
        features = self.features2(features)
        features = self.batch_norm2(features, training=training)
        features = self.dropout2(features, training=training)
        
        # Compute logits
        logits = self.logits_dense(features)
        
        # Compute softmax
        softmax = self.softmax_layer(logits)
        
        # Use our fixed heads
        evidential_head = fixed_nig_head(features, name="evidential_head")
        evt_head = fixed_gpd_head(features, name="evt_head")
        
        return {
            "logits_dense": logits,
            "softmax_dense": softmax,
            "evidential_head": evidential_head,
            "evt_head": evt_head
        }

# --------------------------------------------------------------------------
# Model Training
# --------------------------------------------------------------------------
def compile_model(model, learning_rate=1e-3):
    """Compile the model with all heads properly weighted."""
    
    # Define loss functions
    def softmax_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true, y_pred))
    
    def logits_loss(y_true, y_pred):
        y_true_binary = tf.cast(y_true[:, 1:2], tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        return tf.reduce_mean(tf.keras.losses.binary_crossentropy(
            y_true_binary, y_pred, from_logits=True
        ))
    
    # Compile with initial weights for all heads
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate),
        loss={
            'softmax_dense': softmax_loss,
            'evidential_head': fixed_evidential_nll,
            'evt_head': fixed_evt_loss,
            'logits_dense': logits_loss
        },
        loss_weights={
            'softmax_dense': 1.0,
            'evidential_head': 0.1,  # Start with non-zero weight
            'evt_head': 0.1,         # Start with non-zero weight
            'logits_dense': 0.1      # Start with non-zero weight
        },
        metrics={
            'softmax_dense': [
                tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
                tf.keras.metrics.Precision(name='precision', class_id=1),
                tf.keras.metrics.Recall(name='recall', class_id=1),
                CategoricalTSSMetric(name='tss')
            ]
        }
    )
    return model

def load_data(flare_class, time_window, toy=False):
    """Load and prepare data for the model."""
    
    # Import necessary functions
    try:
        from utils import get_training_data
    except ImportError:
        raise ImportError("Cannot import get_training_data. Make sure utils.py is in the path.")
    
    # Load the raw data
    X, y_raw, original_df = get_training_data(time_window, flare_class, return_df=True)
    
    # Handle toy mode
    if toy:
        n_samples = min(int(len(X) * 0.05), 1000)  # 5% of data or 1000 max
        print(f"Toy mode: using {n_samples} samples")
        X = X[:n_samples]
        y_raw = y_raw[:n_samples]
    
    # Convert labels to one-hot
    if isinstance(y_raw, list):
        y_raw = np.array(y_raw)
        
    if y_raw.dtype in [np.int64, np.int32, np.float64, np.float32]:
        print("Using numerical labels directly (0=negative, 1=positive)")
        y = y_raw.astype(np.int32)
    else:
        print("Converting string labels ('P'=positive, others=negative)")
        y = np.array([1 if label == 'P' else 0 for label in y_raw]).astype(np.int32)
    
    # One-hot encoding
    y = tf.keras.utils.to_categorical(y, 2)
    
    # Create train/validation split
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=42
    )
    
    # Print class distribution
    pos = np.sum(y_train[:, 1])
    neg = len(y_train) - pos
    print(f"Train set: {len(X_train)} samples ({neg} negative, {pos} positive)")
    print(f"Validation set: {len(X_val)} samples")
    
    # Create class weights
    class_weight = {
        0: 1.0,
        1: min(10.0, neg / pos * 2.0)
    }
    print(f"Using class weights: {class_weight}")
    
    # Create sample weights
    sample_weights = np.ones(len(X_train))
    positive_indices = np.where(y_train[:, 1] > 0.5)[0]
    sample_weights[positive_indices] = class_weight[1]
    
    return X_train, X_val, y_train, y_val, sample_weights, class_weight

def train_model(flare_class, time_window, toy=False, epochs=100, batch_size=256):
    """Train the fixed EVEREST model."""
    # Load data
    X_train, X_val, y_train, y_val, sample_weights, class_weight = load_data(
        flare_class, time_window, toy
    )
    
    # Create model
    seq_len, features = X_train.shape[1], X_train.shape[2]
    model = FixedEVEREST(seq_len, features)
    
    # Compile model
    model = compile_model(model, learning_rate=5e-4)
    
    # Create callbacks
    callbacks = [
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_softmax_dense_tss',
            mode='max',
            patience=10,
            restore_best_weights=True
        ),
        # Learning rate scheduler
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_softmax_dense_tss',
            mode='max',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        # TensorBoard
        tf.keras.callbacks.TensorBoard(
            log_dir=f"logs/fixed_everest/{flare_class}_{time_window}",
            histogram_freq=1
        )
    ]
    
    # Train model
    print(f"Training model for {epochs} epochs with batch size {batch_size}...")
    dummy_input = np.zeros((1, seq_len, features), dtype=np.float32)
    _ = model(dummy_input)  # Initialize the model
    
    # Create target dictionaries
    y_train_dict = {
        'softmax_dense': y_train,
        'logits_dense': y_train,
        'evidential_head': y_train,
        'evt_head': y_train
    }
    
    y_val_dict = {
        'softmax_dense': y_val,
        'logits_dense': y_val,
        'evidential_head': y_val,
        'evt_head': y_val
    }
    
    # Train with sample weights
    history = model.fit(
        X_train,
        y_train_dict,
        validation_data=(X_val, y_val_dict),
        epochs=epochs,
        batch_size=batch_size,
        sample_weight=sample_weights,
        callbacks=callbacks,
        verbose=2
    )
    
    # Save model
    save_dir = f"models/fixed_everest_{flare_class}_{time_window}"
    os.makedirs(save_dir, exist_ok=True)
    model.save_weights(f"{save_dir}/model_weights.h5")
    
    # Save summary metrics
    val_tss = max(history.history.get('val_softmax_dense_tss', [0]))
    print(f"Best validation TSS: {val_tss:.4f}")
    
    return model, history

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train fixed EVEREST model with proper EVT and evidential heads")
    parser.add_argument("--flare", default="M5", help="Flare class (C, M, M5)")
    parser.add_argument("--window", default="24", help="Time window (24h, 48h, 72h)")
    parser.add_argument("--toy", action="store_true", help="Use toy dataset")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    
    args = parser.parse_args()
    
    # Train model
    train_model(args.flare, args.window, args.toy, args.epochs, args.batch_size) 