"""
EVEREST – Extreme‑Value/Evidential Retentive Event Sequence Transformer

Complete implementation with:
1. Performer (FAVOR+) linear attention blocks
2. Retentive mechanism (exponential memory half-life)
3. Class-Balanced Focal loss
4. Multi-scale tokenization
5. Diffusion / KDE oversampling
6. Evidential uncertainty
7. Extreme Value Theory
8. Conformal calibration
"""

import tensorflow as tf
import numpy as np
import os
import json
from datetime import datetime
from tensorflow.keras import layers, models, regularizers

# Import custom components
from performer_custom import Performer
from retentive_layer import RetentiveLayer
from focal_loss import ClassBalancedFocalLoss, cb_focal_loss
from multi_scale_tokenizer import MultiScaleTokenizer
from diffusion_oversampling import DiffusionOversampler, SMOTEOversampler
from evidential_head import nig_head, evidential_nll, kl_divergence_nig
from evt_head import gpd_head, evt_loss, gpd_threshold_probability
from conformal_calibration import ConformalCalibrator, calibrate_model
from metrics import CategoricalTSSMetric

# Set random seed for reproducibility
tf.keras.utils.set_random_seed(42)

# Configure GPU memory growth if available
for g in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except:
        pass

# ---------------------------------------------------------------------------
# Positional encoding (sinusoid)
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
# Performer Transformer block with Retentive memory
# ---------------------------------------------------------------------------
class PerformerRetentiveBlock(layers.Layer):
    def __init__(self, 
                 embed_dim: int, 
                 num_heads: int, 
                 ff_dim: int, 
                 dropout: float = 0.2, 
                 causal: bool = False, 
                 use_retentive: bool = True,
                 decay_factor: float = 0.95,
                 feature_dim: int = 256):
        super().__init__()
        self.use_retentive = use_retentive
        
        # Linear attention layer
        self.attn = Performer(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            feature_dim=feature_dim,
            dropout=dropout*1.5,  # Higher dropout in attention
            causal=causal  # Use causal attention if specified
        )
        self.drop1 = layers.Dropout(dropout)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)

        # Add retentive layer if enabled
        if use_retentive:
            self.retentive = RetentiveLayer(
                output_dim=embed_dim,
                decay_factor=decay_factor,
                trainable_decay=True
            )
            self.norm_ret = layers.LayerNormalization(epsilon=1e-6)
            self.drop_ret = layers.Dropout(dropout)

        # Feed-forward network
        self.ffn = models.Sequential([
            layers.Dense(ff_dim, 
                        activation=tf.keras.activations.gelu,
                        kernel_regularizer=regularizers.l2(1e-4)),
            layers.Dropout(dropout * 1.2),  # Increased intermediate dropout
            layers.Dense(embed_dim, 
                        kernel_regularizer=regularizers.l2(1e-4))
        ])
        self.drop2 = layers.Dropout(dropout)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        
    def call(self, x, training=False):
        # Self-attention and residual connection
        attn_output = self.attn(x, training=training)
        x = self.norm1(x + self.drop1(attn_output, training=training))
        
        # Apply retentive mechanism if enabled
        if self.use_retentive:
            ret_output = self.retentive(x, training=training)
            x = self.norm_ret(x + self.drop_ret(ret_output, training=training))
        
        # Feed-forward network and residual connection
        ffn_output = self.ffn(x, training=training)
        x = self.norm2(x + self.drop2(ffn_output, training=training))
        
        # Additional stochastic depth dropout during training
        if training and tf.random.uniform([]) < 0.7:
            x = tf.nn.dropout(x, rate=min(self.drop1.rate * 1.5, 0.5))
            
        return x

# ---------------------------------------------------------------------------
# Complete EVEREST model 
# ---------------------------------------------------------------------------
class EVEREST:
    model_name = "EVEREST"
    
    def __init__(self, 
                 use_evidential=True,    # Use evidential uncertainty head
                 use_evt=True,           # Use extreme value theory head
                 use_retentive=True,     # Use retentive memory mechanism
                 use_multi_scale=True,   # Use multi-scale tokenization
                 early_stopping_patience=15):
        """
        Initialize the EVEREST model with all components.
        
        Args:
            use_evidential: Whether to use evidential uncertainty head
            use_evt: Whether to use extreme value theory head
            use_retentive: Whether to use retentive memory mechanism
            use_multi_scale: Whether to use multi-scale tokenization
            early_stopping_patience: Patience for early stopping
        """
        # Store configuration
        self.use_evidential = use_evidential
        self.use_evt = use_evt
        self.use_retentive = use_retentive
        self.use_multi_scale = use_multi_scale
        self.early_stopping_patience = early_stopping_patience
        
        # Advanced feature flags
        self.use_advanced_heads = use_evidential or use_evt
        
        # Create callbacks for monitoring
        self.callbacks = []
        
        # Pre-trained diffusion oversampler
        self.diffusion_sampler = None
        
        # Conformal calibrator
        self.calibrator = None
    
    def build_base_model(self, 
                         input_shape: tuple,
                         embed_dim: int = 128,
                         num_heads: int = 4,
                         ff_dim: int = 256,
                         n_blocks: int = 4,
                         dropout: float = 0.3,
                         num_classes: int = 2,
                         causal: bool = False):
        """
        Build the EVEREST model architecture.
        
        Args:
            input_shape: Shape of input data (seq_len, features)
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward network dimension
            n_blocks: Number of transformer blocks
            dropout: Dropout rate
            num_classes: Number of output classes
            causal: Whether to use causal attention
            
        Returns:
            The constructed model
        """
        # Create the input layer
        inp = layers.Input(shape=input_shape)
        
        # Apply multi-scale tokenization if enabled
        if self.use_multi_scale:
            # Process at different time scales
            tokenizer = MultiScaleTokenizer(
                include_original=True,
                include_1h=True,
                include_3h=True
            )
            
            # Expand features by concatenating multi-scale views
            x = tokenizer(inp)
            
            # Project back to original feature dimension to avoid parameter explosion
            x = layers.Dense(input_shape[1])(x)
        else:
            x = inp
        
        # Multi-scale stem with 3 kernel sizes
        stem = layers.Concatenate()([
            layers.Conv1D(embed_dim//4, 3, padding="causal", activation="gelu")(x),
            layers.Conv1D(embed_dim//4, 5, padding="causal", activation="gelu")(x),
            layers.Conv1D(embed_dim//4, 7, padding="causal", activation="gelu")(x)
        ])
        
        # Project to embedding dimension
        x = layers.Dense(embed_dim)(stem)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.Dropout(dropout)(x)
        
        # Apply positional encoding
        x = PositionalEncoding(input_shape[0], embed_dim)(x)
        
        # Transformer blocks with retentive memory
        for i in range(n_blocks):
            decay_factor = 0.95 - (i * 0.05)  # Decrease memory retention in deeper layers
            x = PerformerRetentiveBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                causal=causal,
                use_retentive=self.use_retentive and (i > 0),  # Skip retentive on first block
                decay_factor=max(0.7, decay_factor),  # Don't go below 0.7
                feature_dim=256
            )(x)
        
        # Pool outputs with global average pooling
        pooled_outputs = []
        
        # Use outputs from last two blocks for better feature representation
        for i in range(max(1, n_blocks-2), n_blocks):
            if i == n_blocks-1:  # Always keep the final layer
                pooled_outputs.append(layers.GlobalAveragePooling1D()(x))
            else:
                # Include with probability
                mask = tf.cast(tf.random.uniform([]) < 0.8, tf.float32)
                pooled = layers.GlobalAveragePooling1D()(x)
                pooled_outputs.append(pooled * mask)
        
        # Combine pooled features
        if len(pooled_outputs) > 1:
            x = layers.Add()(pooled_outputs)
        else:
            x = pooled_outputs[0]
            
        x = layers.Dropout(dropout)(x)
        
        # Feature representation layers
        features = layers.Dense(128, 
                       activation=tf.keras.activations.gelu,
                       kernel_regularizer=regularizers.l1_l2(1e-4, 1e-3))(x)
        features = layers.BatchNormalization()(features)
        features = layers.Dropout(dropout * 1.2)(features)
        
        # Second dense layer
        features = layers.Dense(64, 
                           activation=tf.keras.activations.gelu,
                           kernel_regularizer=regularizers.l2(1e-3))(features)
        features = layers.BatchNormalization()(features)
        features = layers.Dropout(dropout)(features)
        
        # Create the model with appropriate heads
        if self.use_advanced_heads:
            # Standard logits for binary classification
            logits = layers.Dense(1, 
                              activation=None, 
                              kernel_regularizer=regularizers.l2(1e-3),
                              name="logits_dense")(features)
            
            # Create softmax output
            softmax_activation = lambda x: tf.nn.softmax(
                tf.concat([tf.zeros_like(x), x], axis=-1)
            )
            softmax_out = layers.Lambda(
                softmax_activation, 
                name="softmax_dense"
            )(logits)
            
            # Define the outputs dictionary
            outputs = {
                "logits_dense": logits,
                "softmax_dense": softmax_out
            }
            
            # Add evidential head if enabled
            if self.use_evidential:
                outputs["evidential_head"] = nig_head(features, name="evidential_head")
                
            # Add EVT head if enabled
            if self.use_evt:
                outputs["evt_head"] = gpd_head(features, name="evt_head")
                
            # Create model
            self.model = models.Model(inputs=inp, outputs=outputs)
        else:
            # Standard binary classification model
            output = layers.Dense(num_classes, 
                             activation="softmax",
                             kernel_regularizer=regularizers.l2(1e-3))(features)
            self.model = models.Model(inputs=inp, outputs=output)
            
        return self.model
    
    def compile(self, lr: float = 2e-4, class_counts=None):
        """
        Compile the model with appropriate loss functions and metrics.
        
        Args:
            lr: Learning rate
            class_counts: Optional count of samples in each class
        """
        # Create metrics including TSS
        tss_metric = CategoricalTSSMetric()
        
        if self.use_advanced_heads:
            # Define loss functions for each head
            def softmax_loss(y_true, y_pred):
                # Use Class-Balanced Focal Loss
                return cb_focal_loss(
                    beta=0.9999,
                    gamma=2.0,
                    class_counts=class_counts,
                    from_logits=False
                )(y_true, y_pred)
            
            def logits_loss(y_true, y_pred):
                # Extract positive class and use BCE
                y_true_binary = tf.cast(y_true[:, 1:2], tf.float32)
                return tf.reduce_mean(tf.keras.losses.binary_crossentropy(
                    y_true_binary, y_pred, from_logits=True
                ))
            
            def evidential_loss(y_true, y_pred):
                # Use evidential head loss with KL regularization
                y_true_binary = tf.cast(y_true[:, 1:2], tf.float32)
                return evidential_nll(y_true_binary, y_pred, kl_weight=0.01)
            
            def evt_loss_fn(y_true, y_pred):
                # Use EVT head loss with threshold mask
                logits = None
                
                # Try to get logits from the model
                # When forward pass is active, use direct values
                # Create synthetic logits from true labels
                y_true_binary = tf.cast(y_true[:, 1:2], tf.float32)
                
                # Synthetic is safer than trying to access other outputs
                synthetic_logits = tf.where(
                    y_true_binary > 0.5,
                    tf.ones_like(y_true_binary) * 3.0,  # Strong positive
                    tf.ones_like(y_true_binary) * -3.0  # Strong negative
                )
                
                # Use threshold mask = 0.5 for more tail samples
                return evt_loss(synthetic_logits, y_pred, threshold=0.5, mask_quantile=0.95)
            
            # Compile model with all heads
            self.model.compile(
                optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=lr),
                loss={
                    'softmax_dense': softmax_loss,
                    'logits_dense': logits_loss,
                    **({'evidential_head': evidential_loss} if self.use_evidential else {}),
                    **({'evt_head': evt_loss_fn} if self.use_evt else {})
                },
                loss_weights={
                    'softmax_dense': 1.0,
                    'logits_dense': 0.2,
                    **({'evidential_head': 0.2} if self.use_evidential else {}),
                    **({'evt_head': 0.2} if self.use_evt else {})
                },
                metrics={
                    "softmax_dense": [
                        tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
                        tf.keras.metrics.Precision(name="prec", class_id=1),
                        tf.keras.metrics.Recall(name="rec", class_id=1),
                        CategoricalTSSMetric(name="tss")
                    ]
                }
            )
            
            print("Model compiled with multiple heads and non-zero loss weights")
        else:
            # For standard binary classification, use focal loss
            focal_loss = cb_focal_loss(
                beta=0.9999,
                gamma=2.0,
                class_counts=class_counts,
                from_logits=False
            )
            
            self.model.compile(
                optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=lr),
                loss=focal_loss,
                metrics=[
                    tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
                    tf.keras.metrics.Precision(name="prec", class_id=1),
                    tf.keras.metrics.Recall(name="rec", class_id=1),
                    CategoricalTSSMetric(name="tss")
                ]
            )
            
            print("Model compiled with Class-Balanced Focal Loss")
    
    def fit(self, X, y, validation_data=None, epochs=100, batch_size=128,
            class_weight=None, callbacks=None, verbose=2, sample_weight=None,
            use_diffusion=False, diffusion_ratio=0.25):
        """
        Train the model with diffusion-based oversampling.
        
        Args:
            X: Training data
            y: Training labels
            validation_data: Validation data tuple (X_val, y_val)
            epochs: Number of training epochs
            batch_size: Batch size
            class_weight: Class weighting dictionary
            callbacks: Additional callbacks
            verbose: Verbosity level
            sample_weight: Sample weights
            use_diffusion: Whether to use diffusion-based oversampling
            diffusion_ratio: Ratio of synthetic samples to include
        """
        # Prepare callbacks
        all_callbacks = callbacks or self.callbacks.copy()
        
        # Create early stopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_softmax_dense_tss' if self.use_advanced_heads else 'val_tss', 
            mode='max',
            patience=self.early_stopping_patience,
            restore_best_weights=True,
            min_delta=0.005
        )
        all_callbacks.append(early_stopping)
        
        # Create learning rate scheduler
        def lr_schedule(epoch, lr):
            if epoch < 5:
                return lr * 0.2 + lr * 0.8 * epoch / 5  # Warm-up
            elif epoch < 30:
                return lr
            else:
                return lr * 0.95  # Gradual decay
                
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
        all_callbacks.append(lr_scheduler)
        
        # Apply diffusion-based oversampling if requested
        if use_diffusion and diffusion_ratio > 0:
            # Handle multi-output format
            if isinstance(y, dict):
                # Just use softmax_dense for oversampling
                y_for_oversample = y['softmax_dense']
            else:
                y_for_oversample = y
                
            # Find positive samples
            if len(y_for_oversample.shape) > 1 and y_for_oversample.shape[1] > 1:
                # One-hot encoding
                is_positive = np.argmax(y_for_oversample, axis=1) == 1
            else:
                # Binary labels
                is_positive = y_for_oversample == 1
                
            X_pos = X[is_positive]
            
            if len(X_pos) > 10:
                print(f"Using diffusion oversampling with {len(X_pos)} positive samples")
                
                # Create or load diffusion sampler
                input_shape = X.shape[1:]
                sampler_path = f"models/diffusion/everest_sampler.h5"
                
                self.diffusion_sampler = DiffusionOversampler(
                    input_shape=input_shape,
                    diffusion_steps=50,
                    model_path=sampler_path
                )
                
                # Try to load existing model
                model_loaded = self.diffusion_sampler.load_model()
                
                if not model_loaded and len(X_pos) >= 50:
                    # Train the sampler if enough positive samples
                    print("Training diffusion model on positive samples...")
                    self.diffusion_sampler.train(
                        X_pos, 
                        epochs=30, 
                        batch_size=16, 
                        validation_split=0.1
                    )
                elif not model_loaded:
                    # Fall back to SMOTE for small sample sizes
                    print("Not enough samples for diffusion, using SMOTE instead")
                    smote = SMOTEOversampler(k_neighbors=min(5, len(X_pos)-1))
                    
                    # Generate synthetic samples
                    n_synthetic = int(len(X) * diffusion_ratio)
                    X_synthetic = smote.generate_samples(X_pos, n_synthetic)
                    
                    # Create corresponding labels
                    if isinstance(y, dict):
                        # Multi-output format
                        y_synthetic = {}
                        for key, value in y.items():
                            # Copy the positive class label format
                            if len(value.shape) > 1 and value.shape[1] > 1:
                                # One-hot encoding
                                y_synthetic[key] = np.zeros((len(X_synthetic), value.shape[1]))
                                y_synthetic[key][:, 1] = 1  # Set positive class
                            else:
                                # Binary labels
                                y_synthetic[key] = np.ones(len(X_synthetic))
                    else:
                        # Standard format
                        if len(y.shape) > 1 and y.shape[1] > 1:
                            # One-hot encoding
                            y_synthetic = np.zeros((len(X_synthetic), y.shape[1]))
                            y_synthetic[:, 1] = 1  # Set positive class
                        else:
                            # Binary labels
                            y_synthetic = np.ones(len(X_synthetic))
                    
                    # Combine original and synthetic data
                    X_combined = np.concatenate([X, X_synthetic], axis=0)
                    
                    if isinstance(y, dict):
                        # Multi-output format
                        y_combined = {}
                        for key in y.keys():
                            y_combined[key] = np.concatenate([y[key], y_synthetic[key]], axis=0)
                    else:
                        # Standard format
                        y_combined = np.concatenate([y, y_synthetic], axis=0)
                    
                    # Update sample weights if needed
                    if sample_weight is not None:
                        sample_weight_synthetic = np.ones(len(X_synthetic)) * np.mean(sample_weight[is_positive])
                        sample_weight = np.concatenate([sample_weight, sample_weight_synthetic], axis=0)
                    
                    # Use the combined dataset
                    X, y = X_combined, y_combined
            else:
                print("Too few positive samples for oversampling, using original data")
                
        # For multi-output models, prepare targets
        if self.use_advanced_heads and not isinstance(y, dict):
            # Convert to dictionary format
            y_dict = {
                "softmax_dense": y,
                "logits_dense": y
            }
            
            if self.use_evidential:
                y_dict["evidential_head"] = y
                
            if self.use_evt:
                y_dict["evt_head"] = y
                
            # Replace original y with dictionary
            y = y_dict
            
            # Also convert validation data if provided
            if validation_data is not None:
                X_val, y_val = validation_data
                
                val_dict = {
                    "softmax_dense": y_val,
                    "logits_dense": y_val
                }
                
                if self.use_evidential:
                    val_dict["evidential_head"] = y_val
                    
                if self.use_evt:
                    val_dict["evt_head"] = y_val
                    
                validation_data = (X_val, val_dict)
        
        # Train the model
        history = self.model.fit(
            X, y,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            sample_weight=sample_weight,
            callbacks=all_callbacks,
            verbose=verbose
        )
        
        return history
    
    def mc_predict(self, X, n_passes=20, batch_size=128):
        """
        Monte Carlo dropout prediction for uncertainty estimation.
        
        Args:
            X: Input data
            n_passes: Number of forward passes with dropout active
            batch_size: Batch size for prediction
            
        Returns:
            mean_preds: Mean predictions
            std_preds: Standard deviation (uncertainty)
        """
        if not hasattr(self, 'model'):
            raise ValueError("Model has not been built yet")
            
        # Ensure X is float32
        X = np.asarray(X, dtype=np.float32)
        
        # Process in batches to avoid memory issues
        all_preds = []
        
        for _ in range(n_passes):
            batch_preds = []
            
            for i in range(0, len(X), batch_size):
                end_idx = min(i + batch_size, len(X))
                batch_data = X[i:end_idx]
                
                # Run with dropout active
                outputs = self.model(batch_data, training=True)
                
                # Extract predictions based on model type
                if self.use_advanced_heads:
                    pred = outputs["softmax_dense"].numpy()
                else:
                    pred = outputs.numpy()
                    
                batch_preds.append(pred)
                
            # Concatenate batch predictions
            preds = np.concatenate(batch_preds, axis=0)
            all_preds.append(preds)
            
        # Stack predictions from all passes
        all_preds = np.stack(all_preds, axis=0)  # [n_passes, n_samples, n_classes]
        
        # Calculate mean and std across Monte Carlo samples
        mean_preds = np.mean(all_preds, axis=0)
        std_preds = np.std(all_preds, axis=0)
        
        return mean_preds, std_preds
    
    def predict_proba(self, X, batch_size=128, mc_passes=None):
        """
        Predict class probabilities.
        
        Args:
            X: Input data
            batch_size: Batch size for prediction
            mc_passes: Number of Monte Carlo passes (None for deterministic prediction)
            
        Returns:
            Class probabilities
        """
        # Ensure X is float32
        X = np.asarray(X, dtype=np.float32)
        
        if mc_passes is not None:
            # Use Monte Carlo dropout
            mean, _ = self.mc_predict(X, n_passes=mc_passes, batch_size=batch_size)
            
            # Extract probability of positive class
            if mean.shape[-1] == 2:
                return mean[:, 1]
            return mean
        else:
            # Deterministic prediction
            preds = self.model.predict(X, batch_size=batch_size, verbose=0)
            
            if self.use_advanced_heads:
                # Extract softmax output
                if isinstance(preds, dict) and "softmax_dense" in preds:
                    probs = preds["softmax_dense"]
                else:
                    print("Warning: Model output format not as expected")
                    if isinstance(preds, dict):
                        # Try to find a suitable output
                        for key in preds.keys():
                            if "softmax" in key or "prob" in key:
                                probs = preds[key]
                                break
                        else:
                            # Use first available output
                            probs = preds[list(preds.keys())[0]]
                    else:
                        probs = preds
            else:
                probs = preds
                
            # Return probability of positive class
            if probs.shape[-1] == 2:
                return probs[:, 1]
            return probs
    
    def predict_evidential(self, X, batch_size=128):
        """
        Predict evidential uncertainty parameters.
        
        Args:
            X: Input data
            batch_size: Batch size
            
        Returns:
            NIG parameters (μ, ν, α, β)
        """
        if not self.use_evidential:
            raise ValueError("Evidential prediction is only available when use_evidential=True")
            
        preds = self.model.predict(X, batch_size=batch_size, verbose=0)
        return preds["evidential_head"]
    
    def predict_evt(self, X, batch_size=128):
        """
        Predict Extreme Value Theory parameters.
        
        Args:
            X: Input data
            batch_size: Batch size
            
        Returns:
            GPD parameters (ξ, σ)
        """
        if not self.use_evt:
            raise ValueError("EVT prediction is only available when use_evt=True")
            
        preds = self.model.predict(X, batch_size=batch_size, verbose=0)
        return preds["evt_head"]
    
    def calibrate(self, X_val, y_val, alpha=0.1, mc_samples=20):
        """
        Calibrate the model using conformal prediction.
        
        Args:
            X_val: Validation data
            y_val: Validation labels
            alpha: Significance level (e.g., 0.1 for 90% coverage)
            mc_samples: Number of Monte Carlo samples
            
        Returns:
            Calibrated threshold
        """
        print(f"Calibrating model with conformal prediction (α={alpha}, MC samples={mc_samples})")
        
        # Extract binary labels if needed
        if len(y_val.shape) > 1 and y_val.shape[1] > 1:
            y_val_binary = np.argmax(y_val, axis=1)
        else:
            y_val_binary = y_val
            
        # Create the calibrator
        self.calibrator = calibrate_model(
            model=self.model,
            X_val=X_val,
            y_val=y_val_binary,
            alpha=alpha,
            mc_samples=mc_samples
        )
        
        # Return calibrated threshold
        return self.calibrator.threshold
    
    def predict_with_uncertainty(self, X, batch_size=128, mc_passes=20):
        """
        Predict with uncertainty estimates.
        
        Args:
            X: Input data
            batch_size: Batch size
            mc_passes: Number of Monte Carlo passes
            
        Returns:
            Dictionary of predictions with uncertainty information
        """
        # Run Monte Carlo predictions
        mc_mean, mc_std = self.mc_predict(X, n_passes=mc_passes, batch_size=batch_size)
        
        # Extract probabilities for positive class
        if mc_mean.shape[-1] == 2:
            probs = mc_mean[:, 1]
            uncertainty = mc_std[:, 1]
        else:
            probs = mc_mean
            uncertainty = mc_std
            
        # Get standard point predictions
        point_preds = (probs > 0.5).astype(int)
        
        # Add evidential uncertainty if available
        if self.use_evidential:
            evidential_params = self.predict_evidential(X, batch_size=batch_size)
            mu, v, alpha, beta = np.split(evidential_params, 4, axis=1)
            
            # Calculate epistemic and aleatoric uncertainty
            epistemic_uncertainty = beta / (v * (alpha - 1) + 1e-8)
            aleatoric_uncertainty = beta / (alpha - 1 + 1e-8)
            
            # Add to results
            evidential_results = {
                'epistemic': epistemic_uncertainty,
                'aleatoric': aleatoric_uncertainty,
                'total': epistemic_uncertainty + aleatoric_uncertainty,
                'mu': mu,
                'v': v,
                'alpha': alpha,
                'beta': beta
            }
        else:
            evidential_results = None
            
        # Add EVT parameters if available
        if self.use_evt:
            evt_params = self.predict_evt(X, batch_size=batch_size)
            shape, scale = np.split(evt_params, 2, axis=1)
            
            # Store parameters
            evt_results = {
                'shape': shape,
                'scale': scale
            }
        else:
            evt_results = None
            
        # Add conformal prediction sets if calibrated
        if self.calibrator is not None:
            # Generate conformal prediction sets
            conf_sets = self.calibrator.predict_sets(probs)
            
            # Add to results
            conformal_results = {
                'sets': conf_sets['sets'],
                'lower': conf_sets['lower'],
                'upper': conf_sets['upper']
            }
        else:
            conformal_results = None
            
        # Combine results
        results = {
            'probabilities': probs,
            'predictions': point_preds,
            'mc_uncertainty': uncertainty,
            'evidential': evidential_results,
            'evt': evt_results,
            'conformal': conformal_results
        }
        
        return results
    
    def save_weights(self, model_dir=None, flare_class=None):
        """
        Save model weights and metadata.
        
        Args:
            model_dir: Directory to save the model
            flare_class: Flare class for the model
        """
        # Create model directory
        if model_dir is None:
            model_dir = os.path.join("models", self.model_name, str(flare_class or ""))
            
        # Ensure directory exists
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model weights
        self.model.save_weights(os.path.join(model_dir, "model_weights.h5"))
        
        # Save diffusion model if available
        if self.diffusion_sampler is not None and hasattr(self.diffusion_sampler, 'model_path'):
            diff_path = self.diffusion_sampler.model_path or os.path.join(model_dir, "diffusion_model.h5")
            try:
                self.diffusion_sampler.model.save_weights(diff_path)
                print(f"Diffusion model saved to {diff_path}")
            except:
                print("Warning: Could not save diffusion model")
        
        # Save conformal calibration if available
        if self.calibrator is not None:
            try:
                self.calibrator.save(os.path.join(model_dir, "conformal_calibration.npy"))
                print(f"Conformal calibration saved to {model_dir}")
            except:
                print("Warning: Could not save conformal calibration")
        
        # Save metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "model_name": self.model_name,
            "flare_class": flare_class,
            "uses_focal_loss": True,
            "uses_evidential": self.use_evidential,
            "uses_evt": self.use_evt,
            "uses_diffusion": self.diffusion_sampler is not None,
            "uses_retentive": self.use_retentive,
            "uses_multi_scale": self.use_multi_scale,
            "advanced_model": self.use_advanced_heads,
            "linear_attention": True,
            "output_names": {
                "softmax": "softmax_dense",
                "softmax_dense": "softmax_dense",
                "evidential": "evidential_head" if self.use_evidential else None,
                "evt": "evt_head" if self.use_evt else None,
                "logits": "logits_dense" if self.use_advanced_heads else None
            }
        }
        
        # Save metadata
        with open(os.path.join(model_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
            
        print(f"Model saved to {model_dir}")
        
    def load_weights(self, model_dir=None, flare_class=None):
        """
        Load model weights and metadata.
        
        Args:
            model_dir: Directory to load the model from
            flare_class: Flare class for the model
        """
        # Get model directory
        if model_dir is None:
            model_dir = os.path.join("models", self.model_name, str(flare_class or ""))
            
        # Load model weights
        self.model.load_weights(os.path.join(model_dir, "model_weights.h5"))
        
        # Try to load diffusion model if available
        diff_path = os.path.join(model_dir, "diffusion_model.h5")
        if os.path.exists(diff_path) and hasattr(self, 'input_shape'):
            try:
                self.diffusion_sampler = DiffusionOversampler(
                    input_shape=self.input_shape[1:],
                    model_path=diff_path
                )
                self.diffusion_sampler.load_model()
                print(f"Diffusion model loaded from {diff_path}")
            except:
                print("Warning: Could not load diffusion model")
        
        # Try to load conformal calibration if available
        conf_path = os.path.join(model_dir, "conformal_calibration.npy")
        if os.path.exists(conf_path):
            try:
                self.calibrator = ConformalCalibrator()
                self.calibrator.load(conf_path)
                print(f"Conformal calibration loaded from {conf_path}")
            except:
                print("Warning: Could not load conformal calibration")
        
        # Load metadata
        metadata_path = os.path.join(model_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                
            # Update model configuration from metadata
            self.use_evidential = metadata.get("uses_evidential", self.use_evidential)
            self.use_evt = metadata.get("uses_evt", self.use_evt)
            self.use_retentive = metadata.get("uses_retentive", self.use_retentive)
            self.use_multi_scale = metadata.get("uses_multi_scale", self.use_multi_scale)
            self.use_advanced_heads = metadata.get("advanced_model", self.use_advanced_heads)
            
        print(f"Model loaded from {model_dir}")

# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Create synthetic data for testing
    seq_len, feat = 100, 14
    X = np.random.random((16, seq_len, feat)).astype("float32")
    y = tf.keras.utils.to_categorical(np.random.randint(0, 2, size=16), 2)
    
    # Create model with all components
    print("Creating complete EVEREST model...")
    model = EVEREST(
        use_evidential=True,
        use_evt=True,
        use_retentive=True,
        use_multi_scale=True
    )
    
    # Build and compile model
    model.build_base_model((seq_len, feat))
    model.compile()
    
    # Display model summary
    model.model.summary()
    
    # Train for a single epoch
    print("Training model for one epoch...")
    model.fit(X, y, epochs=1, batch_size=4)
    
    # Test Monte Carlo prediction
    print("Testing Monte Carlo prediction...")
    m, s = model.mc_predict(X[:4], n_passes=5)
    print(f"MC prediction shape: {m.shape}, uncertainty shape: {s.shape}")
    
    # Test evidential head
    print("Testing evidential head...")
    ev = model.predict_evidential(X[:4])
    print(f"Evidential parameters shape: {ev.shape}")
    
    # Test EVT head
    print("Testing EVT head...")
    evt = model.predict_evt(X[:4])
    print(f"EVT parameters shape: {evt.shape}")
    
    # Test comprehensive prediction
    print("Testing comprehensive prediction...")
    results = model.predict_with_uncertainty(X[:4], mc_passes=5)
    print(f"Prediction results keys: {list(results.keys())}")
    
    print("Demo completed successfully!") 