"""
 Alternative transformer-based model with improved capacity for time-series classification.
 @author: by Antanas Zilinskas
"""

import json
import os
import shutil
import warnings
from datetime import datetime

import numpy as np
import tensorflow as tf
<<<<<<< Updated upstream
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Import tensorflow_addons for focal loss
try:
    import tensorflow_addons as tfa
except ImportError:
    print("tensorflow_addons not found. Installing...")
    import subprocess

    subprocess.check_call(["pip", "install", "tensorflow-addons"])
    import tensorflow_addons as tfa

# Set up mixed precision (for improved performance on MPS/M2)
tf.keras.mixed_precision.set_global_policy("mixed_float16")
print(
    "Mixed precision enabled. Current policy:",
    tf.keras.mixed_precision.global_policy(),
)


# Set GPU memory growth (this works for both GPU/MPS on Apple Silicon)
physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, enable=True)
    print(
        f"SUCCESS: Found and set memory growth for {len(physical_devices)} GPU device(s)."
    )
else:
    print("WARNING: GPU device not found.")

# Custom TSS metric for optimization


class TrueSkillStatisticMetric(tf.keras.metrics.Metric):
    def __init__(self, name="tss", **kwargs):
        super(TrueSkillStatisticMetric, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="tp", initializer="zeros")
        self.true_negatives = self.add_weight(name="tn", initializer="zeros")
        self.false_positives = self.add_weight(name="fp", initializer="zeros")
        self.false_negatives = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.argmax(y_true, axis=1), tf.bool)
        y_pred = tf.cast(tf.argmax(y_pred, axis=1), tf.bool)

        true_positives = tf.cast(
            tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True)),
            tf.float32,
        )
        true_negatives = tf.cast(
            tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, False)),
            tf.float32,
        )
        false_positives = tf.cast(
            tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, True)),
            tf.float32,
        )
        false_negatives = tf.cast(
            tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, False)),
            tf.float32,
        )

        self.true_positives.assign_add(tf.reduce_sum(true_positives))
        self.true_negatives.assign_add(tf.reduce_sum(true_negatives))
        self.false_positives.assign_add(tf.reduce_sum(false_positives))
        self.false_negatives.assign_add(tf.reduce_sum(false_negatives))

    def result(self):
        sensitivity = self.true_positives / (
            self.true_positives + self.false_negatives + tf.keras.backend.epsilon()
        )
        specificity = self.true_negatives / (
            self.true_negatives + self.false_positives + tf.keras.backend.epsilon()
        )
        return sensitivity + specificity - 1.0

    def reset_state(self):
        self.true_positives.assign(0)
        self.true_negatives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)

=======
# Enable mixed precision
from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')
print("Mixed precision enabled with policy:", tf.keras.mixed_precision.global_policy())

from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.mixed_precision import LossScaleOptimizer
import numpy as np
import shutil
>>>>>>> Stashed changes

# Set GPU memory growth (this works for both GPU/MPS on Apple Silicon)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, enable=True)
    print(f"SUCCESS: Found and set memory growth for {len(physical_devices)} GPU device(s).")
else:
    print("WARNING: GPU device not found. Using CPU.")

# Custom casting layer that works with Keras
class CastLayer(layers.Layer):
    def __init__(self, dtype):
        super(CastLayer, self).__init__()
        self.dtype = dtype
        
    def call(self, inputs):
        return tf.cast(inputs, self.dtype)

# -----------------------------
# Positional Encoding Layer
# -----------------------------


class PositionalEncoding(layers.Layer):
    def __init__(self, max_len, embed_dim):
        super(PositionalEncoding, self).__init__()
        # Precompute positional encoding in float32 and cast to float16 when needed
        self.pos_encoding = self.positional_encoding(max_len, embed_dim)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model,
        )
        # apply sin to even indices; cos to odd indices
        sines = np.sin(angle_rads[:, 0::2])
        cosines = np.cos(angle_rads[:, 1::2])
        pos_encoding = np.concatenate([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
<<<<<<< Updated upstream
        # Cast the positional encoding to the same dtype as inputs (for mixed
        # precision)
        pos_encoding = tf.cast(self.pos_encoding[:, :seq_len, :], dtype=inputs.dtype)
=======
        # Cast positional encoding to match input dtype
        pos_encoding = tf.cast(self.pos_encoding[:, :seq_len, :], inputs.dtype)
>>>>>>> Stashed changes
        return inputs + pos_encoding


# -----------------------------
# Improved Transformer Block
# -----------------------------


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.2):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        # Use GELU activation in the feed-forward network for smoother
        # nonlinearities.
        self.ffn = models.Sequential(
            [
                layers.Dense(ff_dim, activation=tf.keras.activations.gelu),
                layers.Dropout(dropout_rate),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


# -----------------------------
# Improved SolarKnowledge Model Class
# -----------------------------


class SolarKnowledge:
    model = None
    model_name = "SolarKnowledge"
    callbacks = None
    input_tensor = None

    def __init__(self, early_stopping_patience=3):
        self.model_name = "SolarKnowledge"
        self.callbacks = [
            EarlyStopping(
                monitor="loss",
                patience=early_stopping_patience,
                restore_best_weights=True,
            )
        ]

<<<<<<< Updated upstream
    def build_base_model(
        self,
        input_shape,
        embed_dim=128,  # Increased embedding dimension
        num_heads=4,
        ff_dim=256,  # Increased feed-forward dimension
        num_transformer_blocks=6,  # Use more transformer blocks
        dropout_rate=0.2,
        num_classes=2,
    ):
=======
    def build_base_model(self, input_shape, 
                         embed_dim=128,
                         num_heads=4, 
                         ff_dim=256,
                         num_transformer_blocks=6,
                         dropout_rate=0.2,
                         num_classes=2):
>>>>>>> Stashed changes
        """
        Build a transformer-based model for time-series classification.
        input_shape: tuple (timesteps, features)
        """
        # Create the input layer
        inputs = layers.Input(shape=input_shape, dtype='float32')
        self.input_tensor = inputs
        
        # Use a proper Keras layer for casting to float16
        x = CastLayer(dtype='float16')(inputs)

        # Project the input features into a higher-dimensional embedding space.
        x = layers.Dense(embed_dim)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.Dropout(dropout_rate)(x)

        # Add positional encoding.
        x = PositionalEncoding(max_len=input_shape[0], embed_dim=embed_dim)(x)

        # Apply several transformer blocks.
        for i in range(num_transformer_blocks):
            x = TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)(x)

        # Global average pooling and a dense classification head.
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(dropout_rate)(x)
<<<<<<< Updated upstream
        x = layers.Dense(
            128,
            activation=tf.keras.activations.gelu,
            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
        )(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(
            num_classes,
            activation="softmax",
            activity_regularizer=regularizers.l2(1e-5),
        )(x)
=======
        
        # More aggressive regularization in intermediate layer
        x = layers.Dense(128, activation=tf.keras.activations.gelu,
                       kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-3))(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.Dropout(dropout_rate + 0.1)(x)  # Slightly higher dropout
        
        # The last layer automatically casts to float32 for numerical stability
        outputs = layers.Dense(num_classes, activation='softmax',
                             kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-3),
                             activity_regularizer=regularizers.l2(1e-4))(x)
>>>>>>> Stashed changes

        self.model = models.Model(inputs=inputs, outputs=outputs)
        return self.model

    def summary(self):
        if self.model is not None:
            self.model.summary()
        else:
            print("Model is not built yet!")

<<<<<<< Updated upstream
    def compile(
        self,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
        learning_rate=1e-4,
        use_focal_loss=True,
    ):
        """
        Compile the model with specified loss and metrics.
=======
    def compile(self, loss='categorical_crossentropy', metrics=['accuracy'], learning_rate=1e-4):
        # Set up optimizer with loss scaling for mixed precision
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        optimizer = LossScaleOptimizer(optimizer)
        
        # Compile the model
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
>>>>>>> Stashed changes

        Args:
            loss: Loss function to use. If use_focal_loss is True, this will be overridden.
            metrics: List of metrics to track
            learning_rate: Learning rate for the optimizer
            use_focal_loss: Whether to use focal loss (better for imbalanced data)
        """
        # Create TSS metric
        tss_metric = TrueSkillStatisticMetric()

        # Use focal loss for rare event prediction if specified
        if use_focal_loss:
            # Use CategoricalFocalLoss which is compatible with softmax outputs and one-hot encoded labels
            # Implementing focal loss manually for categorical crossentropy
            def categorical_focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
                """
                Focal loss for multi-class classification with one-hot encoded labels
                """
                # Standard categorical crossentropy
                cross_entropy = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

                # Get the predicted probability for the correct class
                y_pred_softmax = tf.nn.softmax(y_pred, axis=-1)
                p_t = tf.reduce_sum(y_true * y_pred_softmax, axis=-1)

                # Apply the focal term
                focal_term = tf.pow(1 - p_t, gamma)

                # Apply class weights if using alpha
                if alpha > 0:
                    alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
                    alpha_weight = tf.reduce_sum(alpha_factor, axis=-1)
                    focal_loss = alpha_weight * focal_term * cross_entropy
                else:
                    focal_loss = focal_term * cross_entropy

                return focal_loss

            loss = categorical_focal_loss
            print("Using Categorical Focal Loss for rare event awareness")

        # Add TSS to metrics
        if "tss" not in metrics and tss_metric not in metrics:
            metrics = metrics + [tss_metric]

        self.model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=metrics,
        )

    def fit(
        self,
        X_train,
        y_train,
        X_valid=None,
        y_valid=None,
        epochs=100,
        verbose=2,
        batch_size=512,
        class_weight=None,
    ):
        """
        Train the model with optional class weights for imbalanced data.

        Args:
            X_train, y_train: Training data
            X_valid, y_valid: Validation data (optional)
            epochs: Number of training epochs
            verbose: Verbosity level
            batch_size: Batch size for training
            class_weight: Optional dictionary mapping class indices to weights
        """
        validation_data = None
        if (X_valid is not None) and (y_valid is not None):
            validation_data = (X_valid, y_valid)

        # If class_weight is not provided but we want to handle rare events,
        # create a default class weight that emphasizes the positive class
        # (flares)
        if class_weight is None:
            # Default weight for imbalanced binary classification
            # (adjust based on your specific class ratios)
            class_weight = {0: 1.0, 1: 10.0}
            print(f"Using default class weights: {class_weight}")

        return self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            verbose=verbose,
            batch_size=batch_size,
            callbacks=self.callbacks,
            validation_data=validation_data,
            class_weight=class_weight,
        )

    def predict(self, X_test, batch_size=1024, verbose=0):
        """Standard prediction - no MC dropout"""
        predictions = self.model.predict(X_test, verbose=verbose, batch_size=batch_size)
        return predictions

    def mc_predict(self, X_test, n_passes=20, batch_size=1024, verbose=0):
        """
        Monte Carlo dropout prediction - keeps dropout active during inference
        to get uncertainty estimates.

        Args:
            X_test: Input data
            n_passes: Number of forward passes with dropout
            batch_size: Batch size for prediction
            verbose: Verbosity level

        Returns:
            mean_preds: Mean of the predictions across all passes
            std_preds: Standard deviation of predictions (uncertainty)
        """
        if verbose > 0:
            print(f"Performing {n_passes} MC dropout passes...")

        # Initialize list to store predictions
        all_preds = []

        # Perform multiple forward passes with dropout enabled
        for i in range(n_passes):
            if verbose > 0 and i % 5 == 0:
                print(f"MC pass {i+1}/{n_passes}")

            # Use tf.function for better performance
            @tf.function
            def predict_with_dropout(x, training=True):
                return self.model(x, training=training)

            # Process in batches to avoid OOM errors
            preds_batches = []
            for j in range(0, len(X_test), batch_size):
                batch_end = min(j + batch_size, len(X_test))
                X_batch = X_test[j:batch_end]
                # training=True keeps dropout active
                pred_batch = predict_with_dropout(X_batch, training=True).numpy()
                preds_batches.append(pred_batch)

            # Combine batches
            preds = np.concatenate(preds_batches, axis=0)
            all_preds.append(preds)

        # Convert to numpy array for easier calculations
        all_preds = np.array(all_preds)

        # Calculate mean and std
        mean_preds = np.mean(all_preds, axis=0)
        std_preds = np.std(all_preds, axis=0)

        return mean_preds, std_preds

    def save_weights(self, flare_class=None, w_dir=None, verbose=True):
        if w_dir is None and flare_class is None:
            print("You must specify flare_class or w_dir to save the model weights.")
            exit()
        if w_dir is None:
            weight_dir = os.path.join("models", self.model_name, str(flare_class))
        else:
            weight_dir = w_dir
        if os.path.exists(weight_dir):
            shutil.rmtree(weight_dir)
        os.makedirs(weight_dir)
        if verbose:
            print("Saving model weights to directory:", weight_dir)
        weight_file = os.path.join(weight_dir, "model_weights.weights.h5")
        self.model.save_weights(weight_file)

        # Generate a timestamp for this model version
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        metadata = {
            "timestamp": timestamp,
            "model_name": self.model_name,
            "flare_class": flare_class,
            "uses_focal_loss": True,
            "mc_dropout_enabled": True,
        }

        # Save metadata
        with open(os.path.join(weight_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f)

    def load_weights(self, flare_class=None, w_dir=None, timestamp=None, verbose=True):
        if w_dir is None and flare_class is None:
            print("You must specify flare_class or w_dir to load the model weights.")
            exit()
        if w_dir is None:
            weight_dir = os.path.join("models", self.model_name, str(flare_class))
        else:
            weight_dir = w_dir
        if verbose:
            print("Loading weights from model dir:", weight_dir)
        if not os.path.exists(weight_dir):
            print("Model weights directory:", weight_dir, "does not exist!")
            exit()
        if self.model is None:
            print("You must build the model first before loading weights.")
            exit()

        # If a specific timestamp is requested, try to find that version
        if timestamp:
            # Logic for loading a specific timestamped version would go here
            # For now, we'll just use the standard file path
            pass

        filepath = os.path.join(weight_dir, "model_weights.weights.h5")
        status = self.model.load_weights(filepath)
        if status is not None:
            status.expect_partial()

    def load_model(self, input_shape, flare_class, w_dir=None, verbose=True):
        self.build_base_model(input_shape)
        self.compile()
        self.load_weights(flare_class, w_dir=w_dir, verbose=verbose)

    def get_model(self):
        return self.model

    def update_results(self, metrics_dict):
        """Update model metadata with test results"""
        # This function can be expanded to save metrics to the model's metadata
        pass


if __name__ == "__main__":
    # Example usage for debugging: build, compile, and show summary.
    # For example, input_shape is (timesteps, features) e.g., (100, 14)
    example_input_shape = (100, 14)
    model_instance = SolarKnowledge(early_stopping_patience=3)
    model_instance.build_base_model(example_input_shape)
    model_instance.compile(use_focal_loss=True)
    model_instance.summary()

    # Test MC dropout prediction
    X_test = np.random.random((10, 100, 14))
    mean_preds, std_preds = model_instance.mc_predict(X_test, n_passes=5, verbose=1)
    print(f"Mean predictions shape: {mean_preds.shape}")
    print(f"Std predictions shape: {std_preds.shape}")
