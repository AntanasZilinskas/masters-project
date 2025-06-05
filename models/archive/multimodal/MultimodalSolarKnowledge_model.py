"""
Multimodal SolarKnowledge model that combines SHARP parameters and SDO images
for solar flare prediction.

Author: Antanas Zilinskas
"""

import os
import shutil
import warnings

import numpy as np
import tensorflow as tf
from SolarKnowledge_model import PositionalEncoding, TransformerBlock
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Set up mixed precision (for improved performance on MPS/M2)
tf.keras.mixed_precision.set_global_policy("mixed_float16")
print(
    "Mixed precision enabled. Current policy:",
    tf.keras.mixed_precision.global_policy(),
)


# Import the original SolarKnowledge components

# Set GPU memory growth
physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, enable=True)
    print(
        f"SUCCESS: Found and set memory growth for {len(physical_devices)} GPU device(s)."
    )
else:
    print("WARNING: GPU device not found.")

# -----------------------------
# CNN for Image Feature Extraction
# -----------------------------


class ImageEncoder(layers.Layer):
    def __init__(self, embed_dim=128):
        super(ImageEncoder, self).__init__()
        self.embed_dim = embed_dim

        # CNN layers for feature extraction
        self.conv1 = layers.Conv2D(
            32, kernel_size=3, strides=2, padding="same", activation="relu"
        )
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(
            64, kernel_size=3, strides=2, padding="same", activation="relu"
        )
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(
            128, kernel_size=3, strides=2, padding="same", activation="relu"
        )
        self.bn3 = layers.BatchNormalization()
        self.conv4 = layers.Conv2D(
            256, kernel_size=3, strides=2, padding="same", activation="relu"
        )
        self.bn4 = layers.BatchNormalization()

        # Global pooling to get a fixed-size representation
        self.global_pool = layers.GlobalAveragePooling2D()

        # Project to the embedding dimension
        self.projection = layers.Dense(embed_dim)

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.conv4(x)
        x = self.bn4(x, training=training)
        x = self.global_pool(x)
        return self.projection(x)


# -----------------------------
# Multimodal Fusion Layer
# -----------------------------


class MultimodalFusion(layers.Layer):
    def __init__(self, embed_dim):
        super(MultimodalFusion, self).__init__()
        self.attention = layers.MultiHeadAttention(num_heads=4, key_dim=embed_dim)
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
        self.dense1 = layers.Dense(embed_dim * 2, activation="relu")
        self.dense2 = layers.Dense(embed_dim)

    def call(self, time_series_features, image_features, training=False):
        # Expand image features to match time series sequence length
        batch_size = tf.shape(time_series_features)[0]
        seq_len = tf.shape(time_series_features)[1]

        # Reshape image features to [batch_size, 1, embed_dim]
        image_features = tf.reshape(image_features, [batch_size, 1, -1])

        # Repeat image features for each time step
        image_features = tf.repeat(image_features, seq_len, axis=1)

        # Concatenate time series and image features
        combined_features = tf.concat([time_series_features, image_features], axis=-1)

        # Apply self-attention to learn cross-modal interactions
        attention_output = self.attention(
            combined_features,
            combined_features,
            combined_features,
            training=training,
        )

        # Add & normalize
        normalized = self.layernorm(combined_features + attention_output)

        # Apply feed-forward network for fusion
        fusion = self.dense1(normalized)
        fusion = self.dense2(fusion)

        return fusion


# -----------------------------
# Multimodal SolarKnowledge Model
# -----------------------------


class MultimodalSolarKnowledge:
    def __init__(
        self,
        embed_dim=128,
        num_heads=8,
        ff_dim=256,
        num_transformer_blocks=4,
        dropout_rate=0.1,
        early_stopping_patience=5,
    ):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.dropout_rate = dropout_rate
        self.early_stopping_patience = early_stopping_patience
        self.model = None
        self.model_name = "MultimodalSolarKnowledge"

        # Define callbacks
        self.callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=self.early_stopping_patience,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                verbose=1,
                min_lr=1e-6,
            ),
        ]

    def build_base_model(self, time_series_shape, image_shape=(256, 256, 1)):
        """
        Build the multimodal model architecture.

        Parameters:
        -----------
        time_series_shape : tuple
            Shape of time series input (seq_len, features)
        image_shape : tuple
            Shape of image input (height, width, channels)
        """
        # Time series input
        time_series_inputs = layers.Input(
            shape=time_series_shape, name="time_series_input"
        )

        # Image input
        image_inputs = layers.Input(shape=image_shape, name="image_input")

        # Process time series with transformer
        x1 = layers.Dense(self.embed_dim)(time_series_inputs)
        x1 = PositionalEncoding(time_series_shape[0], self.embed_dim)(x1)

        for _ in range(self.num_transformer_blocks):
            x1 = TransformerBlock(
                self.embed_dim, self.num_heads, self.ff_dim, self.dropout_rate
            )(x1)

        # Process image with CNN
        image_encoder = ImageEncoder(self.embed_dim)
        x2 = image_encoder(image_inputs)

        # Fuse modalities
        fusion_layer = MultimodalFusion(self.embed_dim)
        fused_features = fusion_layer(x1, x2)

        # Global average pooling over sequence dimension
        x = layers.GlobalAveragePooling1D()(fused_features)

        # Final classification layers
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(2, activation="softmax")(x)

        # Create model
        self.model = models.Model(
            inputs=[time_series_inputs, image_inputs],
            outputs=outputs,
            name=self.model_name,
        )

        return self.model

    def compile(self, learning_rate=1e-4):
        """Compile the model with appropriate optimizer and loss function."""
        if self.model is None:
            raise ValueError("Model must be built before compilation")

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        self.model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

    def summary(self):
        """Print model summary."""
        if self.model is None:
            raise ValueError("Model must be built before getting summary")
        self.model.summary()

    def train(
        self,
        X_time_series,
        X_images,
        y,
        validation_split=0.1,
        epochs=100,
        batch_size=32,
        verbose=1,
    ):
        """
        Train the model on multimodal data.

        Parameters:
        -----------
        X_time_series : numpy.ndarray
            Time series input data
        X_images : numpy.ndarray
            Image input data
        y : numpy.ndarray
            Target labels (one-hot encoded)
        """
        if self.model is None:
            raise ValueError("Model must be built and compiled before training")

        history = self.model.fit(
            {"time_series_input": X_time_series, "image_input": X_images},
            y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=self.callbacks,
            verbose=verbose,
        )

        return history

    def predict(self, X_time_series, X_images, batch_size=32, verbose=0):
        """
        Make predictions with the model.

        Parameters:
        -----------
        X_time_series : numpy.ndarray
            Time series input data
        X_images : numpy.ndarray
            Image input data
        """
        if self.model is None:
            raise ValueError("Model must be built before prediction")

        predictions = self.model.predict(
            {"time_series_input": X_time_series, "image_input": X_images},
            batch_size=batch_size,
            verbose=verbose,
        )

        return predictions

    def save_weights(self, flare_class=None, w_dir=None, verbose=True):
        """Save model weights to disk."""
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

    def load_weights(self, flare_class=None, w_dir=None, verbose=True):
        """Load model weights from disk."""
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
        filepath = os.path.join(weight_dir, "model_weights.weights.h5")
        status = self.model.load_weights(filepath)
        if status is not None:
            status.expect_partial()

    def load_model(
        self,
        time_series_shape,
        image_shape,
        flare_class,
        w_dir=None,
        verbose=True,
    ):
        """Build, compile, and load weights for the model."""
        self.build_base_model(time_series_shape, image_shape)
        self.compile()
        self.load_weights(flare_class, w_dir=w_dir, verbose=verbose)

    def get_model(self):
        """Return the Keras model."""
        return self.model


if __name__ == "__main__":
    # Example usage for debugging
    time_series_shape = (100, 14)  # (timesteps, features)
    image_shape = (256, 256, 1)  # (height, width, channels)

    model_instance = MultimodalSolarKnowledge(early_stopping_patience=5)
    model_instance.build_base_model(time_series_shape, image_shape)
    model_instance.compile()
    model_instance.summary()
