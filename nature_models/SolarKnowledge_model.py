"""
 (c) Copyright 2023
 All rights reserved
 Programs written by Yasser Abduallah
 Department of Computer Science
 New Jersey Institute of Technology
 University Heights, Newark, NJ 07102, USA

 Permission to use, copy, modify, and distribute this
 software and its documentation for any purpose and without
 fee is hereby granted, provided that this copyright
 notice appears in all copies. Programmer(s) makes no
 representations about the suitability of this
 software for any purpose.  It is provided "as is" without
 express or implied warranty.

 Alternative transformer-based model with improved capacity for time-series classification.
 @author: Yasser Abduallah (modified)
"""

import shutil
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers, models, regularizers
import tensorflow as tf
import os
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Set up mixed precision (for improved performance on MPS/M2)
tf.keras.mixed_precision.set_global_policy("mixed_float16")
print(
    "Mixed precision enabled. Current policy:", tf.keras.mixed_precision.global_policy()
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

# -----------------------------
# Positional Encoding Layer
# -----------------------------


class PositionalEncoding(layers.Layer):
    def __init__(self, max_len, embed_dim):
        super(PositionalEncoding, self).__init__()
        # Precompute positional encoding (in float32) and cast later in call().
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
        # Cast the positional encoding to the same dtype as inputs (for mixed precision)
        pos_encoding = tf.cast(self.pos_encoding[:, :seq_len, :], dtype=inputs.dtype)
        return inputs + pos_encoding


# -----------------------------
# Improved Transformer Block
# -----------------------------


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.2):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        # Use GELU activation in the feed-forward network for smoother nonlinearities.
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
        """
        Build a transformer-based model for time-series classification.
        input_shape: tuple (timesteps, features)
        """
        inputs = layers.Input(shape=input_shape)
        self.input_tensor = inputs

        # Project the input features into a higher-dimensional embedding space.
        x = layers.Dense(embed_dim)(inputs)
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

        self.model = models.Model(inputs=inputs, outputs=outputs)
        return self.model

    def summary(self):
        if self.model is not None:
            self.model.summary()
        else:
            print("Model is not built yet!")

    def compile(
        self, loss="categorical_crossentropy", metrics=["accuracy"], learning_rate=1e-4
    ):
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
    ):
        validation_data = None
        if (X_valid is not None) and (y_valid is not None):
            validation_data = (X_valid, y_valid)
        self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            verbose=verbose,
            batch_size=batch_size,
            callbacks=self.callbacks,
            validation_data=validation_data,
        )

    def predict(self, X_test, batch_size=1024, verbose=0):
        predictions = self.model.predict(X_test, verbose=verbose, batch_size=batch_size)
        return predictions

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

    def load_weights(self, flare_class=None, w_dir=None, verbose=True):
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

    def load_model(self, input_shape, flare_class, w_dir=None, verbose=True):
        self.build_base_model(input_shape)
        self.compile()
        self.load_weights(flare_class, w_dir=w_dir, verbose=verbose)

    def get_model(self):
        return self.model


if __name__ == "__main__":
    # Example usage for debugging: build, compile, and show summary.
    # For example, input_shape is (timesteps, features) e.g., (100, 14)
    example_input_shape = (100, 14)
    model_instance = SolarKnowledge(early_stopping_patience=3)
    model_instance.build_base_model(example_input_shape)
    model_instance.compile()
    model_instance.summary()
