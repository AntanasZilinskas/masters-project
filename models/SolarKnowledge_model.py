'''
 author: Antanas Zilinskas
'''

import warnings
warnings.filterwarnings("ignore")
import os
import platform
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

# Configure for Apple Silicon (M1/M2/M3)
apple_silicon = False
using_mps = False

# Check for Apple Silicon
if platform.system() == 'Darwin' and platform.processor() == 'arm':
    apple_silicon = True
    print("Apple Silicon detected")
    
    # Enable Metal Performance Shaders (MPS)
    try:
        # These environment variables help with MPS performance
        os.environ['TF_METAL_DEVICE_FORCE'] = '1'  # Force Metal device usage
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations which can cause issues
        
        # Check for GPU availability - standard method
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            using_mps = True
            print(f"GPU devices found: {len(physical_devices)}")
        else:
            # If standard detection doesn't work, try forcing MPS mode
            print("Standard GPU detection didn't find devices, forcing MPS mode")
            using_mps = True  # Force MPS mode
    except Exception as e:
        print(f"Error configuring Metal Performance Shaders: {e}")
        print("Falling back to CPU")
else:
    # Standard GPU configuration for non-Apple hardware
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, enable=True)
            print(f"GPU devices found: {len(physical_devices)}")
            using_mps = True  # Not MPS but we are using GPU
        except Exception as e:
            print(f"Error configuring GPU: {e}")
    else:
        print("WARNING: GPU device not found. Using CPU.")

# IMPORTANT: Force float32 precision for stability with MPS backend
print("Using float32 precision for maximum compatibility")
tf.keras.mixed_precision.set_global_policy('float32')

print("Current precision policy:", tf.keras.mixed_precision.global_policy())

from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import shutil
import json
from datetime import datetime

# -----------------------------
# Positional Encoding Layer
# -----------------------------
class PositionalEncoding(layers.Layer):
    def __init__(self, max_len, embed_dim):
        super(PositionalEncoding, self).__init__()
        # Use float32 for positional encoding to match our global policy
        self.pos_encoding = self.positional_encoding(max_len, embed_dim)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)
        # apply sin to even indices; cos to odd indices
        sines = np.sin(angle_rads[:, 0::2])
        cosines = np.cos(angle_rads[:, 1::2])
        pos_encoding = np.concatenate([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[np.newaxis, ...]
        # Explicitly use float32
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        # Ensure both inputs and positional encoding are float32
        inputs = tf.cast(inputs, tf.float32)
        pos_encoding = self.pos_encoding[:, :seq_len, :]
        return inputs + pos_encoding

# -----------------------------
# Improved Transformer Block
# -----------------------------
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.2):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        # Use GELU activation in the feed-forward network for smoother nonlinearities.
        self.ffn = models.Sequential([
            layers.Dense(ff_dim, activation=tf.keras.activations.gelu),
            layers.Dropout(dropout_rate),
            layers.Dense(embed_dim)
        ])
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

    def __init__(self, early_stopping_patience=3, description=None):
        self.model_name = "SolarKnowledge"
        self.callbacks = [EarlyStopping(monitor='loss', patience=early_stopping_patience, restore_best_weights=True)]
        self.description = description if description else "SolarKnowledge transformer-based model"
        self.metadata = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "description": self.description
        }

    def build_base_model(self, input_shape, 
                         embed_dim=128,        # Increased embedding dimension
                         num_heads=4, 
                         ff_dim=256,           # Increased feed-forward dimension
                         num_transformer_blocks=6,  # Use more transformer blocks
                         dropout_rate=0.2,
                         num_classes=2):
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
        x = layers.Dense(128, activation=tf.keras.activations.gelu,
                         kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(num_classes, activation='softmax',
                               activity_regularizer=regularizers.l2(1e-5))(x)

        self.model = models.Model(inputs=inputs, outputs=outputs)
        return self.model

    def summary(self):
        if self.model is not None:
            self.model.summary()
        else:
            print("Model is not built yet!")

    def compile(self, loss='categorical_crossentropy', metrics=['accuracy'], learning_rate=1e-4):
        # Store training parameters in metadata
        self.metadata["training"] = {
            "optimizer": "Adam",
            "learning_rate": learning_rate,
            "loss": loss,
            "metrics": metrics
        }
        
        # Create a standard optimizer - using float32 throughout
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        print("Using standard float32 optimizer")
        
        # Compile the model with the appropriate optimizer
        self.model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=metrics)

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, epochs=100, verbose=2, batch_size=512):
        validation_data = None
        if (X_valid is not None) and (y_valid is not None):
            validation_data = (X_valid, y_valid)
        self.model.fit(X_train, y_train,
                       epochs=epochs,
                       verbose=verbose,
                       batch_size=batch_size,
                       callbacks=self.callbacks,
                       validation_data=validation_data)

    def predict(self, X_test, batch_size=1024, verbose=0):
        predictions = self.model.predict(X_test,
                                           verbose=verbose,
                                           batch_size=batch_size)
        return predictions

    def save_weights(self, flare_class=None, w_dir=None, verbose=True):
        if w_dir is None and flare_class is None:
            print("You must specify flare_class or w_dir to save the model weights.")
            exit()
        if w_dir is None:
            weight_dir = os.path.join('models', self.model_name, str(flare_class))
        else:
            weight_dir = w_dir
        if os.path.exists(weight_dir):
            shutil.rmtree(weight_dir)
        os.makedirs(weight_dir)
        if verbose:
            print('Saving model weights to directory:', weight_dir)
        weight_file = os.path.join(weight_dir, 'model_weights.weights.h5')
        self.model.save_weights(weight_file)
        
        # Save metadata for tracking
        metadata_file = os.path.join(weight_dir, f'metadata_{self.metadata["timestamp"]}.json')
        latest_metadata_file = os.path.join(weight_dir, 'metadata_latest.json')
        
        # Collect model architecture details
        if self.model is not None:
            self.metadata["model_architecture"] = {
                "num_transformer_blocks": 6,  # This should be dynamically set during build_base_model
                "embed_dim": 128,
                "num_heads": 4,
                "ff_dim": 256,
                "dropout_rate": 0.2,
                "total_params": self.model.count_params()
            }
        
        # Add flare class and time window to metadata
        if flare_class:
            self.metadata["flare_class"] = flare_class
            # Extract time window from directory path if available
            if w_dir:
                parts = w_dir.split(os.sep)
                if len(parts) >= 2:
                    try:
                        self.metadata["time_window"] = parts[-2]
                    except:
                        pass
        
        # Save metadata
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # Also save as latest for easy access
        with open(latest_metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def load_weights(self, flare_class=None, w_dir=None, timestamp=None, verbose=True):
        if w_dir is None and flare_class is None:
            print("You must specify flare_class or w_dir to load the model weights.")
            exit()
        if w_dir is None:
            weight_dir = os.path.join('models', self.model_name, str(flare_class))
        else:
            weight_dir = w_dir
        if verbose:
            print('Loading weights from model dir:', weight_dir)
        if not os.path.exists(weight_dir):
            print('Model weights directory:', weight_dir, 'does not exist!')
            exit()
        if self.model is None:
            print("You must build the model first before loading weights.")
            exit()
        
        # Determine which weights file to load
        if timestamp:
            weight_file = os.path.join(weight_dir, f'model_weights_{timestamp}.weights.h5')
            metadata_file = os.path.join(weight_dir, f'metadata_{timestamp}.json')
        else:
            weight_file = os.path.join(weight_dir, 'model_weights.weights.h5')
            metadata_file = os.path.join(weight_dir, 'metadata_latest.json')
        
        # Check if the specific weights file exists
        if not os.path.exists(weight_file):
            # Fall back to latest weights
            if timestamp:
                print(f"Weights file for timestamp {timestamp} not found, using latest weights.")
                weight_file = os.path.join(weight_dir, 'model_weights.weights.h5')
            else:
                print(f"Weights file {weight_file} does not exist!")
                exit()
        
        # Load weights
        status = self.model.load_weights(weight_file)
        if status is not None:
            status.expect_partial()
        
        # Load metadata if available
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    self.metadata = json.load(f)
                    self.metadata["file_path"] = metadata_file  # Store file path for updates
                if verbose:
                    print(f"Loaded metadata from {metadata_file}")
            except Exception as e:
                print(f"Error loading metadata: {str(e)}")
        else:
            if verbose:
                print(f"No metadata file found at {metadata_file}")
            
        return self.model

    def load_model(self, input_shape, flare_class, w_dir=None, verbose=True):
        self.build_base_model(input_shape)
        self.compile()
        self.load_weights(flare_class, w_dir=w_dir, verbose=verbose)

    def get_model(self):
        return self.model

    def update_results(self, results_dict, verbose=True):
        """Update the model metadata with test results."""
        if not isinstance(results_dict, dict):
            print("Results must be provided as a dictionary")
            return
        
        # Store results in metadata
        self.metadata["results"] = results_dict
        
        # Save updated metadata if it was loaded from a file
        if "file_path" in self.metadata:
            metadata_file = self.metadata["file_path"]
            try:
                with open(metadata_file, 'w') as f:
                    json.dump(self.metadata, f, indent=2)
                if verbose:
                    print(f"Updated metadata in {metadata_file}")
                    
                # Update the latest metadata file as well
                latest_file = os.path.join(os.path.dirname(metadata_file), "metadata_latest.json")
                with open(latest_file, 'w') as f:
                    json.dump(self.metadata, f, indent=2)
            except Exception as e:
                print(f"Error updating metadata file: {str(e)}")
        else:
            if verbose:
                print("No metadata file path available. Results saved in memory only.")


if __name__ == '__main__':
    # Example usage for debugging: build, compile, and show summary.
    # For example, input_shape is (timesteps, features) e.g., (100, 14)
    example_input_shape = (100, 14)
    model_instance = SolarKnowledge(early_stopping_patience=3)
    model_instance.build_base_model(example_input_shape)
    model_instance.compile()
    model_instance.summary() 