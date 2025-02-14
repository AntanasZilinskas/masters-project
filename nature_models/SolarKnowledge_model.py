'''
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

 @author: Yasser Abduallah
'''

import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import shutil

if tf.test.gpu_device_name() != '/device:GPU:0':
    print('WARNING: GPU device not found.')
else:
    print('SUCCESS: Found GPU: {}'.format(tf.test.gpu_device_name()))
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# -----------------------------
# Positional Encoding Layer
# -----------------------------
class PositionalEncoding(layers.Layer):
    def __init__(self, max_len, embed_dim):
        super(PositionalEncoding, self).__init__()
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
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pos_encoding[:, :seq_len, :]

# -----------------------------
# Transformer Block
# -----------------------------
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = models.Sequential([
            layers.Dense(ff_dim, activation="relu"),
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
# SolarKnowledge Model Class
# -----------------------------
class SolarKnowledge:
    model = None
    model_name = "SolarKnowledge"
    callbacks = None
    input_tensor = None

    def __init__(self, early_stopping_patience=3):
        self.model_name = "SolarKnowledge"
        self.callbacks = [EarlyStopping(monitor='loss', patience=early_stopping_patience, restore_best_weights=True)]

    def build_base_model(self, input_shape, 
                         embed_dim=64, 
                         num_heads=4, 
                         ff_dim=128, 
                         num_transformer_blocks=4,
                         dropout_rate=0.1,
                         num_classes=2):
        """
        Build a transformer-based model for time-series classification.
        input_shape: tuple (timesteps, features)
        """
        inputs = layers.Input(shape=input_shape)
        self.input_tensor = inputs

        # Project input features to the embed_dim space
        x = layers.Dense(embed_dim)(inputs)
        # Add positional encoding to capture order information
        x = PositionalEncoding(max_len=input_shape[0], embed_dim=embed_dim)(x)

        # Apply a series of transformer blocks
        for i in range(num_transformer_blocks):
            x = TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)(x)

        # Global pooling to collapse the sequence dimension
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(64, activation='relu',
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
        self.model.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
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

    def load_weights(self, flare_class=None, w_dir=None, verbose=True):
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
        self.model.load_weights(os.path.join(weight_dir, 'model_weights')).expect_partial()

    def load_model(self, input_shape, flare_class, w_dir=None, verbose=True):
        self.build_base_model(input_shape)
        self.compile()
        self.load_weights(flare_class, w_dir=w_dir, verbose=verbose)

    def get_model(self):
        return self.model


if __name__ == '__main__':
    # Example usage for debugging: build, compile, and show summary.
    # Here, input_shape is (timesteps, features), e.g., (100, 14)
    example_input_shape = (100, 14)
    model_instance = SolarKnowledge(early_stopping_patience=3)
    model_instance.build_base_model(example_input_shape)
    model_instance.compile()
    model_instance.summary() 