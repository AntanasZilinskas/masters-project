"""
This script runs all training processes for the multimodal SolarKnowledge model
combining SHARP parameters and SDO/HMI magnetogram images.

Author: Antanas Zilinskas
"""

import os
import warnings

import numpy as np
import tensorflow as tf
from multimodal_utils import get_multimodal_training_data
from MultimodalSolarKnowledge_model import MultimodalSolarKnowledge

from utils import data_transform, log, supported_flare_class

warnings.filterwarnings("ignore")


def train(time_window, flare_class):
    log(
        "Multimodal training initiated for time window: "
        + str(time_window)
        + " and flare class: "
        + flare_class,
        verbose=True,
    )

    # Load multimodal training data
    X_time_series, X_images, y_train = get_multimodal_training_data(
        time_window, flare_class
    )
    y_train_tr = data_transform(y_train)  # One-hot encode labels

    # Define model parameters
    epochs = 100
    time_series_shape = (X_time_series.shape[1], X_time_series.shape[2])
    image_shape = X_images.shape[1:]  # (height, width, channels)

    # Create model instance
    model = MultimodalSolarKnowledge(
        embed_dim=128,
        num_heads=8,
        ff_dim=256,
        num_transformer_blocks=4,
        dropout_rate=0.1,
        early_stopping_patience=5,
    )

    # Build and compile model
    model.build_base_model(time_series_shape, image_shape)
    model.compile(learning_rate=1e-4)

    # Add learning rate scheduler
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="loss", factor=0.5, patience=3, verbose=1, min_lr=1e-6
    )

    # Combine existing callbacks with the learning rate scheduler
    callbacks = model.callbacks + [reduce_lr]

    # Train model
    model.train(
        X_time_series=X_time_series,
        X_images=X_images,
        y=y_train_tr,
        validation_split=0.1,
        epochs=epochs,
        batch_size=32,
        verbose=1,
        callbacks=callbacks,
    )

    # Save model weights
    w_dir = os.path.join(
        "models", "multimodal", str(time_window), str(flare_class)
    )
    model.save_weights(flare_class=flare_class, w_dir=w_dir)

    log(
        f"Multimodal model training completed for {time_window}h and flare class {flare_class}",
        verbose=True,
    )


if __name__ == "__main__":
    # Loop over the defined time windows and flare classes
    for time_window in [24, 48, 72]:
        for flare_class in ["C", "M", "M5"]:
            if flare_class not in supported_flare_class:
                print(
                    "Unsupported flare class:",
                    flare_class,
                    "It must be one of:",
                    ", ".join(supported_flare_class),
                )
                continue
            train(str(time_window), flare_class)
            log(
                "===========================================================\n\n",
                verbose=True,
            )
