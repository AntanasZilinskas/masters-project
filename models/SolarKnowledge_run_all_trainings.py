"""
 author: Antanas Zilinskas

 This script runs all training processes for flare class: C, M, M5 and time window: 24, 48, 72
 using the transformer-based SolarKnowledge model.
 Extended callbacks are added (EarlyStopping, ReduceLROnPlateau) to help the model converge further.

 Improvements:
 - Uses focal loss to handle class imbalance
 - Applies class weights based on flare class rarity
 - Enables Monte Carlo dropout for better uncertainty estimation
"""

import argparse
<<<<<<< Updated upstream
import os
import warnings

import numpy as np
import tensorflow as tf
from model_tracking import (
    compare_models,
    get_latest_version,
    get_next_version,
    save_model_with_metadata,
)
from SolarKnowledge_model import SolarKnowledge

from utils import data_transform, get_training_data, log, supported_flare_class

warnings.filterwarnings("ignore")


def train(
    time_window,
    flare_class,
    version=None,
    description=None,
    auto_increment=True,
):
    log(
        "Training is initiated for time window: "
        + str(time_window)
        + " and flare class: "
        + flare_class,
        verbose=True,
    )

    # Determine version automatically if not specified or auto_increment is
    # True
=======
import tensorflow as tf
import numpy as np
from utils import get_training_data, data_transform, log, supported_flare_class
from SolarKnowledge_model import SolarKnowledge
from model_tracking import (
    save_model_with_metadata, 
    compare_models, 
    get_next_version,
    get_latest_version
)

def get_focal_loss(gamma=2.0, alpha=0.25):
    """
    Create a focal loss function to better handle class imbalance.
    
    Args:
        gamma: Focusing parameter that places more emphasis on hard examples
        alpha: Weighting factor for the positive class
        
    Returns:
        A callable focal loss function
    """
    def focal_loss(y_true, y_pred):
        # Clip to prevent NaN's and Inf's
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        # Calculate cross entropy
        cross_entropy = -y_true * tf.math.log(y_pred)
        
        # Calculate focal loss
        loss = alpha * tf.math.pow(1 - y_pred, gamma) * cross_entropy
        
        # Sum over classes, average over batch
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
    
    return focal_loss

def train(time_window, flare_class, version=None, description=None, auto_increment=True):
    log('Training is initiated for time window: ' + str(time_window) + ' and flare class: ' + flare_class, verbose=True)
    
    # Determine version automatically if not specified or auto_increment is True
>>>>>>> Stashed changes
    if version is None or auto_increment:
        version = get_next_version(flare_class, time_window)
        log(
            f"Automatically using version v{version} (next available)",
            verbose=True,
        )
    else:
        log(f"Using specified version v{version}", verbose=True)

    # Check for previous version to include in description
    prev_version = get_latest_version(flare_class, time_window)
    if prev_version and not description:
        description = f"Iteration on v{prev_version} model for {flare_class}-class flares with {time_window}h window"
    elif not description:
        description = f"Initial model for {flare_class}-class flares with {time_window}h prediction window"

    # Load training data and transform the labels (one-hot encoding)
    X_train, y_train = get_training_data(time_window, flare_class)
    y_train_tr = data_transform(y_train)
<<<<<<< Updated upstream
=======
    
    # For mixed precision, we need to clean the data first then convert to float32
    # (The model will internally use float16 for computation)
    if not isinstance(X_train, np.ndarray):
        X_train = np.array(X_train)
    if not isinstance(y_train_tr, np.ndarray):
        y_train_tr = np.array(y_train_tr)
    
    # Always use float32 for inputs when using mixed precision
    # (TensorFlow will handle the float16 conversions internally)
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    y_train_tr = tf.convert_to_tensor(y_train_tr, dtype=tf.float32)
    
    log(f"Input data shapes - X_train: {X_train.shape}, y_train_tr: {y_train_tr.shape}", verbose=True)
    log(f"Input data types - X_train: {X_train.dtype}, y_train_tr: {y_train_tr.dtype}", verbose=True)
    
    # Fixed at 100 epochs like in the original working model
    epochs = 100  
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    # Create an instance of the SolarKnowledge transformer-based model.
    model = SolarKnowledge(early_stopping_patience=3)  # Original early stopping patience
    model.build_base_model(input_shape)  # Build the model
    model.compile()
>>>>>>> Stashed changes

    epochs = 100  # extend the number of epochs to let the model converge further
    input_shape = (X_train.shape[1], X_train.shape[2])

    # Calculate class weights based on class distribution
    # This helps with the imbalanced nature of flare prediction
    class_counts = np.sum(y_train_tr, axis=0)
    n_samples = len(y_train)
    n_classes = len(class_counts)

    # Calculate weights inversely proportional to class frequencies
    class_weight = {}
    for i in range(n_classes):
        # More aggressive weighting for very rare classes like M5
        if flare_class == "M5":
            # Higher weight for the positive class in M5 flares (very rare)
            class_weight[i] = (
                n_samples / (n_classes * class_counts[i]) if i == 1 else 1.0
            )
        elif flare_class == "M":
            # Moderate weight for M-class flares (rare)
            class_weight[i] = (
                n_samples / (n_classes * class_counts[i]) * 0.8 if i == 1 else 1.0
            )
        else:
            # Lower weight for C-class flares (more common)
            class_weight[i] = (
                n_samples / (n_classes * class_counts[i]) * 0.6 if i == 1 else 1.0
            )

    log(f"Class distribution: {class_counts}", verbose=True)
    log(f"Using class weights: {class_weight}", verbose=True)

    # Create an instance of the SolarKnowledge transformer-based model.
    # increased patience to allow more epochs before stopping
    model = SolarKnowledge(early_stopping_patience=5)
    model.build_base_model(input_shape)  # Build the model

    # Compile model with focal loss for better handling of imbalanced data
    model.compile(use_focal_loss=True)

    # Add an additional ReduceLROnPlateau callback to lower the learning rate
    # when training plateaus.
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="loss", factor=0.5, patience=3, verbose=1, min_lr=1e-6
    )
    
    # Combine existing callbacks with the learning rate scheduler
    callbacks = model.callbacks + [reduce_lr]
    
    # Train the model and store the history
<<<<<<< Updated upstream
    log(
        f"Starting training for {flare_class}-class flares with {time_window}h window",
        verbose=True,
    )

    history = model.model.fit(
        X_train,
        y_train_tr,
        epochs=epochs,
        verbose=2,
        batch_size=512,
        callbacks=callbacks,
        class_weight=class_weight,
    )

    # Get performance metrics from training history
    metrics = {}
    if history.history and "accuracy" in history.history:
        metrics["final_training_accuracy"] = history.history["accuracy"][-1]
        metrics["final_training_loss"] = history.history["loss"][-1]
        metrics["epochs_trained"] = len(history.history["accuracy"])

        # Add TSS if available
        if "tss" in history.history:
            metrics["final_training_tss"] = history.history["tss"][-1]

    # Create hyperparameters dictionary
    hyperparams = {
        "learning_rate": 1e-4,
        "batch_size": 512,
        "early_stopping_patience": 5,
        "epochs": epochs,
        "num_transformer_blocks": 6,
        "embed_dim": 128,
        "num_heads": 4,
        "ff_dim": 256,
        "dropout_rate": 0.2,
        "focal_loss": True,
        "focal_loss_alpha": 0.25,
        "focal_loss_gamma": 2.0,
        "class_weights": class_weight,
=======
    log(f"Starting training for {flare_class}-class flares with {time_window}h window", verbose=True)
    history = model.model.fit(X_train, y_train_tr,
                    epochs=epochs,
                    verbose=2,
                    batch_size=512,
                    callbacks=callbacks)
    
    # Get performance metrics from training history
    metrics = {}
    if history.history and 'accuracy' in history.history:
        # Convert tensor metrics to Python values for serialization
        metrics['final_training_accuracy'] = float(history.history['accuracy'][-1])
        metrics['final_training_loss'] = float(history.history['loss'][-1])
        metrics['epochs_trained'] = len(history.history['accuracy'])
    
    # Create hyperparameters dictionary
    hyperparams = {
        'learning_rate': 1e-4,
        'batch_size': 512,
        'early_stopping_patience': 3,
        'epochs': epochs,
        'num_transformer_blocks': 6,
        'embed_dim': 128,
        'num_heads': 4,
        'ff_dim': 256,
        'dropout_rate': 0.2,
        'precision': 'mixed_float16'  # Document the precision policy used
>>>>>>> Stashed changes
    }

    # Include information about previous version in metadata if it exists
    if prev_version:
        hyperparams["previous_version"] = prev_version

    # Save model with all metadata
    model_dir = save_model_with_metadata(
        model=model,
        metrics=metrics,
        hyperparams=hyperparams,
        history=history,
        version=version,
        flare_class=flare_class,
        time_window=time_window,
        description=description,
    )

    log(f"Model saved to {model_dir}", verbose=True)
    return model_dir, version


if __name__ == "__main__":
    # Add command line arguments
    parser = argparse.ArgumentParser(
        description="Train SolarKnowledge models for solar flare prediction"
    )
    parser.add_argument(
        "--version",
        "-v",
        type=str,
        help="Model version identifier (auto-incremented by default)",
    )
    parser.add_argument(
        "--description",
        "-d",
        type=str,
        help="Description of the model being trained",
    )
    parser.add_argument(
        "--specific-flare",
        "-f",
        type=str,
        help="Train only for a specific flare class (C, M, or M5)",
    )
    parser.add_argument(
        "--specific-window",
        "-w",
        type=str,
        help="Train only for a specific time window (24, 48, or 72)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all models after training",
    )
    parser.add_argument(
        "--no-auto-increment",
        action="store_true",
        help="Do not auto-increment version",
    )
    args = parser.parse_args()

    # Determine which flare classes and time windows to train for
    flare_classes = [args.specific_flare] if args.specific_flare else ["C", "M", "M5"]
    time_windows = [args.specific_window] if args.specific_window else [24, 48, 72]

    # Train models
    trained_models = []
    versions = []

    for time_window in time_windows:
        for flare_class in flare_classes:
            if flare_class not in supported_flare_class:
                print(
                    "Unsupported flare class:",
                    flare_class,
                    "It must be one of:",
                    ", ".join(supported_flare_class),
                )
                continue

            model_dir, version = train(
                str(time_window),
                flare_class,
                version=args.version,
                description=args.description,
                auto_increment=not args.no_auto_increment,
            )

            trained_models.append(
                {
                    "time_window": time_window,
                    "flare_class": flare_class,
                    "model_dir": model_dir,
                    "version": version,
                }
            )
            versions.append(version)
            log(
                "===========================================================\n\n",
                verbose=True,
            )

    # Compare models if requested
    if args.compare and trained_models:
        log("\nModel Comparison:", verbose=True)
        # Use the actual versions that were used (might be auto-incremented)
        comparison = compare_models(
            list(set(versions)),  # Unique versions
            flare_classes,
            [str(tw) for tw in time_windows],
        )
        print(comparison)
