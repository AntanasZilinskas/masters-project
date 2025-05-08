"""
 author: Antanas Zilinskas


 This script runs all training processes for flare class: C, M, M5 and time window: 24, 48, 72
 using the transformer-based SolarKnowledge model in PyTorch.
 Extended callbacks are added (EarlyStopping, ReduceLROnPlateau) to help the model converge further.

 Improvements:
 - Uses focal loss to handle class imbalance
 - Applies class weights based on flare class rarity
 - Enables Monte Carlo dropout for better uncertainty estimation
 - Uses TensorFlow-compatible weight initialization for better convergence
"""

import argparse
import os
import warnings

import numpy as np
import torch
from model_tracking import (
    compare_models,
    get_latest_version,
    get_next_version,
    save_model_with_metadata,
)
from SolarKnowledge_model_pytorch import SolarKnowledge

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
                n_samples / (n_classes * class_counts[i]) * 0.8
                if i == 1
                else 1.0
            )
        else:
            # Lower weight for C-class flares (more common)
            class_weight[i] = (
                n_samples / (n_classes * class_counts[i]) * 0.6
                if i == 1
                else 1.0
            )

    log(f"Class distribution: {class_counts}", verbose=True)
    log(f"Using class weights: {class_weight}", verbose=True)

    # Create an instance of the SolarKnowledge transformer-based model
    # increased patience to allow more epochs before stopping
    model = SolarKnowledge(early_stopping_patience=5)
    model.build_base_model(input_shape)  # Build the model

    # Compile model with focal loss for better handling of imbalanced data
    model.compile(use_focal_loss=True)

    # Set up learning rate scheduler - PyTorch equivalent of ReduceLROnPlateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        model.optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3, 
        verbose=True, 
        min_lr=1e-6
    )

    # Train the model and store the history
    log(
        f"Starting training for {flare_class}-class flares with {time_window}h window",
        verbose=True,
    )

    # Custom fit method to use scheduler
    def scheduler_step(loss):
        scheduler.step(loss)

    # Train the model, using the learning rate scheduler
    history = model.fit(
        X_train,
        y_train_tr,
        epochs=epochs,
        verbose=2,
        batch_size=512,
        class_weight=class_weight,
        callbacks={'lr_scheduler': scheduler_step}  # Pass scheduler as callback
    )

    # Get performance metrics from training history
    metrics = {}
    if history and 'accuracy' in history:
        metrics["final_training_accuracy"] = history['accuracy'][-1]
        metrics["final_training_loss"] = history['loss'][-1]
        metrics["epochs_trained"] = len(history['accuracy'])

        # Add TSS if available
        if 'tss' in history and history['tss'] is not None:
            metrics["final_training_tss"] = history['tss'][-1]

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
        "framework": "pytorch",
        "weight_initialization": "tf_compatible",  # Using TensorFlow-compatible initialization
        "gradient_clipping": True,
        "max_grad_norm": 1.0,
        "input_shape": input_shape,  # Add input shape for model metadata
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
    flare_classes = (
        [args.specific_flare] if args.specific_flare else ["C", "M", "M5"]
    )
    time_windows = (
        [args.specific_window] if args.specific_window else [24, 48, 72]
    )

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