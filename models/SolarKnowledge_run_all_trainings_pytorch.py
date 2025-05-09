"""
 author: Antanas Zilinskas
 Based on work by Yasser Abduallah

 This script runs all training processes for flare class: C, M, M5 and time window: 24, 48, 72
 using the transformer-based SolarKnowledge model in PyTorch.
 Extended callbacks are added (EarlyStopping, ReduceLROnPlateau) to help the model converge further.

 Improvements:
 - Uses focal loss to handle class imbalance
 - Applies class weights based on flare class rarity
 - Enables Monte Carlo dropout for better uncertainty estimation
 - Uses TensorFlow-compatible weight initialization for better convergence
 - Enhanced with batch normalization, residual connections and AdamW optimizer
 - Uses cosine annealing with warm restarts for better convergence
"""

import argparse
import os
import warnings
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts, OneCycleLR
from model_tracking import (
    compare_models,
    get_latest_version,
    get_next_version,
    save_model_with_metadata,
)
from SolarKnowledge_model_pytorch import SolarKnowledge, set_seed

from utils import data_transform, get_training_data, log, supported_flare_class

warnings.filterwarnings("ignore")

# Set random seed for reproducibility
RANDOM_SEED = 42
set_seed(RANDOM_SEED)  # Use the set_seed function from the model file
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def train(
    time_window,
    flare_class,
    version=None,
    description=None,
    auto_increment=True,
    # Add custom parameter options for notebook integration
    custom_model=None,
    custom_hyperparams=None,
    epochs=100,
    scheduler_type="reduce_on_plateau",
    scheduler_params=None,
    batch_size=512,
    learning_rate=3e-4,
    embed_dim=128,
    transformer_blocks=6,
    use_batch_norm=True,
    use_focal_loss=True,
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

    # Use provided epochs or default
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

    # Use custom model if provided, otherwise create a new one
    if custom_model is not None:
        model = custom_model
    else:
        # Create an instance of the SolarKnowledge transformer-based model
        model = SolarKnowledge(early_stopping_patience=5)  # Match TensorFlow patience
        
        # Build the model with provided parameters
        model.build_base_model(
            input_shape=input_shape,
            embed_dim=128,           # Fixed to exactly match TensorFlow model
            num_heads=4,             # Fixed to exactly match TensorFlow model
            ff_dim=256,              # Fixed to exactly match TensorFlow model
            num_transformer_blocks=6, # Changed from 4 to 6 to match TensorFlow exactly
            dropout_rate=0.2,
            num_classes=2
        )

        # Compile model with specified settings
        model.compile(
            use_focal_loss=use_focal_loss,
            learning_rate=learning_rate,
            weight_decay=1e-4
        )

    # Configure scheduler parameters if not provided
    if scheduler_params is None:
        if scheduler_type == "cosine_with_restarts":
            scheduler_params = {
                "T_0": 10,        # Initial cycle length
                "T_mult": 2,      # Cycle length multiplier
                "min_lr": 1e-7    # Minimum learning rate
            }
        elif scheduler_type == "reduce_on_plateau":
            scheduler_params = {
                "mode": "min",    # Reduce LR on plateau (minimize loss)
                "factor": 0.5,    # Factor by which LR will be reduced
                "patience": 5,    # Wait for 5 epochs with no improvement
                "min_lr": 1e-6    # Minimum learning rate
            }
        elif scheduler_type == "one_cycle":  # Add One Cycle LR
            scheduler_params = {
                "max_lr": learning_rate * 10,  # Maximum learning rate at peak
                "pct_start": 0.3,              # Percentage of cycle spent increasing LR
                "div_factor": 25,              # Initial learning rate is max_lr/div_factor
                "final_div_factor": 1e4,       # Final learning rate is max_lr/final_div_factor
            }

    # Setup learning rate scheduler if requested
    if scheduler_type == "cosine_with_restarts":
        scheduler = CosineAnnealingWarmRestarts(
            model.optimizer,
            T_0=scheduler_params["T_0"],
            T_mult=scheduler_params["T_mult"],
            eta_min=scheduler_params["min_lr"],
        )
        # Wrap scheduler in a function that will be called each epoch
        scheduler_fn = lambda epoch_loss: scheduler.step()
    elif scheduler_type == "reduce_on_plateau":
        scheduler = ReduceLROnPlateau(
            model.optimizer,
            mode=scheduler_params["mode"],
            factor=scheduler_params["factor"],
            patience=scheduler_params["patience"],
            min_lr=scheduler_params["min_lr"],
            verbose=True,
        )
        # Wrap scheduler in a function that will be called each epoch
        scheduler_fn = lambda epoch_loss: scheduler.step(epoch_loss)
    elif scheduler_type == "one_cycle":
        # Calculate total steps for OneCycleLR
        steps_per_epoch = len(X_train) // batch_size
        total_steps = steps_per_epoch * epochs
        
        scheduler = OneCycleLR(
            model.optimizer,
            max_lr=scheduler_params["max_lr"],
            total_steps=total_steps,
            pct_start=scheduler_params["pct_start"],
            div_factor=scheduler_params["div_factor"],
            final_div_factor=scheduler_params["final_div_factor"],
        )
        # OneCycleLR needs to be called after every batch, not epoch
        # So we'll return None here and handle the stepping in the model's _train_epoch method
        scheduler_fn = None
        
        # Update model to step the scheduler after each batch
        model.scheduler = scheduler
        model.use_batch_scheduler = True
    else:
        scheduler_fn = None

    # Train the model and store the history
    log(
        f"Starting training for {flare_class}-class flares with {time_window}h window",
        verbose=True,
    )

    # Train the model using the specified scheduler
    history = model.fit(
        X_train,
        y_train_tr,
        epochs=epochs,
        verbose=2,
        batch_size=batch_size,
        class_weight=class_weight,
        scheduler_type=scheduler_type,
        scheduler_params=scheduler_params
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

    # Use custom hyperparams if provided, otherwise create new ones
    if custom_hyperparams is not None:
        hyperparams = custom_hyperparams
    else:
        # Create hyperparameters dictionary
        hyperparams = {
            "learning_rate": learning_rate,
            "weight_decay": 0.0,        # Match TensorFlow Adam (no weight decay)
            "batch_size": batch_size,
            "early_stopping_patience": 5,
            "epochs": epochs,
            "num_transformer_blocks": 6,  # Changed from 4 to 6 to match TensorFlow model
            "embed_dim": 128,             # Fixed to match TensorFlow model
            "num_heads": 4,               # Fixed to match TensorFlow model
            "ff_dim": 256,                # Fixed to match TensorFlow model
            "dropout_rate": 0.2,
            "focal_loss": use_focal_loss,
            "focal_loss_alpha": 0.25,
            "focal_loss_gamma": 2.0,
            "class_weights": class_weight,
            "framework": "pytorch",
            "weight_initialization": "tf_compatible",
            "gradient_clipping": True,
            "max_grad_norm": 1.0,
            "input_shape": input_shape,
            "scheduler": scheduler_type,
            "scheduler_params": scheduler_params,
            "use_batch_norm": use_batch_norm,
            "optimizer": "Adam",         # Changed from AdamW to Adam to match TensorFlow
            "random_seed": RANDOM_SEED,
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
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        help="Batch size for training",
    )
    parser.add_argument(
        "--flare-classes",
        type=str,
        nargs="+",
        default=["M", "C"],
        help="Flare classes to train on",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="one_cycle",  # Changed default from reduce_on_plateau to one_cycle
        choices=["cosine_with_restarts", "reduce_on_plateau", "one_cycle"],
        help="Learning rate scheduler type",
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
                batch_size=args.batch_size or 512,
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