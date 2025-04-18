'''
 author: Antanas Zilinskas
 Based on work by Yasser Abduallah
 
 This script runs all training processes for flare class: C, M, M5 and time window: 24, 48, 72
 using the transformer-based SolarKnowledge model.
 Extended callbacks are added (EarlyStopping, ReduceLROnPlateau) to help the model converge further.
'''

import warnings 
warnings.filterwarnings('ignore')
import os
import argparse
import tensorflow as tf
from utils import get_training_data, data_transform, log, supported_flare_class
from SolarKnowledge_model import SolarKnowledge
from model_tracking import (
    save_model_with_metadata, 
    compare_models, 
    get_next_version,
    get_latest_version
)

def train(time_window, flare_class, version=None, description=None, auto_increment=True):
    log('Training is initiated for time window: ' + str(time_window) + ' and flare class: ' + flare_class, verbose=True)
    
    # Determine version automatically if not specified or auto_increment is True
    if version is None or auto_increment:
        version = get_next_version(flare_class, time_window)
        log(f"Automatically using version v{version} (next available)", verbose=True)
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
    
    log(f"Input data shapes - X_train: {X_train.shape}, y_train_tr: {y_train_tr.shape}", verbose=True)
    
    epochs = 200  # extend the number of epochs to allow more complete convergence
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    # Create an instance of the SolarKnowledge transformer-based model.
    model = SolarKnowledge(early_stopping_patience=5)  # increased patience to allow more epochs before stopping
    model.build_base_model(input_shape)  # Build the model
    model.compile()

    # Add an additional ReduceLROnPlateau callback to lower the learning rate when training plateaus.
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss', 
        factor=0.5, 
        patience=3, 
        verbose=1, 
        min_lr=1e-6
    )

    # Combine existing callbacks with the learning rate scheduler
    callbacks = model.callbacks + [reduce_lr]

    # Train the model and store the history
    log(f"Starting training for {flare_class}-class flares with {time_window}h window", verbose=True)
    
    # Use larger batch size to utilize available RAM more effectively
    batch_size = 2048  # Increased from 512 to use more RAM
    log(f"Using larger batch size ({batch_size}) to utilize available RAM", verbose=True)
    
    history = model.model.fit(X_train, y_train_tr,
                    epochs=epochs,
                    verbose=2,
                    batch_size=batch_size,
                    callbacks=callbacks)
    
    # Get performance metrics from training history
    metrics = {}
    if history.history and 'accuracy' in history.history:
        metrics['final_training_accuracy'] = history.history['accuracy'][-1]
        metrics['final_training_loss'] = history.history['loss'][-1]
        metrics['epochs_trained'] = len(history.history['accuracy'])
    
    # Create hyperparameters dictionary
    hyperparams = {
        'learning_rate': 1e-4,
        'batch_size': batch_size,
        'early_stopping_patience': 5,
        'epochs': epochs,
        'num_transformer_blocks': 6,
        'embed_dim': 128,
        'num_heads': 4,
        'ff_dim': 256,
        'dropout_rate': 0.2
    }
    
    # Include information about previous version in metadata if it exists
    if prev_version:
        hyperparams['previous_version'] = prev_version
    
    # Save model with all metadata
    model_dir = save_model_with_metadata(
        model=model,
        metrics=metrics,
        hyperparams=hyperparams, 
        history=history,
        version=version,
        flare_class=flare_class,
        time_window=time_window,
        description=description
    )
    
    log(f"Model saved to {model_dir}", verbose=True)
    return model_dir, version

if __name__ == '__main__':
    # Add command line arguments
    parser = argparse.ArgumentParser(description='Train SolarKnowledge models for solar flare prediction')
    parser.add_argument('--version', '-v', type=str, help='Model version identifier (auto-incremented by default)')
    parser.add_argument('--description', '-d', type=str, help='Description of the model being trained')
    parser.add_argument('--specific-flare', '-f', type=str, help='Train only for a specific flare class (C, M, or M5)')
    parser.add_argument('--specific-window', '-w', type=str, help='Train only for a specific time window (24, 48, or 72)')
    parser.add_argument('--compare', action='store_true', help='Compare all models after training')
    parser.add_argument('--no-auto-increment', action='store_true', help='Do not auto-increment version')
    args = parser.parse_args()
    
    # Determine which flare classes and time windows to train for
    flare_classes = [args.specific_flare] if args.specific_flare else ['C', 'M', 'M5']
    time_windows = [args.specific_window] if args.specific_window else [24, 48, 72]
    
    # Train models
    trained_models = []
    versions = []
    
    for time_window in time_windows:
        for flare_class in flare_classes:
            if flare_class not in supported_flare_class:
                print('Unsupported flare class:', flare_class, 'It must be one of:', ', '.join(supported_flare_class))
                continue
                
            model_dir, version = train(
                str(time_window), 
                flare_class, 
                version=args.version,
                description=args.description,
                auto_increment=not args.no_auto_increment
            )
            
            trained_models.append({
                'time_window': time_window,
                'flare_class': flare_class,
                'model_dir': model_dir,
                'version': version
            })
            versions.append(version)
            log('===========================================================\n\n', verbose=True)
    
    # Compare models if requested
    if args.compare and trained_models:
        log("\nModel Comparison:", verbose=True)
        # Use the actual versions that were used (might be auto-incremented)
        comparison = compare_models(
            list(set(versions)),  # Unique versions
            flare_classes, 
            [str(tw) for tw in time_windows]
        )
        print(comparison) 