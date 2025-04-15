'''
 This script runs all training processes for flare class: C, M, M5 and time window: 24, 48, 72
 using the transformer-based SolarKnowledge model.
 Extended callbacks are added (EarlyStopping, ReduceLROnPlateau) to help the model converge further.
 author: Antanas Zilinskas
'''

import warnings 
warnings.filterwarnings('ignore')
import os
import argparse
import platform
import time
import numpy as np
from utils import get_training_data, data_transform, log, supported_flare_class
from SolarKnowledge_model import SolarKnowledge
import tensorflow as tf

# Function to check Apple Silicon and MPS availability
def check_gpu_availability():
    print(f"Python version: {platform.python_version()}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Operating system: {platform.system()} {platform.release()}")
    print(f"Processor: {platform.processor()}")
    
    # Check TensorFlow device placement
    try:
        # We need a larger tensor and more iterations for accurate benchmarking
        print("\nRunning GPU performance test...")
        
        # Create larger tensors (4000x4000) for a more meaningful test
        a = tf.random.normal([4000, 4000], dtype=tf.float32)
        b = tf.random.normal([4000, 4000], dtype=tf.float32)
        
        # Warm-up run (first run on GPU is often slow due to compilation)
        with tf.device('/GPU:0'):
            warmup = tf.matmul(a, b)
            # Force execution of the operation
            _ = warmup.numpy().mean()
        
        # Time GPU execution with multiple iterations
        with tf.device('/GPU:0'):
            start_time = time.time()
            # Run the operation multiple times
            for _ in range(3):
                c = tf.matmul(a, b)
                # Force execution of the operation
                _ = c.numpy().mean()
            gpu_time = (time.time() - start_time) / 3  # Average time
            print(f"GPU device: {c.device}")
        
        # Time CPU execution
        with tf.device('/CPU:0'):
            start_time = time.time()
            # Just do one iteration on CPU as it's slower
            c_cpu = tf.matmul(a, b)
            # Force execution of the operation
            _ = c_cpu.numpy().mean()
            cpu_time = time.time() - start_time
            print(f"CPU device: {c_cpu.device}")
        
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        
        print(f"\nPerformance test results:")
        print(f"GPU/MPS time: {gpu_time:.4f}s")
        print(f"CPU time: {cpu_time:.4f}s")
        print(f"Speedup: {speedup:.2f}x")
        
        if speedup > 1.5:
            print("✅ GPU/MPS IS BEING USED SUCCESSFULLY")
        else:
            print("⚠️ GPU/MPS usage detected but performance improvement is limited")
            print("   This is normal for small operations or initial runs")
            
        # Show where computations are being placed
        print("\nDevice for neural network operations:")
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(1000, activation='relu', input_shape=(1000,)),
            tf.keras.layers.Dense(1000, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        x = tf.random.normal([32, 1000])
        with tf.device('/GPU:0'):
            result = model(x)
            print(f"Model inference device: {result.device}")
            
        return True
    except Exception as e:
        print(f"Error testing GPU performance: {e}")
        return False

# Run the GPU check before starting training
check_gpu_availability()

def train(time_window, flare_class, model_description=None):
    log('Training is initiated for time window: ' + str(time_window) + ' and flare class: ' + flare_class, verbose=True)
    
    # Create a detailed model description if one is not provided
    if model_description is None:
        model_description = f"SolarKnowledge model for {flare_class}-class flares with {time_window}h prediction window"
    
    # Load training data and transform the labels (one-hot encoding)
    X_train, y_train = get_training_data(time_window, flare_class)
    y_train_tr = data_transform(y_train)
    
    # Ensure data is in float32 format for consistent typing
    if not isinstance(X_train, tf.Tensor):
        X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    else:
        X_train = tf.cast(X_train, tf.float32)
        
    if not isinstance(y_train_tr, tf.Tensor):
        y_train_tr = tf.convert_to_tensor(y_train_tr, dtype=tf.float32)
    else:
        y_train_tr = tf.cast(y_train_tr, tf.float32)
    
    log(f"X_train shape: {X_train.shape}, dtype: {X_train.dtype}", verbose=True)
    log(f"y_train_tr shape: {y_train_tr.shape}, dtype: {y_train_tr.dtype}", verbose=True)
    
    # Check where the tensors are placed
    log("Checking tensor device placement...", verbose=True)
    with tf.device('/GPU:0'):
        try:
            # Try a small operation to verify GPU usage
            a = tf.random.normal([100, 100], dtype=tf.float32)
            b = tf.matmul(a, a)
            log(f"Test tensor device: {b.device}", verbose=True)
            if 'GPU' in b.device or 'MPS' in b.device or 'Metal' in b.device:
                log("✅ GPU/MPS is available for training", verbose=True)
            else:
                log("⚠️ Training will use CPU (GPU not detected in tensor operations)", verbose=True)
        except Exception as e:
            log(f"Error in device check: {e}", verbose=True)
    
    epochs = 100  # extend the number of epochs to let the model converge further
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    # Create an instance of the SolarKnowledge transformer-based model with description
    model = SolarKnowledge(early_stopping_patience=5, description=model_description)
    
    # Build the model with explicit architecture parameters for tracking
    model.build_base_model(
        input_shape=input_shape,
        embed_dim=128,
        num_heads=4,
        ff_dim=256,
        num_transformer_blocks=6,
        dropout_rate=0.2,
        num_classes=2
    )
    
    # Compile the model - this will track compilation parameters
    model.compile()

    # Add an additional ReduceLROnPlateau callback to lower the learning rate when training plateaus.
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss', 
        factor=0.5, 
        patience=3, 
        verbose=1, 
        min_lr=1e-6
    )
    
    # Add TensorBoard callback to monitor GPU usage
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=f'logs/{time_window}_{flare_class}_{time.strftime("%Y%m%d-%H%M%S")}',
        histogram_freq=1,
        profile_batch='500,520'  # Profile a few batches to check GPU usage
    )

    # Combine existing callbacks with the learning rate scheduler and monitoring
    callbacks = model.callbacks + [reduce_lr, tensorboard_callback]
    
    # Log start time for performance monitoring
    start_time = time.time()

    # Train the model - this will track training metrics
    try:
        log("Starting model training...", verbose=True)
        
        # Set memory growth for GPU if available
        try:
            physical_devices = tf.config.list_physical_devices('GPU')
            if physical_devices:
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
                log(f"Memory growth enabled for {len(physical_devices)} GPU devices", verbose=True)
        except Exception as e:
            log(f"Could not set memory growth: {str(e)}", verbose=True)
        
        # Use with_options to ensure the tensors are placed on GPU
        options = tf.data.Options()
        options.experimental_device = '/GPU:0'
        
        # Create a dataset to better control the data flow to GPU
        dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train_tr))
        dataset = dataset.batch(512).with_options(options).prefetch(tf.data.AUTOTUNE)
        
        history = model.model.fit(
            dataset,
            epochs=epochs,
            verbose=2,
            callbacks=callbacks
        )
        
        # Calculate and log training time
        training_time = time.time() - start_time
        log(f"Training completed in {training_time:.2f} seconds", verbose=True)
        
        # Store training metrics in metadata
        if hasattr(model, 'metadata'):
            model.metadata['training']['training_time_seconds'] = training_time
            model.metadata['training']['training_samples'] = len(X_train)
            model.metadata['training']['actual_epochs'] = len(history.history['loss'])
            model.metadata['training']['final_loss'] = history.history['loss'][-1]
            if 'accuracy' in history.history:
                model.metadata['training']['final_accuracy'] = history.history['accuracy'][-1]
                
    except Exception as e:
        log(f"Error during training: {str(e)}", verbose=True)
        # Try with smaller batch size if there's an error
        log("Trying with smaller batch size and direct tensor input...", verbose=True)
        try:
            # Try with direct tensor input rather than Dataset
            history = model.model.fit(
                X_train, y_train_tr,
                epochs=epochs,
                verbose=2,
                batch_size=256,
                callbacks=callbacks
            )
                        
            # Calculate and log training time
            training_time = time.time() - start_time
            log(f"Training completed in {training_time:.2f} seconds", verbose=True)
            
            # Store training metrics with reduced batch size
            if hasattr(model, 'metadata'):
                model.metadata['training']['training_time_seconds'] = training_time
                model.metadata['training']['batch_size'] = 256  # Override the default 512
                model.metadata['training']['training_samples'] = len(X_train)
                model.metadata['training']['actual_epochs'] = len(history.history['loss'])
                model.metadata['training']['final_loss'] = history.history['loss'][-1]
                if 'accuracy' in history.history:
                    model.metadata['training']['final_accuracy'] = history.history['accuracy'][-1]
                    
        except Exception as e:
            log(f"Still encountering error: {str(e)}", verbose=True)
            raise e
    
    # Construct a directory path for saving the weights.
    w_dir = os.path.join('weights', str(time_window), str(flare_class))
    
    # Ensure the directory exists
    os.makedirs(w_dir, exist_ok=True)
    
    # Save weights - this will create timestamped files
    model.save_weights(flare_class=flare_class, w_dir=w_dir)
    
    log(f'Model saved with timestamp: {model.metadata["timestamp"]}', verbose=True)
    return model.metadata["timestamp"]

if __name__ == '__main__':
    # Add command line arguments for optional model description
    parser = argparse.ArgumentParser(description='Train SolarKnowledge models for solar flare prediction')
    parser.add_argument('--description', '-d', type=str, help='Description of the model being trained')
    args = parser.parse_args()
    
    model_description = args.description
    
    # Loop over the defined time windows and flare classes 
    for time_window in [24, 48, 72]:
        for flare_class in ['C', 'M', 'M5']:
            if flare_class not in supported_flare_class:
                print('Unsupported flare class:', flare_class, 'It must be one of:', ', '.join(supported_flare_class))
                continue
            timestamp = train(str(time_window), flare_class, model_description)
            log(f'Model timestamp: {timestamp}', verbose=True)
            log('===========================================================\n\n', verbose=True) 