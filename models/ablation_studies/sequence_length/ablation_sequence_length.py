'''
 author: Antanas Zilinskas
 
 Ablation Study: Sequence Length
 
 This script tests the impact of different sequence lengths on model performance.
 It resamples the input sequences to various lengths and compares model performance
 for each flare class and time window.
 
 Usage:
   python ablation_sequence_length.py --flare-class M5 --time-window 24
   python ablation_sequence_length.py --all  # Run for all combinations
'''

import os
import sys
import json
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, balanced_accuracy_score, confusion_matrix

# Add parent directory to path so we can import from models module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import project utilities
from utils import get_training_data, get_testing_data, data_transform, log, supported_flare_class
from SolarKnowledge_model import SolarKnowledge

# Configure GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, enable=True)
    print(f"Found and configured {len(physical_devices)} GPU device(s).")
else:
    print("No GPU devices found.")

# Set up mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')
print("Mixed precision enabled. Current policy:", tf.keras.mixed_precision.global_policy())

# Sequence lengths to test
SEQUENCE_LENGTHS = [5, 8, 10, 12, 15, 20, 25, 30, 40, 50]

def resample_sequence(sequence, target_length):
    """
    Resample a sequence to a target length using linear interpolation.
    
    Args:
        sequence: Input sequence of shape (original_length, features)
        target_length: Desired sequence length
        
    Returns:
        Resampled sequence of shape (target_length, features)
    """
    original_length = sequence.shape[0]
    num_features = sequence.shape[1]
    
    # If sequences are same length, return original
    if original_length == target_length:
        return sequence
    
    # Create a new array for the resampled sequence
    resampled = np.zeros((target_length, num_features))
    
    # Create indices for the original and target sequences
    orig_indices = np.arange(original_length)
    target_indices = np.linspace(0, original_length - 1, target_length)
    
    # Interpolate each feature
    for i in range(num_features):
        resampled[:, i] = np.interp(target_indices, orig_indices, sequence[:, i])
        
    return resampled

def resample_dataset(X, target_length):
    """
    Resample all sequences in a dataset to the target length.
    
    Args:
        X: Dataset of shape (num_samples, original_length, features)
        target_length: Desired sequence length
        
    Returns:
        Resampled dataset of shape (num_samples, target_length, features)
    """
    num_samples = X.shape[0]
    num_features = X.shape[2]
    
    # Create new array for resampled data
    X_resampled = np.zeros((num_samples, target_length, num_features))
    
    # Resample each sequence
    for i in range(num_samples):
        X_resampled[i] = resample_sequence(X[i], target_length)
    
    return X_resampled

def run_ablation_study(flare_class, time_window):
    """
    Run ablation study for a specific flare class and time window.
    
    Args:
        flare_class: Flare class to test (C, M, or M5)
        time_window: Prediction window in hours (24, 48, or 72)
    """
    log(f"Running sequence length ablation study for {flare_class}-class flares with {time_window}h window", verbose=True)
    
    # Load data
    X_train, y_train = get_training_data(time_window, flare_class)
    X_test, y_test = get_testing_data(time_window, flare_class)
    
    # Transform labels
    y_train_tr = data_transform(y_train)
    
    # Convert test labels for evaluation
    y_test = np.array(y_test)
    if y_test.ndim > 1:
        y_true = np.argmax(y_test, axis=-1)
    else:
        y_true = y_test
    
    # Get original sequence length
    original_length = X_train.shape[1]
    log(f"Original sequence length: {original_length}", verbose=True)
    
    # Dictionary to store results
    results = {
        'flare_class': flare_class,
        'time_window': time_window,
        'original_length': original_length,
        'metrics': {}
    }
    
    # Iterate through sequence lengths
    for seq_length in SEQUENCE_LENGTHS:
        log(f"Testing sequence length: {seq_length}", verbose=True)
        
        # Skip if sequence length is too large compared to original - allowing up to 5x now
        # since we know the original length is small (10)
        if seq_length > original_length * 5:
            log(f"Skipping length {seq_length} (more than 5x original length)", verbose=True)
            continue
            
        # Skip if length is too small (less than 3)
        if seq_length < 3:
            log(f"Skipping length {seq_length} (too small for meaningful analysis)", verbose=True)
            continue
        
        # Resample data
        X_train_resampled = resample_dataset(X_train, seq_length)
        X_test_resampled = resample_dataset(X_test, seq_length)
        
        log(f"Resampled data shapes - X_train: {X_train_resampled.shape}, X_test: {X_test_resampled.shape}", verbose=True)
        
        # Create and compile model
        input_shape = (seq_length, X_train.shape[2])
        model = SolarKnowledge(early_stopping_patience=5)
        model.build_base_model(input_shape)
        model.compile()
        
        # Train model
        log(f"Training model with sequence length {seq_length}...", verbose=True)
        history = model.model.fit(
            X_train_resampled, 
            y_train_tr,
            epochs=50,  # Fewer epochs for quicker ablation study
            batch_size=512,
            verbose=1,
            callbacks=model.callbacks
        )
        
        # Evaluate model
        predictions = model.predict(X_test_resampled)
        predicted_classes = np.argmax(predictions, axis=-1)
        
        # Calculate metrics
        acc = accuracy_score(y_true, predicted_classes)
        prec = precision_score(y_true, predicted_classes)
        rec = recall_score(y_true, predicted_classes)
        bal_acc = balanced_accuracy_score(y_true, predicted_classes)
        
        # Calculate TSS
        cm = confusion_matrix(y_true, predicted_classes)
        sensitivity = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
        specificity = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
        tss = sensitivity + specificity - 1
        
        # Store metrics
        results['metrics'][seq_length] = {
            'accuracy': round(float(acc), 4),
            'precision': round(float(prec), 4),
            'recall': round(float(rec), 4),
            'balanced_accuracy': round(float(bal_acc), 4),
            'TSS': round(float(tss), 4),
            'confusion_matrix': cm.tolist(),
            'epochs_trained': len(history.history['accuracy']),
            'final_loss': float(history.history['loss'][-1]),
            'training_time': None  # Will be filled in during the study
        }
        
        log(f"Completed sequence length {seq_length} with accuracy {acc:.4f}, precision {prec:.4f}, recall {rec:.4f}, TSS {tss:.4f}", verbose=True)
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    results_dir = os.path.join("models", "ablation_studies", "sequence_length", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    result_file = os.path.join(results_dir, f"seq_length_ablation_{flare_class}_{time_window}h_{timestamp}.json")
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    log(f"Saved results to {result_file}", verbose=True)
    
    # Visualize results
    visualize_results(results, results_dir, timestamp)
    
    return results

def visualize_results(results, output_dir, timestamp):
    """
    Create visualizations of ablation study results.
    
    Args:
        results: Results dictionary from ablation study
        output_dir: Directory to save visualizations
        timestamp: Timestamp to use in filenames
    """
    flare_class = results['flare_class']
    time_window = results['time_window']
    
    # Extract data for plotting
    seq_lengths = sorted([int(k) for k in results['metrics'].keys()])
    accuracy = [results['metrics'][str(s)]['accuracy'] for s in seq_lengths]
    precision = [results['metrics'][str(s)]['precision'] for s in seq_lengths]
    recall = [results['metrics'][str(s)]['recall'] for s in seq_lengths]
    tss = [results['metrics'][str(s)]['TSS'] for s in seq_lengths]
    
    # Set up the figure
    plt.figure(figsize=(12, 8))
    
    # Plot metrics
    plt.plot(seq_lengths, accuracy, 'o-', label='Accuracy')
    plt.plot(seq_lengths, precision, 's-', label='Precision')
    plt.plot(seq_lengths, recall, '^-', label='Recall')
    plt.plot(seq_lengths, tss, 'D-', label='TSS')
    
    # Add original length marker
    plt.axvline(x=results['original_length'], color='r', linestyle='--', 
                label=f"Original Length ({results['original_length']})")
    
    # Customize plot
    plt.title(f"Impact of Sequence Length on {flare_class}-class Flare Prediction ({time_window}h window)")
    plt.xlabel("Sequence Length")
    plt.ylabel("Metric Value")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save figure
    plot_file = os.path.join(output_dir, f"seq_length_plot_{flare_class}_{time_window}h_{timestamp}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    log(f"Saved visualization to {plot_file}", verbose=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Perform sequence length ablation study for solar flare prediction")
    parser.add_argument("--flare-class", "-f", choices=supported_flare_class, help="Flare class to test")
    parser.add_argument("--time-window", "-t", choices=['24', '48', '72'], help="Prediction window in hours")
    parser.add_argument("--all", action="store_true", help="Run for all flare classes and time windows")
    
    args = parser.parse_args()
    
    if args.all:
        # Run for all combinations
        for flare_class in ['C', 'M', 'M5']:
            for time_window in ['24', '48', '72']:
                run_ablation_study(flare_class, time_window)
    elif args.flare_class and args.time_window:
        # Run for specific combination
        run_ablation_study(args.flare_class, args.time_window)
    else:
        parser.error("Either specify both --flare-class and --time-window, or use --all") 