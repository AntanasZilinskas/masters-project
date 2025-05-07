#!/usr/bin/env python
"""
Test script for EVEREST models (all flare classes and time windows)

This script will:
1. Find and load previously trained EVEREST models
2. Test them on the test set for their respective flare class and time window
3. Compute and record performance metrics
4. Update the model's metadata with the test results
5. Generate uncertainty plots if requested
"""

import argparse
import os
import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
import tensorflow as tf

# Import EVEREST model, testing functions, and utilities
from everest_model import EVEREST
from test_everest import test, compute_roc_curve
from utils import get_testing_data, supported_flare_class
from model_tracking import load_model

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Default configuration
FLARE_CLASSES = ["C", "M", "M5"]  # Flare classes to test
TIME_WINDOWS = ["24", "48", "72"]  # Time windows in hours
MC_PASSES = 20  # Number of Monte Carlo passes for uncertainty estimation
TEST_BATCH_SIZE = 256  # Batch size for testing

# This dictionary will hold metrics for all models
all_metrics = {}

def find_model_path(flare_class, time_window, version=None):
    """
    Find the path to the model weights for the given flare class and time window.
    
    Args:
        flare_class: Flare class (e.g., "M5", "M", "C")
        time_window: Time window in hours (e.g., "24", "48", "72")
        version: Optional model version. If None, use the latest version.
        
    Returns:
        Path to model directory, or None if not found
    """
    # Check for models in trained_models directory (new structure)
    base_dir = os.path.join("models", "trained_models")
    
    if version:
        # If version is specified, look for that specific version
        model_dir = os.path.join(base_dir, f"EVEREST-v{version}-{flare_class}-{time_window}h")
        if os.path.exists(model_dir):
            return model_dir
    else:
        # Look for all matching directories and get the latest version
        import glob
        pattern = os.path.join(base_dir, f"EVEREST-v*-{flare_class}-{time_window}h")
        matching_dirs = glob.glob(pattern)
        
        if matching_dirs:
            # Extract version numbers
            versions = []
            for dir_path in matching_dirs:
                dir_name = os.path.basename(dir_path)
                parts = dir_name.split('-')
                if len(parts) >= 2 and parts[1].startswith('v'):
                    try:
                        version_str = parts[1][1:]  # Remove the 'v' prefix
                        version_num = float(version_str)
                        versions.append((version_num, dir_path))
                    except ValueError:
                        continue
            
            if versions:
                # Sort by version number and get the highest one
                versions.sort(reverse=True)
                return versions[0][1]  # Return the directory path
    
    # Also check old structure as fallback
    old_dir = os.path.join("models", "EVEREST", str(flare_class))
    if os.path.exists(old_dir):
        return old_dir
    
    return None

def create_uncertainty_plots(probabilities, uncertainties, true_labels, predicted_labels, 
                            flare_class, time_window, output_dir):
    """
    Create plots to visualize prediction uncertainties.
    
    Args:
        probabilities: Predicted probabilities for all classes
        uncertainties: Standard deviations from MC dropout
        true_labels: Ground truth labels
        predicted_labels: Predicted labels
        flare_class: Flare class
        time_window: Time window
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract positive class probability and uncertainty
    if probabilities.ndim > 1 and probabilities.shape[1] > 1:
        pos_probs = probabilities[:, 1]  # Probability of positive class
    else:
        pos_probs = probabilities  # Already single class probability
    
    if uncertainties.ndim > 1 and uncertainties.shape[1] > 1:
        pos_uncertainties = uncertainties[:, 1]  # Uncertainty of positive class
    else:
        pos_uncertainties = uncertainties  # Already single class uncertainty
    
    # Convert labels if needed
    if true_labels.ndim > 1:
        true_labels = np.argmax(true_labels, axis=1)
    if predicted_labels.ndim > 1:
        predicted_labels = np.argmax(predicted_labels, axis=1)
    
    # 1. Create confidence vs. uncertainty scatter plot
    plt.figure(figsize=(10, 8))
    correct = (true_labels == predicted_labels)
    
    # Plot incorrect predictions
    plt.scatter(pos_probs[~correct], pos_uncertainties[~correct], 
                color='red', alpha=0.6, label='Incorrect predictions')
    
    # Plot correct predictions
    plt.scatter(pos_probs[correct], pos_uncertainties[correct], 
                color='blue', alpha=0.3, label='Correct predictions')
    
    plt.xlabel('Predicted Probability (Positive Class)')
    plt.ylabel('Prediction Uncertainty (Std. Dev.)')
    plt.title(f'Prediction Confidence vs. Uncertainty ({flare_class}-class, {time_window}h window)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add density contours
    from scipy.stats import gaussian_kde
    try:
        # For correct predictions
        if sum(correct) > 10:  # Only if we have enough points
            xy_correct = np.vstack([pos_probs[correct], pos_uncertainties[correct]])
            z_correct = gaussian_kde(xy_correct)(xy_correct)
            idx_correct = z_correct.argsort()
            x_correct, y_correct, z_correct = xy_correct[0][idx_correct], xy_correct[1][idx_correct], z_correct[idx_correct]
            plt.scatter(x_correct, y_correct, c=z_correct, cmap='Blues', s=50, alpha=0.3, edgecolors='none')
        
        # For incorrect predictions
        if sum(~correct) > 10:  # Only if we have enough points
            xy_incorrect = np.vstack([pos_probs[~correct], pos_uncertainties[~correct]])
            z_incorrect = gaussian_kde(xy_incorrect)(xy_incorrect)
            idx_incorrect = z_incorrect.argsort()
            x_incorrect, y_incorrect, z_incorrect = xy_incorrect[0][idx_incorrect], xy_incorrect[1][idx_incorrect], z_incorrect[idx_incorrect]
            plt.scatter(x_incorrect, y_incorrect, c=z_incorrect, cmap='Reds', s=50, alpha=0.3, edgecolors='none')
    except Exception as e:
        print(f"Could not add density contours: {e}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"confidence_uncertainty_{flare_class}_{time_window}h.png"), dpi=300)
    
    # 2. Create uncertainty distribution plot
    plt.figure(figsize=(10, 6))
    
    # Plot uncertainty distributions for correct and incorrect predictions
    sns.histplot(pos_uncertainties[correct], color='blue', alpha=0.6, label='Correct predictions', kde=True, stat='density')
    sns.histplot(pos_uncertainties[~correct], color='red', alpha=0.6, label='Incorrect predictions', kde=True, stat='density')
    
    plt.xlabel('Prediction Uncertainty (Std. Dev.)')
    plt.ylabel('Density')
    plt.title(f'Uncertainty Distribution ({flare_class}-class, {time_window}h window)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"uncertainty_distribution_{flare_class}_{time_window}h.png"), dpi=300)
    
    # 3. Create ROC curve with uncertainty
    plt.figure(figsize=(10, 6))
    
    # Calculate ROC curve data
    roc_data = compute_roc_curve(pos_probs, true_labels)
    thresholds = roc_data['thresholds']
    tss_values = roc_data['tss']
    precision_values = roc_data['precision']
    recall_values = roc_data['recall']
    
    # Plot TSS curve
    plt.plot(thresholds, tss_values, 'b-', label='TSS')
    plt.plot(thresholds, precision_values, 'g-', label='Precision')
    plt.plot(thresholds, recall_values, 'r-', label='Recall')
    
    # Mark best threshold
    best_idx = np.argmax(tss_values)
    best_thr = thresholds[best_idx]
    best_tss = tss_values[best_idx]
    
    plt.axvline(x=best_thr, color='k', linestyle='--', alpha=0.7)
    plt.plot(best_thr, best_tss, 'ro', markersize=8)
    plt.annotate(f'Best TSS: {best_tss:.2f}\nat threshold: {best_thr:.2f}', 
                xy=(best_thr, best_tss), xytext=(best_thr+0.1, best_tss-0.1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    plt.xlabel('Threshold')
    plt.ylabel('Metric Value')
    plt.title(f'TSS, Precision, and Recall vs. Threshold ({flare_class}-class, {time_window}h window)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"tss_curve_{flare_class}_{time_window}h.png"), dpi=300)
    
    print(f"Saved uncertainty plots to {output_dir}")
    plt.close('all')

def test_everest_model(flare_class, time_window, version=None, mc_passes=20, 
                      batch_size=256, generate_plots=True, update_metadata=True):
    """
    Test an EVEREST model for the specified flare class and time window.
    
    Args:
        flare_class: Flare class (e.g., "M5", "M", "C")
        time_window: Time window in hours (e.g., "24", "48", "72")
        version: Optional model version
        mc_passes: Number of Monte Carlo passes for uncertainty estimation
        batch_size: Batch size for testing
        generate_plots: Whether to generate uncertainty plots
        update_metadata: Whether to update the model metadata with test results
        
    Returns:
        Dictionary of test metrics
    """
    print(f"\n{'='*80}")
    print(f"Testing EVEREST model for {flare_class}-class flares with {time_window}h window")
    print(f"{'='*80}")
    
    # Find the model path
    model_path = find_model_path(flare_class, time_window, version)
    if not model_path:
        print(f"No model found for {flare_class}-class flares with {time_window}h window")
        return None
    
    print(f"Found model at: {model_path}")
    
    try:
        # Load the testing data
        X_test, y_test_raw = get_testing_data(time_window, flare_class)
        
        # Handle different label formats
        if y_test_raw.dtype == np.int64 or y_test_raw.dtype == np.int32 or y_test_raw.dtype == np.float64 or y_test_raw.dtype == np.float32:
            y_test = y_test_raw.astype("int")
        else:
            y_test = np.array([1 if label == 'P' else 0 for label in y_test_raw]).astype("int")
        
        # If y_test is a single column, convert to one-hot
        if y_test.ndim == 1:
            y_test = tf.keras.utils.to_categorical(y_test, 2)
        
        print(f"Test data shape: X={X_test.shape}, y={y_test.shape}")
        print(f"Class distribution: {np.bincount(np.argmax(y_test, axis=1))}")
        
        # Load model and metadata
        model, metadata, model_version = load_model(model_path, version, flare_class, time_window)
        
        # Determine if this is an advanced model
        uses_evidential = metadata.get('hyperparameters', {}).get('uses_evidential', False)
        uses_evt = metadata.get('hyperparameters', {}).get('uses_evt', False)
        is_advanced = uses_evidential and uses_evt
        
        print(f"Testing with {'advanced' if is_advanced else 'standard'} model configuration")
        
        # Get threshold from metadata or use default
        threshold = metadata.get('performance', {}).get('val_best_thr', 0.5)
        print(f"Using threshold: {threshold:.2f}")
        
        # Perform Monte Carlo predictions
        print(f"Performing {mc_passes} Monte-Carlo passes for uncertainty estimation...")
        mean_preds, std_preds = model.mc_predict(X_test, n_passes=mc_passes, batch_size=batch_size)
        
        if is_advanced:
            # For advanced model, get softmax output for class 1
            probs = mean_preds[:, 1]
            uncertainties = std_preds[:, 1]
        else:
            # For standard model, get class 1 probability directly
            probs = mean_preds[:, 1]
            uncertainties = std_preds[:, 1]
        
        # Apply threshold for classification
        y_pred = (probs > threshold).astype(int)
        
        # Convert y_test to class indices if it's one-hot
        y_true = np.argmax(y_test, axis=1) if y_test.ndim > 1 else y_test
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        if cm.size < 4:  # Handle the case with only one class
            if cm.size == 1:
                if y_true[0] == 0:  # Only negative samples
                    tn, fp, fn, tp = cm[0, 0], 0, 0, 0
                else:  # Only positive samples
                    tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
            else:
                tn, fp, fn, tp = cm.ravel() + [0] * (4 - cm.size)
        else:
            tn, fp, fn, tp = cm.ravel()
        
        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        tss = recall + specificity - 1  # TSS = sensitivity + specificity - 1
        hss = 2 * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
        
        # Print results
        print(f"\nTest Results:")
        print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"TSS: {tss:.4f}")
        print(f"HSS: {hss:.4f}")
        print(f"Average uncertainty: {np.mean(uncertainties):.4f}")
        
        # Add metrics for advanced models
        if is_advanced and isinstance(model, EVEREST):
            try:
                # Get evidential parameters
                evidential_params = model.predict_evidential(X_test, batch_size=batch_size)
                mu, v, alpha, beta = np.hsplit(evidential_params, 4)
                
                # Calculate epistemic uncertainty
                epistemic = beta / v / (alpha - 1)
                
                # Calculate aleatoric uncertainty
                aleatoric = beta / (alpha - 1)
                
                print("\nEvidential uncertainty metrics:")
                print(f"  Mean epistemic uncertainty: {np.mean(epistemic):.4f}")
                print(f"  Mean aleatoric uncertainty: {np.mean(aleatoric):.4f}")
                
                # Get EVT parameters
                evt_params = model.predict_evt(X_test, batch_size=batch_size)
                xi, sigma = np.hsplit(evt_params, 2)
                
                print("\nEVT parameter statistics:")
                print(f"  Shape (ξ): mean={np.mean(xi):.4f}, std={np.std(xi):.4f}")
                print(f"  Scale (σ): mean={np.mean(sigma):.4f}, std={np.std(sigma):.4f}")
                
                # Add to metrics
                advanced_metrics = {
                    'ev_epistemic': float(np.mean(epistemic)),
                    'ev_aleatoric': float(np.mean(aleatoric)),
                    'ev_total': float(np.mean(epistemic + aleatoric)),
                    'evt_xi_mean': float(np.mean(xi)),
                    'evt_xi_std': float(np.std(xi)),
                    'evt_sigma_mean': float(np.mean(sigma)),
                    'evt_sigma_std': float(np.std(sigma))
                }
            except Exception as e:
                print(f"Error calculating advanced metrics: {e}")
                advanced_metrics = {}
        else:
            advanced_metrics = {}
        
        # Create metrics dictionary
        metrics = {
            'test_accuracy': float(accuracy),
            'test_precision': float(precision),
            'test_recall': float(recall),
            'test_f1': float(f1),
            'test_tss': float(tss),
            'test_hss': float(hss),
            'test_threshold': float(threshold),
            'test_tp': int(tp),
            'test_fp': int(fp),
            'test_tn': int(tn),
            'test_fn': int(fn),
            'test_uncertainty': float(np.mean(uncertainties)),
            'test_date': datetime.now().isoformat(),
            'mc_passes': mc_passes
        }
        
        # Add advanced metrics if available
        metrics.update(advanced_metrics)
        
        # Generate uncertainty plots
        if generate_plots:
            plots_dir = os.path.join(model_path, "test_plots")
            create_uncertainty_plots(
                mean_preds, std_preds, y_true, y_pred,
                flare_class, time_window, plots_dir
            )
        
        # Update model metadata
        if update_metadata:
            metadata_file = os.path.join(model_path, "metadata.json")
            if os.path.exists(metadata_file):
                try:
                    # Load existing metadata
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    # Add or update test results
                    if 'test_results' not in metadata:
                        metadata['test_results'] = {}
                    
                    # Use test date as key
                    test_key = datetime.now().strftime("%Y%m%d%H%M%S")
                    metadata['test_results'][test_key] = metrics
                    
                    # Add latest test results
                    metadata['latest_test'] = metrics
                    
                    # Write updated metadata
                    with open(metadata_file, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    print(f"Updated metadata with test results at {metadata_file}")
                except Exception as e:
                    print(f"Error updating metadata: {e}")
            
        # Store metrics in the global dictionary
        if str(time_window) not in all_metrics:
            all_metrics[str(time_window)] = {}
        all_metrics[str(time_window)][flare_class] = metrics
        
        return metrics
    
    except Exception as e:
        import traceback
        print(f"Error testing model: {e}")
        traceback.print_exc()
        return None

def run_all_tests(flare_classes=None, time_windows=None, version=None, 
                 mc_passes=20, batch_size=256, generate_plots=True, update_metadata=True):
    """
    Run tests for all specified flare classes and time windows.
    
    Args:
        flare_classes: List of flare classes to test. If None, test all.
        time_windows: List of time windows to test. If None, test all.
        version: Optional model version
        mc_passes: Number of Monte Carlo passes for uncertainty estimation
        batch_size: Batch size for testing
        generate_plots: Whether to generate uncertainty plots
        update_metadata: Whether to update the model metadata with test results
        
    Returns:
        Dictionary of all test metrics
    """
    # Use default values if not specified
    if flare_classes is None:
        flare_classes = FLARE_CLASSES
    if time_windows is None:
        time_windows = TIME_WINDOWS
    
    all_results = {}
    
    # Test each combination
    for time_window in time_windows:
        if time_window not in all_results:
            all_results[time_window] = {}
        
        for flare_class in flare_classes:
            print(f"\nTesting {flare_class}-class flares with {time_window}h window...")
            
            # Skip unsupported flare classes
            if flare_class not in supported_flare_class:
                print(f"Skipping unsupported flare class: {flare_class}")
                continue
            
            # Run the test
            metrics = test_everest_model(
                flare_class, time_window, 
                version=version,
                mc_passes=mc_passes,
                batch_size=batch_size,
                generate_plots=generate_plots,
                update_metadata=update_metadata
            )
            
            if metrics:
                all_results[time_window][flare_class] = metrics
    
    # Save the combined results
    save_results(all_results)
    
    return all_results

def save_results(results):
    """
    Save the combined test results to a file.
    
    Args:
        results: Dictionary of test results
    """
    # Create results directory if it doesn't exist
    results_dir = os.path.join("models", "test_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save as JSON
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    results_file = os.path.join(results_dir, f"everest_test_results_{timestamp}.json")
    
    try:
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nSaved combined test results to {results_file}")
    except Exception as e:
        print(f"Error saving results: {e}")

def print_results_table(results):
    """
    Print a formatted table of test results.
    
    Args:
        results: Dictionary of test results
    """
    print("\n" + "="*80)
    print("EVEREST TEST RESULTS SUMMARY")
    print("="*80)
    
    # Header
    print(f"{'Time Window':<12} {'Flare Class':<10} {'TSS':<8} {'F1':<8} {'Precision':<10} {'Recall':<8} {'Accuracy':<10}")
    print("-"*80)
    
    # Sort by time window and flare class
    for time_window in sorted(results.keys()):
        for flare_class in sorted(results[time_window].keys()):
            metrics = results[time_window][flare_class]
            
            # Get metrics or display N/A if not available
            tss = f"{metrics.get('test_tss', 'N/A'):.4f}" if isinstance(metrics.get('test_tss'), (int, float)) else "N/A"
            f1 = f"{metrics.get('test_f1', 'N/A'):.4f}" if isinstance(metrics.get('test_f1'), (int, float)) else "N/A"
            precision = f"{metrics.get('test_precision', 'N/A'):.4f}" if isinstance(metrics.get('test_precision'), (int, float)) else "N/A"
            recall = f"{metrics.get('test_recall', 'N/A'):.4f}" if isinstance(metrics.get('test_recall'), (int, float)) else "N/A"
            accuracy = f"{metrics.get('test_accuracy', 'N/A'):.4f}" if isinstance(metrics.get('test_accuracy'), (int, float)) else "N/A"
            
            print(f"{time_window:<12} {flare_class:<10} {tss:<8} {f1:<8} {precision:<10} {recall:<8} {accuracy:<10}")
    
    print("="*80)

def main():
    """Main function to run tests on EVEREST models"""
    parser = argparse.ArgumentParser(description="Test EVEREST models for solar flare prediction")
    parser.add_argument("--flare", "-f", type=str, help="Specific flare class to test (default: all)")
    parser.add_argument("--window", "-w", type=str, help="Specific time window to test (default: all)")
    parser.add_argument("--version", "-v", type=str, help="Model version to test (default: latest)")
    parser.add_argument("--mc-passes", "-m", type=int, default=20, help="Number of Monte Carlo passes (default: 20)")
    parser.add_argument("--batch-size", "-b", type=int, default=256, help="Batch size for testing (default: 256)")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")
    parser.add_argument("--no-update", action="store_true", help="Skip updating metadata")
    
    args = parser.parse_args()
    
    # Set up parameters
    flare_classes = [args.flare] if args.flare else FLARE_CLASSES
    time_windows = [args.window] if args.window else TIME_WINDOWS
    generate_plots = not args.no_plots
    update_metadata = not args.no_update
    
    # Run tests
    print(f"Running tests for EVEREST models...")
    print(f"Flare classes: {', '.join(flare_classes)}")
    print(f"Time windows: {', '.join(time_windows)}")
    print(f"MC passes: {args.mc_passes}")
    print(f"Generating plots: {generate_plots}")
    print(f"Updating metadata: {update_metadata}")
    
    # Run the tests
    results = run_all_tests(
        flare_classes=flare_classes,
        time_windows=time_windows,
        version=args.version,
        mc_passes=args.mc_passes,
        batch_size=args.batch_size,
        generate_plots=generate_plots,
        update_metadata=update_metadata
    )
    
    # Print results table
    print_results_table(results)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 