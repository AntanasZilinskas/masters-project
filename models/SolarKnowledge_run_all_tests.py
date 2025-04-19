'''
 author: Antanas Zilinskas
 
 This script tests SolarKnowledge models for flare class: C, M, M5 and time window: 24, 48, 72.
 
 Usage:
   python SolarKnowledge_run_all_tests.py --latest  # Test the latest version of each model
   python SolarKnowledge_run_all_tests.py --version 1.3  # Test a specific version
 
 The script will save results to this_work_results.json and a version-specific file.
'''

import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import json
import argparse
import glob
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, balanced_accuracy_score, confusion_matrix

# Import utility functions and configuration from your project
from utils import get_testing_data, log, supported_flare_class
from SolarKnowledge_model import SolarKnowledge

# This dictionary will hold your "This work" performance for each time window.
# Structure: { "24": { "C": {metrics}, "M": {metrics}, "M5": {metrics} },
#              "48": { ... },
#              "72": { ... } }
all_metrics = {}

def find_latest_model_version(flare_class, time_window):
    """Find the latest model version for a specific flare class and time window"""
    # Check if we're in the models directory already or at project root
    if os.path.exists("models/models"):
        models_dir = "models/models"
    elif os.path.exists("models"):
        models_dir = "models"
    else:
        log(f"Cannot find models directory", verbose=True)
        return None
        
    # Pattern for model directories: SolarKnowledge-v{version}-{flare_class}-{time_window}h
    pattern = f"SolarKnowledge-v*-{flare_class}-{time_window}h"
    full_pattern = os.path.join(models_dir, pattern)
    matching_dirs = glob.glob(full_pattern)
    
    if not matching_dirs:
        return None
    
    # Extract version numbers and find the highest one
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
    
    if not versions:
        return None
    
    # Sort by version number and get the highest one
    versions.sort(reverse=True)
    return versions[0][1]  # Return the directory path

def test_model(time_window, flare_class, timestamp=None, use_latest=False):
    log("Testing initiated for time window: " + str(time_window) + " and flare class: " + flare_class, verbose=True)
    
    # Load the testing data using your project's utility function
    X_test, y_test = get_testing_data(time_window, flare_class)
    # Convert y_test to a NumPy array so we can check its dimensions
    y_test = np.array(y_test)
    # If your test labels are one-hot encoded, convert them to class indices.
    if y_test.ndim > 1:
        y_true = np.argmax(y_test, axis=-1)
    else:
        y_true = y_test
    
    input_shape = (X_test.shape[1], X_test.shape[2])
    
    # Build and compile the model
    model = SolarKnowledge(early_stopping_patience=5)
    model.build_base_model(input_shape)
    model.compile()
    
    # Find the latest model version in the correct directory structure
    latest_model_dir = find_latest_model_version(flare_class, time_window)
    
    if latest_model_dir and os.path.exists(latest_model_dir):
        weight_dir = latest_model_dir
        log(f"Using model at: {weight_dir}", verbose=True)
    else:
        log(f"No model found for flare class {flare_class} with time window {time_window}", verbose=True)
        # Ensure we record placeholders for missing models.
        time_key = str(time_window)
        if time_key not in all_metrics:
            all_metrics[time_key] = {}
        all_metrics[time_key][flare_class] = {
            "accuracy": "N/A",
            "precision": "N/A",
            "recall": "N/A",
            "balanced_accuracy": "N/A",
            "TSS": "N/A"
        }
        return

    print("Loading weights from model dir:", weight_dir)
    # Try to load weights from the model directory
    weight_file = os.path.join(weight_dir, 'model_weights.weights.h5')
    if not os.path.exists(weight_file):
        log(f"Error: Weight file not found at {weight_file}", verbose=True)
        return

    try:
        status = model.model.load_weights(weight_file)
        if status is not None:
            status.expect_partial()
    except Exception as e:
        log(f"Error loading weights: {str(e)}", verbose=True)
        return
    
    # Run predictions on test data
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=-1)
    
    # Calculate basic metrics:
    acc = accuracy_score(y_true, predicted_classes)
    prec = precision_score(y_true, predicted_classes)
    rec = recall_score(y_true, predicted_classes)
    bal_acc = balanced_accuracy_score(y_true, predicted_classes)
    # Compute TSS: sensitivity + specificity - 1. For binary classification,
    # sensitivity = recall for class 1, specificity = recall for class 0.
    cm = confusion_matrix(y_true, predicted_classes)
    sensitivity = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
    specificity = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
    TSS = sensitivity + specificity - 1

    print("==============================================")
    print(f"Accuracy for flare class {flare_class} with time window {time_window}: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_true, predicted_classes))
    print("==============================================\n\n")
    
    # Store metrics for the given time window and flare class
    metrics_dict = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "TSS": round(TSS, 4)
    }
    
    # Get confusion matrix as a list for easier JSON serialization
    metrics_dict["confusion_matrix"] = cm.tolist()
    
    # Add test data size information
    metrics_dict["test_samples"] = len(y_true)
    metrics_dict["positive_samples"] = int(np.sum(y_true))
    metrics_dict["negative_samples"] = len(y_true) - int(np.sum(y_true))
    
    # Get model version from directory name
    model_name = os.path.basename(weight_dir)
    version = "unknown"
    if "-v" in model_name:
        version = model_name.split("-v")[1].split("-")[0]
    
    # Add version and test date to metrics
    from datetime import datetime
    metrics_dict["version"] = version
    metrics_dict["test_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Store in consolidated metrics for all models
    time_key = str(time_window)
    if time_key not in all_metrics:
        all_metrics[time_key] = {}
    all_metrics[time_key][flare_class] = metrics_dict
    
    # Update model's metadata file with test results
    metadata_file = os.path.join(weight_dir, 'metadata.json')
    if os.path.exists(metadata_file):
        try:
            # Load existing metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                
            # Add or update test results
            if 'test_results' not in metadata:
                metadata['test_results'] = {}
            
            # Use test date as a key to allow multiple test runs
            test_key = datetime.now().strftime("%Y%m%d%H%M%S")
            metadata['test_results'][test_key] = metrics_dict
            
            # Add latest test results at the top level for easy access
            metadata['latest_test'] = metrics_dict
            
            # Write updated metadata back to the file
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=4)
                
            log(f"Updated metadata file at {metadata_file} with test results", verbose=True)
        except Exception as e:
            log(f"Error updating metadata file: {str(e)}", verbose=True)
    else:
        log(f"No metadata file found at {metadata_file}", verbose=True)
    
    return metrics_dict

if __name__ == '__main__':
    # Add command line argument for timestamp
    parser = argparse.ArgumentParser(description="Test SolarKnowledge models for solar flare prediction")
    parser.add_argument("--timestamp", "-t", help="Specific model timestamp to test")
    parser.add_argument("--latest", action="store_true", help="Test the latest model version")
    parser.add_argument("--version", "-v", help="Test a specific model version")
    args = parser.parse_args()
    
    # Can't use multiple selection methods together
    if sum([bool(args.timestamp), args.latest, bool(args.version)]) > 1:
        print("Error: Cannot use multiple model selection options together. Use only one of: --timestamp, --latest, --version")
        exit(1)
    
    # Loop over the desired time windows and flare classes
    for time_window in [24, 48, 72]:
        for flare_class in ['C', 'M', 'M5']:
            if flare_class not in supported_flare_class:
                print("Unsupported flare class:", flare_class, "It must be one of:", ", ".join(supported_flare_class))
                continue
            test_model(str(time_window), flare_class, args.timestamp, args.latest)
            log("===========================================================\n\n", verbose=True)
    
    # Save the metrics for all time windows into a JSON file
    output_file = "this_work_results.json"
    with open(output_file, "w") as f:
        json.dump(all_metrics, f, indent=4)
    print(f"Saved test metrics for 'This work' into {output_file}")
    
    # If a specific timestamp was tested, also save results to a timestamped file
    if args.timestamp:
        timestamped_output = f"results_{args.timestamp}.json"
        with open(timestamped_output, "w") as f:
            json.dump(all_metrics, f, indent=4)
        print(f"Saved test metrics for timestamp {args.timestamp} into {timestamped_output}")
    elif args.latest:
        latest_output = "results_latest.json"
        with open(latest_output, "w") as f:
            json.dump(all_metrics, f, indent=4)
        print(f"Saved test metrics for latest version into {latest_output}")
    elif args.version:
        version_output = f"results_v{args.version}.json"
        with open(version_output, "w") as f:
            json.dump(all_metrics, f, indent=4)
        print(f"Saved test metrics for version v{args.version} into {version_output}") 