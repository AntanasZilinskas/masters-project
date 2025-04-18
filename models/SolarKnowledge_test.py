'''
 author: Antanas Zilinskas
 Based on work by Yasser Abduallah
 
 This script tests the SolarKnowledge model by loading pre-trained weights
 and evaluating on test datasets for various time windows and flare classes.
 It integrates with the model tracking system to record comprehensive results.
'''

import warnings
warnings.filterwarnings("ignore")
import os
import argparse
import numpy as np
import json
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score
)

# Import utility functions and our model
from utils import get_testing_data, log, supported_flare_class
from SolarKnowledge_model import SolarKnowledge
from model_tracking import (
    load_model_metadata,
    plot_confusion_matrix,
    compare_models,
    get_latest_version,
    list_available_models
)

def test_model(version, time_window, flare_class, update_metadata=True, use_latest=False):
    """
    Test a specific model version and record metrics.
    
    Args:
        version: Model version string
        time_window: Time window (24, 48, or 72)
        flare_class: Flare class (C, M, or M5)
        update_metadata: Whether to update the model's metadata with test results
        use_latest: Whether to use the latest available version for this flare class and time window
    
    Returns:
        Dictionary of performance metrics
    """
    # Find the latest version if requested
    if use_latest:
        latest_version = get_latest_version(flare_class, time_window)
        if latest_version:
            version = latest_version
            log(f"Using latest version v{version} for {flare_class}-class flares with {time_window}h window", verbose=True)
        else:
            log(f"No models found for {flare_class}-class flares with {time_window}h window", verbose=True)
            return {
                "version": "N/A",
                "flare_class": flare_class,
                "time_window": time_window,
                "error": "No models found"
            }
    
    log(f"Testing model v{version} for {flare_class}-class flares with {time_window}h window", verbose=True)
    
    # Load the testing data
    X_test, y_test = get_testing_data(time_window, flare_class)
    
    # Convert y_test to a NumPy array and get class indices
    y_test = np.array(y_test)
    if y_test.ndim > 1:
        y_true = np.argmax(y_test, axis=-1)
    else:
        y_true = y_test
    
    input_shape = (X_test.shape[1], X_test.shape[2])
    
    # Define model directory
    model_dir = f"models/SolarKnowledge-v{version}-{flare_class}-{time_window}h"
    
    # Check if model exists
    if not os.path.exists(model_dir):
        log(f"⚠️ Model directory not found: {model_dir}", verbose=True)
        return {
            "version": version,
            "flare_class": flare_class,
            "time_window": time_window,
            "error": "Model not found"
        }
    
    # Build and compile the model
    model = SolarKnowledge(early_stopping_patience=5)
    model.build_base_model(input_shape)
    model.compile()
    
    # Load weights - the model directory structure is different with the tracking system
    weights_path = os.path.join(model_dir, "model_weights.weights.h5")
    if not os.path.exists(weights_path):
        log(f"⚠️ Model weights not found: {weights_path}", verbose=True)
        return {
            "version": version,
            "flare_class": flare_class,
            "time_window": time_window,
            "error": "Weights not found"
        }
    
    log(f"Loading weights from: {weights_path}", verbose=True)
    model.model.load_weights(weights_path)
    
    # Make predictions
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=-1)
    
    # Calculate performance metrics
    metrics = {
        "accuracy": float(accuracy_score(y_true, predicted_classes)),
        "precision": float(precision_score(y_true, predicted_classes)),
        "recall": float(recall_score(y_true, predicted_classes)),
        "f1_score": float(f1_score(y_true, predicted_classes)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted_classes))
    }
    
    # Calculate TSS (True Skill Statistic)
    cm = confusion_matrix(y_true, predicted_classes)
    sensitivity = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
    specificity = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
    metrics["TSS"] = float(sensitivity + specificity - 1)
    
    # Print results
    log(f"Test results for v{version}-{flare_class}-{time_window}h:", verbose=True)
    for metric, value in metrics.items():
        log(f"  {metric}: {value:.4f}", verbose=True)
    
    # Print detailed classification report
    log("\nDetailed Classification Report:", verbose=True)
    log(classification_report(y_true, predicted_classes), verbose=True)
    
    # Generate and save confusion matrix
    plot_confusion_matrix(y_true, predicted_classes, model_dir, normalize=True)
    log(f"Confusion matrix saved to {model_dir}/confusion_matrix.png", verbose=True)
    
    # Update model metadata with test results if requested
    if update_metadata:
        try:
            # Load existing metadata
            metadata_path = os.path.join(model_dir, "metadata.json")
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            # Update with test results
            metadata["test_results"] = metrics
            
            # Save updated metadata
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
                
            log(f"✅ Updated metadata with test results", verbose=True)
        except Exception as e:
            log(f"⚠️ Error updating metadata: {str(e)}", verbose=True)
    
    # Return all metrics
    return {
        "version": version,
        "flare_class": flare_class,
        "time_window": time_window,
        **metrics
    }

def test_all_models(version=None, update_metadata=True, use_latest=False):
    """
    Test all models of a specific version or the latest version for each configuration.
    
    Args:
        version: Model version to test (if None and use_latest is True, will use latest for each config)
        update_metadata: Whether to update model metadata with test results
        use_latest: Whether to use the latest version for each flare class and time window
    
    Returns:
        Dictionary of all test results
    """
    all_results = {}
    versions_used = set()
    
    for time_window in ["24", "48", "72"]:
        if time_window not in all_results:
            all_results[time_window] = {}
            
        for flare_class in ["C", "M", "M5"]:
            if flare_class not in supported_flare_class:
                continue
            
            current_version = version
            if use_latest or version is None:
                latest_version = get_latest_version(flare_class, time_window)
                if latest_version:
                    current_version = latest_version
                    log(f"Using latest version v{current_version} for {flare_class}-class, {time_window}h window", verbose=True)
                else:
                    log(f"No models found for {flare_class}-class, {time_window}h window", verbose=True)
                    continue
            
            log(f"Testing model for time window {time_window} and flare class {flare_class}", verbose=True)
            results = test_model(current_version, time_window, flare_class, update_metadata)
            all_results[time_window][flare_class] = results
            versions_used.add(current_version)
            log("===========================================================\n", verbose=True)
    
    # Save overall results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if version:
        output_file = f"test_results_v{version}_{timestamp}.json"
    else:
        output_file = f"test_results_latest_{timestamp}.json"
    
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=4)
    log(f"✅ Saved all test results to {output_file}", verbose=True)
    
    return all_results, list(versions_used)

def list_and_compare_models():
    """List all available models and compare them."""
    models = list_available_models()
    
    if not models:
        log("No models found.", verbose=True)
        return
    
    # Group by flare class and time window
    grouped = {}
    for model in models:
        key = f"{model['flare_class']}-{model['time_window']}"
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(model)
    
    # Print grouped models
    log("\nAvailable Models:", verbose=True)
    for key, models_list in grouped.items():
        flare_class, time_window = key.split("-")
        log(f"\n{flare_class}-class flares with {time_window}h window:", verbose=True)
        
        # Sort by version
        models_list.sort(key=lambda x: [int(p) for p in x["version"].split(".")])
        
        for model in models_list:
            accuracy = model["accuracy"]
            if isinstance(accuracy, float):
                accuracy = f"{accuracy:.4f}"
            log(f"  v{model['version']} - Created: {model['timestamp'][:10]} - Accuracy: {accuracy}", verbose=True)
    
    # Collect unique versions and generate comparison
    versions = list(set(model["version"] for model in models))
    flare_classes = list(set(model["flare_class"] for model in models))
    time_windows = list(set(model["time_window"] for model in models))
    
    if versions:
        log("\nModel Comparison Table:", verbose=True)
        comparison = compare_models(versions, flare_classes, time_windows)
        log(comparison, verbose=True)

if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser(description="Test SolarKnowledge models")
    parser.add_argument("--version", "-v", type=str, help="Model version to test")
    parser.add_argument("--flare-class", "-f", type=str, help="Specific flare class to test (C, M, or M5)")
    parser.add_argument("--time-window", "-w", type=str, help="Specific time window to test (24, 48, or 72)")
    parser.add_argument("--no-update", action="store_true", help="Don't update metadata with test results")
    parser.add_argument("--latest", "-l", action="store_true", help="Use latest version for each configuration")
    parser.add_argument("--list", action="store_true", help="List all available models and compare them")
    args = parser.parse_args()
    
    # Import datetime here to avoid conflict with earlier import
    from datetime import datetime
    
    if args.list:
        # Just list and compare models without testing
        list_and_compare_models()
    elif args.flare_class and args.time_window:
        # Test specific model
        test_model(
            args.version, 
            args.time_window, 
            args.flare_class, 
            update_metadata=not args.no_update,
            use_latest=args.latest
        )
    else:
        # Test all models
        all_results, versions_used = test_all_models(
            version=args.version, 
            update_metadata=not args.no_update,
            use_latest=args.latest
        )
        
        # Generate comparison table
        log("\nModel Comparison:", verbose=True)
        comparison = compare_models(
            versions_used,
            ["C", "M", "M5"], 
            ["24", "48", "72"]
        )
        log(comparison, verbose=True) 