#!/usr/bin/env python3
'''
 (c) Copyright 2023
 All rights reserved
 Programs written by Yasser Abduallah
 Department of Computer Science
 New Jersey Institute of Technology
 University Heights, Newark, NJ 07102, USA

 This script tests the transformer-based SolarKnowledge model by loading pre-trained
 weights and evaluating on the test datasets for various time windows and flare classes.
 It prints the accuracy and classification report for every combination.
 Additionally, it computes key metrics and writes out a JSON file that stores these results
 for each time window so that they can be automatically integrated into the final report.
 @author: Yasser Abduallah
'''

import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import json
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, balanced_accuracy_score, confusion_matrix

# Import utility functions and configuration from your project
from utils import get_testing_data, log, supported_flare_class
from SolarKnowledge_model import SolarKnowledge

# This dictionary will hold your "This work" performance for each time window.
# Structure: { "24": { "C": {metrics}, "M": {metrics}, "M5": {metrics} },
#              "48": { ... },
#              "72": { ... } }
all_metrics = {}

def test_model(time_window, flare_class):
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
    
    # Define the weights directory
    weight_dir = os.path.join("models", str(time_window), flare_class)
    if not os.path.exists(weight_dir):
        print(f"Warning: Model weights directory: {weight_dir} does not exist! Skipping test for time window {time_window} and flare class {flare_class}.")
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
    model.load_weights(flare_class=flare_class, w_dir=weight_dir, verbose=True)
    
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
    
    # Store metrics for the given time window and flare class.
    time_key = str(time_window)
    if time_key not in all_metrics:
        all_metrics[time_key] = {}
    all_metrics[time_key][flare_class] = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "TSS": round(TSS, 4)
    }

if __name__ == '__main__':
    # Loop over the desired time windows and flare classes.
    for time_window in [24, 48, 72]:
        for flare_class in ['C', 'M', 'M5']:
            if flare_class not in supported_flare_class:
                print("Unsupported flare class:", flare_class, "It must be one of:", ", ".join(supported_flare_class))
                continue
            test_model(str(time_window), flare_class)
            log("===========================================================\n\n", verbose=True)
    
    # Save the metrics for all time windows into a JSON file.
    output_file = "this_work_results.json"
    with open(output_file, "w") as f:
        json.dump(all_metrics, f, indent=4)
    print(f"Saved test metrics for 'This work' into {output_file}") 