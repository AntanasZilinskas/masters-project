"""
This script runs all testing processes for the multimodal SolarKnowledge model
combining SHARP parameters and SDO/HMI magnetogram images.

Author: Antanas Zilinskas
"""

import json
import os
import warnings

import numpy as np
from multimodal_utils import get_multimodal_testing_data
from MultimodalSolarKnowledge_model import MultimodalSolarKnowledge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)

from utils import log, supported_flare_class

warnings.filterwarnings("ignore")

# Import utility functions and configuration

# Dictionary to hold performance metrics
all_metrics = {}


def test_model(time_window, flare_class):
    log(
        "Multimodal testing initiated for time window: "
        + str(time_window)
        + " and flare class: "
        + flare_class,
        verbose=True,
    )

    # Load the testing data
    X_time_series, X_images, y_test = get_multimodal_testing_data(
        time_window, flare_class
    )

    # Convert y_test to class indices if one-hot encoded
    if y_test.ndim > 1:
        y_true = np.argmax(y_test, axis=-1)
    else:
        y_true = y_test

    # Define input shapes
    time_series_shape = (X_time_series.shape[1], X_time_series.shape[2])
    image_shape = X_images.shape[1:]

    # Build and compile the model
    model = MultimodalSolarKnowledge(
        embed_dim=128,
        num_heads=8,
        ff_dim=256,
        num_transformer_blocks=4,
        dropout_rate=0.1,
        early_stopping_patience=5,
    )
    model.build_base_model(time_series_shape, image_shape)
    model.compile()

    # Define the weights directory
    weight_dir = os.path.join(
        "models", "multimodal", str(time_window), flare_class
    )
    if not os.path.exists(weight_dir):
        print(
            f"Warning: Model weights directory: {weight_dir} does not exist! Skipping test for time window {time_window} and flare class {flare_class}."
        )
        # Record placeholders for missing models
        time_key = str(time_window)
        if time_key not in all_metrics:
            all_metrics[time_key] = {}
        all_metrics[time_key][flare_class] = {
            "accuracy": "N/A",
            "precision": "N/A",
            "recall": "N/A",
            "balanced_accuracy": "N/A",
            "TSS": "N/A",
        }
        return

    print("Loading weights from model dir:", weight_dir)
    model.load_weights(flare_class=flare_class, w_dir=weight_dir, verbose=True)

    # Run predictions
    predictions = model.predict(X_time_series, X_images)
    predicted_classes = np.argmax(predictions, axis=-1)

    # Calculate metrics
    acc = accuracy_score(y_true, predicted_classes)
    prec = precision_score(y_true, predicted_classes)
    rec = recall_score(y_true, predicted_classes)
    bal_acc = balanced_accuracy_score(y_true, predicted_classes)

    # Compute TSS
    cm = confusion_matrix(y_true, predicted_classes)
    sensitivity = (
        cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
    )
    specificity = (
        cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
    )
    TSS = sensitivity + specificity - 1

    print("==============================================")
    print(
        f"Multimodal accuracy for flare class {flare_class} with time window {time_window}: {acc:.4f}"
    )
    print("Classification Report:")
    print(classification_report(y_true, predicted_classes))
    print("Confusion Matrix:")
    print(cm)
    print(f"TSS: {TSS:.4f}")
    print("==============================================\n\n")

    # Store metrics
    time_key = str(time_window)
    if time_key not in all_metrics:
        all_metrics[time_key] = {}
    all_metrics[time_key][flare_class] = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "TSS": round(TSS, 4),
    }


if __name__ == "__main__":
    # Loop over the desired time windows and flare classes
    for time_window in [24, 48, 72]:
        for flare_class in ["C", "M", "M5"]:
            if flare_class not in supported_flare_class:
                print(
                    "Unsupported flare class:",
                    flare_class,
                    "It must be one of:",
                    ", ".join(supported_flare_class),
                )
                continue
            test_model(str(time_window), flare_class)
            log(
                "===========================================================\n\n",
                verbose=True,
            )

    # Save the metrics to a JSON file
    output_file = "multimodal_results.json"
    with open(output_file, "w") as f:
        json.dump(all_metrics, f, indent=4)
    print(f"Saved multimodal test metrics into {output_file}")
