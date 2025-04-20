"""
Model Tracking and Management for SolarKnowledge

This module provides tools for tracking model versions, saving metadata,
generating model cards, and maintaining a structured model directory.
"""

import csv
import json
import os
import shutil
import subprocess
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def create_model_dir(version, flare_class, time_window):
    """Create a standardized model directory structure."""
    model_dir = (
        f"models/SolarKnowledge-v{version}-{flare_class}-{time_window}h"
    )
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


def get_git_info():
    """Get the current git commit hash and branch if available."""
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode()
            .strip()
        )
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"]
            )
            .decode()
            .strip()
        )
        return {"commit": commit, "branch": branch}
    except (subprocess.SubprocessError, FileNotFoundError):
        return {"commit": "unknown", "branch": "unknown"}


def save_model_with_metadata(
    model,
    metrics,
    hyperparams,
    history,
    version,
    flare_class,
    time_window,
    description=None,
):
    """
    Save a model with comprehensive metadata tracking.

    Args:
        model: The trained SolarKnowledge model instance
        metrics: Dictionary of evaluation metrics
        hyperparams: Dictionary of hyperparameters used for training
        history: Training history from model.fit
        version: Model version string (e.g., "1.0")
        flare_class: Flare class target
        time_window: Time window used
        description: Optional model description
    """
    # Create model directory
    model_dir = create_model_dir(version, flare_class, time_window)

    # Save model weights
    weights_path = os.path.join(model_dir, "model_weights.weights.h5")
    model.model.save_weights(weights_path)

    # Extract architecture details
    architecture = {
        "name": "SolarKnowledge",
        "input_shape": model.model.input_shape[1:],
        "num_params": model.model.count_params(),
        "precision": str(model.model.dtype_policy)
        if hasattr(model.model, "dtype_policy")
        else "float32",
    }

    # Create metadata
    metadata = {
        "version": version,
        "timestamp": datetime.now().isoformat(),
        "description": description
        or f"SolarKnowledge model for {flare_class}-class flares with {time_window}h prediction window",
        "flare_class": flare_class,
        "time_window": time_window,
        "hyperparameters": hyperparams,
        "performance": metrics,
        "git_info": get_git_info(),
        "architecture": architecture,
    }

    # Save metadata
    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Save training history
    if history and hasattr(history, "history"):
        save_training_history(history.history, model_dir)

    # Generate model card
    generate_model_card(metadata, metrics, model_dir)

    print(f"Model saved to {model_dir}")
    return model_dir


def save_training_history(history, model_dir):
    """Save training history as CSV and generate learning curves plot."""
    # Save as CSV
    with open(
        os.path.join(model_dir, "training_history.csv"), "w", newline=""
    ) as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(["epoch"] + list(history.keys()))
        # Write data
        for epoch in range(len(next(iter(history.values())))):
            writer.writerow(
                [epoch] + [history[k][epoch] for k in history.keys()]
            )

    # Generate plot
    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    if "loss" in history:
        plt.plot(history["loss"], label="Training Loss")
    if "val_loss" in history:
        plt.plot(history["val_loss"], label="Validation Loss")
    plt.title("Loss During Training")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    if "accuracy" in history:
        plt.plot(history["accuracy"], label="Training Accuracy")
    if "val_accuracy" in history:
        plt.plot(history["val_accuracy"], label="Validation Accuracy")
    plt.title("Accuracy During Training")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "learning_curves.png"))
    plt.close()


def generate_model_card(metadata, metrics, model_dir):
    """Generate a model card markdown document."""
    timestamp = datetime.fromisoformat(metadata["timestamp"]).strftime(
        "%Y-%m-%d %H:%M"
    )

    # Format metrics for display
    metrics_text = ""
    for metric, value in metrics.items():
        if isinstance(value, float):
            metrics_text += f"- **{metric}**: {value:.4f}\n"
        else:
            metrics_text += f"- **{metric}**: {value}\n"

    model_card = f"""# SolarKnowledge Model v{metadata['version']}

## Overview
- **Created**: {timestamp}
- **Type**: Solar flare prediction model
- **Target**: {metadata['flare_class']}-class flares
- **Time Window**: {metadata['time_window']} hours

## Description
{metadata['description']}

## Performance Metrics
{metrics_text}

## Training Details
- **Architecture**: SolarKnowledge Transformer Model
- **Parameters**: {metadata['architecture']['num_params']:,}
- **Precision**: {metadata['architecture']['precision']}

## Hyperparameters
"""

    # Add hyperparameters
    for param, value in metadata["hyperparameters"].items():
        model_card += f"- **{param}**: {value}\n"

    # Add git info
    model_card += f"""
## Version Control
- **Git Commit**: {metadata['git_info']['commit']}
- **Git Branch**: {metadata['git_info']['branch']}

## Usage
```python
from SolarKnowledge_model import SolarKnowledge

# Load the model
model = SolarKnowledge()
model.load_model(
    input_shape={metadata['architecture']['input_shape']},
    flare_class="{metadata['flare_class']}",
    w_dir="{os.path.basename(model_dir)}"
)

# Make predictions
predictions = model.predict(X_test)
```
"""

    # Write model card to file
    with open(os.path.join(model_dir, "model_card.md"), "w") as f:
        f.write(model_card)


def load_model_metadata(version, flare_class, time_window):
    """Load metadata for a specific model version."""
    model_dir = create_model_dir(version, flare_class, time_window)
    metadata_path = os.path.join(model_dir, "metadata.json")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"No metadata found for model v{version}-{flare_class}-{time_window}h"
        )

    with open(metadata_path, "r") as f:
        return json.load(f)


def list_available_models():
    """List all available models in the models directory."""
    try:
        model_dirs = [
            d for d in os.listdir("models") if d.startswith("SolarKnowledge-v")
        ]
        models = []

        for d in model_dirs:
            parts = d.split("-")
            if len(parts) >= 3:
                version = parts[1][1:]  # Remove 'v' prefix
                flare_class = parts[2]
                time_window = parts[3][:-1]  # Remove 'h' suffix

                metadata_path = os.path.join("models", d, "metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)

                    models.append(
                        {
                            "version": version,
                            "flare_class": flare_class,
                            "time_window": time_window,
                            "timestamp": metadata.get("timestamp", "unknown"),
                            "accuracy": metadata.get("performance", {}).get(
                                "accuracy", "unknown"
                            ),
                        }
                    )
                else:
                    models.append(
                        {
                            "version": version,
                            "flare_class": flare_class,
                            "time_window": time_window,
                            "timestamp": "unknown",
                            "accuracy": "unknown",
                        }
                    )

        return sorted(
            models,
            key=lambda x: (x["flare_class"], x["time_window"], x["version"]),
        )
    except FileNotFoundError:
        return []


def plot_confusion_matrix(y_true, y_pred, model_dir, normalize=False):
    """
    Generate and save a confusion matrix visualization.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        model_dir: Directory to save the plot
        normalize: Whether to normalize the confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    classes = ["Negative", "Positive"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    plt.savefig(os.path.join(model_dir, "confusion_matrix.png"), dpi=200)
    plt.close()


def compare_models(versions, flare_classes, time_windows):
    """
    Generate a comparison table of different model versions.

    Args:
        versions: List of version strings
        flare_classes: List of flare classes
        time_windows: List of time windows

    Returns:
        Markdown table comparing the models
    """
    headers = [
        "Version",
        "Flare Class",
        "Time Window",
        "Accuracy",
        "TSS",
        "Timestamp",
    ]
    rows = []

    for v in versions:
        for fc in flare_classes:
            for tw in time_windows:
                try:
                    metadata = load_model_metadata(v, fc, tw)
                    performance = metadata.get("performance", {})
                    timestamp = datetime.fromisoformat(
                        metadata["timestamp"]
                    ).strftime("%Y-%m-%d")

                    rows.append(
                        [
                            v,
                            fc,
                            f"{tw}h",
                            f"{performance.get('accuracy', 'N/A'):.4f}"
                            if isinstance(performance.get("accuracy"), float)
                            else "N/A",
                            f"{performance.get('TSS', 'N/A'):.4f}"
                            if isinstance(performance.get("TSS"), float)
                            else "N/A",
                            timestamp,
                        ]
                    )
                except FileNotFoundError:
                    rows.append([v, fc, f"{tw}h", "N/A", "N/A", "N/A"])

    # Create markdown table
    table = " | ".join(headers) + "\n"
    table += " | ".join(["---"] * len(headers)) + "\n"

    for row in rows:
        table += " | ".join(str(cell) for cell in row) + "\n"

    return table


def get_next_version(flare_class, time_window):
    """
    Determine the next available version number for a specific flare class and time window.

    Args:
        flare_class: Flare class (C, M, or M5)
        time_window: Time window (24, 48, or 72)

    Returns:
        Next available version number as a string (e.g., "1.0")
    """
    try:
        # Find all model directories matching the pattern
        pattern = f"SolarKnowledge-v*-{flare_class}-{time_window}h"
        matching_dirs = []

        # Look through the models directory
        for d in os.listdir("models"):
            # Skip if not a directory
            if not os.path.isdir(os.path.join("models", d)):
                continue

            # Check if the directory matches our pattern
            if d.startswith("SolarKnowledge-v") and d.endswith(
                f"-{flare_class}-{time_window}h"
            ):
                matching_dirs.append(d)

        if not matching_dirs:
            return "1.0"  # First version

        # Extract version numbers from directory names
        versions = []
        for d in matching_dirs:
            parts = d.split("-")
            if len(parts) >= 2:
                v = parts[1][1:]  # Remove 'v' prefix
                try:
                    # Parse version numbers
                    if "." in v:
                        major, minor = v.split(".")
                        versions.append((int(major), int(minor)))
                    else:
                        versions.append((int(v), 0))
                except ValueError:
                    continue

        if not versions:
            return "1.0"  # Default if can't parse any versions

        # Find the highest version
        versions.sort(reverse=True)
        highest_major, highest_minor = versions[0]

        # Increment the minor version
        new_minor = highest_minor + 1

        # If minor version gets too high, increment major version
        if new_minor >= 10:
            return f"{highest_major + 1}.0"
        else:
            return f"{highest_major}.{new_minor}"
    except Exception as e:
        print(f"Error determining next version: {e}")
        return "1.0"  # Default to 1.0 if anything goes wrong


def get_latest_version(flare_class, time_window):
    """
    Get the latest available version for a specific flare class and time window.

    Args:
        flare_class: Flare class (C, M, or M5)
        time_window: Time window (24, 48, or 72)

    Returns:
        Latest version number as a string (e.g., "1.0") or None if no models exist
    """
    next_version = get_next_version(flare_class, time_window)
    if next_version == "1.0":
        return None

    # Parse the next version to find the previous one
    if "." in next_version:
        major, minor = next_version.split(".")
        minor = int(minor)

        if minor > 0:
            return f"{major}.{minor - 1}"
        else:
            # Check if previous major version exists
            major = int(major)
            if major > 1:
                # Try to find the highest minor version of the previous major
                # version
                prev_major = str(major - 1)
                for i in range(9, -1, -1):  # Check from 9 down to 0
                    version = f"{prev_major}.{i}"
                    if os.path.exists(
                        f"models/SolarKnowledge-v{version}-{flare_class}-{time_window}h"
                    ):
                        return version

                # If no minor version found, default to .0
                return f"{prev_major}.0"

    return "1.0"


if __name__ == "__main__":
    # Example usage
    models = list_available_models()
    if models:
        print("Available Models:")
        for model in models:
            print(
                f"v{model['version']} - {model['flare_class']} - {model['time_window']}h - Accuracy: {model['accuracy']}"
            )
    else:
        print("No models found.")
