"""
Model Tracking and Management for EVEREST

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
import glob
import pandas as pd
import torch
from sklearn.calibration import calibration_curve
import platform
import time
import sklearn
import seaborn as sns


def create_model_dir(version, flare_class, time_window):
    """Create a standardized model directory structure."""
    # Create the parent models directory if it doesn't exist
    if not os.path.exists("models"):
        os.makedirs("models", exist_ok=True)
        
    model_dir = (
        f"models/EVEREST-v{version}-{flare_class}-{time_window}h"
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
    # Optional evaluation artefacts
    y_true=None,
    y_pred=None,
    y_scores=None,
    evt_scores=None,
    sample_input=None,
    latency_repeats: int = 100,
    # Interpretability/uncertainty artefacts
    att_X_batch=None,
    att_y_true=None,
    att_y_pred=None,
    att_y_score=None,
    evidential_out=None,
):
    """
    Save a model with comprehensive metadata tracking.

    Args:
        model: The trained EVEREST model instance
        metrics: Dictionary of evaluation metrics
        hyperparams: Dictionary of hyperparameters used for training
        history: Training history from model.fit
        version: Model version string (e.g., "1.0")
        flare_class: Flare class target
        time_window: Time window used
        description: Optional model description
        y_true: Ground truth labels for classification
        y_pred: Predicted labels for classification
        y_scores: Predicted probabilities for classification
        evt_scores: EVT scores for classification
        sample_input: Input sample for latency benchmarking
        latency_repeats: Number of repeats for latency benchmarking
        att_X_batch: Batch of inputs for attention heatmaps (B, T, F)
        att_y_true: True labels for attention batch
        att_y_pred: Predicted labels for attention batch
        att_y_score: Confidence scores for attention batch
        evidential_out: Evidential model outputs (mu, v, alpha, beta) for violin plots
    """
    # Create model directory
    model_dir = create_model_dir(version, flare_class, time_window)

    # Detect if it's a PyTorch or TensorFlow model
    is_pytorch_model = hasattr(model, 'pytorch_model') or isinstance(model.model, torch.nn.Module)

    # Save model weights
    if is_pytorch_model:
        weights_path = os.path.join(model_dir, "model_weights.pt")
        if hasattr(model, 'save_weights'):
            # Use the model's save_weights method if available
            model.save_weights(flare_class=flare_class, w_dir=model_dir)
        else:
            # Fallback to direct PyTorch save
            torch.save(model.model.state_dict(), weights_path)
            
        # Extract architecture details for PyTorch model
        architecture = {
            "name": "EVEREST (PyTorch)",
            "input_shape": hyperparams.get("input_shape", "unknown"),
            "num_params": sum(p.numel() for p in model.model.parameters()),
            "precision": str(next(model.model.parameters()).dtype),
        }
    else:
        # Original TensorFlow saving logic
        weights_path = os.path.join(model_dir, "model_weights.weights.h5")
        model.model.save_weights(weights_path)

        # Extract architecture details for TensorFlow model
        architecture = {
            "name": "EVEREST",
            "input_shape": model.model.input_shape[1:],
            "num_params": model.model.count_params(),
            "precision": str(model.model.dtype_policy)
            if hasattr(model.model, "dtype_policy")
            else "float32",
        }

    # ------------------------------------------------------------------
    # Optional: Benchmark inference latency (before metadata creation so we can log it)
    # ------------------------------------------------------------------
    latency = None
    if sample_input is not None:
        latency = benchmark_inference_latency(
            model, sample_input, repeats=latency_repeats
        )

    # Create metadata
    metadata = {
        "version": version,
        "timestamp": datetime.now().isoformat(),
        "description": description
        or f"EVEREST model for {flare_class}-class flares with {time_window}h prediction window",
        "flare_class": flare_class,
        "time_window": time_window,
        "hyperparameters": hyperparams,
        "performance": metrics,
        "git_info": get_git_info(),
        "architecture": architecture,
        "framework": "PyTorch" if is_pytorch_model else "TensorFlow",
        # Optional latency benchmark (seconds per batch of len(sample_input))
        **({"latency_sec_per_batch32": latency} if latency is not None else {}),
    }

    # Save metadata
    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Save training history
    if history:
        if is_pytorch_model:
            # Save PyTorch history format (which might be a dict already)
            save_training_history(history, model_dir)
        elif hasattr(history, "history"):
            # Save TensorFlow history format
            save_training_history(history.history, model_dir)

    # Generate model card
    generate_model_card(metadata, metrics, model_dir)

    # ------------------------------------------------------------------
    # Additional artefacts: classification report, predictions, calibration, EVT, env
    # ------------------------------------------------------------------
    if (y_true is not None) and (y_pred is not None):
        save_classification_report(y_true, y_pred, model_dir)
        save_predictions_csv(y_true, y_pred, y_scores, model_dir)
        if y_scores is not None:
            save_calibration_curve(y_true, y_scores, model_dir)

    if evt_scores is not None:
        save_evt_tail_distribution(evt_scores, model_dir)

    # Always save environment info for reproducibility
    save_environment_info(model_dir)

    # Attention heatmaps
    if (
        (att_X_batch is not None)
        and (att_y_true is not None)
        and (att_y_pred is not None)
        and (att_y_score is not None)
    ):
        save_attention_heatmaps(
            model,
            att_X_batch,
            att_y_true,
            att_y_pred,
            att_y_score,
            model_dir,
        )

    # Evidential uncertainty violin plot
    if (evidential_out is not None) and (y_true is not None) and (y_pred is not None):
        save_uncertainty_violinplots(
            evidential_out, y_true, y_pred, model_dir
        )

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

    model_card = f"""# EVEREST Model v{metadata['version']}

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
- **Architecture**: EVEREST Transformer Model
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
from EVEREST_model import EVEREST

# Load the model
model = EVEREST()
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
            d for d in os.listdir("models") if d.startswith("EVEREST-v")
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


def get_latest_version(flare_class, time_window):
    """Get latest version for a specific flare class and time window"""
    # Check both potential model directory paths
    potential_dirs = ["models"]
    if os.path.exists(os.path.join("models", "models")):
        potential_dirs.append(os.path.join("models", "models"))
    
    # Look in all potential directories
    versions = []
    for base_dir in potential_dirs:
        pattern = os.path.join(base_dir, f"EVEREST-v*-{flare_class}-{time_window}h")
        dirs = glob.glob(pattern)
        
        for dir_path in dirs:
            dir_name = os.path.basename(dir_path)
            parts = dir_name.split("-")
            if len(parts) >= 2 and parts[1].startswith("v"):
                try:
                    version_str = parts[1][1:]  # Remove the 'v' prefix
                    version_num = float(version_str)
                    # Round to 1 decimal place to ensure consistency
                    version_num = round(version_num, 1)
                    versions.append(version_num)
                except ValueError:
                    continue
    
    # Return the highest version or None if no versions found
    return max(versions) if versions else None


def get_next_version(flare_class, time_window):
    """Get the next version number for a specific flare class and time window"""
    latest_version = get_latest_version(flare_class, time_window)
    if latest_version is None:
        return 1.0
    # Round to 1 decimal place to avoid floating point precision issues
    return round(latest_version + 0.1, 1)


# ----------------------------------------------------------------------
# New helper functions for extended tracking
# ----------------------------------------------------------------------


def save_classification_report(y_true, y_pred, model_dir):
    """Save sklearn classification report as JSON."""
    report = classification_report(y_true, y_pred, output_dict=True)
    with open(
        os.path.join(model_dir, "classification_report.json"), "w"
    ) as f:
        json.dump(report, f, indent=2)


def save_predictions_csv(y_true, y_pred, y_scores, model_dir):
    """Save per-sample predictions, probabilities, and ground truth to CSV."""
    df = pd.DataFrame({"true": y_true, "pred": y_pred})
    if y_scores is not None:
        df["proba"] = y_scores
    df.to_csv(os.path.join(model_dir, "predictions.csv"), index=False)


def save_calibration_curve(y_true, y_scores, model_dir, n_bins: int = 10):
    """Plot and save a reliability diagram (calibration curve)."""
    if y_scores is None:
        return
    prob_true, prob_pred = calibration_curve(y_true, y_scores, n_bins=n_bins)
    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    plt.title("Reliability Diagram")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "calibration_curve.png"))
    plt.close()


def save_evt_tail_distribution(evt_scores, model_dir):
    """Plot histogram/KDE of EVT tail scores and save figure."""
    if evt_scores is None:
        return

    plt.figure()

    # Support either dict[class -> scores] or flat list/array
    if isinstance(evt_scores, dict):
        for label, scores in evt_scores.items():
            plt.hist(
                scores,
                bins=30,
                alpha=0.6,
                label=str(label),
                density=True,
            )
        plt.legend(title="Class")
    else:
        plt.hist(evt_scores, bins=30, density=True)

    plt.title("EVT Tail Score Distribution")
    plt.xlabel("Score")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "evt_tail_distribution.png"))
    plt.close()


def benchmark_inference_latency(model, sample_input, repeats: int = 100):
    """Compute average inference latency per batch.

    Args:
        model: Trained model (should expose predict or predict_proba or __call__)
        sample_input: Batched inputs (e.g., X_test[:32])
        repeats: Number of forward passes to average over
    Returns:
        float: Average time in seconds per batch
    """
    start = time.time()
    for _ in range(repeats):
        if hasattr(model, "predict_proba"):
            _ = model.predict_proba(sample_input)
        elif hasattr(model, "predict"):
            _ = model.predict(sample_input)
        else:
            # Fallback to PyTorch forward pass
            with torch.no_grad():
                _ = model(sample_input)
    latency = (time.time() - start) / repeats
    return latency


def save_environment_info(model_dir):
    """Save Python and library version information for reproducibility."""
    env_info = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
    }
    with open(os.path.join(model_dir, "env.json"), "w") as f:
        json.dump(env_info, f, indent=2)


# ----------------------------------------------------------------------
# Interpretability: Attention Heatmaps
# ----------------------------------------------------------------------


def save_attention_heatmaps(
    model,
    X_batch,
    y_true,
    y_pred,
    y_score,
    model_dir,
    max_samples: int = 10,
):
    """Generate and save attention heatmaps for a batch of samples.

    Assumes the model exposes .model.att_pool following the RETPlus/EVERES
    architecture.
    """
    model.model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_batch, dtype=torch.float32).to(
            next(model.model.parameters()).device
        )
        # Forward pass through embedding stack up to attention pooling
        embedding = model.model.embedding(X_tensor)
        normed = model.model.norm(embedding)
        dropped = model.model.drop(normed)
        pos = model.model.pos(dropped)
        x = pos
        for blk in model.model.transformers:
            x = blk(x)
        att_weights = (
            model.model.att_pool(x).squeeze(-1).cpu().numpy()
        )  # shape (B, T)

    for i in range(min(len(X_batch), max_samples)):
        plt.figure(figsize=(6, 1.5))
        plt.title(
            f"Attention Weights\nTrue={y_true[i]} Pred={y_pred[i]} Conf={y_score[i]:.2f}"
        )
        plt.imshow(att_weights[i][None, :], cmap="hot", aspect="auto")
        plt.colorbar(label="Attention Weight")
        plt.xticks(range(X_batch.shape[1]))
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(
            os.path.join(model_dir, f"attention_heatmap_{i}.png"), dpi=200
        )
        plt.close()


# ----------------------------------------------------------------------
# Evidential Uncertainty Decomposition
# ----------------------------------------------------------------------


def compute_nig_uncertainties(evid_out):
    """Compute Total, Epistemic, and Aleatoric variance from NIG parameters."""
    mu, v, alpha, beta = [evid_out[:, i] for i in range(4)]
    eps = 1e-6
    total_var = beta / ((alpha - 1) * v + eps)
    epistemic_var = beta / ((alpha - 1) * v * alpha + eps)
    aleatoric_var = total_var - epistemic_var
    return total_var, epistemic_var, aleatoric_var


def save_uncertainty_violinplots(
    evid_out,
    y_true,
    y_pred,
    model_dir,
):
    """Save violin plots separating epistemic and aleatoric uncertainty."""
    total_var, epistemic_var, aleatoric_var = compute_nig_uncertainties(
        evid_out
    )

    df = pd.DataFrame(
        {
            "Total": total_var,
            "Epistemic": epistemic_var,
            "Aleatoric": aleatoric_var,
            "TrueLabel": y_true,
            "Correct": (y_true == y_pred).astype(int),
        }
    )

    plt.figure(figsize=(10, 5))
    sns.violinplot(
        data=pd.melt(
            df,
            id_vars=["Correct"],
            value_vars=["Epistemic", "Aleatoric"],
        ),
        x="variable",
        y="value",
        hue="Correct",
        split=True,
    )
    plt.title("Uncertainty Components by Prediction Correctness")
    plt.ylabel("Variance")
    plt.xlabel("Uncertainty Type")
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "uncertainty_violinplot.png"), dpi=200)
    plt.close()


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
