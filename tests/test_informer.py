#!/usr/bin/env python
"""Test script for the Informer model.

This script tests a saved Informer model by loading it from a checkpoint,
unpacking model settings, and evaluating its prediction capabilities.
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

from models.archive.informer import GOESParquetDataset, Informer, select_device

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)


def load_model(model_checkpoint, device, **model_kwargs):
    """Load a trained Informer model from a checkpoint.

    Args:
        model_checkpoint: Path to the saved model checkpoint
        device: Device to load the model onto (CPU or GPU)
        **model_kwargs: Additional model parameters

    Returns:
        A tuple of (model, metadata)
    """
    checkpoint = torch.load(model_checkpoint, map_location=device)
    metadata = None
    if (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
        and "metadata" in checkpoint
    ):
        metadata = checkpoint["metadata"]
        print("Loaded metadata from checkpoint:")
        print(metadata)
        # Use metadata to override model parameters if not provided.
        meta_model_kwargs = metadata.get("model_kwargs", {})
        for key, value in meta_model_kwargs.items():
            if key not in model_kwargs or model_kwargs[key] is None:
                model_kwargs[key] = value
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model = Informer(**model_kwargs).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, metadata


def get_sample(dataset, sample_index=0):
    """Get a single sample from the dataset.

    Args:
        dataset: The dataset to sample from
        sample_index: Index of the sample to retrieve

    Returns:
        A tuple of (input_features, target_values)
    """
    x, y = dataset[sample_index]
    return x, y


def plot_results(x, y_true, y_pred):
    """Plot the context, ground truth, and predictions.

    Args:
        x: Context (input) data
        y_true: Ground truth values
        y_pred: Predicted values
    """
    # Plot the context (input) followed by the forecast.
    context_time = np.arange(len(x))
    forecast_time = np.arange(len(x), len(x) + len(y_true))

    plt.figure(figsize=(10, 6))
    plt.plot(context_time, x, label="Context (Input)", linewidth=2)
    plt.plot(forecast_time, y_true, label="Ground Truth", marker="o")
    plt.plot(forecast_time, y_pred, label="Prediction", marker="x")
    plt.xlabel("Time step")
    plt.ylabel("Value (Original Scale)")
    plt.title(
        "Informer Forecast: Context, Ground Truth, and Prediction (Unscaled)"
    )
    plt.legend()
    plt.grid(True)
    plt.show()


def main(args):
    """Run the main testing workflow.

    Args:
        args: Command line arguments
    """
    device = select_device()
    # Determine which checkpoint to use.
    if args.model_archive:
        candidate = os.path.join(args.model_archive, "final_model.pth")
        if not os.path.exists(candidate):
            candidate = os.path.join(args.model_archive, "best_model.pth")
        elif not os.path.exists(candidate):
            candidate = os.path.join(args.model_archive, "best_model.pth")
        model_checkpoint = candidate
    else:
        model_checkpoint = args.model_checkpoint

    # Start with an empty model_kwargs dict so that load_model can fill it
    # from metadata.
    model_kwargs = {}
    model, metadata = load_model(model_checkpoint, device, **model_kwargs)

    # Determine lookback and forecast lengths. If metadata is present, it will
    # be used.
    if metadata and "model_kwargs" in metadata:
        model_params = metadata["model_kwargs"]
        lookback_len = model_params.get("lookback_len", args.lookback_len)
        forecast_len = model_params.get("forecast_len", args.forecast_len)
    else:
        lookback_len = args.lookback_len
        forecast_len = args.forecast_len

    # Load the validation dataset using the (possibly metadata-based)
    # parameters.
    dataset = GOESParquetDataset(
        parquet_file=args.parquet_file,
        lookback_len=lookback_len,
        forecast_len=forecast_len,
        train=False,
        train_split=0.8,
    )
    if args.max_samples:
        dataset.indices = dataset.indices[: args.max_samples]

    # Get a specific sample.
    x, y_true = get_sample(dataset, sample_index=args.sample_index)
    x = x.unsqueeze(0).to(device)  # shape: [1, lookback_len]

    # Autoregressive decoding: build the decoder input one step at a time.
    forecast_steps = y_true.shape[0]
    predictions = []

    # Option 1: Use a longer warm start.
    # Increase the window size (experiment with 10 or more timesteps)
    init_window = 1000
    if x.shape[1] >= init_window:
        dec_input = x[:, -init_window:].clone()  # [1, init_window]
    else:
        dec_input = x.clone()

    # Option 2: Hybrid decoding – for the first few forecast steps use teacher
    # forcing if ground truth is available.
    # use teacher forcing for first 3 forecast steps, for example.
    initial_steps = min(3, forecast_steps)
    for t in range(initial_steps):
        # Use actual ground truth forecast values as seed for these steps.
        # (This assumes you want to see smoother transition; for a pure test you'd use autoregressive only.)
        gt_val = y_true[t].unsqueeze(0).unsqueeze(1).to(device)  # shape: [1,1]
        predictions.append(gt_val)
        dec_input = torch.cat([dec_input, gt_val], dim=1)

    # Now switch to autoregressive decoding for the remaining steps.
    for t in range(initial_steps, forecast_steps):
        current_length = dec_input.shape[1]
        dummy = torch.zeros(1, current_length + 1, device=device)
        dummy[:, :current_length] = dec_input
        with torch.no_grad():
            pred_full = model(x, dummy)  # shape: [1, current_length+1]
        pred_next = pred_full[:, -1].unsqueeze(1)  # [1,1]
        predictions.append(pred_next)
        dec_input = torch.cat([dec_input, pred_next], dim=1)
    y_pred = torch.cat(predictions, dim=1)
    y_pred = y_pred.squeeze(0).cpu().numpy()
    y_true = y_true.cpu().numpy()
    x = x.squeeze(0).cpu().numpy()

    # Inverse the normalization: unstandardize then invert the log transform.
    x_unscaled = x * dataset.std + dataset.mean
    y_true_unscaled = y_true * dataset.std + dataset.mean
    y_pred_unscaled = y_pred * dataset.std + dataset.mean

    x_orig = np.expm1(x_unscaled)
    y_true_orig = np.expm1(y_true_unscaled)
    y_pred_orig = np.expm1(y_pred_unscaled)

    plot_results(x_orig, y_true_orig, y_pred_orig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test a saved Informer model with automated metadata extraction"
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint (e.g. best_model.pth); ignored if --model_archive is provided.",
    )
    parser.add_argument(
        "--model_archive",
        type=str,
        default=None,
        help="Path to the model archive directory; if provided, the script uses final_model.pth or best_model.pth.",
    )
    parser.add_argument(
        "--parquet_file",
        type=str,
        required=True,
        help="Path to the Parquet file for data loading.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional limit on the number of samples to use from the dataset.",
    )
    parser.add_argument(
        "--sample_index",
        type=int,
        default=0,
        help="Index of the sample to use for plotting (default: 0).",
    )
    parser.add_argument(
        "--lookback_len",
        type=int,
        default=72,
        help="Default lookback length (in hours) if metadata not available.",
    )
    parser.add_argument(
        "--forecast_len",
        type=int,
        default=2,
        help="Default forecast length (in hours) if metadata not available.",
    )
    args = parser.parse_args()
    main(args)
