#!/usr/bin/env python
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
"""
Script to test a saved Informer model.
It loads a model from a specified checkpoint and a sample from a provided
Parquet file, then plots the context (input), prediction, and ground truth.
"""

import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np

from models.informer import Informer, GOESParquetDataset, select_device

def load_model(model_checkpoint, device, **model_kwargs):
    model = Informer(**model_kwargs).to(device)
    state_dict = torch.load(model_checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

def get_sample(dataset, sample_index=0):
    # Get the sample at the provided index.
    x, y = dataset[sample_index]
    return x, y

def plot_results(x, y_true, y_pred):
    # Concatenate the sequences for plotting: context from index 0 to len(x)-1
    # and forecast from len(x) to len(x)+len(y)
    context_time = np.arange(len(x))
    forecast_time = np.arange(len(x), len(x) + len(y_true))
    
    plt.figure(figsize=(10,6))
    plt.plot(context_time, x, label="Context (Input)", linewidth=2)
    plt.plot(forecast_time, y_true, label="Ground Truth", marker="o")
    plt.plot(forecast_time, y_pred, label="Prediction", marker="x")
    plt.xlabel("Time step")
    plt.ylabel("Value (Original Scale)")
    plt.title("Informer Forecast: Context, Ground Truth, and Prediction (Unscaled)")
    plt.legend()
    plt.grid(True)
    plt.show()
    
def main(args):
    device = select_device()
    # Model parameters MUST match those used during training of the checkpoint in the archive.
    model_kwargs = {
        "d_model": 128,
        "n_heads": 8,
        "d_ff": 256,
        "enc_layers": 3,
        "dec_layers": 2,
        "dropout": 0.1,
        "lookback_len": 24,   # 24 hours of context
        "forecast_len": 12    # 12 hours forecast
    }
    
    if args.model_archive is not None:
        candidate = os.path.join(args.model_archive, "final_model.pth")
        if not os.path.exists(candidate):
            # Fallback to best_model.pth if final_model.pth does not exist.
            candidate = os.path.join(args.model_archive, "epoch_2_model.pth")
        if not os.path.exists(candidate):
            candidate = os.path.join(args.model_archive, "best_model.pth")
        model_checkpoint = candidate
    else:
        model_checkpoint = args.model_checkpoint
    model = load_model(model_checkpoint, device, **model_kwargs)
    
    # Load validation dataset (using GOESParquetDataset)
    dataset = GOESParquetDataset(
        parquet_file=args.parquet_file,
        lookback_len=model_kwargs["lookback_len"],
        forecast_len=model_kwargs["forecast_len"],
        train=False,
        train_split=0.8
    )
    
    # For quick testing, truncate the dataset sample count if needed.
    if args.max_samples is not None:
        dataset.indices = dataset.indices[:args.max_samples]
    
    x, y_true = get_sample(dataset, sample_index=args.sample_index)
    # x and y_true are torch tensors of shape [context] and [forecast] respectively.
    x = x.unsqueeze(0).to(device)  # shape: [1, lookback_len]
    # For inference, we will use a dummy (zeros) forecast input.
    tgt_dummy = torch.zeros(1, y_true.shape[0]).to(device)
    
    with torch.no_grad():
        y_pred = model(x, tgt_dummy)
    # y_pred is [1, forecast_len]
    y_pred = y_pred.squeeze(0).cpu().numpy()
    y_true = y_true.cpu().numpy()
    x = x.squeeze(0).cpu().numpy()
    
    # Inverse-transform the normalized values back to the log1p domain.
    x_unscaled = x * dataset.std + dataset.mean
    y_true_unscaled = y_true * dataset.std + dataset.mean
    y_pred_unscaled = y_pred * dataset.std + dataset.mean

    # Then, apply the inverse of the log1p transform to get back to the original scale.
    x_orig = np.expm1(x_unscaled)
    y_true_orig = np.expm1(y_true_unscaled)
    y_pred_orig = np.expm1(y_pred_unscaled)

    plot_results(x_orig, y_true_orig, y_pred_orig)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a saved Informer model")
    parser.add_argument("--model_checkpoint", type=str, required=True,
                        help="Path to the model checkpoint (e.g. best_model.pth). Ignored if --model_archive is provided.")
    parser.add_argument("--model_archive", type=str, default=None,
                        help="Path to model archive directory; if provided, final_model.pth from that directory will be used.")
    parser.add_argument("--parquet_file", type=str, required=True,
                        help="Path to the parquet file for data loading")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Optional limit on the number of samples to use from the dataset")
    parser.add_argument("--sample_index", type=int, default=0,
                        help="Index of the sample to use for plotting (default: 0)")
    args = parser.parse_args()
    
    main(args) 