# -----------------------------
# File: train_everest.py  (light wrapper around existing train script)
# -----------------------------
"""
Usage:  python train_everest.py  --specific-flare M5 --specific-window 24
This script is identical to SolarKnowledge_run_all_trainings.py but imports
EVEREST instead of SolarKnowledge and passes the same hyper‑parameters.
"""
import argparse, numpy as np, tensorflow as tf
from utils import get_training_data, data_transform, log, supported_flare_class
from model_tracking import save_model_with_metadata, get_next_version, get_latest_version
from everest_model import EVEREST

def train(time_window, flare_class, auto_increment=True):
    X, y_raw = get_training_data(time_window, flare_class)
    
    # Debug: Print raw labels distribution
    print(f"DEBUG - Raw labels: {np.unique(y_raw, return_counts=True)}")
    
    # Handle different label formats - could be strings ('N'/'P') or integers (0/1)
    if y_raw.dtype == np.int64 or y_raw.dtype == np.int32 or y_raw.dtype == np.float64 or y_raw.dtype == np.float32:
        # Labels are already numerical
        print("Using numerical labels directly (0=negative, 1=positive)")
        y = y_raw.astype("float32")
    else:
        # Labels are strings, convert 'P' to 1, everything else to 0
        print("Converting string labels ('P'=positive, others=negative)")
        y = np.array([1 if label == 'P' else 0 for label in y_raw]).astype("float32")
    
    # Debug: After conversion 
    print(f"DEBUG - After conversion, we have {np.sum(y)} positive examples")
    
    input_shape = (X.shape[1], X.shape[2])
    
    # Get class distribution
    pos = np.sum(y)
    neg = len(y) - pos
    
    # Check if we have any positive examples
    if pos == 0:
        print(f"ERROR: No positive examples found for flare class {flare_class}!")
        print("Try a different flare class (e.g., 'M' instead of 'M5') or window size.")
        print("Available flare classes:", supported_flare_class)
        return None
    
    # Calculate appropriate alpha parameter for focal loss based on class distribution
    # Higher alpha gives more weight to the positive class
    alpha = min(0.9, 0.25 + 0.5 * (neg / pos) / (neg / pos + 100))
    
    # Print class distribution info
    print(f"Class counts - Negative: {neg}, Positive: {pos}")
    print(f"Class imbalance ratio: {neg/pos:.2f}")
    print(f"Using focal loss alpha: {alpha:.4f}")
    
    # Create model with calculated alpha
    model = EVEREST()
    model.build_base_model(input_shape)
    model.compile(alpha=alpha, gamma=1.5)  # Using gamma=1.5 for better stability
    
    version = get_next_version(flare_class, time_window) if auto_increment else "dev"
    hist = model.fit(X, y, epochs=100)
    
    # Get the final metrics
    metrics = {}
    for metric in ["loss", "prec", "rec", "tss"]:
        if metric in hist.history:
            metrics[f"final_{metric}"] = hist.history[metric][-1]
    
    hp = {"focal_alpha": alpha, "focal_gamma": 1.5, "linear_attention": True}
    save_model_with_metadata(model, metrics, hp, hist, version, flare_class, time_window,
                             "EVEREST SHARP‑only experiment")
    return model

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--specific-flare", "-f", default="M5")
    p.add_argument("--specific-window", "-w", default="24")
    a = p.parse_args()
    train(a.specific_window, a.specific_flare)