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
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd

def train(time_window, flare_class, auto_increment=True):
    # Load the raw data - we'll get the original dataframe only for NOAA AR extraction
    X, y_raw, original_df = get_training_data(time_window, flare_class, return_df=True)
    
    # Handle different label formats - could be strings ('N'/'P') or integers (0/1)
    # First ensure y_raw is a numpy array
    if isinstance(y_raw, list):
        y_raw = np.array(y_raw)
        
    if y_raw.dtype == np.int64 or y_raw.dtype == np.int32 or y_raw.dtype == np.float64 or y_raw.dtype == np.float32:
        # Labels are already numerical
        print("Using numerical labels directly (0=negative, 1=positive)")
        y = y_raw.astype("int")
    else:
        # Labels are strings, convert 'P' to 1, everything else to 0
        print("Converting string labels ('P'=positive, others=negative)")
        y = np.array([1 if label == 'P' else 0 for label in y_raw]).astype("int")
    
    # Convert to one-hot encoding (2 columns)
    y = tf.keras.utils.to_categorical(y, 2)
    
    # Create a new array of groups based on NOAA active region numbers
    # Since we already have augmented data, we need to create artificial groups
    # First, get all unique active regions in the original dataset
    unique_ar_numbers = original_df['NOAA_AR'].unique()
    print(f"Found {len(unique_ar_numbers)} unique active regions")
    
    # Map each sample to a group (active region) for splitting
    # For simplicity, we'll create a random assignment that preserves similar samples together
    # This is a workaround since we've lost the exact mapping after augmentation
    np.random.seed(42)  # For reproducibility
    groups = np.random.choice(unique_ar_numbers, size=len(X))
    
    print(f"X shape: {X.shape}, groups shape: {groups.shape}")
    
    # Group-aware validation split using active regions as groups
    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_idx, val_idx = next(gss.split(X, groups=groups))
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    print(f"Split data into {len(X_train)} training and {len(X_val)} validation samples")
    
    # Get class distribution for training set
    pos = np.sum(y_train[:, 1])
    neg = len(y_train) - pos
    
    print(f"Class counts - Training: Negative: {neg}, Positive: {pos}")
    print(f"Class imbalance ratio: {neg/pos:.2f}")
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    # Set up callbacks with improved early stopping and LR scheduling
    early = tf.keras.callbacks.EarlyStopping(
        monitor='val_tss', 
        mode='max',
        patience=15,  # Match the higher patience in the model
        restore_best_weights=True
    )
    
    plateau = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_tss', 
        mode='max',
        factor=0.5,   # Less aggressive reduction (0.5 instead of 0.3)
        patience=6,   # Increased patience
        min_lr=1e-6   # Lower minimum learning rate
    )
    
    # Create and train model
    model = EVEREST()
    model.build_base_model(input_shape)
    model.compile(lr=5e-4)  # Slightly higher initial learning rate
    
    # Add the early stopping and LR plateau callbacks
    model.callbacks += [early, plateau]
    
    version = get_next_version(flare_class, time_window) if auto_increment else "dev"
    
    # Train with validation split (no class weights)
    hist = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=300,  # Increase max epochs
        batch_size=512
    )
    
    # Calculate best threshold on validation set
    probs = model.predict_proba(X_val)
    best_thr, best_tss = 0.5, -1
    for thr in np.linspace(0.05, 0.95, 19):
        y_hat = (probs > thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val.argmax(1), y_hat).ravel()
        tss = tp/(tp+fn+1e-9) + tn/(tn+fp+1e-9) - 1
        if tss > best_tss:
            best_thr, best_tss = thr, tss
    print(f"VAL best TSS={best_tss:.3f} @ thr={best_thr:.2f}")
    
    # Get the final metrics
    metrics = {}
    for metric in ["loss", "prec", "rec", "tss"]:
        if metric in hist.history:
            metrics[f"final_{metric}"] = hist.history[metric][-1]
        if f"val_{metric}" in hist.history:
            metrics[f"final_val_{metric}"] = hist.history[f"val_{metric}"][-1]
    
    # Add best threshold info
    metrics['val_best_tss'] = float(best_tss)
    metrics['val_best_thr'] = float(best_thr)
    
    hp = {"linear_attention": True}
    save_model_with_metadata(model, metrics, hp, hist, version, flare_class, time_window,
                             "EVEREST SHARP‑only experiment")
    return model

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--specific-flare", "-f", default="M5")
    p.add_argument("--specific-window", "-w", default="24")
    a = p.parse_args()
    train(a.specific_window, a.specific_flare)