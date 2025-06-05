#!/usr/bin/env python3
"""
Prospective Case Study: July 12, 2012 X1.4 Flare Analysis
Evaluates EVEREST model performance on the historical Carrington-class near-miss event.
"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import sys
from sklearn.metrics import roc_curve, confusion_matrix

# Add the models directory to path to import the required modules
sys.path.append('./models')

try:
    from solarknowledge_ret_plus import RETPlusWrapper
    from utils import get_training_data, get_testing_data
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure you're running from the project root directory")
    sys.exit(1)


def load_july_12_data():
    """Load and prepare the July 12, 2012 data for analysis."""
    # Load the M5 72h training data
    data = pd.read_csv('Nature_data/training_data_M5_72.csv')
    
    # Convert date column and handle timezone
    data['DATE__OBS'] = pd.to_datetime(data['DATE__OBS'], utc=True)
    
    # Focus on July 12, 2012 event (around the X1.4 flare time: 12:52 PM EDT = 16:52 UTC)
    flare_time = pd.to_datetime('2012-07-12 16:52:00', utc=True)
    
    # Get data around the flare time (72h prediction window)
    start_time = flare_time - timedelta(hours=72)  # 72h before flare
    end_time = flare_time + timedelta(hours=24)    # 24h after flare for context
    
    event_data = data[
        (data['DATE__OBS'] >= start_time) & 
        (data['DATE__OBS'] <= end_time)
    ].copy()
    
    print(f"Found {len(event_data)} data points around July 12, 2012 flare")
    print(f"Date range: {event_data['DATE__OBS'].min()} to {event_data['DATE__OBS'].max()}")
    
    # Show the positive samples around flare time
    positive_samples = event_data[event_data['Flare'] == 'P']
    print(f"\nPositive samples around flare time:")
    print(positive_samples[['DATE__OBS', 'Flare', 'NOAA_AR', 'HARPNUM']].head(10))
    
    return event_data, flare_time


def prepare_sequences_for_model(data, seq_len=10):
    """Prepare sequences in the format expected by EVEREST model (10, 9)."""
    # These are the exact 9 features used by the model (from utils.py)
    feature_cols = ['TOTUSJH', 'TOTUSJZ', 'USFLUX', 'TOTBSQ', 'R_VALUE', 'TOTPOT', 'SAVNCPP', 'AREA_ACR', 'ABSNJZH']
    
    # Check if all required features are present
    missing_features = [col for col in feature_cols if col not in data.columns]
    if missing_features:
        print(f"Missing features: {missing_features}")
        return None, None, None
    
    # Extract only the required features
    features = data[feature_cols].copy()
    
    # Convert to numeric, handling any string values
    for col in feature_cols:
        features[col] = pd.to_numeric(features[col], errors='coerce')
    
    # Fill any NaN values with 0 (mask_value)
    features = features.fillna(0)
    
    # Create sequences
    sequences = []
    timestamps = []
    labels = []
    
    for i in range(seq_len, len(data)):
        seq = features.iloc[i-seq_len:i].values
        sequences.append(seq)
        timestamps.append(data['DATE__OBS'].iloc[i])
        labels.append(1 if data['Flare'].iloc[i] == 'P' else 0)
    
    return np.array(sequences), timestamps, labels


def load_everest_model():
    """Load the EVEREST M5 72h model using the RETPlusWrapper."""
    print("Loading EVEREST M5 72h model...")
    
    # Check if model weights exist
    model_path = 'tests/model_weights_EVEREST_72h_M5.pt'
    if not os.path.exists(model_path):
        print(f"ERROR: Model weights not found at {model_path}")
        return None
    
    try:
        # Initialize the RETPlusWrapper with the expected input shape
        input_shape = (10, 9)  # (seq_len, features) as shown in the notebook
        print(f"Using input shape: {input_shape}")
        
        model = RETPlusWrapper(input_shape)
        
        # Load the trained weights
        model.load(model_path)
        
        print(f"✓ EVEREST model loaded successfully")
        return model
        
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return None


def run_prospective_analysis():
    """Main function to run the prospective case study analysis."""
    print("=" * 60)
    print("EVEREST Prospective Case Study: July 12, 2012 X1.4 Flare")
    print("=" * 60)
    
    # Load the EVEREST model
    print("\n1. Loading EVEREST model...")
    model = load_everest_model()
    if model is None:
        return None
    
    # Load July 12, 2012 data
    print("\n2. Loading July 12, 2012 event data...")
    try:
        event_data, flare_time = load_july_12_data()
    except Exception as e:
        print(f"ERROR loading data: {e}")
        return None
    
    # Prepare sequences for prediction
    print("\n3. Preparing sequences for prediction...")
    try:
        sequences, timestamps, true_labels = prepare_sequences_for_model(event_data, seq_len=10)
        print(f"✓ Created {len(sequences)} sequences with shape {sequences.shape}")
    except Exception as e:
        print(f"ERROR preparing sequences: {e}")
        return None
    
    # Run model predictions
    print("\n4. Running EVEREST predictions...")
    try:
        # Get probability predictions
        pred_probs = model.predict_proba(sequences)
        
        # If pred_probs has multiple columns, take the positive class probability
        if pred_probs.ndim > 1 and pred_probs.shape[1] > 1:
            pred_probs = pred_probs[:, 1]  # Probability of class 1 (flare)
        elif pred_probs.ndim > 1:
            pred_probs = pred_probs.squeeze()
        
        print(f"✓ Completed predictions for all sequences")
        print(f"  Prediction shape: {pred_probs.shape}")
        print(f"  Mean prediction probability: {np.mean(pred_probs):.4f}")
        print(f"  Min/Max predictions: {np.min(pred_probs):.4f} / {np.max(pred_probs):.4f}")
        
    except Exception as e:
        print(f"ERROR during prediction: {e}")
        return None
    
    # Analysis of results
    print("\n5. Analyzing results...")
    
    # Convert labels to numpy array
    true_labels = np.array(true_labels)
    
    # Find optimal threshold using ROC analysis
    fpr, tpr, thresholds = roc_curve(true_labels, pred_probs)
    
    # Use Youden's J statistic to find optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"Optimal threshold τ*: {optimal_threshold:.4f}")
    
    # Create binary predictions
    binary_preds = (pred_probs >= optimal_threshold).astype(int)
    
    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, binary_preds)
    
    print(f"\nConfusion Matrix at τ=τ*:")
    print(f"                 Predicted")
    print(f"Actual    Flare    Quiet    Total")
    if cm.shape == (2, 2):
        print(f"Flare     {cm[1,1]:4d}    {cm[1,0]:4d}    {cm[1,1] + cm[1,0]:4d}")
        print(f"Quiet     {cm[0,1]:4d}    {cm[0,0]:4d}    {cm[0,1] + cm[0,0]:4d}")
        
        # Calculate metrics
        tp, fp, tn, fn = cm[1,1], cm[0,1], cm[0,0], cm[1,0]
    else:
        print("Warning: Unexpected confusion matrix shape")
        print(f"Confusion matrix:\n{cm}")
        # Handle case where there might be only one class
        if len(np.unique(true_labels)) == 1:
            if true_labels[0] == 1:  # All positive
                tp = np.sum(binary_preds == 1)
                fp = np.sum(binary_preds == 0)
                tn = fn = 0
            else:  # All negative
                tn = np.sum(binary_preds == 0)
                fn = np.sum(binary_preds == 1)
                tp = fp = 0
        else:
            tp = fp = tn = fn = 0
    
    # Performance metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    print(f"\nPerformance Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Timeline analysis around the flare
    print(f"\n6. Timeline analysis around flare time ({flare_time})...")
    
    # Find predictions around the actual flare time
    flare_window = []
    for i, ts in enumerate(timestamps):
        time_diff = (ts - flare_time).total_seconds() / 3600  # hours
        if -72 <= time_diff <= 24:  # 72h before to 24h after
            flare_window.append({
                'timestamp': ts,
                'hours_to_flare': time_diff,
                'prediction': pred_probs[i],
                'true_label': true_labels[i],
                'binary_pred': binary_preds[i]
            })
    
    # Find when model first crosses threshold before flare
    threshold_crossings = []
    for entry in flare_window:
        if entry['hours_to_flare'] < 0 and entry['prediction'] >= optimal_threshold:
            threshold_crossings.append(entry)
    
    alert_lead_time = 0
    if threshold_crossings:
        earliest_alert = min(threshold_crossings, key=lambda x: x['hours_to_flare'])
        alert_lead_time = -earliest_alert['hours_to_flare']  # Make positive
        print(f"✓ Model first crossed threshold {alert_lead_time:.1f} hours before flare")
        print(f"  Alert time: {earliest_alert['timestamp']}")
        print(f"  Prediction probability: {earliest_alert['prediction']:.4f}")
        
        # Check if this meets the 24h requirement
        if alert_lead_time >= 24:
            print(f"✓ MEETS 24h requirement! (Lead time: {alert_lead_time:.1f}h)")
        else:
            print(f"✗ Does not meet 24h requirement (Lead time: {alert_lead_time:.1f}h)")
    else:
        print("✗ Model never crossed threshold before the flare")
    
    # Find earliest prediction above different thresholds
    print(f"\nAlert lead times at different thresholds:")
    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5]:
        crossings = [entry for entry in flare_window 
                    if entry['hours_to_flare'] < 0 and entry['prediction'] >= thresh]
        if crossings:
            earliest = min(crossings, key=lambda x: x['hours_to_flare'])
            lead_time = -earliest['hours_to_flare']
            print(f"  Threshold {thresh:.1f}: {lead_time:.1f}h lead time")
        else:
            print(f"  Threshold {thresh:.1f}: No alert")
    
    # Create visualization
    print(f"\n7. Creating timeline visualization...")
    try:
        create_timeline_plot(flare_window, flare_time, optimal_threshold)
        print("✓ Timeline plot saved as 'july_12_2012_prospective_timeline.png'")
    except Exception as e:
        print(f"Warning: Could not create plot: {e}")
    
    # Summary for the paper table
    print(f"\n" + "="*60)
    print("SUMMARY FOR CASE STUDY TABLE:")
    print(f"="*60)
    print(f"Confusion Matrix at τ=τ*:")
    print(f"Flare predictions: TP={tp}, FN={fn}")
    print(f"Quiet predictions: FP={fp}, TN={tn}")
    print(f"Total quiet samples (24h): {tn + fp}")
    
    if threshold_crossings:
        print(f"\nAlert Performance:")
        print(f"Lead time: {alert_lead_time:.1f} hours")
        print(f"Meets 24h requirement: {'Yes' if alert_lead_time >= 24 else 'No'}")
    
    return {
        'confusion_matrix': cm,
        'optimal_threshold': optimal_threshold,
        'alert_lead_time': alert_lead_time,
        'meets_requirement': alert_lead_time >= 24 if threshold_crossings else False,
        'predictions': pred_probs,
        'timestamps': timestamps,
        'true_labels': true_labels
    }


def create_timeline_plot(flare_window, flare_time, threshold):
    """Create a timeline plot similar to Figure referenced in the case study."""
    plt.figure(figsize=(12, 8))
    
    # Extract data for plotting
    hours = [entry['hours_to_flare'] for entry in flare_window]
    predictions = [entry['prediction'] for entry in flare_window]
    true_labels = [entry['true_label'] for entry in flare_window]
    
    # Plot EVEREST predictions
    plt.subplot(2, 1, 1)
    plt.plot(hours, predictions, 'b-', linewidth=2, label='EVEREST Probability', alpha=0.8)
    
    plt.axhline(y=threshold, color='orange', linestyle='--', alpha=0.7, 
                label=f'Threshold τ*={threshold:.3f}')
    plt.axvline(x=0, color='red', linestyle='-', alpha=0.8, label='X1.4 Flare Onset')
    plt.fill_between(hours, 0, predictions, alpha=0.3, color='blue')
    
    plt.xlabel('Hours to Flare')
    plt.ylabel('Flare Probability')
    plt.title('EVEREST Prospective Replay: July 12, 2012 X1.4 Flare')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-72, 24)
    plt.ylim(0, 1)
    
    # Plot ground truth
    plt.subplot(2, 1, 2)
    flare_times = [h for h, label in zip(hours, true_labels) if label == 1]
    if flare_times:
        plt.scatter(flare_times, [1]*len(flare_times), color='red', s=100, alpha=0.8, 
                   label='Actual Flares')
    plt.axvline(x=0, color='red', linestyle='-', alpha=0.8, label='X1.4 Flare Onset')
    
    plt.xlabel('Hours to Flare')
    plt.ylabel('Ground Truth')
    plt.title('Ground Truth Flare Activity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-72, 24)
    plt.ylim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.savefig('july_12_2012_prospective_timeline.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    results = run_prospective_analysis() 