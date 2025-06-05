#!/usr/bin/env python
"""
Main training script for SKAB industrial anomaly prediction using the EVEREST model.

This script combines all valve datasets chronologically for training a unified model
that can detect anomalies across different valve scenarios. It preserves temporal
relationships in the data and uses all available features.

Key aspects:
- Chronological ordering of all data
- Full 24-timestep sequences
- All 16 features utilized
- Balanced model architecture for generalization
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import glob
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Create the target directory
os.makedirs("trained_models_unified", exist_ok=True)

# Tell PyTorch to use single process DataLoader
os.environ["PYTORCH_WORKERS"] = "0"

# Import the modules
from everest import RETPlusWrapper, device
import model_tracking

def main():
    # --- Configuration ---
    input_shape = (24, 16)      # 24 timesteps, 16 features
    
    # Medium-sized model - balancing complexity and generalization
    embed_dim = 96              # Between small (64) and large (128)
    num_heads = 3               # Between small (2) and large (4)
    ff_dim = 192                # Between small (128) and large (256)
    num_blocks = 4              # Same as large model for capacity
    dropout = 0.25              # Between small (0.3) and large (0.2)
    
    # Training parameters
    epochs = 100                
    batch_size = 32             
    patience = 20               
    learning_rate = 2e-5        # Slower learning rate for better generalization
    
    print(f"Training UNIFIED CHRONOLOGICAL EVEREST model on ALL valve datasets")
    print(f"Using input shape: {input_shape}")
    print(f"Using device: {device}")
    
    # Load and combine all data chronologically
    print("\n=== Loading and combining all valve data chronologically ===")
    
    # Get all available valve experiments
    valve_scenarios = ["valve1", "valve2"]
    all_experiments = {}
    
    for scenario in valve_scenarios:
        data_files = glob.glob(f"../data/SKAB/*_data_{scenario}_*.csv")
        experiments = list(set([os.path.basename(f).split('_')[-1].split('.')[0] for f in data_files]))
        all_experiments[scenario] = experiments
        print(f"Found {len(experiments)} experiments for {scenario}: {experiments}")
    
    # Load all data from all scenarios (both training and testing)
    all_data = []
    
    for scenario in valve_scenarios:
        for experiment in all_experiments[scenario]:
            for data_type in ["training", "testing"]:
                file_path = f"../data/SKAB/{data_type}_data_{scenario}_{experiment}.csv"
                
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    
                    # Add scenario and experiment info
                    df['scenario'] = scenario
                    df['experiment'] = experiment
                    df['data_type'] = data_type
                    
                    all_data.append(df)
                    print(f"Loaded {len(df)} samples from {data_type}_{scenario}_{experiment}")
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Combined dataset: {len(combined_df)} samples")
    
    # Check the format of timestamps
    print("Sample timestamps:", combined_df['timestamp'].iloc[:5].values)
    
    # Create a proper datetime column for sorting
    # Parse the timestamps which look like "20200309_133504_0" (YYYYMMDD_HHMMSS_X)
    # First, add a properly formatted datetime column
    combined_df['datetime'] = combined_df['timestamp'].apply(parse_custom_timestamp)
    
    # Sort by timestamp to preserve chronological order
    combined_df = combined_df.sort_values('datetime')
    
    # Extract samples based on HARPNUM grouping
    print("\n=== Creating chronologically ordered sequences ===")
    X, y, metadata = extract_sequences(combined_df)
    
    # Split the data into training and testing sets while preserving chronological order
    # Use a time-based split instead of random split
    split_idx = int(len(X) * 0.8)  # 80% for training
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    meta_train, meta_test = metadata[:split_idx], metadata[split_idx:]
    
    print(f"Training set: {len(X_train)} samples ({sum(y_train)} positive, {len(y_train) - sum(y_train)} negative)")
    print(f"Testing set: {len(X_test)} samples ({sum(y_test)} positive, {len(y_test) - sum(y_test)} negative)")
    
    # Initialize the model
    model = RETPlusWrapper(
        input_shape=input_shape,
        early_stopping_patience=patience,
        loss_weights={"focal": 0.80, "evid": 0.10, "evt": 0.10},
    )
    
    # Override the model with custom parameters
    model.model = model.model.__class__(
        input_shape=input_shape,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_blocks=num_blocks,
        dropout=dropout,
        use_attention_bottleneck=True,
        use_evidential=True, 
        use_evt=True,
        use_precursor=True
    ).to(device)
    
    # Set the attribute after initialization
    model.train_with_no_workers = True
    
    # Configure optimizer
    model.optimizer = torch.optim.AdamW(
        model.model.parameters(), 
        lr=learning_rate,
        weight_decay=1e-4
    )
    
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        model.optimizer, 
        mode='max',
        factor=0.5,
        patience=10,
        verbose=True,
        min_lr=1e-6
    )
    
    # Train the model
    print("\n=== Training unified model ===")
    model_dir = train_with_lr_scheduler(
        model, 
        X_train, 
        y_train, 
        epochs=epochs, 
        batch_size=batch_size, 
        scheduler=scheduler,
        patience=patience
    )
    
    # Evaluate the model
    print("\n=== Evaluating unified model ===")
    evaluate_model(model, X_test, y_test, meta_test)
    
    # Analyze model performance per valve type
    analyze_performance_by_valve(model, X_test, y_test, meta_test)
    
    return model, model_dir

def parse_custom_timestamp(timestamp_str):
    """
    Parse timestamps in the format "20200309_133504_0" (YYYYMMDD_HHMMSS_X)
    Returns a datetime object.
    """
    try:
        # First, split by underscore
        parts = timestamp_str.split('_')
        
        # Check if we have the expected format
        if len(parts) >= 2:
            date_part = parts[0]
            time_part = parts[1]
            
            # Extract date components
            year = int(date_part[:4])
            month = int(date_part[4:6])
            day = int(date_part[6:8])
            
            # Extract time components
            hour = int(time_part[:2])
            minute = int(time_part[2:4])
            second = int(time_part[4:6])
            
            # Create datetime object
            return datetime(year, month, day, hour, minute, second)
        else:
            # Fallback to a default date if the format is unexpected
            print(f"Warning: Unexpected timestamp format: {timestamp_str}")
            return datetime(2000, 1, 1)
    except Exception as e:
        # Fallback to a default date if there's an error in parsing
        print(f"Error parsing timestamp {timestamp_str}: {e}")
        return datetime(2000, 1, 1)

def extract_sequences(df):
    """Extract feature sequences from the combined dataframe"""
    # Extract features (all columns except metadata)
    feature_cols = [col for col in df.columns 
                   if col not in ['class', 'timestamp', 'datetime', 'step', 'HARPNUM', 
                                 'scenario', 'experiment', 'data_type']]
    
    # Group by HARPNUM to create time series
    grouped = df.groupby('HARPNUM')
    
    # Get unique sequence IDs
    harps = list(grouped.groups.keys())
    
    X = []
    y = []
    metadata = []  # Store metadata for each sequence for later analysis
    
    for harpnum in harps:
        group = grouped.get_group(harpnum)
        
        # Sort by step to ensure correct sequence
        group = group.sort_values('step')
        
        # Check if we have enough steps
        if len(group) < 24:  # We need all 24 steps
            continue
            
        # Get the label (same for all rows in the group)
        label = 1 if group['class'].iloc[0] == 'F' else 0
        
        # Extract features for this sequence
        seq = group[feature_cols].values
        
        # Make sure we have exactly 24 steps
        if len(seq) > 24:
            seq = seq[:24]
            
        # Store metadata
        meta = {
            'scenario': group['scenario'].iloc[0],
            'experiment': group['experiment'].iloc[0],
            'timestamp': group['timestamp'].iloc[0],
            'data_type': group['data_type'].iloc[0]
        }
        
        X.append(seq)
        y.append(label)
        metadata.append(meta)
    
    return np.array(X), np.array(y), metadata

def train_with_lr_scheduler(model, X_train, y_train, epochs, batch_size, scheduler, patience):
    """Custom training function that includes learning rate scheduling"""
    # Create a temporary TensorDataset for training
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    
    # Create a DataLoader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        num_workers=0,
    )
    
    # Set up tracking variables
    best_tss = -1e8
    patience_counter = 0
    best_weights = None
    best_epoch = -1
    
    # Training loop
    for epoch in range(epochs):
        model.model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        TP = TN = FP = FN = 0
        
        # Process each batch
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Forward pass
            model.optimizer.zero_grad()
            outputs = model.model(X_batch)
            
            # Compute loss
            from everest import composite_loss
            gamma = min(2.0, 2.0 * epoch / 50)  # Gradually increase gamma
            loss = composite_loss(
                y_batch,
                outputs,
                gamma=gamma,
                weights={"focal": 0.80, "evid": 0.10, "evt": 0.10, "prec": 0.05},
            )
            
            # Backward pass and optimize
            loss.backward()
            model.optimizer.step()
            
            # Accumulate metrics
            epoch_loss += loss.item()
            preds = (torch.sigmoid(outputs["logits"]) > 0.5).int().squeeze()
            y_true = y_batch.int().squeeze()
            
            # Handle case where batch size is 1
            if preds.dim() == 0:
                preds = preds.unsqueeze(0)
                y_true = y_true.unsqueeze(0)
                
            TP += ((preds == 1) & (y_true == 1)).sum().item()
            TN += ((preds == 0) & (y_true == 0)).sum().item()
            FP += ((preds == 1) & (y_true == 0)).sum().item()
            FN += ((preds == 0) & (y_true == 1)).sum().item()
            
            correct += (preds == y_true).sum().item()
            total += y_batch.size(0)
        
        # Calculate epoch metrics
        avg_loss = epoch_loss / len(loader)
        accuracy = correct / total if total > 0 else 0.0
        sensitivity = TP / (TP + FN + 1e-8)
        specificity = TN / (TN + FP + 1e-8)
        tss = sensitivity + specificity - 1.0
        
        # Update model history
        model.history["loss"].append(avg_loss)
        model.history["accuracy"].append(accuracy)
        model.history["tss"].append(tss)
        
        # Print progress
        current_lr = model.optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f} - acc: {accuracy:.4f} - "
              f"tss: {tss:.4f} - gamma: {gamma:.2f} - lr: {current_lr:.6f}")
        
        # Update learning rate scheduler
        scheduler.step(tss)
        
        # Early stopping check
        if tss > best_tss:
            best_tss = tss
            best_weights = model.model.state_dict().copy()
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs!")
                print(f"Restoring best model from epoch {best_epoch+1} with TSS {best_tss:.4f}")
                break
    
    # Restore best weights
    if best_weights is not None:
        model.model.load_state_dict(best_weights)
    
    # Save model with metadata
    model_dir = model.save(
        version=1,
        flare_class="unified_chronological",
        time_window=24,
        # Provide evaluation data for artefact generation
        X_eval=X_train,
        y_eval=y_train,
    )
    
    return model_dir

def evaluate_model(model, X_test, y_test, metadata=None):
    """Evaluate the model on the test data"""
    # Make predictions
    y_pred_proba = model.predict_proba(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate overall metrics
    accuracy = (y_pred.flatten() == y_test).mean()
    
    # Calculate TSS
    pos_idx = np.where(y_test == 1)[0]
    neg_idx = np.where(y_test == 0)[0]
    
    if len(pos_idx) == 0:
        sensitivity = 1.0
    else:
        sensitivity = np.mean(y_pred.flatten()[pos_idx])
        
    if len(neg_idx) == 0:
        specificity = 1.0
    else:
        specificity = 1 - np.mean(y_pred.flatten()[neg_idx])
    
    tss = sensitivity + specificity - 1
    
    # Calculate more detailed metrics
    TP = np.sum((y_pred.flatten() == 1) & (y_test == 1))
    TN = np.sum((y_pred.flatten() == 0) & (y_test == 0))
    FP = np.sum((y_pred.flatten() == 1) & (y_test == 0))
    FN = np.sum((y_pred.flatten() == 0) & (y_test == 1))
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Overall test accuracy: {accuracy:.4f}")
    print(f"Overall test TSS: {tss:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    print(f"True Positives: {TP}, False Positives: {FP}")
    print(f"True Negatives: {TN}, False Negatives: {FN}")
    
    # Plot confusion matrix
    plot_confusion_matrix(TP, FP, TN, FN)
    
    # Plot ROC curve
    plot_roc_curve(y_test, y_pred_proba.flatten())
    
    return {
        "accuracy": accuracy,
        "tss": tss,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": TP,
        "tn": TN,
        "fp": FP,
        "fn": FN
    }

def analyze_performance_by_valve(model, X_test, y_test, metadata):
    """Analyze model performance broken down by valve type"""
    # Get predictions
    y_pred_proba = model.predict_proba(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Group results by scenario and experiment
    scenario_results = {}
    
    # Get unique scenario+experiment combinations
    unique_scenarios = set()
    for meta in metadata:
        scenario_key = f"{meta['scenario']}_{meta['experiment']}"
        unique_scenarios.add(scenario_key)
    
    # Analyze each scenario
    print("\n=== Performance by valve scenario ===")
    for scenario_key in sorted(unique_scenarios):
        # Find indices for this scenario
        scenario_indices = [i for i, meta in enumerate(metadata) 
                           if f"{meta['scenario']}_{meta['experiment']}" == scenario_key]
        
        if not scenario_indices:
            continue
            
        # Extract predictions and ground truth for this scenario
        scenario_y_true = y_test[scenario_indices]
        scenario_y_pred = y_pred.flatten()[scenario_indices]
        scenario_y_proba = y_pred_proba.flatten()[scenario_indices]
        
        # Calculate metrics
        if len(scenario_y_true) == 0:
            continue
            
        scenario_accuracy = (scenario_y_pred == scenario_y_true).mean()
        
        # Calculate TSS for this scenario
        scenario_pos_idx = np.where(scenario_y_true == 1)[0]
        scenario_neg_idx = np.where(scenario_y_true == 0)[0]
        
        if len(scenario_pos_idx) == 0:
            scenario_sensitivity = 1.0
        else:
            scenario_sensitivity = np.mean(scenario_y_pred[scenario_pos_idx])
            
        if len(scenario_neg_idx) == 0:
            scenario_specificity = 1.0
        else:
            scenario_specificity = 1 - np.mean(scenario_y_pred[scenario_neg_idx])
        
        scenario_tss = scenario_sensitivity + scenario_specificity - 1
        
        # Calculate more detailed metrics
        TP = np.sum((scenario_y_pred == 1) & (scenario_y_true == 1))
        TN = np.sum((scenario_y_pred == 0) & (scenario_y_true == 0))
        FP = np.sum((scenario_y_pred == 1) & (scenario_y_true == 0))
        FN = np.sum((scenario_y_pred == 0) & (scenario_y_true == 1))
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{scenario_key}: Accuracy = {scenario_accuracy:.4f}, TSS = {scenario_tss:.4f}, F1 = {f1:.4f}")
        
        # Store results
        scenario_results[scenario_key] = {
            "accuracy": scenario_accuracy,
            "tss": scenario_tss,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "total_samples": len(scenario_y_true),
            "positive_samples": sum(scenario_y_true),
            "negative_samples": len(scenario_y_true) - sum(scenario_y_true),
        }
    
    # Plot comparative bar chart
    plot_scenario_comparison(scenario_results)
    
    return scenario_results

def plot_confusion_matrix(TP, FP, TN, FN):
    """Plot a confusion matrix"""
    plt.figure(figsize=(8, 6))
    cm = np.array([[TN, FP], [FN, TP]])
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = ['Normal (0)', 'Anomaly (1)']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('dataset_analysis/unified_confusion_matrix.png')
    plt.close()

def plot_roc_curve(y_true, y_scores):
    """Plot ROC curve"""
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('dataset_analysis/unified_roc_curve.png')
    plt.close()

def plot_scenario_comparison(scenario_results):
    """Plot comparative bar chart of performance across scenarios"""
    plt.figure(figsize=(12, 8))
    
    # Define metrics to plot
    metrics = ["accuracy", "tss", "f1"]
    metric_titles = ["Accuracy", "TSS", "F1 Score"]
    
    # Extract scenario names and sort them
    scenarios = sorted(list(scenario_results.keys()))
    
    # Define colors for different scenario types
    colors = {
        "valve1_0": "blue", "valve1_10": "blue", "valve1_11": "blue", "valve1_12": "blue",  # Easy scenarios
        "valve1_1": "red",  # Difficult scenarios
        "valve2_0": "orange",  # Difficult scenarios
        "valve2_1": "green"  # Moderate scenarios
    }
    
    # Create subplots
    for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
        plt.subplot(1, 3, i+1)
        
        # Get values for the metric
        values = [scenario_results[s][metric] for s in scenarios]
        
        # Get colors for each scenario
        bar_colors = [colors.get(s, "gray") for s in scenarios]
        
        # Create the bar chart
        bars = plt.bar(scenarios, values, color=bar_colors)
        
        # Add a horizontal line at 0.5 for reference
        plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        
        # Add a horizontal line at 0 for TSS
        if metric == "tss":
            plt.axhline(y=0.0, color='red', linestyle='-', alpha=0.5)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f"{height:.2f}", ha='center', va='bottom', rotation=0)
        
        # Formatting
        plt.title(title)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0 if metric != "tss" else -0.2, 1.1)  # Allow negative values for TSS
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('dataset_analysis/unified_scenario_comparison.png')
    plt.close()

if __name__ == "__main__":
    # Make sure the output directories exist
    os.makedirs("dataset_analysis", exist_ok=True)
    os.makedirs("trained_models_unified", exist_ok=True)
    
    # Run the main function
    main() 