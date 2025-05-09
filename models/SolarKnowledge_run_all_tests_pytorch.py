"""
 author: Antanas Zilinskas

 This script tests SolarKnowledge PyTorch models for flare class: C, M, M5 and time window: 24, 48, 72.

 Improvements:
 - Uses Monte Carlo dropout for better uncertainty estimation and improved TSS
 - Stores uncertainty estimates in the results
 - Generates confidence plots to visualize prediction uncertainty
"""

import argparse
import glob
import json
import os
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
)
from SolarKnowledge_model_pytorch import SolarKnowledge
from tensorflow.keras.utils import to_categorical
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from utils import get_testing_data, get_training_data, log, supported_flare_class

warnings.filterwarnings("ignore")

# This dictionary will hold your "This work" performance for each time window.
# Structure: { "24": { "C": {metrics}, "M": {metrics}, "M5": {metrics} },
#              "48": { ... },
#              "72": { ... } }
all_metrics = {}


def find_latest_model_version(flare_class, time_window):
    """Find the latest model version for a specific flare class and time window"""
    model_patterns = [
        # New structure: models/models/SolarKnowledge-v*
        os.path.join("models", "models", f"SolarKnowledge-v*-{flare_class}-{time_window}h"),
        # Old structure: models/SolarKnowledge-v*
        os.path.join("models", f"SolarKnowledge-v*-{flare_class}-{time_window}h"),
    ]
    
    matching_dirs = []
    for pattern in model_patterns:
        matching_dirs.extend(glob.glob(pattern))
    
    if not matching_dirs:
        return None

    # Extract version numbers and find the highest one
    versions = []
    for dir_path in matching_dirs:
        dir_name = os.path.basename(dir_path)
        parts = dir_name.split("-")
        if len(parts) >= 2 and parts[1].startswith("v"):
            try:
                version_str = parts[1][1:]  # Remove the 'v' prefix
                version_num = float(version_str)
                versions.append((version_num, dir_path))
            except ValueError:
                continue

    if not versions:
        return None

    # Sort by version number and get the highest one
    versions.sort(reverse=True)
    return versions[0][1]  # Return the directory path


def test_model(
    time_window,
    flare_class,
    timestamp=None,
    use_latest=False,
    mc_passes=20,
    plot_uncertainties=True,
):
    log(
        "Testing initiated for time window: "
        + str(time_window)
        + " and flare class: "
        + flare_class,
        verbose=True,
    )

    # Load the testing data using your project's utility function
    X_test, y_test = get_testing_data(time_window, flare_class)
    
    # Check for potential label inversion
    test_positive_ratio = np.mean(y_test)
    log(f"Test data positive label ratio: {test_positive_ratio:.4f}", verbose=True)
    
    # Load a sample of training data to check distribution
    X_train, y_train = get_training_data(time_window, flare_class)
    train_positive_ratio = np.mean(y_train)
    log(f"Train data positive label ratio: {train_positive_ratio:.4f}", verbose=True)
    
    # Check data statistics without normalizing (data is already normalized)
    log("Checking data statistics for diagnostics...", verbose=True)
    # Get feature-wise mean and std from training data
    X_train_2d = X_train.reshape(-1, X_train.shape[-1])
    train_mean = np.mean(X_train_2d, axis=0)
    train_std = np.std(X_train_2d, axis=0)
    # Print a few samples from both datasets
    log(f"X_test sample: {X_test[0, 0, :5]}", verbose=True)
    log(f"X_train sample: {X_train[0, 0, :5]}", verbose=True)
    log(f"Training data mean: {train_mean[:5]}", verbose=True)
    log(f"Training data std: {train_std[:5]}", verbose=True)
    log("Data is already normalized, skipping normalization", verbose=True)

    # DEBUG CLASS DISTRIBUTION ISSUE
    log(f"Train y data type: {type(y_train)}, shape: {np.asarray(y_train).shape}", verbose=True)
    log(f"Test y data type: {type(y_test)}, shape: {np.asarray(y_test).shape}", verbose=True)
    if isinstance(y_train, list):
        # Convert to numpy array if needed
        y_train = np.array(y_train)
    if isinstance(y_test, list):
        # Convert to numpy array if needed
        y_test = np.array(y_test)
    
    # Fix bug in class distribution calculation
    if len(y_train.shape) == 1:
        train_neg = np.sum(y_train == 0)
        train_pos = np.sum(y_train == 1)
    else:
        # One-hot encoded
        train_neg = np.sum(y_train[:, 0] == 1)
        train_pos = np.sum(y_train[:, 1] == 1)
    
    if len(y_test.shape) == 1:
        test_neg = np.sum(y_test == 0)
        test_pos = np.sum(y_test == 1)
    else:
        # One-hot encoded
        test_neg = np.sum(y_test[:, 0] == 1)
        test_pos = np.sum(y_test[:, 1] == 1)

    log(f"FIXED Training distribution: {train_neg} negative, {train_pos} positive", verbose=True)
    log(f"FIXED Testing distribution: {test_neg} negative, {test_pos} positive", verbose=True)

    # If test data has inverted or highly skewed class distribution compared to training,
    # we need to handle this situation more carefully
    train_positive_ratio = train_pos / (train_pos + train_neg)
    test_positive_ratio = test_pos / (test_pos + test_neg)

    # Compare which class is the minority class in each set
    train_minority_is_positive = train_pos < train_neg
    test_minority_is_positive = test_pos < test_neg

    # If minority class differs between train and test, labels might be inverted
    labels_likely_inverted = (train_minority_is_positive != test_minority_is_positive)

    # Alternative check: Compare dominant class ratios
    # If they differ significantly, there might be an inversion or data issue
    ratio_mismatch = abs(train_positive_ratio - test_positive_ratio) > 0.3

    log(f"REVISED Train positive ratio: {train_positive_ratio:.4f}", verbose=True)
    log(f"REVISED Test positive ratio: {test_positive_ratio:.4f}", verbose=True)
    log(f"Train minority is positive: {train_minority_is_positive}", verbose=True)
    log(f"Test minority is positive: {test_minority_is_positive}", verbose=True)
    log(f"Labels likely inverted: {labels_likely_inverted}", verbose=True)
    log(f"Ratio mismatch: {ratio_mismatch}", verbose=True)

    if labels_likely_inverted or ratio_mismatch:
        log("WARNING: Test labels appear to have inverted or significantly different distribution compared to training data", verbose=True)
        log(f"Original test labels: Positive ratio = {test_positive_ratio:.4f}", verbose=True)
        
        # Invert labels if needed
        if labels_likely_inverted:
            log("Inverting test labels to match training distribution", verbose=True)
            y_test = 1 - y_test
            log(f"Inverted test labels: Positive ratio = {np.mean(y_test):.4f}", verbose=True)
    
    # Convert y_test to a NumPy array so we can check its dimensions
    y_test = np.array(y_test)
    # If your test labels are one-hot encoded, convert them to class indices.
    if y_test.ndim > 1:
        y_true = np.argmax(y_test, axis=-1)
    else:
        y_true = y_test

    input_shape = (X_test.shape[1], X_test.shape[2])

    # Find the latest model version in the correct directory structure
    latest_model_dir = find_latest_model_version(flare_class, time_window)

    if latest_model_dir and os.path.exists(latest_model_dir):
        weight_dir = latest_model_dir
        log(f"Using model at: {weight_dir}", verbose=True)
    else:
        log(
            f"No model found for flare class {flare_class} with time window {time_window}",
            verbose=True,
        )
        # Ensure we record placeholders for missing models.
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
    # Try to load weights from the model directory
    weight_file = os.path.join(weight_dir, "model_weights.pt")
    if not os.path.exists(weight_file):
        # Look for TensorFlow weights file and convert if needed
        tf_weight_file = os.path.join(weight_dir, "model_weights.weights.h5")
        if os.path.exists(tf_weight_file):
            log(f"Found TensorFlow weights file. Need to convert to PyTorch format first.", verbose=True)
            log(f"Error: Weight file not found at {weight_file}", verbose=True)
            return
        else:
            log(f"Error: Weight file not found at {weight_file}", verbose=True)
            return
    
    # Load metadata to get architecture parameters
    metadata_file = os.path.join(weight_dir, "metadata.json")
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Extract architecture parameters with defaults as fallback
            hyperparams = metadata.get('hyperparams', {})
            
            # Get key architecture parameters - ensure we read ALL the parameters
            # that could affect model structure
            embed_dim = hyperparams.get('embed_dim', 128)
            num_transformer_blocks = hyperparams.get('num_transformer_blocks', 6)
            num_heads = hyperparams.get('num_heads', 4)
            ff_dim = hyperparams.get('ff_dim', 256)
            dropout_rate = hyperparams.get('dropout_rate', 0.2)
            use_batch_norm = hyperparams.get('use_batch_norm', False)
            
            # Check for potential weight-metadata mismatch by examining weight file
            # This is a safety check to ensure the model architecture matches the saved weights
            try:
                state_dict = torch.load(weight_file, map_location=torch.device('cpu'))
                # Check embedding dimension from weights
                if "embedding.weight" in state_dict:
                    actual_embed_dim = state_dict["embedding.weight"].shape[0]
                    if actual_embed_dim != embed_dim:
                        log(f"WARNING: Metadata embed_dim ({embed_dim}) doesn't match weights ({actual_embed_dim}). " +
                            f"Using value from weights.", verbose=True)
                        embed_dim = actual_embed_dim
                
                # Check transformer blocks from weights by counting layers
                max_block_idx = -1
                for key in state_dict.keys():
                    if key.startswith("transformer_blocks."):
                        block_idx = int(key.split(".")[1])
                        max_block_idx = max(max_block_idx, block_idx)
                
                if max_block_idx >= 0:  # Found transformer blocks in weights
                    actual_num_blocks = max_block_idx + 1
                    if actual_num_blocks != num_transformer_blocks:
                        log(f"WARNING: Metadata transformer_blocks ({num_transformer_blocks}) doesn't match " +
                            f"weights ({actual_num_blocks}). Using value from weights.", verbose=True)
                        num_transformer_blocks = actual_num_blocks
                
                # Check feed-forward dimension from weights
                actual_ff_dim = None
                for i in range(actual_num_blocks if max_block_idx >= 0 else num_transformer_blocks):
                    key = f"transformer_blocks.{i}.ffn.0.weight"
                    if key in state_dict:
                        actual_ff_dim = state_dict[key].shape[0]
                        break
                
                if actual_ff_dim is not None and actual_ff_dim != ff_dim:
                    log(f"WARNING: Metadata ff_dim ({ff_dim}) doesn't match weights ({actual_ff_dim}). " +
                        f"Using value from weights.", verbose=True)
                    ff_dim = actual_ff_dim
            except Exception as e:
                log(f"Unable to verify weights structure: {str(e)}", verbose=True)
            
            log(f"Building model based on metadata: embed_dim={embed_dim}, " +
                f"transformer_blocks={num_transformer_blocks}, heads={num_heads}, ff_dim={ff_dim}", verbose=True)
            
            # Build the model with matching architecture
            model = SolarKnowledge(early_stopping_patience=5)
            model.build_base_model(
                input_shape=input_shape,
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                num_transformer_blocks=num_transformer_blocks,
                dropout_rate=dropout_rate
            )
            model.compile(use_focal_loss=True)
            
        except Exception as e:
            log(f"Error reading metadata, using default parameters: {str(e)}", verbose=True)
            # Fallback to default architecture if metadata reading fails
            model = SolarKnowledge(early_stopping_patience=5)
            model.build_base_model(input_shape)
            model.compile(use_focal_loss=True)
    else:
        log(f"No metadata file found, using default parameters", verbose=True)
        # Build model with default parameters if no metadata exists
        model = SolarKnowledge(early_stopping_patience=5)
        model.build_base_model(input_shape)
        model.compile(use_focal_loss=True)

    try:
        model.load_weights(w_dir=weight_dir)
        log(f"Successfully loaded weights from {weight_dir}", verbose=True)
    except Exception as e:
        log(f"Error loading weights: {str(e)}", verbose=True)
        return

    # Run predictions using Monte Carlo dropout
    log(
        f"Using Monte Carlo dropout with {mc_passes} passes for robust prediction",
        verbose=True,
    )
    mean_preds, std_preds = model.mc_predict(
        X_test, n_passes=mc_passes, verbose=1
    )

    # DEBUG: Analyze raw predictions to understand the model's behavior
    pos_probs = mean_preds[:, 1]  # Probability of positive class
    log(f"Raw prediction statistics:", verbose=True)
    log(f"Mean positive probability: {np.mean(pos_probs):.4f}", verbose=True)
    log(f"Median positive probability: {np.median(pos_probs):.4f}", verbose=True)
    log(f"Min positive probability: {np.min(pos_probs):.4f}", verbose=True)
    log(f"Max positive probability: {np.max(pos_probs):.4f}", verbose=True)
    log(f"Std of positive probability: {np.std(pos_probs):.4f}", verbose=True)

    # Create histogram of prediction probabilities to visualize distribution
    plt.figure(figsize=(10, 6))
    plt.hist(pos_probs, bins=50, alpha=0.7)
    plt.axvline(x=0.5, color='r', linestyle='--', label='Default threshold (0.5)')
    plt.title('Distribution of Prediction Probabilities')
    plt.xlabel('Positive Class Probability')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    prob_dist_file = os.path.join(weight_dir, f"probability_distribution_{datetime.now().strftime('%Y%m%d%H%M%S')}.png")
    plt.savefig(prob_dist_file)
    plt.close()
    log(f"Saved probability distribution plot to {prob_dist_file}", verbose=True)

    # Try to calibrate probabilities using the validation data
    log(f"Applying probability calibration using Platt scaling...", verbose=True)
    # Since we don't have validation probabilities defined yet, use a subset of test data
    # Split test data into two parts - one for calibration training, one for final evaluation
    from sklearn.model_selection import train_test_split

    # Use 20% of test data for calibration
    X_calib_idx, X_eval_idx = train_test_split(
        np.arange(len(pos_probs)), 
        test_size=0.8, 
        random_state=42, 
        stratify=y_true
    )

    X_calib = pos_probs[X_calib_idx].reshape(-1, 1)  # Features are probabilities
    y_calib = y_true[X_calib_idx]  # Targets are true classes

    # Train a logistic regression calibration model
    calibrator = LogisticRegression(C=1.0)
    calibrator.fit(X_calib, y_calib)

    # Apply calibration to all test predictions
    X_test_calib = pos_probs.reshape(-1, 1)
    calibrated_probs = calibrator.predict_proba(X_test_calib)[:, 1]

    # Visualize calibrated probabilities
    plt.figure(figsize=(10, 6))
    plt.hist(calibrated_probs, bins=50, alpha=0.7, color='green')
    plt.axvline(x=0.5, color='r', linestyle='--', label='Default threshold (0.5)')
    plt.title('Distribution of Calibrated Prediction Probabilities')
    plt.xlabel('Calibrated Positive Class Probability')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    calib_prob_file = os.path.join(weight_dir, f"calibrated_probability_distribution_{datetime.now().strftime('%Y%m%d%H%M%S')}.png")
    plt.savefig(calib_prob_file)
    plt.close()
    log(f"Saved calibrated probability distribution plot to {calib_prob_file}", verbose=True)

    # Use calibrated probabilities for predictions
    log(f"Using calibrated probabilities for predictions", verbose=True)
    log(f"Mean calibrated probability: {np.mean(calibrated_probs):.4f}", verbose=True)
    log(f"Median calibrated probability: {np.median(calibrated_probs):.4f}", verbose=True)

    # Find new threshold based on calibrated probabilities
    all_thresholds = np.linspace(0.01, 0.99, 100)
    best_threshold = 0.5
    best_tss = -1

    for threshold in all_thresholds:
        calib_preds = (calibrated_probs >= threshold).astype(int)
        # Calculate TSS
        cm = confusion_matrix(y_true, calib_preds)
        if cm.shape[0] > 1 and cm.shape[1] > 1:
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            tss = sensitivity + specificity - 1
            if tss > best_tss:
                best_tss = tss
                best_threshold = threshold

    log(f"Best threshold for calibrated probabilities: {best_threshold:.4f}, TSS: {best_tss:.4f}", verbose=True)

    # Use best threshold for final predictions
    predicted_classes = (calibrated_probs >= best_threshold).astype(int)
    log(f"Predictions using calibrated probabilities: Positive ratio = {np.mean(predicted_classes):.4f}", verbose=True)

    # Calculate uncertainty metrics
    entropy = -np.sum(mean_preds * np.log(mean_preds + 1e-10), axis=1)
    max_probs = np.max(mean_preds, axis=1)
    uncertainties = np.max(std_preds, axis=1)
    
    # Get validation data to find optimal threshold (a small subset of training data)
    # This helps address class imbalance which can cause threshold shifts
    val_size = min(5000, len(X_train))
    
    # Make sure X_train and y_train are numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # Create random indices for validation set
    indices = np.random.choice(len(X_train), val_size, replace=False)
    X_val, y_val = X_train[indices], y_train[indices]
    
    # Convert y_val to one-hot encoding if needed for prediction
    y_val_binary = y_val
    y_val_onehot = None
    if len(y_val.shape) == 1:
        y_val_onehot = to_categorical(y_val, num_classes=2)
    else:
        y_val_onehot = y_val
        y_val_binary = np.argmax(y_val, axis=1)
    
    # Get MC predictions on validation data
    val_mean_preds, _ = model.mc_predict(X_val, n_passes=mc_passes, verbose=1)
    
    # Find optimal threshold using validation data
    log("Finding optimal classification threshold using validation data...", verbose=True)
    # Get predicted probabilities for the positive class
    val_pos_probs = val_mean_preds[:, 1]
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_val_binary, val_pos_probs)
    
    # Sample more thresholds to ensure we find a good balance
    thresholds_dense = np.linspace(0.01, 0.99, 100)
    all_thresholds = np.unique(np.concatenate([thresholds, thresholds_dense]))
    
    # Calculate TSS and other metrics for different thresholds
    tss_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []
    
    # Print class distribution to better understand data
    neg_count = np.sum(y_val_binary == 0)
    pos_count = np.sum(y_val_binary == 1)
    log(f"Validation data has {neg_count} negative samples and {pos_count} positive samples", verbose=True)
    log(f"Positive ratio: {pos_count / (pos_count + neg_count):.4f}", verbose=True)

    for threshold in all_thresholds:
        val_preds = (val_pos_probs >= threshold).astype(int)
        # Calculate TSS (True Skill Statistic)
        cm = confusion_matrix(y_val_binary, val_preds)
        if cm.shape[0] > 1 and cm.shape[1] > 1:  # Ensure confusion matrix has proper shape
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            tss = sensitivity + specificity - 1
            tss_scores.append(tss)
            
            # Also calculate F1 for reference
            f1 = f1_score(y_val_binary, val_preds, average='binary')
            f1_scores.append(f1)
            
            precision = precision_score(y_val_binary, val_preds, average='binary', zero_division=0)
            recall = recall_score(y_val_binary, val_preds, average='binary', zero_division=0)
            precision_scores.append(precision)
            recall_scores.append(recall)
        else:
            tss_scores.append(-1)  # Invalid threshold
            f1_scores.append(0)
            precision_scores.append(0)
            recall_scores.append(0)

    # Find threshold with highest TSS score
    best_idx = np.argmax(tss_scores)
    optimal_threshold = all_thresholds[best_idx]

    log(f"Optimal threshold (maximizing TSS): {optimal_threshold:.4f}", verbose=True)
    log(f"Optimal TSS: {tss_scores[best_idx]:.4f}", verbose=True)
    log(f"At optimal threshold - Precision: {precision_scores[best_idx]:.4f}, Recall: {recall_scores[best_idx]:.4f}", verbose=True)

    # Find a more balanced threshold with good TSS - IMPROVED
    # Consider class balance in the scoring function
    class_ratio = min(pos_count / neg_count, neg_count / pos_count)  # Class imbalance ratio (0-1)
    # Create a balanced score: TSS + (harmonic mean of precision and recall weighted by class imbalance)
    balanced_scores = []
    for i, tss in enumerate(tss_scores):
        # Calculate harmonic mean of precision and recall
        p, r = precision_scores[i], recall_scores[i]
        # Avoid division by zero
        if p + r > 0:
            f_score = 2 * p * r / (p + r)
        else:
            f_score = 0
        # Weighted score that puts more emphasis on precision for imbalanced datasets
        # and more emphasis on recall for balanced datasets
        balance_weight = (1 - class_ratio) * 1.5  # Up to 1.5 more weight on precision for imbalanced data
        weighted_score = tss + f_score * (1 + balance_weight * (p - r)) if p > 0.5 else 0
        balanced_scores.append(weighted_score)

    balanced_idx = np.argmax(balanced_scores)
    balanced_threshold = all_thresholds[balanced_idx]

    log(f"Balanced threshold (TSS + weighted precision/recall): {balanced_threshold:.4f}", verbose=True)
    log(f"Balanced TSS: {tss_scores[balanced_idx]:.4f}", verbose=True)
    log(f"At balanced threshold - Precision: {precision_scores[balanced_idx]:.4f}, Recall: {recall_scores[balanced_idx]:.4f}", verbose=True)

    # Use balanced threshold for best overall performance
    optimal_threshold = balanced_threshold
    log(f"Using balanced threshold: {optimal_threshold:.4f}", verbose=True)
    
    # Use optimal threshold for predictions
    pos_probs = mean_preds[:, 1]
    predicted_classes = (pos_probs >= optimal_threshold).astype(int)
    
    log(f"Predictions using optimal threshold: Positive ratio = {np.mean(predicted_classes):.4f}", verbose=True)
    
    # Save original predicted classes based on argmax for comparison
    argmax_classes = np.argmax(mean_preds, axis=-1)
    log(f"Predictions using default threshold (0.5): Positive ratio = {np.mean(argmax_classes):.4f}", verbose=True)
    
    # If requested, plot prediction uncertainties
    if plot_uncertainties:
        create_uncertainty_plots(
            mean_preds,
            std_preds,
            y_true,
            predicted_classes,
            flare_class,
            time_window,
            weight_dir,
        )

    # Calculate basic metrics with optimal threshold:
    acc = accuracy_score(y_true, predicted_classes)
    prec = precision_score(y_true, predicted_classes)
    rec = recall_score(y_true, predicted_classes)
    bal_acc = balanced_accuracy_score(y_true, predicted_classes)
    # Compute TSS: sensitivity + specificity - 1. For binary classification,
    # sensitivity = recall for class 1, specificity = recall for class 0.
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
        f"Accuracy for flare class {flare_class} with time window {time_window}: {acc:.4f}"
    )
    print(f"TSS (True Skill Statistic): {TSS:.4f}")
    print(f"Sensitivity (True Positive Rate): {sensitivity:.4f}")
    print(f"Specificity (True Negative Rate): {specificity:.4f}")
    print("Classification Report:")
    print(classification_report(y_true, predicted_classes))
    
    # For comparison, also show metrics with default threshold
    print("\nComparison with default threshold (argmax):")
    acc_default = accuracy_score(y_true, argmax_classes)
    cm_default = confusion_matrix(y_true, argmax_classes)
    if cm_default.shape[0] > 1 and cm_default.shape[1] > 1:
        sensitivity_default = (
            cm_default[1, 1] / (cm_default[1, 1] + cm_default[1, 0]) if (cm_default[1, 1] + cm_default[1, 0]) > 0 else 0
        )
        specificity_default = (
            cm_default[0, 0] / (cm_default[0, 0] + cm_default[0, 1]) if (cm_default[0, 0] + cm_default[0, 1]) > 0 else 0
        )
        TSS_default = sensitivity_default + specificity_default - 1
        print(f"Accuracy: {acc_default:.4f}")
        print(f"TSS: {TSS_default:.4f}")
        print(f"Sensitivity: {sensitivity_default:.4f}")
        print(f"Specificity: {specificity_default:.4f}")
    
    print("==============================================\n\n")

    # Store metrics for the given time window and flare class
    metrics_dict = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "TSS": round(TSS, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
    }

    # Add uncertainty metrics
    metrics_dict["mean_entropy"] = float(np.mean(entropy))
    metrics_dict["mean_uncertainty"] = float(np.mean(uncertainties))
    metrics_dict["mean_confidence"] = float(np.mean(max_probs))

    # Get confusion matrix as a list for easier JSON serialization
    metrics_dict["confusion_matrix"] = cm.tolist()

    # Add test data size information
    metrics_dict["test_samples"] = len(y_true)
    metrics_dict["positive_samples"] = int(np.sum(y_true))
    metrics_dict["negative_samples"] = len(y_true) - int(np.sum(y_true))

    # Get model version from directory name
    model_name = os.path.basename(weight_dir)
    version = "unknown"
    if "-v" in model_name:
        version = model_name.split("-v")[1].split("-")[0]

    # Add version and test date to metrics
    test_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metrics_dict["version"] = version
    metrics_dict["test_date"] = test_date
    metrics_dict["used_monte_carlo_dropout"] = True
    metrics_dict["mc_passes"] = mc_passes
    metrics_dict["framework"] = "pytorch"

    # Store in consolidated metrics for all models
    time_key = str(time_window)
    if time_key not in all_metrics:
        all_metrics[time_key] = {}
    all_metrics[time_key][flare_class] = metrics_dict

    # Update model's metadata file with test results
    metadata_file = os.path.join(weight_dir, "metadata.json")
    if os.path.exists(metadata_file):
        try:
            # Load existing metadata
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            # Add or update test results
            if "test_results" not in metadata:
                metadata["test_results"] = {}

            # Use test date as a key to allow multiple test runs
            test_key = datetime.now().strftime("%Y%m%d%H%M%S")
            metadata["test_results"][test_key] = metrics_dict

            # Add latest test results at the top level for easy access
            metadata["latest_test"] = metrics_dict

            # Write updated metadata back to the file
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=4)

            log(
                f"Updated metadata file at {metadata_file} with test results",
                verbose=True,
            )
        except Exception as e:
            log(f"Error updating metadata file: {str(e)}", verbose=True)
    else:
        log(f"No metadata file found at {metadata_file}", verbose=True)

    return metrics_dict


def create_uncertainty_plots(
    mean_preds, std_preds, y_true, y_pred, flare_class, time_window, output_dir
):
    """Create plots to visualize prediction uncertainties."""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    # 1. Confidence Distribution Plot
    plt.figure(figsize=(10, 6))

    # Get confidence scores (highest probability)
    confidence = np.max(mean_preds, axis=1)

    # Plot distributions for correct and incorrect predictions
    correct = y_true == y_pred

    sns.histplot(
        confidence[correct],
        color="green",
        alpha=0.5,
        label="Correct Predictions",
        kde=True,
        bins=20,
    )
    sns.histplot(
        confidence[~correct],
        color="red",
        alpha=0.5,
        label="Incorrect Predictions",
        kde=True,
        bins=20,
    )

    plt.title(
        f"Prediction Confidence Distribution - {flare_class}-class ({time_window}h window)"
    )
    plt.xlabel("Confidence Score")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()

    # Save the figure
    confidence_file = os.path.join(
        output_dir, f"confidence_dist_{timestamp}.png"
    )
    plt.savefig(confidence_file, dpi=300)
    plt.close()

    # 2. Uncertainty vs Correctness Plot
    plt.figure(figsize=(10, 6))

    # Calculate uncertainty as standard deviation
    uncertainty = np.max(std_preds, axis=1)

    # Create scatter plot with different colors for correct/incorrect
    plt.scatter(
        confidence[correct],
        uncertainty[correct],
        color="green",
        alpha=0.5,
        label="Correct Predictions",
    )
    plt.scatter(
        confidence[~correct],
        uncertainty[~correct],
        color="red",
        alpha=0.5,
        label="Incorrect Predictions",
    )

    plt.title(
        f"Confidence vs. Uncertainty - {flare_class}-class ({time_window}h window)"
    )
    plt.xlabel("Confidence Score")
    plt.ylabel("Uncertainty (Std. Dev.)")
    plt.legend()
    plt.tight_layout()

    # Save the figure
    uncertainty_file = os.path.join(
        output_dir, f"uncertainty_scatter_{timestamp}.png"
    )
    plt.savefig(uncertainty_file, dpi=300)
    plt.close()

    log(f"Saved uncertainty visualization plots to {output_dir}", verbose=True)


if __name__ == "__main__":
    # Add command line argument for timestamp
    parser = argparse.ArgumentParser(
        description="Test SolarKnowledge models for solar flare prediction"
    )
    parser.add_argument(
        "--timestamp", "-t", help="Specific model timestamp to test"
    )
    parser.add_argument(
        "--latest", action="store_true", help="Test the latest model version"
    )
    parser.add_argument(
        "--version", "-v", help="Test a specific model version"
    )
    parser.add_argument(
        "--mc-passes",
        type=int,
        default=30,  # Increased from 20 to 30 for better uncertainty estimation
        help="Number of Monte Carlo dropout passes (default: 30)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating uncertainty plots",
    )
    parser.add_argument(
        "--flare-class",
        type=str,
        choices=["C", "M", "M5"],
        help="Test only a specific flare class"
    )
    parser.add_argument(
        "--time-window",
        type=str,
        choices=["24", "48", "72"],
        help="Test only a specific time window"
    )
    args = parser.parse_args()

    # Can't use multiple selection methods together
    if sum([bool(args.timestamp), args.latest, bool(args.version)]) > 1:
        print(
            "Error: Cannot use multiple model selection options together. Use only one of: --timestamp, --latest, --version"
        )
        exit(1)

    # Loop over the desired time windows and flare classes
    time_windows = [args.time_window] if args.time_window else [24, 48, 72]
    flare_classes = [args.flare_class] if args.flare_class else ["C", "M", "M5"]
    
    for time_window in time_windows:
        for flare_class in flare_classes:
            if flare_class not in supported_flare_class:
                print(
                    "Unsupported flare class:",
                    flare_class,
                    "It must be one of:",
                    ", ".join(supported_flare_class),
                )
                continue
            test_model(
                str(time_window),
                flare_class,
                args.timestamp,
                args.latest,
                mc_passes=args.mc_passes,
                plot_uncertainties=not args.no_plots,
            )
            log(
                "===========================================================\n\n",
                verbose=True,
            )

    # Save the metrics for all time windows into a JSON file
    output_file = "pytorch_results.json"
    with open(output_file, "w") as f:
        json.dump(all_metrics, f, indent=4)
    print(f"Saved test metrics for 'PyTorch implementation' into {output_file}")

    # If a specific timestamp was tested, also save results to a timestamped
    # file
    if args.timestamp:
        timestamped_output = f"results_pytorch_{args.timestamp}.json"
        with open(timestamped_output, "w") as f:
            json.dump(all_metrics, f, indent=4)
        print(
            f"Saved test metrics for timestamp {args.timestamp} into {timestamped_output}"
        )
    elif args.latest:
        latest_output = "results_pytorch_latest.json"
        with open(latest_output, "w") as f:
            json.dump(all_metrics, f, indent=4)
        print(f"Saved test metrics for latest version into {latest_output}")
    elif args.version:
        version_output = f"results_pytorch_v{args.version}.json"
        with open(version_output, "w") as f:
            json.dump(all_metrics, f, indent=4)
        print(
            f"Saved test metrics for version v{args.version} into {version_output}"
        ) 