#!/usr/bin/env python
'''
Ablation Study for SolarKnowledge Model
This script systematically tests different model configurations for the 24h M-class prediction task
and reports their TSS values for comparison.

Author: Antanas Zilinskas
'''

import sys
import os
import time
import json
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
from pathlib import Path

# Custom JSON encoder to handle NumPy types


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, tf.Tensor):
            return obj.numpy().tolist()
        return super(NumpyEncoder, self).default(obj)


# Get the project root directory
project_root = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__))))
print(f"Project root: {project_root}")

# Add project root to path
sys.path.insert(0, project_root)

# Try alternative import paths if needed
try:
    from models.SolarKnowledge_model import SolarKnowledge
    from models.utils import get_training_data, get_testing_data, data_transform, log
    print("Successfully imported model and utilities")
except ImportError as e:
    print(f"Error importing modules: {e}")

    # Try alternative import paths
    try:
        sys.path.append(os.path.join(project_root, 'models'))
        # Try direct import from models.utils
        from models.utils import get_training_data, get_testing_data, data_transform, log
        print("Successfully imported utilities from models.utils")
    except ImportError as e2:
        print(f"Still can't import modules: {e2}")
        print(f"Python path: {sys.path}")
        print("Make sure the models.utils module is properly installed")

        # List available modules in models directory if it exists
        models_dir = os.path.join(project_root, 'models')
        if os.path.exists(models_dir):
            print(f"Contents of models directory:")
            for file in os.listdir(models_dir):
                print(f"  - {file}")
        sys.exit(1)

# Configure TensorFlow to use Memory Growth if GPU is available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"Memory growth enabled for {len(physical_devices)} GPU devices")
    except Exception as e:
        print(f"Error configuring GPU: {e}")

# Use float32 for all operations
tf.keras.mixed_precision.set_global_policy('float32')

# Ablation study configurations
CONFIGURATIONS = [
    {
        'name': 'Full model',
        'description': '(Conv1D + BN) + LSTM + 4 TEBs + heavy dropout',
        'use_conv': True,
        'use_lstm': True,
        'teb_layers': 4,
        'dropout_rate': 0.2,
        'use_class_weighting': True
    },
    {
        'name': 'No LSTM',
        'description': 'only conv + BN, then TEBs',
        'use_conv': True,
        'use_lstm': False,
        'teb_layers': 4,
        'dropout_rate': 0.2,
        'use_class_weighting': True
    },
    {
        'name': 'No conv',
        'description': 'BN then LSTM',
        'use_conv': False,
        'use_lstm': True,
        'teb_layers': 4,
        'dropout_rate': 0.2,
        'use_class_weighting': True
    },
    {
        'name': 'Reduced TEBs',
        'description': '2 layers instead of 4',
        'use_conv': True,
        'use_lstm': True,
        'teb_layers': 2,
        'dropout_rate': 0.2,
        'use_class_weighting': True
    },
    {
        'name': 'No class weighting',
        'description': 'No class weights for imbalanced data',
        'use_conv': True,
        'use_lstm': True,
        'teb_layers': 4,
        'dropout_rate': 0.2,
        'use_class_weighting': False
    },
    {
        'name': 'Light dropout',
        'description': 'dropout = 0.1 (lighter)',
        'use_conv': True,
        'use_lstm': True,
        'teb_layers': 4,
        'dropout_rate': 0.1,
        'use_class_weighting': True
    }
]


class AblationModel(SolarKnowledge):
    """Extended SolarKnowledge model with ablation options"""

    def build_ablation_model(self, input_shape,
                             embed_dim=128,
                             num_heads=4,
                             ff_dim=256,
                             num_transformer_blocks=4,
                             dropout_rate=0.2,
                             num_classes=2,
                             use_conv=True,
                             use_lstm=True,
                             use_class_weighting=True):
        """Build model with ablation options"""
        # Store architecture details
        self.metadata["model_architecture"] = {
            "num_transformer_blocks": num_transformer_blocks,
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "ff_dim": ff_dim,
            "dropout_rate": dropout_rate,
            "use_conv": use_conv,
            "use_lstm": use_lstm,
            "use_class_weighting": use_class_weighting
        }

        inputs = tf.keras.layers.Input(shape=input_shape)
        self.input_tensor = inputs
        x = inputs

        # Conv1D + BatchNormalization feature extraction
        if use_conv:
            x = tf.keras.layers.Conv1D(
                filters=embed_dim, kernel_size=3, padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)

        # LSTM layer
        if use_lstm:
            x = tf.keras.layers.LSTM(embed_dim, return_sequences=True)(x)

        # Dense embedding if no conv or LSTM was used
        if not use_conv and not use_lstm:
            x = tf.keras.layers.Dense(embed_dim)(x)
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

        # Apply dropout after input processing
        x = tf.keras.layers.Dropout(dropout_rate)(x)

        # Transformer Encoder Blocks (TEBs)
        for i in range(num_transformer_blocks):
            x = self.transformer_block(
                x, embed_dim, num_heads, ff_dim, dropout_rate)

        # Global pooling and classification head
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.use_class_weighting = use_class_weighting
        return self.model

    def transformer_block(
            self,
            inputs,
            embed_dim,
            num_heads,
            ff_dim,
            dropout_rate):
        """Transformer encoder block implementation"""
        # Multi-head attention
        attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim //
            num_heads)(
            inputs,
            inputs)
        attention = tf.keras.layers.Dropout(dropout_rate)(attention)
        # Add & normalize (residual connection)
        attention = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)(inputs + attention)

        # Feed-forward network
        ffn = tf.keras.layers.Dense(ff_dim, activation='relu')(attention)
        ffn = tf.keras.layers.Dropout(dropout_rate)(ffn)
        ffn = tf.keras.layers.Dense(embed_dim)(ffn)
        ffn = tf.keras.layers.Dropout(dropout_rate)(ffn)
        # Add & normalize (residual connection)
        return tf.keras.layers.LayerNormalization(
            epsilon=1e-6)(attention + ffn)

    def compile(
            self,
            loss='categorical_crossentropy',
            metrics=['accuracy'],
            learning_rate=1e-4):
        """Compile the model with appropriate settings"""
        self.metadata["training"] = {
            "optimizer": "Adam",
            "learning_rate": learning_rate,
            "loss": loss,
            "metrics": metrics,
            "use_class_weighting": self.use_class_weighting
        }

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )


def calculate_tss(y_true, y_pred):
    """Calculate True Skill Statistic (TSS) from binary predictions"""
    cm = confusion_matrix(y_true, y_pred)
    # For binary classification: sensitivity = recall for positive class (1)
    sensitivity = cm[1, 1] / \
        (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
    # Specificity = recall for negative class (0)
    specificity = cm[0, 0] / \
        (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
    # TSS = sensitivity + specificity - 1
    return sensitivity + specificity - 1


def calculate_hss(y_true, y_pred):
    """Calculate Heidke Skill Score (HSS) from binary predictions
    HSS measures the fractional improvement of the forecast over random chance"""
    cm = confusion_matrix(y_true, y_pred)
    a = cm[0, 0]  # True negatives
    b = cm[0, 1]  # False positives
    c = cm[1, 0]  # False negatives
    d = cm[1, 1]  # True positives

    num = 2 * (a * d - b * c)
    den = (a + c) * (c + d) + (a + b) * (b + d)

    return num / den if den > 0 else 0


def calculate_metrics(y_true, y_pred_probs):
    """Calculate comprehensive metrics for binary classification"""
    # Convert probabilities to binary predictions
    y_pred_classes = np.argmax(y_pred_probs, axis=1)

    # If y_true is one-hot encoded, convert to class indices
    if len(np.array(y_true).shape) > 1 and np.array(y_true).shape[1] > 1:
        y_true_classes = np.argmax(np.array(y_true), axis=1)
    else:
        y_true_classes = np.array(y_true)

    # Calculate metrics
    metrics = {}

    # Basic metrics
    metrics['accuracy'] = np.mean(y_true_classes == y_pred_classes)
    metrics['precision'] = precision_score(
        y_true_classes, y_pred_classes, zero_division=0)
    metrics['recall'] = recall_score(
        y_true_classes, y_pred_classes, zero_division=0)
    metrics['f1'] = f1_score(y_true_classes, y_pred_classes, zero_division=0)

    # Skill scores
    metrics['tss'] = calculate_tss(y_true_classes, y_pred_classes)
    metrics['hss'] = calculate_hss(y_true_classes, y_pred_classes)

    # AUC-ROC (requires probabilities of positive class)
    try:
        metrics['auc'] = roc_auc_score(y_true_classes, y_pred_probs[:, 1])
    except BaseException:
        metrics['auc'] = 0  # In case of single-class predictions

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true_classes, y_pred_classes).ravel()
    metrics['tn'] = tn
    metrics['fp'] = fp
    metrics['fn'] = fn
    metrics['tp'] = tp

    return metrics


def plot_learning_curves(history, fold_idx, config_name, output_dir):
    """Plot and save learning curves to detect overfitting"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 8))

    # Plot training & validation loss
    plt.subplot(2, 1, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Loss Curves - {config_name} (Fold {fold_idx+1})')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    # Plot training & validation accuracy
    plt.subplot(2, 1, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Accuracy Curves - {config_name} (Fold {fold_idx+1})')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    # Save figure
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            output_dir,
            f'{config_name.replace(" ", "_")}_fold_{fold_idx+1}_learning_curves.png'))
    plt.close()


def assess_overfitting(history):
    """Assess overfitting by comparing final training and validation metrics"""
    # Calculate average of last 5 epochs to smooth fluctuations
    n_last = min(5, len(history.history['loss']))

    # Loss gap
    train_loss = np.mean(history.history['loss'][-n_last:])
    val_loss = np.mean(history.history['val_loss'][-n_last:])
    loss_gap = train_loss - val_loss

    # Accuracy gap
    train_acc = np.mean(history.history['accuracy'][-n_last:])
    val_acc = np.mean(history.history['val_accuracy'][-n_last:])
    acc_gap = train_acc - val_acc

    # Detect if validation loss started increasing
    val_losses = history.history['val_loss']
    min_val_loss_idx = np.argmin(val_losses)
    early_stopping_triggered = min_val_loss_idx < len(val_losses) - 1

    return {
        'train_loss': train_loss,
        'val_loss': val_loss,
        'loss_gap': loss_gap,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'acc_gap': acc_gap,
        'min_val_loss_epoch': min_val_loss_idx + 1,
        'early_stopping_triggered': early_stopping_triggered,
        # Threshold for significant overfitting
        'overfitting_detected': acc_gap > 0.05 or early_stopping_triggered
    }


def run_ablation_study(
        time_window="24",
        flare_class="M",
        epochs=50,
        n_folds=5):
    """Run ablation study with k-fold cross-validation for all configurations and report results"""
    all_results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results/ablation", timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Load data once
    print(f"Loading data for {time_window}h {flare_class}-class prediction...")
    try:
        X_train_all, y_train_all = get_training_data(time_window, flare_class)
        y_train_tr_all = data_transform(y_train_all)
        X_test, y_test = get_testing_data(time_window, flare_class)

        # Ensure data is in float32 format
        X_train_all = tf.cast(X_train_all, tf.float32)
        y_train_tr_all = tf.cast(y_train_tr_all, tf.float32)
        X_test = tf.cast(X_test, tf.float32)

        # Convert TensorFlow tensors to NumPy arrays for KFold indexing
        X_train_all_np = X_train_all.numpy()
        y_train_tr_all_np = y_train_tr_all.numpy()

        print(f"Data loaded successfully:")
        print(
            f"X_train shape: {X_train_all.shape}, y_train shape: {y_train_tr_all.shape}")
        print(
            f"X_test shape: {X_test.shape}, y_test shape: {np.array(y_test).shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return []

    # Set up k-fold cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    # For each configuration
    for config_idx, config in enumerate(CONFIGURATIONS):
        print(
            f"\n[{config_idx+1}/{len(CONFIGURATIONS)}] Testing configuration: {config['name']}")
        print(f"Description: {config['description']}")

        config_results = {
            'config_name': config['name'],
            'description': config['description'],
            'params': config.copy(),
            'folds': [],
            'test_metrics': {},
            'timestamp': timestamp
        }

        # Run k-fold cross-validation
        fold_train_metrics = []
        fold_val_metrics = []
        fold_models = []
        fold_histories = []

        for fold_idx, (train_idx, val_idx) in enumerate(
                kf.split(X_train_all_np)):
            print(f"\nTraining fold {fold_idx+1}/{n_folds}")

            # Split data using NumPy arrays
            X_train, X_val = X_train_all_np[train_idx], X_train_all_np[val_idx]
            y_train, y_val = y_train_tr_all_np[train_idx], y_train_tr_all_np[val_idx]

            # Convert back to TensorFlow tensors if needed
            X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
            X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
            y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
            y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)

            # Calculate class weights for imbalanced data
            class_counts = np.sum(y_train, axis=0)
            n_samples = len(y_train)
            class_weights = {
                0: n_samples / (2 * class_counts[0]) if class_counts[0] > 0 else 1.0,
                1: n_samples / (2 * class_counts[1]) if class_counts[1] > 0 else 1.0
            }

            print(f"Class weights: {class_weights}")

            # Get input shape from the training data
            input_shape = (X_train.shape[1], X_train.shape[2])

            # Create model with the specific configuration
            model = AblationModel(
                early_stopping_patience=5,
                description=f"Ablation: {config['description']} - Fold {fold_idx+1}")

            # Build the model with ablation options
            try:
                model.build_ablation_model(
                    input_shape=input_shape,
                    embed_dim=128,
                    num_heads=4,
                    ff_dim=256,
                    num_transformer_blocks=config['teb_layers'],
                    dropout_rate=config['dropout_rate'],
                    num_classes=2,
                    use_conv=config['use_conv'],
                    use_lstm=config['use_lstm'],
                    use_class_weighting=config['use_class_weighting']
                )

                # Compile the model
                model.compile()

                # Add callbacks
                callbacks = [
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=5,
                        restore_best_weights=True),
                    tf.keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=3,
                        verbose=1,
                        min_lr=1e-6),
                    tf.keras.callbacks.CSVLogger(
                        os.path.join(
                            output_dir,
                            f"{config['name'].replace(' ', '_')}_fold_{fold_idx+1}_training_log.csv"))]

            except Exception as e:
                print(f"Error building model: {e}")
                continue

            # Training with or without class weights
            try:
                if config['use_class_weighting']:
                    print("Training with class weights")
                    history = model.model.fit(
                        X_train, y_train,
                        epochs=epochs,
                        verbose=1,
                        batch_size=512,
                        validation_data=(X_val, y_val),
                        class_weight=class_weights,
                        callbacks=callbacks
                    )
                else:
                    print("Training without class weights")
                    history = model.model.fit(
                        X_train, y_train,
                        epochs=epochs,
                        verbose=1,
                        batch_size=512,
                        validation_data=(X_val, y_val),
                        callbacks=callbacks
                    )

                # Plot and save learning curves
                plot_learning_curves(
                    history, fold_idx, config['name'], output_dir)

                # Assess overfitting
                overfitting_assessment = assess_overfitting(history)
                print(f"Overfitting assessment for fold {fold_idx+1}:")
                for key, value in overfitting_assessment.items():
                    print(f"  {key}: {value}")

                # Save history for later analysis
                fold_histories.append(history.history)

            except Exception as e:
                print(f"Error during training: {e}")
                continue

            # Evaluate on validation set
            y_val_pred = model.model.predict(X_val)
            val_metrics = calculate_metrics(y_val, y_val_pred)

            # Evaluate on training set (to check for overfitting)
            y_train_pred = model.model.predict(X_train)
            train_metrics = calculate_metrics(y_train, y_train_pred)

            # Store results for this fold
            fold_results = {
                'fold': fold_idx + 1,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'overfitting_assessment': overfitting_assessment,
                'model_summary': str(model.model.summary())
            }

            config_results['folds'].append(fold_results)
            fold_train_metrics.append(train_metrics)
            fold_val_metrics.append(val_metrics)
            fold_models.append(model)

            print(f"Fold {fold_idx+1} validation metrics:")
            for metric_name, metric_value in val_metrics.items():
                print(f"  {metric_name}: {metric_value:.4f}")

        # Calculate average metrics across folds
        if fold_val_metrics:
            avg_val_metrics = {}
            for metric in fold_val_metrics[0].keys():
                if metric not in ['tn', 'fp', 'fn',
                                  'tp']:  # Skip confusion matrix counts
                    avg_val_metrics[metric] = np.mean(
                        [fold[metric] for fold in fold_val_metrics])
                    std_val_metrics = np.std(
                        [fold[metric] for fold in fold_val_metrics])
                    avg_val_metrics[f"{metric}_std"] = std_val_metrics

            config_results['avg_val_metrics'] = avg_val_metrics

            # Print average validation metrics
            print(f"\nAverage validation metrics for {config['name']}:")
            for metric_name, metric_value in avg_val_metrics.items():
                if not metric_name.endswith('_std'):
                    print(
                        f"  {metric_name}: {metric_value:.4f} ± {avg_val_metrics[f'{metric_name}_std']:.4f}")

            # Find best fold based on validation TSS
            best_fold_idx = np.argmax(
                [fold['val_metrics']['tss'] for fold in config_results['folds']])
            config_results['best_fold'] = best_fold_idx + 1
            print(
                f"Best fold: {config_results['best_fold']} with TSS: {config_results['folds'][best_fold_idx]['val_metrics']['tss']:.4f}")

            # Use best model to evaluate on test set
            best_model = fold_models[best_fold_idx]
            y_test_pred = best_model.model.predict(X_test)
            test_metrics = calculate_metrics(y_test, y_test_pred)
            config_results['test_metrics'] = test_metrics

            # Save best model
            best_model_path = os.path.join(
                output_dir, f"{config['name'].replace(' ', '_')}_best_model")
            try:
                best_model.model.save(best_model_path)
                print(f"Best model saved to {best_model_path}")
            except Exception as e:
                print(f"Error saving model: {e}")

            # Print test metrics for best model
            print(f"\nTest metrics for best model ({config['name']}):")
            for metric_name, metric_value in test_metrics.items():
                if metric_name not in ['tn', 'fp', 'fn', 'tp']:
                    print(f"  {metric_name}: {metric_value:.4f}")

        all_results.append(config_results)

        # Save intermediate results after each configuration
        with open(os.path.join(output_dir, f"ablation_results_intermediate_{config_idx+1}.json"), 'w') as f:
            json.dump(config_results, f, indent=2, cls=NumpyEncoder)

    # Compare configurations using statistical tests
    if all_results:
        print("\n=== Statistical Comparison of Configurations ===")

        # Prepare data for statistical tests
        tss_by_config = {}
        for result in all_results:
            config_name = result['config_name']
            tss_values = [fold['val_metrics']['tss']
                          for fold in result['folds']]
            tss_by_config[config_name] = tss_values

        # Perform paired t-tests between the best configuration and others
        baseline_config = 'Full model'
        if baseline_config in tss_by_config:
            baseline_tss = tss_by_config[baseline_config]
            for config_name, tss_values in tss_by_config.items():
                if config_name != baseline_config:
                    # Ensure we compare only the common folds
                    min_folds = min(len(baseline_tss), len(tss_values))
                    if min_folds > 1:  # Need at least 2 samples for t-test
                        t_stat, p_value = stats.ttest_rel(
                            baseline_tss[:min_folds], tss_values[:min_folds])
                        significance = "significant" if p_value < 0.05 else "not significant"
                        print(
                            f"{baseline_config} vs {config_name}: p={p_value:.4f} ({significance})")

    # Save final comprehensive results
    output_file = os.path.join(
        output_dir,
        f"ablation_results_{time_window}h_{flare_class}_class_comprehensive.json")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)

    # Create summary table for LaTeX
    create_summary_table(all_results, output_dir, time_window, flare_class)

    # Generate comparison plots
    create_comparison_plots(all_results, output_dir, time_window, flare_class)

    return all_results


def create_summary_table(results, output_dir, time_window, flare_class):
    """Create and save a summary table of results for LaTeX"""
    table_data = []

    for result in results:
        config_name = result['config_name']
        description = result['description']

        # Get metrics from best model on test set
        test_metrics = result.get('test_metrics', {})
        tss = test_metrics.get('tss', float('nan'))
        hss = test_metrics.get('hss', float('nan'))
        auc = test_metrics.get('auc', float('nan'))
        f1 = test_metrics.get('f1', float('nan'))

        # Get validation metrics (average across folds)
        avg_val_metrics = result.get('avg_val_metrics', {})
        val_tss = avg_val_metrics.get('tss', float('nan'))
        val_tss_std = avg_val_metrics.get('tss_std', float('nan'))

        # Check for overfitting
        train_val_gap = []
        for fold in result.get('folds', []):
            if 'overfitting_assessment' in fold:
                train_val_gap.append(
                    fold['overfitting_assessment'].get(
                        'acc_gap', 0))
        avg_gap = np.mean(train_val_gap) if train_val_gap else float('nan')

        row = {
            'Configuration': config_name,
            'Description': description,
            'Val TSS': f"{val_tss:.3f} ± {val_tss_std:.3f}",
            'Test TSS': f"{tss:.3f}",
            'Test HSS': f"{hss:.3f}",
            'Test AUC': f"{auc:.3f}",
            'Test F1': f"{f1:.3f}",
            'Train-Val Gap': f"{avg_gap:.3f}"
        }
        table_data.append(row)

    # Sort by test TSS (descending)
    table_data = sorted(
        table_data,
        key=lambda x: float(
            x['Test TSS'].split()[0]),
        reverse=True)

    # Write LaTeX table
    with open(os.path.join(output_dir, f"ablation_summary_{time_window}h_{flare_class}_class.tex"), 'w') as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write(
            "\\caption{Ablation Study Results for " +
            f"{time_window}h {flare_class}-class Prediction" +
            "}\n")
        f.write("\\label{tab:ablation_results}\n")
        f.write("\\begin{tabular}{p{3cm}|p{3.5cm}|c|c|c|c|c|c}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Configuration} & \\textbf{Description} & \\textbf{Val TSS} & \\textbf{Test TSS} & \\textbf{Test HSS} & \\textbf{Test AUC} & \\textbf{Test F1} & \\textbf{Overfit} \\\\\n")
        f.write("\\midrule\n")

        for row in table_data:
            line = f"{row['Configuration']} & {row['Description']} & {row['Val TSS']} & {row['Test TSS']} & {row['Test HSS']} & {row['Test AUC']} & {row['Test F1']} & {row['Train-Val Gap']} \\\\\n"
            f.write(line)

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    # Also create a CSV version
    pd.DataFrame(table_data).to_csv(
        os.path.join(
            output_dir,
            f"ablation_summary_{time_window}h_{flare_class}_class.csv"),
        index=False)

    # Print the table to console
    print("\n=== ABLATION STUDY SUMMARY ===")
    headers = list(table_data[0].keys())
    row_format = "{:<15} {:<20} {:<15} {:<10} {:<10} {:<10} {:<10} {:<10}"
    print(row_format.format(*headers))
    print("-" * 100)
    for row in table_data:
        values = [row[h] for h in headers]
        print(row_format.format(*values))


def create_comparison_plots(results, output_dir, time_window, flare_class):
    """Create comparison plots for different configurations"""
    if not results:
        return

    # Extract configuration names and metrics
    config_names = [r['config_name'] for r in results]
    test_tss = [r.get('test_metrics', {}).get('tss', 0) for r in results]
    val_tss = [r.get('avg_val_metrics', {}).get('tss', 0) for r in results]
    val_tss_std = [
        r.get(
            'avg_val_metrics',
            {}).get(
            'tss_std',
            0) for r in results]

    test_hss = [r.get('test_metrics', {}).get('hss', 0) for r in results]
    test_auc = [r.get('test_metrics', {}).get('auc', 0) for r in results]
    test_f1 = [r.get('test_metrics', {}).get('f1', 0) for r in results]

    # Comparison of TSS (validation vs test)
    plt.figure(figsize=(12, 8))
    x = np.arange(len(config_names))
    width = 0.35

    plt.bar(x - width / 2, val_tss, width, yerr=val_tss_std,
            label='Validation TSS', color='royalblue', alpha=0.7)
    plt.bar(x + width / 2, test_tss, width,
            label='Test TSS', color='darkorange', alpha=0.7)

    plt.xlabel('Configuration', fontsize=14)
    plt.ylabel('TSS', fontsize=14)
    plt.title(
        f'Validation vs Test TSS for {time_window}h {flare_class}-class Prediction',
        fontsize=16)
    plt.xticks(x, config_names, rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim([0, 1.0])
    plt.legend(fontsize=12)
    plt.tight_layout()

    # Add TSS values on top of each bar
    for i, v in enumerate(val_tss):
        plt.text(i - width / 2, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
    for i, v in enumerate(test_tss):
        plt.text(i + width / 2, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)

    plt.savefig(
        os.path.join(
            output_dir,
            f"tss_comparison_{time_window}h_{flare_class}_class.png"),
        dpi=300)
    plt.close()

    # Comparison of different metrics on test set
    plt.figure(figsize=(12, 8))
    x = np.arange(len(config_names))
    width = 0.2

    plt.bar(x - width * 1.5, test_tss, width, label='TSS', color='royalblue')
    plt.bar(x - width / 2, test_hss, width, label='HSS', color='darkorange')
    plt.bar(x + width / 2, test_auc, width, label='AUC', color='forestgreen')
    plt.bar(x + width * 1.5, test_f1, width, label='F1', color='firebrick')

    plt.xlabel('Configuration', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.title(
        f'Test Metrics for {time_window}h {flare_class}-class Prediction',
        fontsize=16)
    plt.xticks(x, config_names, rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim([0, 1.0])
    plt.legend(fontsize=12)
    plt.tight_layout()

    # Add score values on top of each bar
    for i, v in enumerate(test_tss):
        plt.text(
            i - width * 1.5,
            v + 0.02,
            f'{v:.3f}',
            ha='center',
            fontsize=10)
    for i, v in enumerate(test_hss):
        plt.text(i - width / 2, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
    for i, v in enumerate(test_auc):
        plt.text(i + width / 2, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
    for i, v in enumerate(test_f1):
        plt.text(
            i + width * 1.5,
            v + 0.02,
            f'{v:.3f}',
            ha='center',
            fontsize=10)

    plt.savefig(
        os.path.join(
            output_dir,
            f"metrics_comparison_{time_window}h_{flare_class}_class.png"),
        dpi=300)
    plt.close()

    # Create ablation feature heatmap
    create_ablation_heatmap(results, output_dir, time_window, flare_class)


def create_ablation_heatmap(results, output_dir, time_window, flare_class):
    """Create a heatmap visualization of ablation study results"""
    # Extract configuration details and TSS values
    config_names = []
    tss_values = []
    features = ['LSTM', 'Conv1D', 'TEBs=4', 'Class weighting', 'Heavy dropout']

    # Prepare data for heatmap
    feature_matrix = []

    for result in results:
        config_name = result['config_name']
        config_names.append(config_name)

        # Get test TSS value
        tss = result.get('test_metrics', {}).get('tss', 0)
        tss_values.append(tss)

        # Determine which features are present
        params = result.get('params', {})
        row = [
            1 if params.get('use_lstm', False) else 0,
            1 if params.get('use_conv', False) else 0,
            1 if params.get('teb_layers', 0) == 4 else 0,
            1 if params.get('use_class_weighting', False) else 0,
            # 1 for heavy dropout (0.2), 0 for light (0.1)
            1 if params.get('dropout_rate', 0) == 0.2 else 0
        ]
        feature_matrix.append(row)

    # Convert to numpy array for heatmap
    feature_matrix = np.array(feature_matrix)

    # Create figure with custom size
    plt.figure(figsize=(14, 10))

    # Create heatmap with improved readability
    cmap = plt.cm.Blues
    heatmap = plt.pcolormesh(
        feature_matrix,
        cmap=cmap,
        edgecolors='white',
        linewidth=2)

    # Add feature labels on top
    plt.xticks(np.arange(len(features)) + 0.5, features, fontsize=14)

    # Add configuration names on left with TSS values
    config_labels = [
        f"{name}\n(TSS: {tss:.3f})" for name,
        tss in zip(
            config_names,
            tss_values)]
    plt.yticks(np.arange(len(config_names)) + 0.5, config_labels, fontsize=14)

    # Add title and labels
    plt.title(
        f'Ablation Study Features for {time_window}h {flare_class}-class Prediction',
        fontsize=16)

    # Add colorbar
    cbar = plt.colorbar(heatmap)
    cbar.set_label('Feature Present', fontsize=14)

    # Tight layout
    plt.tight_layout()

    # Save figure with high resolution
    plt.savefig(
        os.path.join(
            output_dir,
            f"ablation_feature_heatmap_{time_window}h_{flare_class}_class.png"),
        dpi=300,
        bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    print("Starting ablation study...")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")

    # Set up proper directories
    for directory in ['results', 'results/ablation']:
        Path(directory).mkdir(exist_ok=True)

    # Run ablation study for 24h M-class prediction with 5-fold
    # cross-validation
    results = run_ablation_study(
        time_window="24",
        flare_class="M",
        epochs=50,
        n_folds=5)

    if results:
        # Final summary already created by run_ablation_study
        print("\nAblation study completed successfully.")
    else:
        print("Ablation study did not produce any results. Check the errors above.")
