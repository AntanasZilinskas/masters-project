#!/usr/bin/env python
"""
Demonstration of the complete EVEREST model and its components

This script shows how to:
1. Create an EVEREST model with all components
2. Train the model on synthetic data
3. Generate predictions with uncertainty
4. Visualize model outputs and confidence
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc

# Import our custom modules
from complete_everest import EVEREST
from utils import get_training_data, get_testing_data
from metrics import CategoricalTSSMetric, calculate_tss_from_cm

# Set random seed for reproducibility
tf.keras.utils.set_random_seed(42)
np.random.seed(42)

# Configure GPU memory growth if available
for g in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except:
        pass

def demo_model_components():
    """Demonstrate individual EVEREST components."""
    # Create synthetic data
    seq_len, feat_dim = 144, 14  # 24 hours (6 samples per hour), 14 features
    X = np.random.normal(0, 1, (10, seq_len, feat_dim)).astype(np.float32)
    
    # Create sample data
    print("\n=== Demonstrating Individual Components ===")
    
    try:
        # Demonstrate Performer (linear attention)
        from performer_custom import Performer
        
        print("\n--- Performer (Linear Attention) ---")
        performer = Performer(num_heads=4, key_dim=32, feature_dim=64)
        # Build model
        input_tensor = tf.keras.layers.Input((seq_len, feat_dim))
        output_tensor = performer(input_tensor)
        model = tf.keras.Model(input_tensor, output_tensor)
        # Process sample
        output = model(X)
        print(f"Input shape: {X.shape}, Output shape: {output.shape}")
        print("Linear attention successful! Complexity is O(L) instead of O(L²)")
    except ImportError:
        print("Performer module not found")
    
    try:
        # Demonstrate Retentive Layer
        from retentive_layer import RetentiveLayer
        
        print("\n--- Retentive Layer ---")
        retention = RetentiveLayer(decay_factor=0.95)
        # Build model
        input_tensor = tf.keras.layers.Input((seq_len, feat_dim))
        output_tensor = retention(input_tensor)
        model = tf.keras.Model(input_tensor, output_tensor)
        # Process sample
        output = model(X)
        print(f"Input shape: {X.shape}, Output shape: {output.shape}")
        print(f"Retention decay factor: {retention.decay.numpy():.4f}")
    except ImportError:
        print("RetentiveLayer module not found")
    
    try:
        # Demonstrate focal loss
        from focal_loss import ClassBalancedFocalLoss
        
        print("\n--- Class-Balanced Focal Loss ---")
        # Create dummy labels
        y = np.zeros((10, 2))
        y[0:2, 1] = 1  # 2 positive, 8 negative
        
        # Create loss function
        focal_loss = ClassBalancedFocalLoss(beta=0.999, gamma=2.0)
        # Create logits
        logits = np.random.normal(0, 1, (10, 2))
        probas = tf.nn.softmax(logits, axis=-1).numpy()
        
        # Calculate loss
        loss = focal_loss(y, logits)
        print(f"Class counts: Positive={np.sum(y[:,1])}, Negative={np.sum(y[:,0])}")
        print(f"Focal loss: {loss.numpy():.4f}")
    except ImportError:
        print("FocalLoss module not found")
    
    try:
        # Demonstrate multi-scale tokenizer
        from multi_scale_tokenizer import MultiScaleTokenizer
        
        print("\n--- Multi-Scale Tokenizer ---")
        tokenizer = MultiScaleTokenizer()
        # Build model
        input_tensor = tf.keras.layers.Input((seq_len, feat_dim))
        output_tensor = tokenizer(input_tensor)
        model = tf.keras.Model(input_tensor, output_tensor)
        # Process sample
        output = model(X)
        print(f"Input shape: {X.shape}, Output shape: {output.shape}")
        print("Multi-scale tokenization expands features by 3x by incorporating 10min, 1h, and 3h views")
    except ImportError:
        print("MultiScaleTokenizer module not found")
    
    try:
        # Demonstrate evidential head
        from evidential_head import nig_head, evidential_nll
        
        print("\n--- Evidential Uncertainty ---")
        # Build model with evidential head
        input_tensor = tf.keras.layers.Input((feat_dim,))
        output_tensor = nig_head(input_tensor)
        model = tf.keras.Model(input_tensor, output_tensor)
        # Process sample
        x_flat = X[:, 0, :]  # Take first timestep
        evid_params = model(x_flat)
        print(f"Input shape: {x_flat.shape}, Evidential parameters shape: {evid_params.shape}")
        print("Evidential parameters: [μ, ν, α, β]")
        mu, v, alpha, beta = np.split(evid_params.numpy(), 4, axis=1)
        print(f"  μ mean: {np.mean(mu):.4f}")
        print(f"  ν mean: {np.mean(v):.4f}")
        print(f"  α mean: {np.mean(alpha):.4f}")
        print(f"  β mean: {np.mean(beta):.4f}")
        
        # Calculate epistemic and aleatoric uncertainty
        epistemic = beta / (v * (alpha - 1) + 1e-8)
        aleatoric = beta / (alpha - 1 + 1e-8)
        print(f"Epistemic uncertainty: {np.mean(epistemic):.4f}")
        print(f"Aleatoric uncertainty: {np.mean(aleatoric):.4f}")
    except ImportError:
        print("EvidentialHead module not found")
    
    try:
        # Demonstrate EVT head
        from evt_head import gpd_head, evt_loss
        
        print("\n--- Extreme Value Theory (EVT) ---")
        # Build model with EVT head
        input_tensor = tf.keras.layers.Input((feat_dim,))
        output_tensor = gpd_head(input_tensor)
        model = tf.keras.Model(input_tensor, output_tensor)
        # Process sample
        x_flat = X[:, 0, :]  # Take first timestep
        evt_params = model(x_flat)
        print(f"Input shape: {x_flat.shape}, EVT parameters shape: {evt_params.shape}")
        print("EVT parameters: [ξ (shape), σ (scale)]")
        shape, scale = np.split(evt_params.numpy(), 2, axis=1)
        print(f"  ξ mean: {np.mean(shape):.4f}")
        print(f"  σ mean: {np.mean(scale):.4f}")
    except ImportError:
        print("EVTHead module not found")
    
    try:
        # Demonstrate conformal calibration
        from conformal_calibration import ConformalCalibrator
        
        print("\n--- Conformal Calibration ---")
        # Create synthetic predictions and labels
        probs = np.random.uniform(0, 1, 100)
        y_true = np.random.randint(0, 2, 100)
        
        # Create calibrator
        calibrator = ConformalCalibrator(alpha=0.1)
        # Calibrate
        threshold = calibrator.calibrate(probs, y_true)
        print(f"Calibrated threshold: {threshold:.4f}")
        
        # Generate prediction sets
        pred_sets = calibrator.predict_sets(probs)
        coverage = np.mean((y_true == 1) & pred_sets['sets'] | (y_true == 0) & ~pred_sets['sets'])
        print(f"Conformal coverage: {coverage:.4f}")
    except ImportError:
        print("ConformalCalibration module not found")

def train_and_evaluate_model():
    """Train and evaluate a complete EVEREST model."""
    print("\n=== Training and Evaluating Complete EVEREST Model ===")
    
    # Set parameters
    flare_class = "M5"
    time_window = "24"
    
    # Load training and testing data
    X_train, y_train = get_training_data(time_window, flare_class)
    X_test, y_test = get_testing_data(time_window, flare_class)
    
    # Convert to one-hot if needed
    if len(y_train.shape) == 1 or y_train.shape[1] == 1:
        y_train = tf.keras.utils.to_categorical(y_train, 2)
    if len(y_test.shape) == 1 or y_test.shape[1] == 1:
        y_test = tf.keras.utils.to_categorical(y_test, 2)
    
    # Print dataset info
    print(f"Training data: {X_train.shape}, Positive samples: {np.sum(y_train[:, 1])}")
    print(f"Testing data: {X_test.shape}, Positive samples: {np.sum(y_test[:, 1])}")
    
    # Create model
    model = EVEREST(
        use_evidential=True,
        use_evt=True,
        use_retentive=True,
        use_multi_scale=True
    )
    
    # Build and compile model
    model.build_base_model(
        input_shape=X_train.shape[1:],
        embed_dim=64,  # Smaller for demo
        num_heads=2,   # Smaller for demo
        ff_dim=128,    # Smaller for demo
        n_blocks=2,    # Smaller for demo
        dropout=0.3
    )
    
    # Compile with class counts for focal loss
    pos_count = np.sum(y_train[:, 1])
    neg_count = len(y_train) - pos_count
    model.compile(lr=2e-4, class_counts=[neg_count, pos_count])
    
    # Create callbacks
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'models/demo_everest.h5',
        monitor='val_softmax_dense_tss',
        mode='max',
        save_best_weights_only=True,
        verbose=1
    )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_softmax_dense_tss',
        mode='max',
        patience=10,
        restore_best_weights=True
    )
    
    # Train model (for demo, use reduced epochs)
    print("\nTraining model (reduced epochs for demo)...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),  # Use test as validation for demo
        epochs=5,  # Reduced epochs for demo
        batch_size=32,
        callbacks=[checkpoint, early_stopping],
        verbose=1,
        use_diffusion=False,  # Disable diffusion for demo to save time
    )
    
    # Print training results
    final_epoch = len(history.history['loss']) - 1
    print("\nTraining Results:")
    print(f"Train loss: {history.history['loss'][final_epoch]:.4f}")
    if 'softmax_dense_tss' in history.history:
        print(f"Train TSS: {history.history['softmax_dense_tss'][final_epoch]:.4f}")
    print(f"Val loss: {history.history['val_loss'][final_epoch]:.4f}")
    if 'val_softmax_dense_tss' in history.history:
        print(f"Val TSS: {history.history['val_softmax_dense_tss'][final_epoch]:.4f}")
    
    # Calibrate model
    print("\nCalibrating model...")
    threshold = model.calibrate(X_test, y_test, alpha=0.1, mc_samples=5)
    print(f"Calibration threshold: {threshold:.4f}")
    
    # Make predictions with uncertainty
    print("\nGenerating predictions with uncertainty...")
    results = model.predict_with_uncertainty(X_test, mc_passes=5)
    
    # Extract probabilities
    probs = results['probabilities']
    mc_uncertainty = results['mc_uncertainty']
    
    # Get binary predictions using best threshold
    y_pred = (probs > 0.5).astype(int)
    
    # Calculate confusion matrix
    cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    tss = calculate_tss_from_cm(tn, fp, fn, tp)
    
    print("\nTest Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"TSS: {tss:.4f}")
    print(f"Confusion Matrix: [TN={tn}, FP={fp}, FN={fn}, TP={tp}]")
    
    # Extract uncertainty info
    if 'evidential' in results and results['evidential'] is not None:
        ev = results['evidential']
        print("\nEvidential Uncertainty:")
        print(f"Epistemic Uncertainty (mean): {np.mean(ev['epistemic']):.4f}")
        print(f"Aleatoric Uncertainty (mean): {np.mean(ev['aleatoric']):.4f}")
    
    # Plot ROC and PR curves
    plot_curves(np.argmax(y_test, axis=1), probs, mc_uncertainty)
    
    # Plot sample with uncertainty
    if len(X_test) > 0:
        plot_sample_with_uncertainty(X_test[0], results, 0)
    
    return model, results

def plot_curves(y_true, y_pred_proba, uncertainty=None):
    """Plot ROC and Precision-Recall curves."""
    # Create figure
    plt.figure(figsize=(12, 5))
    
    # Plot ROC curve
    plt.subplot(1, 2, 1)
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    # Plot Precision-Recall curve
    plt.subplot(1, 2, 2)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    
    plt.tight_layout()
    plt.savefig('models/demo_everest_curves.png')
    print("Saved ROC and PR curves to models/demo_everest_curves.png")

def plot_sample_with_uncertainty(x_sample, results, sample_idx):
    """Plot a sample with its uncertainty."""
    # Extract uncertainty if available
    uncertainty = None
    if 'mc_uncertainty' in results:
        uncertainty = results['mc_uncertainty'][sample_idx]
    
    # Extract evidential uncertainty if available
    evidential_uncertainty = None
    if 'evidential' in results and results['evidential'] is not None:
        ev = results['evidential']
        evidential_uncertainty = (
            ev['epistemic'][sample_idx] + ev['aleatoric'][sample_idx]
        )
    
    # Extract probability
    probability = results['probabilities'][sample_idx]
    
    # Get flattened features for plotting
    # Assuming x_sample has shape [seq_len, features]
    seq_len, n_features = x_sample.shape
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot features
    plt.subplot(2, 1, 1)
    for i in range(min(5, n_features)):
        plt.plot(x_sample[:, i], label=f'Feature {i}')
    plt.xlabel('Time')
    plt.ylabel('Feature Value')
    plt.title(f'Sample Features (Probability: {probability:.4f})')
    plt.legend()
    
    # Plot uncertainty
    plt.subplot(2, 1, 2)
    plt.axhline(y=probability, color='b', linestyle='-', label=f'Probability: {probability:.4f}')
    
    if uncertainty is not None:
        lower = max(0, probability - uncertainty)
        upper = min(1, probability + uncertainty)
        plt.fill_between(
            [0, 1], [lower, lower], [upper, upper], 
            color='blue', alpha=0.2, 
            label=f'MC Uncertainty: ±{uncertainty:.4f}'
        )
    
    if evidential_uncertainty is not None:
        plt.axhline(
            y=probability - evidential_uncertainty/2, 
            color='r', linestyle='--',
            label=f'Evidential Lower: {probability - evidential_uncertainty/2:.4f}'
        )
        plt.axhline(
            y=probability + evidential_uncertainty/2, 
            color='r', linestyle='--',
            label=f'Evidential Upper: {probability + evidential_uncertainty/2:.4f}'
        )
    
    plt.xlabel('Confidence')
    plt.ylabel('Probability')
    plt.title('Prediction with Uncertainty')
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('models/demo_everest_sample.png')
    print("Saved sample plot to models/demo_everest_sample.png")

def main():
    """Main function to run the demo."""
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Show hardware info
    print("=== Hardware Information ===")
    print(f"TensorFlow version: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU available: {gpus}")
    else:
        print("No GPU available, using CPU")
    
    # Demo individual components
    demo_model_components()
    
    # Train and evaluate the complete model
    model, results = train_and_evaluate_model()
    
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    main() 