"""
Real ECE comparison using actual SHARP test data and trained model weights.

This script loads the real SHARP M5-72h test data and evaluates 
actual calibration performance for the paper.
"""

import sys
import os
import warnings
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import h5py
from sklearn.calibration import calibration_curve

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "models"))

warnings.filterwarnings("ignore")

# Force CPU to avoid device mismatch issues
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# Disable MPS to force CPU usage on Mac
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from utils import load_data, n_features, series_len, start_feature, mask_value


class ECECalculator:
    """Calculate Expected Calibration Error with 15 bins."""
    
    def __init__(self, n_bins: int = 15):
        self.n_bins = n_bins
    
    def __call__(self, probs: np.ndarray, labels: np.ndarray) -> float:
        """Calculate ECE."""
        probs = np.array(probs).squeeze()
        labels = np.array(labels).squeeze()
            
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        ece = 0.0
        
        for i in range(self.n_bins):
            bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i+1]
            in_bin = (probs > bin_lower) & (probs <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = labels[in_bin].mean()
                avg_confidence_in_bin = probs[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece


def load_sharp_test_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load actual SHARP M5-72h test data."""
    
    test_file = project_root / "Nature_data/testing_data_M5_72.csv"
    
    if not test_file.exists():
        raise FileNotFoundError(f"SHARP test data not found: {test_file}")
    
    print(f"Loading SHARP M5-72h test data from: {test_file}")
    
    # Load using the project's data loading utilities
    X, y, _ = load_data(
        datafile=str(test_file),
        flare_label="M5",
        series_len=series_len,
        start_feature=start_feature,
        n_features=n_features,
        mask_value=mask_value,
    )
    
    # Convert labels to binary
    y_binary = np.array([1 if label == "pos" else 0 for label in y])
    
    print(f"Loaded SHARP data:")
    print(f"   Shape: {X.shape}")
    print(f"   Positive samples: {y_binary.sum()}")
    print(f"   Total samples: {len(y_binary)}")
    print(f"   Base rate: {y_binary.mean():.1%}")
    
    return X, y_binary


def load_everest_model():
    """Load actual trained EVEREST model."""
    
    # Use the actual EVEREST weights from the tests directory
    everest_path = project_root / "tests/model_weights_EVEREST_72h_M5.pt"
    
    if not everest_path.exists():
        raise FileNotFoundError(f"EVEREST weights not found: {everest_path}")
    
    print(f"Loading EVEREST model from: {everest_path}")
    
    try:
        from models.solarknowledge_ret_plus import RETPlusWrapper
        
        # Force CPU device
        torch.set_default_device('cpu')
        
        # Create EVEREST model with SHARP input dimensions
        everest_model = RETPlusWrapper(
            input_shape=(10, 9),  # SHARP data is reshaped to (10, 9)
            use_attention_bottleneck=True,
            use_evidential=True,
            use_evt=True,
            use_precursor=True
        )
        
        # Load actual trained weights to CPU
        checkpoint = torch.load(everest_path, map_location="cpu")
        everest_model.model.load_state_dict(checkpoint)
        everest_model.model.eval()
        
        # Move model to CPU explicitly
        everest_model.model = everest_model.model.cpu()
        
        print("✅ EVEREST model loaded successfully")
        
        def predict_proba_everest(X):
            """Get probabilities from EVEREST model."""
            # Reshape SHARP data to expected format
            if X.shape[1:] != (10, 9):
                # Original SHARP data needs reshaping
                print(f"   Reshaping SHARP data from {X.shape} to (N, 10, 9)")
                X_reshaped = X.reshape(X.shape[0], 10, 9)
            else:
                X_reshaped = X
            
            X_tensor = torch.tensor(X_reshaped, dtype=torch.float32).cpu()
            
            with torch.no_grad():
                outputs = everest_model.model(X_tensor)
                
                # Handle multiple output heads
                if isinstance(outputs, dict):
                    # Use main classification logits
                    logits = outputs.get('logits', list(outputs.values())[0])
                else:
                    logits = outputs
                
                # Convert to probabilities
                if len(logits.shape) > 1 and logits.shape[1] > 1:
                    probs = torch.softmax(logits, dim=1)[:, 1]  # Positive class
                else:
                    probs = torch.sigmoid(logits.squeeze())
                
                return probs.cpu().numpy()
        
        return predict_proba_everest
        
    except Exception as e:
        raise Exception(f"Failed to load EVEREST model: {e}")


def load_solarknowledge_model():
    """Load actual trained SolarKnowledge model."""
    
    # Use the actual working TensorFlow weights for M5-72h
    tf_weights_dir = project_root / "models/archive/models_working/72/M5"
    
    if tf_weights_dir.exists():
        print(f"Loading SolarKnowledge model from: {tf_weights_dir}")
        
        try:
            # Import TensorFlow SolarKnowledge - the ORIGINAL class
            from models.SolarKnowledge_model import SolarKnowledge as TFSolarKnowledge
            
            # Create TensorFlow model using the load_model method
            # This is the correct way that handles the exact architecture 
            tf_model = TFSolarKnowledge()
            tf_model.load_model(
                input_shape=(10, 9),
                flare_class="M5",  # This will be ignored since we're specifying w_dir
                w_dir=str(tf_weights_dir),
                verbose=True
            )
            
            print("✅ SolarKnowledge (TensorFlow) model loaded successfully")
            print(f"Model has {tf_model.model.count_params()} parameters")
            
            def predict_proba_solarknowledge(X):
                """Get probabilities from SolarKnowledge model."""
                # Reshape SHARP data to expected format if needed
                if X.shape[1:] != (10, 9):
                    print(f"   Reshaping SHARP data from {X.shape} to (N, 10, 9)")
                    X_reshaped = X.reshape(X.shape[0], 10, 9)
                else:
                    X_reshaped = X
                
                probs = tf_model.predict(X_reshaped)
                
                # Handle output format - should be (N, 2) for binary classification
                if len(probs.shape) > 1 and probs.shape[1] == 2:
                    return probs[:, 1]  # Return positive class probabilities
                else:
                    return probs.squeeze()
            
            return predict_proba_solarknowledge
            
        except Exception as e:
            print(f"⚠️  TensorFlow SolarKnowledge failed: {e}")
            import traceback
            traceback.print_exc()
    
    # If TensorFlow fails, we have a problem since we need the real model
    raise Exception(f"Failed to load actual SolarKnowledge model from {tf_weights_dir}. Cannot proceed with comparison.")


def create_tensorflow_solarknowledge_model():
    """
    Create the SolarKnowledge model architecture in TensorFlow.
    Based on the original implementation from the codebase.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models, regularizers
    
    class PositionalEncoding(layers.Layer):
        """Positional encoding layer for transformer."""
        
        def __init__(self, max_len, embed_dim):
            super(PositionalEncoding, self).__init__()
            self.max_len = max_len
            self.embed_dim = embed_dim
        
        def build(self, input_shape):
            # Create positional encoding matrix
            import numpy as np
            position = np.arange(self.max_len)[:, np.newaxis]
            div_term = np.exp(np.arange(0, self.embed_dim, 2) * -(np.log(10000.0) / self.embed_dim))
            
            pe = np.zeros((self.max_len, self.embed_dim))
            pe[:, 0::2] = np.sin(position * div_term)
            pe[:, 1::2] = np.cos(position * div_term)
            
            # Add batch dimension and create as constant
            self.pe = tf.constant(pe[np.newaxis, :, :], dtype=tf.float32)
        
        def call(self, x):
            return x + self.pe[:, :tf.shape(x)[1], :]


    class TransformerBlock(layers.Layer):
        """Transformer block from the original TensorFlow implementation."""
        
        def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.2):
            super(TransformerBlock, self).__init__()
            self.att = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=embed_dim
            )
            # Use GELU activation in the feed-forward network
            self.ffn = tf.keras.Sequential([
                layers.Dense(ff_dim, activation=tf.keras.activations.gelu),
                layers.Dropout(dropout_rate),
                layers.Dense(embed_dim),
            ])
            self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
            self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
            self.dropout1 = layers.Dropout(dropout_rate)
            self.dropout2 = layers.Dropout(dropout_rate)

        def call(self, inputs, training=False):
            attn_output = self.att(inputs, inputs)
            attn_output = self.dropout1(attn_output, training=training)
            out1 = self.layernorm1(inputs + attn_output)
            ffn_output = self.ffn(out1)
            ffn_output = self.dropout2(ffn_output, training=training)
            return self.layernorm2(out1 + ffn_output)
    
    # Input layer - SHARP magnetogram features (10 snapshots × 9 channels)
    input_shape = (10, 9)
    inputs = layers.Input(shape=input_shape)
    
    # Project the input features into a higher-dimensional embedding space
    embed_dim = 128
    x = layers.Dense(embed_dim)(inputs)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dropout(0.2)(x)
    
    # Add positional encoding
    x = PositionalEncoding(max_len=input_shape[0], embed_dim=embed_dim)(x)
    
    # Apply transformer blocks
    num_heads = 4
    ff_dim = 256
    num_transformer_blocks = 6
    dropout_rate = 0.2
    
    for i in range(num_transformer_blocks):
        x = TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)(x)
    
    # Global average pooling and classification head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(
        128, 
        activation=tf.keras.activations.gelu,
        kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)
    )(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(
        2, 
        activation="softmax",
        activity_regularizer=regularizers.l2(1e-5)
    )(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='SolarKnowledge')
    
    return model


def evaluate_model_calibration(predict_func, X_test, y_test, model_name):
    """Evaluate actual model calibration."""
    
    print(f"\nEvaluating {model_name} calibration...")
    
    # Get predictions from actual model
    probs = predict_func(X_test)
    probs = np.array(probs).squeeze()
    
    print(f"   Prediction range: [{probs.min():.3f}, {probs.max():.3f}]")
    print(f"   Mean prediction: {probs.mean():.3f}")
    print(f"   Actual base rate: {y_test.mean():.3f}")
    
    # Calculate ECE with 15 bins (standard)
    ece_calc = ECECalculator(n_bins=15)
    ece = ece_calc(probs, y_test)
    
    # Calculate reliability curve
    frac_pos, mean_pred = calibration_curve(
        y_test, probs, n_bins=15, strategy="uniform"
    )
    
    # Find over-confidence threshold (gap ≥ 0.1)
    threshold = None
    max_gap = 0
    for pred, frac in zip(mean_pred, frac_pos):
        gap = pred - frac
        max_gap = max(max_gap, gap)
        if gap >= 0.1 and threshold is None:
            threshold = pred
    
    results = {
        'model_name': model_name,
        'ece': ece,
        'probs': probs,
        'mean_pred': mean_pred,
        'frac_pos': frac_pos,
        'threshold': threshold,
        'max_gap': max_gap,
        'n_samples': len(y_test),
        'positive_rate': y_test.mean(),
        'actual_model': True
    }
    
    print(f"   ECE: {ece:.3f}")
    print(f"   Max confidence gap: {max_gap:.3f}")
    
    if threshold is not None:
        print(f"   Over-confidence threshold: p ≳ {threshold:.3f}")
    else:
        print(f"   Well-calibrated (no over-confidence detected)")
    
    return results


def main():
    """Compare actual EVEREST vs SolarKnowledge ECE values on real SHARP data."""
    
    print("=" * 80)
    print("ACTUAL SHARP DATA ECE COMPARISON: EVEREST vs SolarKnowledge")
    print("Using real SHARP M5-72h test data and trained model weights")
    print("=" * 80)
    
    # Load actual SHARP test data
    print("\n1. Loading real SHARP M5-72h test data...")
    try:
        X_test, y_test = load_sharp_test_data()
    except Exception as e:
        print(f"❌ Failed to load SHARP data: {e}")
        return None
    
    print(f"   Test data shape: {X_test.shape}")
    print(f"   Labels shape: {y_test.shape}")
    
    # Load and evaluate actual models
    print("\n2. Loading actual trained models...")
    
    results = {}
    
    # EVEREST model (actual trained weights)
    try:
        everest_predict = load_everest_model()
        results['everest'] = evaluate_model_calibration(
            everest_predict, X_test, y_test, "EVEREST (trained with evidential learning)"
        )
    except Exception as e:
        print(f"❌ EVEREST evaluation failed: {e}")
    
    # SolarKnowledge model (actual trained weights)
    try:
        solarknowledge_predict = load_solarknowledge_model()
        results['solarknowledge'] = evaluate_model_calibration(
            solarknowledge_predict, X_test, y_test, "SolarKnowledge (trained baseline)"
        )
    except Exception as e:
        print(f"❌ SolarKnowledge evaluation failed: {e}")
    
    # Results comparison
    print("\n" + "=" * 80)
    print("ACTUAL SHARP DATA CALIBRATION RESULTS")
    print("=" * 80)
    
    if 'everest' in results and 'solarknowledge' in results:
        ev_ece = results['everest']['ece']
        sk_ece = results['solarknowledge']['ece']
        
        print(f"\n📊 Expected Calibration Error (15-bin) on Real SHARP M5-72h Data:")
        print(f"   SolarKnowledge (baseline):   {sk_ece:.3f}")
        print(f"   EVEREST (evidential):        {ev_ece:.3f}")
        
        improvement = sk_ece - ev_ece
        if improvement > 0:
            improvement_pct = (improvement / sk_ece) * 100
            print(f"\n✅ ECE Improvement:")
            print(f"   Absolute reduction:          {improvement:.3f}")
            print(f"   Relative improvement:        {improvement_pct:.1f}%")
            print(f"   EVEREST achieves better calibration")
        else:
            worsening = abs(improvement)
            worsening_pct = (worsening / sk_ece) * 100
            print(f"\n⚠️  ECE Difference:")
            print(f"   Absolute increase:           {worsening:.3f}")
            print(f"   Relative change:             {worsening_pct:.1f}% worse")
            print(f"   SolarKnowledge is better calibrated")
        
        # Over-confidence analysis
        print(f"\n⚠️  Over-confidence Analysis:")
        sk_threshold = results['solarknowledge']['threshold']
        ev_threshold = results['everest']['threshold']
        
        if sk_threshold is not None:
            print(f"   SolarKnowledge: Over-confident at p ≳ {sk_threshold:.3f}")
        else:
            print(f"   SolarKnowledge: Well-calibrated")
            
        if ev_threshold is not None:
            print(f"   EVEREST: Over-confident at p ≳ {ev_threshold:.3f}")
        else:
            print(f"   EVEREST: Well-calibrated")
        
        # Paper format
        print(f"\n📝 FOR YOUR PAPER PARAGRAPH:")
        if improvement > 0:
            print(f'   "ECE drops from {sk_ece:.3f} to {ev_ece:.3f}"')
            print(f'   "achieving a {improvement_pct:.1f}% improvement in calibration"')
        else:
            print(f'   "ECE changes from {sk_ece:.3f} to {ev_ece:.3f}"')
            
    elif 'everest' in results:
        print(f"\n📊 EVEREST Results Only:")
        print(f"   ECE: {results['everest']['ece']:.3f}")
        
    elif 'solarknowledge' in results:
        print(f"\n📊 SolarKnowledge Results Only:")
        print(f"   ECE: {results['solarknowledge']['ece']:.3f}")
    
    else:
        print("❌ No models could be evaluated")
        return None
    
    # Save actual results
    save_path = Path("calibration_results")
    save_path.mkdir(exist_ok=True)
    
    np.savez(
        save_path / "actual_sharp_ece_comparison.npz",
        **{f"{k}_results": v for k, v in results.items()},
        test_samples=len(y_test),
        positive_rate=y_test.mean(),
        data_source="real_sharp_m5_72h",
        model_source="actual_trained_weights"
    )
    
    print(f"\n📈 Real SHARP results saved to: {save_path}/actual_sharp_ece_comparison.npz")
    
    # Summary for paper
    if 'everest' in results and 'solarknowledge' in results:
        sk_ece = results['solarknowledge']['ece']
        ev_ece = results['everest']['ece']
        
        print("\n" + "=" * 80)
        print("FINAL NUMBERS FOR PAPER")
        print("=" * 80)
        
        if sk_ece > ev_ece:
            improvement_pct = ((sk_ece - ev_ece) / sk_ece) * 100
            print(f"✅ Use these numbers in your paper:")
            print()
            print(f"Using the real SHARP M5-72h test data ({len(y_test)} samples),")
            print(f"ECE for M5-class events consequently drops from {sk_ece:.3f} to {ev_ece:.3f},")
            print("as illustrated in Fig.~\\ref{fig:ece_improvement}, without incurring the")
            print("Monte-Carlo cost of the dropout scheme used in \\textit{SolarKnowledge}.")
            print()
            print(f"Improvement: {improvement_pct:.1f}% reduction in Expected Calibration Error")
        else:
            print("⚠️  Note: Current results show EVEREST has worse calibration than SolarKnowledge.")
            print("   This may indicate the models need retraining or different hyperparameters.")
    
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    main() 