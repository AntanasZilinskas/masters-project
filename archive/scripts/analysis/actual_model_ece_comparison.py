import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Setup paths
project_root = Path(__file__).parent
test_data_path = project_root / "Nature_data/testing_data_M5_72.csv"
everest_weights_path = project_root / "tests/model_weights_EVEREST_72h_M5.pt"
solarknowledge_weights_path = project_root / "models/archive/models_working/72/M5/model_weights.weights.h5"

def load_test_data():
    """Load SHARP M5-72h test data."""
    print("Loading test data...")
    df = pd.read_csv(test_data_path)
    
    # Filter out padding rows
    df = df[df['Flare'] != 'padding'].copy()
    
    # Extract features and labels - using correct column names from the CSV
    feature_columns = ['USFLUX', 'MEANGAM', 'MEANGBT', 'MEANGBZ', 'MEANPOT', 
                      'TOTUSJH', 'TOTPOT', 'ABSNJZH', 'SAVNCPP']
    
    X_test = df[feature_columns].values
    
    # Convert flare labels: P=1 (positive M5 flare), N=0 (no M5 flare)
    y_test = (df['Flare'] == 'P').astype(int).values
    
    # Reshape for models: (samples, timesteps=10, features=9)
    n_samples = len(X_test) // 10
    X_test = X_test[:n_samples*10].reshape(n_samples, 10, 9)
    y_test = y_test[:n_samples*10:10]  # Take every 10th label
    
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    print(f"Positive class samples: {y_test.sum()}/{len(y_test)} ({100*y_test.sum()/len(y_test):.2f}%)")
    
    return X_test, y_test

def load_actual_solarknowledge_model():
    """Load the actual archived SolarKnowledge TensorFlow model."""
    
    # Try loading using original SolarKnowledge class's load_model method
    try:
        print("Attempting to load using original SolarKnowledge.load_model()...")
        import sys
        sys.path.append(str(project_root / "models"))
        from SolarKnowledge_model import SolarKnowledge
        
        sk_model = SolarKnowledge()
        # Use the load_model method which builds, compiles and loads weights
        sk_model.load_model(
            input_shape=(10, 9),
            flare_class="M5", 
            w_dir=str(solarknowledge_weights_path.parent),
            verbose=True
        )
        
        print("✅ SolarKnowledge model loaded successfully using load_model()!")
        print(f"Model has {sk_model.model.count_params():,} parameters")
        return sk_model.model
        
    except Exception as e:
        print(f"❌ Error loading with load_model: {e}")
        
        # Fallback: Try direct weight loading
        try:
            print("Trying direct model construction...")
            import tensorflow as tf
            tf.mixed_precision.set_global_policy('float32')  # Disable mixed precision
            
            from SolarKnowledge_model import SolarKnowledge
            sk_model = SolarKnowledge()
            sk_model.build_base_model(input_shape=(10, 9))
            sk_model.compile()
            
            # Load weights directly
            sk_model.model.load_weights(str(solarknowledge_weights_path))
            print("✅ SolarKnowledge model loaded successfully with float32!")
            print(f"Model has {sk_model.model.count_params():,} parameters")
            return sk_model.model
            
        except Exception as e2:
            print(f"❌ Error with direct loading: {e2}")
            return None

def load_actual_everest_model():
    """Load the actual EVEREST model with weights."""
    # Import the EVEREST model architecture
    import sys
    sys.path.append(str(project_root / "models"))
    from solarknowledge_ret_plus import RETPlusWrapper
    
    print("Loading EVEREST model...")
    
    # Initialize model with correct constructor
    model = RETPlusWrapper(
        input_shape=(10, 9),
        early_stopping_patience=10,
        use_attention_bottleneck=True,
        use_evidential=True,
        use_evt=True,
        use_precursor=True,
        compile_model=False
    )
    
    # Load weights
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    try:
        checkpoint = torch.load(everest_weights_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            model.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.model.load_state_dict(checkpoint)
        
        model.model.to(device)
        model.model.eval()
        
        print("✅ EVEREST model loaded successfully!")
        return model, device
    except Exception as e:
        print(f"❌ Error loading EVEREST weights: {e}")
        return None, None

def calculate_ece_torch(y_true, y_probs, n_bins=15):
    """Calculate ECE for PyTorch model predictions."""
    y_true = torch.tensor(y_true, dtype=torch.float32)
    y_probs = torch.tensor(y_probs, dtype=torch.float32)
    
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = torch.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_probs > bin_lower.item()) & (y_probs <= bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin.item() > 0:
            accuracy_in_bin = y_true[in_bin].float().mean()
            avg_confidence_in_bin = y_probs[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece.item()

def calculate_ece_tf(y_true, y_probs, n_bins=15):
    """Calculate ECE for TensorFlow model predictions."""
    import tensorflow as tf
    
    y_true = tf.cast(y_true, tf.float32)
    y_probs = tf.cast(y_probs, tf.float32)
    
    bin_boundaries = tf.linspace(0.0, 1.0, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = tf.logical_and(y_probs > bin_lower, y_probs <= bin_upper)
        prop_in_bin = tf.reduce_mean(tf.cast(in_bin, tf.float32))
        
        if prop_in_bin > 0:
            accuracy_in_bin = tf.reduce_mean(tf.cast(y_true[in_bin], tf.float32))
            avg_confidence_in_bin = tf.reduce_mean(y_probs[in_bin])
            ece += tf.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece.numpy()

def main():
    print("🔥 ACTUAL MODEL ECE COMPARISON: EVEREST vs SolarKnowledge 🔥")
    print("="*60)
    
    # Load test data
    X_test, y_test = load_test_data()
    
    # Load actual models
    print("\n📥 Loading actual trained models...")
    sk_model = load_actual_solarknowledge_model()
    everest_model, device = load_actual_everest_model()
    
    if sk_model is None or everest_model is None:
        print("❌ Failed to load one or both models")
        return
    
    print("\n🧮 Computing predictions and ECE values...")
    
    # Get SolarKnowledge predictions
    print("Computing SolarKnowledge predictions...")
    sk_probs = sk_model.predict(X_test, verbose=0)
    sk_probs_pos = sk_probs[:, 1]  # Positive class probabilities
    sk_ece = calculate_ece_tf(y_test, sk_probs_pos)
    
    # Get EVEREST predictions  
    print("Computing EVEREST predictions...")
    X_test_torch = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        everest_output = everest_model(X_test_torch)
        if isinstance(everest_output, dict):
            everest_probs = everest_output['classification_probs']
        else:
            everest_probs = everest_output
        
        everest_probs_pos = everest_probs[:, 1].cpu().numpy()
    
    everest_ece = calculate_ece_torch(y_test, everest_probs_pos)
    
    # Calculate improvement
    improvement = ((sk_ece - everest_ece) / sk_ece) * 100
    
    print("\n" + "="*60)
    print("🎯 ACTUAL ECE RESULTS")
    print("="*60)
    print(f"SolarKnowledge ECE: {sk_ece:.6f}")
    print(f"EVEREST ECE:        {everest_ece:.6f}")
    print(f"Improvement:        {improvement:.1f}%")
    print("="*60)
    
    # Save results
    results = {
        'solarknowledge_ece': sk_ece,
        'everest_ece': everest_ece,
        'improvement_percent': improvement,
        'test_samples': len(y_test),
        'positive_samples': int(y_test.sum())
    }
    
    # Write results for paper
    with open(project_root / "actual_ece_results_paper.txt", "w") as f:
        f.write("ACTUAL MODEL ECE COMPARISON RESULTS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Dataset: SHARP M5-72h test data\n")
        f.write(f"Test samples: {len(y_test):,}\n")
        f.write(f"Positive samples: {int(y_test.sum())}\n")
        f.write(f"Positive rate: {100*y_test.sum()/len(y_test):.2f}%\n\n")
        f.write("ECE Results (15-bin calibration):\n")
        f.write(f"SolarKnowledge (TensorFlow): {sk_ece:.6f}\n")
        f.write(f"EVEREST (PyTorch):          {everest_ece:.6f}\n")
        f.write(f"Improvement:                {improvement:.1f}%\n\n")
        f.write("Models used:\n")
        f.write(f"- SolarKnowledge: {solarknowledge_weights_path}\n")
        f.write(f"- EVEREST: {everest_weights_path}\n")
    
    print(f"\n💾 Results saved to: actual_ece_results_paper.txt")
    
    return results

if __name__ == "__main__":
    results = main() 