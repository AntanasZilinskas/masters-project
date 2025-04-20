"""
Setup script for EVEREST model installation

This script helps install the necessary dependencies for the EVEREST model
and provides instructions on how to use it.
"""

import subprocess
import sys
import os

def check_tensorflow():
    """Check if TensorFlow is installed and at the correct version."""
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow is installed (version {tf.__version__})")
        if tf.__version__.startswith("2."):
            return True
        print("  Note: EVEREST works best with TensorFlow 2.x")
        return True
    except ImportError:
        print("✗ TensorFlow not found")
        return False

def install_tensorflow_addons():
    """Try to install TensorFlow Addons for focal loss."""
    try:
        import tensorflow_addons
        print(f"✓ TensorFlow Addons is installed (version {tensorflow_addons.__version__})")
        return True
    except ImportError:
        print("Installing TensorFlow Addons (for focal loss)...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow-addons"])
            import tensorflow_addons
            print(f"✓ TensorFlow Addons installed successfully (version {tensorflow_addons.__version__})")
            return True
        except Exception as e:
            print(f"✗ Could not install TensorFlow Addons: {e}")
            print("  EVEREST will fall back to standard categorical cross-entropy loss.")
            return False

def print_instructions():
    """Print usage instructions for the EVEREST model."""
    print("\n" + "="*80)
    print("EVEREST MODEL USAGE INSTRUCTIONS".center(80))
    print("="*80)
    
    print("""
EVEREST is a drop-in replacement for SolarKnowledge that includes:
 1. Linear-attention blocks (using standard MultiHeadAttention)
 2. Support for focal loss (if TensorFlow Addons is available)
 3. Monte Carlo dropout for uncertainty estimation

Basic Usage:
-----------
from everest_model import EVEREST

# Create a model instance
model = EVEREST()

# Build model (same API as SolarKnowledge)
model.build_base_model(input_shape=(sequence_length, num_features))
model.compile()  # Will use focal loss if available, otherwise standard cross-entropy

# Train model
model.fit(X_train, y_train)

# Make predictions with uncertainty estimation (Monte Carlo dropout)
mean_predictions, uncertainties = model.mc_predict(X_test)

Testing:
--------
Run the test script to verify everything works:
    python models/test_everest.py

Training & Evaluation:
---------------------
The model works with the existing training and testing scripts:
    python models/SolarKnowledge_run_all_trainings.py
    python models/SolarKnowledge_run_all_tests.py
""")
    print("="*80)

def main():
    """Main function to setup EVEREST model."""
    print("Setting up EVEREST model dependencies...")
    
    # Check for TensorFlow
    tf_ok = check_tensorflow()
    if not tf_ok:
        print("Please install TensorFlow 2.x: pip install tensorflow")
        return
    
    # Try to install TensorFlow Addons
    tfa_ok = install_tensorflow_addons()
    
    print("\nSetup complete!")
    if not tfa_ok:
        print("Note: TensorFlow Addons is not installed, EVEREST will use standard loss.")
    
    # Print usage instructions
    print_instructions()

if __name__ == "__main__":
    main() 