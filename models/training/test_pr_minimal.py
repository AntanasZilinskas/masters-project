#!/usr/bin/env python3
"""
Minimal test for precision-recall generation.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add project root to path  
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_basic_loading():
    """Test basic loading without heavy computations."""
    
    print("Testing basic imports...")
    try:
        from models.solarknowledge_ret_plus import RETPlusWrapper
        from models.utils import get_testing_data
        print("✓ Imports successful")
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False
    
    print("Testing data loading...")
    try:
        test_data = get_testing_data(72, 'M5')
        X_test = test_data[0]
        y_test = test_data[1]
        print(f"✓ Data loaded: {X_test.shape}, {len(y_test)} labels")
        print(f"  Positive rate: {np.mean(y_test):.4f}")
        input_shape = (X_test.shape[1], X_test.shape[2])
        print(f"  Input shape: {input_shape}")
    except Exception as e:
        print(f"✗ Data loading error: {e}")
        return False
    
    print("Testing model path...")
    model_path = '../../archive/saved_models/M5_72/run_001/model_weights.pt'
    if os.path.exists(model_path):
        print(f"✓ Model file exists: {model_path}")
    else:
        print(f"✗ Model file missing: {model_path}")
        return False
    
    print("Testing model loading...")
    try:
        wrapper = RETPlusWrapper(
            input_shape=input_shape,
            early_stopping_patience=10,
            use_attention_bottleneck=True,
            use_evidential=True,
            use_evt=True,
            use_precursor=True,
            compile_model=False
        )
        wrapper.load(model_path)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Model loading error: {e}")
        return False
    
    print("Testing small prediction...")
    try:
        # Test with just first 100 samples
        X_small = X_test[:100]
        y_proba_small = wrapper.predict_proba(X_small)
        if y_proba_small.ndim > 1:
            y_proba_small = y_proba_small[:, 1] if y_proba_small.shape[1] > 1 else y_proba_small.ravel()
        
        print(f"✓ Small prediction successful: {y_proba_small.shape}")
        print(f"  Range: [{y_proba_small.min():.4f}, {y_proba_small.max():.4f}]")
        return True
    except Exception as e:
        print(f"✗ Prediction error: {e}")
        return False

if __name__ == "__main__":
    success = test_basic_loading()
    if success:
        print("\n✓ All tests passed! Ready for full precision-recall analysis.")
    else:
        print("\n✗ Tests failed. Check errors above.") 