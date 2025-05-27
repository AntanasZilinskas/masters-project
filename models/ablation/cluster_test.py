#!/usr/bin/env python3
"""
Minimal test script to debug cluster execution issues.
"""

import os
import sys
import traceback

def test_basic_imports():
    """Test basic Python imports."""
    print("🧪 Testing basic imports...")
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
        
        import pandas as pd
        print(f"✅ Pandas: {pd.__version__}")
        
        return True
    except Exception as e:
        print(f"❌ Basic imports failed: {e}")
        traceback.print_exc()
        return False

def test_path_setup():
    """Test Python path setup."""
    print("\n🧪 Testing Python path setup...")
    try:
        print(f"Current working directory: {os.getcwd()}")
        print(f"Python path: {sys.path[:3]}...")  # Show first 3 entries
        
        # Add models directory to path
        models_path = os.path.join(os.getcwd(), 'models')
        if models_path not in sys.path:
            sys.path.append(models_path)
            print(f"✅ Added models path: {models_path}")
        
        return True
    except Exception as e:
        print(f"❌ Path setup failed: {e}")
        traceback.print_exc()
        return False

def test_everest_imports():
    """Test EVEREST-specific imports."""
    print("\n🧪 Testing EVEREST imports...")
    try:
        from solarknowledge_ret_plus import RETPlusWrapper
        print("✅ RETPlusWrapper imported")
        
        from utils import get_training_data, get_testing_data
        print("✅ Utils imported")
        
        return True
    except Exception as e:
        print(f"❌ EVEREST imports failed: {e}")
        traceback.print_exc()
        return False

def test_ablation_imports():
    """Test ablation-specific imports."""
    print("\n🧪 Testing ablation imports...")
    try:
        # Change to ablation directory
        ablation_dir = os.path.join(os.getcwd(), 'models', 'ablation')
        os.chdir(ablation_dir)
        print(f"Changed to: {os.getcwd()}")
        
        from config import OPTIMAL_HYPERPARAMS, ABLATION_VARIANTS
        print("✅ Config imported")
        
        from trainer import AblationTrainer
        print("✅ Trainer imported")
        
        return True
    except Exception as e:
        print(f"❌ Ablation imports failed: {e}")
        traceback.print_exc()
        return False

def test_data_loading():
    """Test data loading."""
    print("\n🧪 Testing data loading...")
    try:
        # Go back to project root
        project_root = os.path.dirname(os.path.dirname(os.getcwd()))
        os.chdir(project_root)
        print(f"Changed to project root: {os.getcwd()}")
        
        # Add models to path again
        models_path = os.path.join(os.getcwd(), 'models')
        if models_path not in sys.path:
            sys.path.append(models_path)
        
        from utils import get_training_data, get_testing_data
        
        # Test data loading
        X_train, y_train = get_training_data("72", "M5")
        if X_train is not None:
            print(f"✅ Training data loaded: {len(X_train)} samples")
        else:
            print("❌ Training data is None")
            return False
            
        return True
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🔬 EVEREST Cluster Debug Test")
    print("=" * 50)
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Environment variables:")
    print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print(f"  PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
    print("")
    
    tests = [
        test_basic_imports,
        test_path_setup,
        test_everest_imports,
        test_ablation_imports,
        test_data_loading
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"❌ Test {test.__name__} failed")
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            traceback.print_exc()
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Cluster environment is ready.")
        return 0
    else:
        print("❌ Some tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 