#!/usr/bin/env python3
"""
Test script for ablation trainer to verify it works correctly.
"""

import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def test_imports():
    """Test that all imports work correctly."""
    print("🧪 Testing imports...")
    
    try:
        from solarknowledge_ret_plus import RETPlusWrapper, RETPlusModel
        print("✅ EVEREST model imports successful")
    except ImportError as e:
        print(f"❌ EVEREST model import failed: {e}")
        return False
    
    try:
        from utils import get_training_data, get_testing_data
        print("✅ Utils imports successful")
    except ImportError as e:
        print(f"❌ Utils import failed: {e}")
        return False
    
    try:
        from config import OPTIMAL_HYPERPARAMS, ABLATION_VARIANTS
        print("✅ Config imports successful")
    except ImportError as e:
        print(f"❌ Config import failed: {e}")
        return False
    
    try:
        from trainer import AblationTrainer, train_ablation_variant
        print("✅ Trainer imports successful")
    except ImportError as e:
        print(f"❌ Trainer import failed: {e}")
        return False
    
    return True

def test_trainer_creation():
    """Test that trainer can be created."""
    print("\n🧪 Testing trainer creation...")
    
    try:
        from trainer import AblationTrainer
        trainer = AblationTrainer("full_model", 0)
        print("✅ Trainer creation successful")
        return True
    except Exception as e:
        print(f"❌ Trainer creation failed: {e}")
        return False

def test_data_loading():
    """Test that data can be loaded."""
    print("\n🧪 Testing data loading...")
    
    try:
        from utils import get_training_data, get_testing_data
        X_train, y_train = get_training_data("72", "M5")
        X_test, y_test = get_testing_data("72", "M5")
        
        if X_train is not None and y_train is not None:
            print(f"✅ Training data loaded: {len(X_train)} samples")
        else:
            print("❌ Training data is None")
            return False
            
        if X_test is not None and y_test is not None:
            print(f"✅ Testing data loaded: {len(X_test)} samples")
        else:
            print("❌ Testing data is None")
            return False
            
        return True
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🔬 EVEREST Ablation Trainer Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_trainer_creation,
        test_data_loading
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Trainer is ready for cluster execution.")
        return 0
    else:
        print("❌ Some tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 