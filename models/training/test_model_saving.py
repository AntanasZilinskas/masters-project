#!/usr/bin/env python3
"""
Test script to verify that model saving works correctly in production training.
This can be run locally to debug model saving issues.
"""

import os
import sys
import shutil
from pathlib import Path

# Add project root to path
sys.path.append(".")


def test_directory_creation():
    """Test that directories are created correctly."""
    print("🗂️ Testing directory creation...")

    from models.training.config import create_output_directories

    # Create directories
    create_output_directories()

    # Check they exist
    required_dirs = [
        "models/training/results",
        "models/training/trained_models",
        "models/training/logs",
        "models/training/plots",
        "models/training/analysis",
    ]

    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"   ✅ {dir_path}")
        else:
            print(f"   ❌ {dir_path}")
            return False

    return True


def test_trainer_initialization():
    """Test ProductionTrainer initialization."""
    print("\n🏗️ Testing trainer initialization...")

    try:
        from models.training.trainer import ProductionTrainer

        trainer = ProductionTrainer("C", "24", 0)

        print(f"   ✅ Trainer created")
        print(f"   📁 Experiment dir: {trainer.experiment_dir}")
        print(f"   📁 Model dir: {trainer.model_dir}")

        # Check if experiment directory was created
        if os.path.exists(trainer.experiment_dir):
            print(f"   ✅ Experiment directory created")
        else:
            print(f"   ❌ Experiment directory not created")
            return False

        return True, trainer

    except Exception as e:
        print(f"   ❌ Failed to create trainer: {e}")
        return False, None


def test_model_creation(trainer):
    """Test model creation."""
    print("\n🤖 Testing model creation...")

    try:
        model = trainer.create_model()
        print(f"   ✅ Model created: {type(model)}")
        return True, model
    except Exception as e:
        print(f"   ❌ Failed to create model: {e}")
        return False, None


def test_data_loading(trainer):
    """Test data loading."""
    print("\n📊 Testing data loading...")

    try:
        X_train, y_train, X_test, y_test = trainer.load_data()
        print(f"   ✅ Data loaded")
        print(f"   📊 Train: {X_train.shape}, Test: {X_test.shape}")
        print(f"   🎯 Train labels: {y_train.shape}, Test labels: {y_test.shape}")
        return True, (X_train, y_train, X_test, y_test)
    except Exception as e:
        print(f"   ❌ Failed to load data: {e}")
        return False, None


def test_copy_mechanism():
    """Test the copy mechanism with dummy files."""
    print("\n📁 Testing file copy mechanism...")

    # Create dummy source directory with some files
    source_dir = "test_source_model"
    target_dir = "models/training/trained_models/test_experiment"

    try:
        # Create source directory with dummy files
        os.makedirs(source_dir, exist_ok=True)

        # Create dummy model files
        with open(os.path.join(source_dir, "model_weights.pt"), "w") as f:
            f.write("dummy model weights")

        with open(os.path.join(source_dir, "metadata.json"), "w") as f:
            f.write('{"version": "test", "accuracy": 0.95}')

        # Test copy
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)

        shutil.copytree(source_dir, target_dir)

        # Verify copy worked
        if os.path.exists(target_dir):
            files = os.listdir(target_dir)
            print(f"   ✅ Copy successful, files: {files}")

            # Clean up
            shutil.rmtree(source_dir)
            shutil.rmtree(target_dir)
            return True
        else:
            print(f"   ❌ Target directory not created")
            return False

    except Exception as e:
        print(f"   ❌ Copy failed: {e}")
        # Clean up on error
        if os.path.exists(source_dir):
            shutil.rmtree(source_dir)
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        return False


def test_production_config():
    """Test production configuration."""
    print("\n⚙️ Testing production configuration...")

    try:
        from models.training.config import OUTPUT_CONFIG, validate_config

        print(f"   📁 Base dir: {OUTPUT_CONFIG['base_dir']}")
        print(f"   💾 Save model artifacts: {OUTPUT_CONFIG['save_model_artifacts']}")

        # Validate config
        validate_config()
        print(f"   ✅ Configuration valid")

        return True

    except Exception as e:
        print(f"   ❌ Configuration test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Testing EVEREST Production Training Model Saving")
    print("=" * 60)

    # Run tests
    tests = [
        ("Directory Creation", test_directory_creation),
        ("Production Config", test_production_config),
        ("File Copy Mechanism", test_copy_mechanism),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   ❌ {test_name} crashed: {e}")
            results.append((test_name, False))

    # Test trainer components
    success, trainer = test_trainer_initialization()
    results.append(("Trainer Initialization", success))

    if trainer:
        success, model = test_model_creation(trainer)
        results.append(("Model Creation", success))

        success, data = test_data_loading(trainer)
        results.append(("Data Loading", success))

    # Print summary
    print("\n" + "=" * 60)
    print("🎯 TEST SUMMARY:")

    passed = 0
    total = len(results)

    for test_name, success in results:
        if success:
            print(f"   ✅ {test_name}")
            passed += 1
        else:
            print(f"   ❌ {test_name}")

    print(f"\nResult: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Model saving should work correctly.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")

    print("\n💡 Next step: Run a real training job to verify end-to-end functionality")


if __name__ == "__main__":
    main()
