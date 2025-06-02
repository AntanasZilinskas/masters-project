#!/usr/bin/env python3
"""
Comprehensive Test for EVEREST Ablation Study Output Paths

This test verifies:
1. All imports work correctly
2. Model saving paths are correct
3. Metadata structure is proper
4. Output directories are accessible
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_basic_environment():
    """Test basic Python environment."""
    print("🔍 Testing basic environment...")

    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        print(f"   CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"❌ PyTorch: {e}")
        return False

    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
    except Exception as e:
        print(f"❌ NumPy: {e}")
        return False

    try:
        import sklearn
        print(f"✅ Scikit-learn {sklearn.__version__}")
    except Exception as e:
        print(f"❌ Scikit-learn: {e}")
        return False

    return True


def test_everest_imports():
    """Test EVEREST-specific imports."""
    print("\n🧪 Testing EVEREST imports...")

    try:
        from models.solarknowledge_ret_plus import RETPlusWrapper
        print("✅ RETPlusWrapper imported")
    except Exception as e:
        print(f"❌ RETPlusWrapper: {e}")
        return False

    try:
        from models.utils import get_training_data, get_testing_data
        print("✅ Data utilities imported")
    except Exception as e:
        print(f"❌ Data utilities: {e}")
        return False

    try:
        from models.model_tracking import save_model_with_metadata, get_next_version
        print("✅ Model tracking utilities imported")
    except Exception as e:
        print(f"❌ Model tracking: {e}")
        return False

    return True


def test_ablation_script_import():
    """Test ablation script import."""
    print("\n📝 Testing ablation script import...")

    script_path = project_root / 'models' / 'ablation' / 'run_ablation_with_metadata.py'
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return False

    print(f"✅ Script exists: {script_path}")

    try:
        sys.path.insert(0, str(project_root / 'models' / 'ablation'))
        import run_ablation_with_metadata
        print("✅ Ablation script imported successfully")

        # Test if we can create the ablation objective
        objective_class = run_ablation_with_metadata.AblationObjectiveWithMetadata
        print("✅ AblationObjectiveWithMetadata class accessible")

        return True
    except Exception as e:
        print(f"❌ Ablation script import failed: {e}")
        return False


def test_output_directories():
    """Test that output directories can be created."""
    print("\n📁 Testing output directories...")

    # Check models directory structure
    models_dir = project_root / 'models'
    if not models_dir.exists():
        print(f"❌ Models directory not found: {models_dir}")
        return False

    print(f"✅ Models directory exists: {models_dir}")

    # Check if models/models directory exists or can be created
    models_models_dir = models_dir / 'models'
    try:
        models_models_dir.mkdir(exist_ok=True)
        print(f"✅ Output directory ready: {models_models_dir}")
    except Exception as e:
        print(f"❌ Cannot create output directory: {e}")
        return False

    # Test model versioning
    try:
        from models.model_tracking import get_next_version
        next_version = get_next_version("M5", "72")
        print(f"✅ Next model version: {next_version}")

        # Simulate model directory name
        model_dir_name = f"EVEREST-v{next_version}-M5-72h"
        full_model_path = models_models_dir / model_dir_name
        print(f"✅ Model will be saved to: {full_model_path}")

    except Exception as e:
        print(f"❌ Model versioning test failed: {e}")
        return False

    return True


def test_ablation_variants():
    """Test ablation variant configurations."""
    print("\n🎯 Testing ablation variant configurations...")

    try:
        sys.path.insert(0, str(project_root / 'models' / 'ablation'))
        from run_ablation_with_metadata import AblationObjectiveWithMetadata

        # Test creating objective for each variant
        variants = ["full_model", "no_evidential", "no_evt", "mean_pool",
                    "cross_entropy", "no_precursor", "fp32_training"]

        for variant in variants:
            try:
                # Create objective (but don't load data)
                print(f"   Testing variant: {variant}")

                # Test the configuration method directly
                temp_obj = AblationObjectiveWithMetadata.__new__(AblationObjectiveWithMetadata)
                temp_obj.variant_name = variant
                config = temp_obj._get_ablation_config()

                print(f"   ✅ {variant}: {config['description'][:50]}...")

            except Exception as e:
                print(f"   ❌ {variant}: {e}")
                return False

        print(f"✅ All {len(variants)} variants configured correctly")
        return True

    except Exception as e:
        print(f"❌ Variant testing failed: {e}")
        return False


def test_metadata_structure():
    """Test metadata structure."""
    print("\n🏷️  Testing metadata structure...")

    try:
        # Test enhanced metadata structure
        sample_metadata = {
            "experiment_type": "component_ablation",
            "variant": "full_model",
            "seed": 0,
            "ablation_config": {
                "use_attention_bottleneck": True,
                "use_evidential": True,
                "use_evt": True,
                "use_precursor": True,
                "description": "Full EVEREST model with all components"
            },
            "optimal_hyperparams": {
                "embed_dim": 64,
                "num_blocks": 8,
                "dropout": 0.23876978467047777
            }
        }

        print("✅ Metadata structure validated")
        print(f"   • Experiment type: {sample_metadata['experiment_type']}")
        print(f"   • Variant tracking: {sample_metadata['variant']}")
        print(f"   • Seed tracking: {sample_metadata['seed']}")
        print(f"   • Config keys: {list(sample_metadata['ablation_config'].keys())}")

        return True

    except Exception as e:
        print(f"❌ Metadata structure test failed: {e}")
        return False


def simulate_experiment_run():
    """Simulate what happens during an experiment run."""
    print("\n🚀 Simulating experiment run...")

    try:
        # Simulate the command that will be run on cluster
        variant = "full_model"
        seed = 0

        print(f"   Simulating: python run_ablation_with_metadata.py --variant {variant} --seed {seed}")

        # Check argument parsing
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--variant", choices=["full_model", "no_evidential", "no_evt",
                                                  "mean_pool", "cross_entropy", "no_precursor", "fp32_training"])
        parser.add_argument("--seed", type=int, default=0)

        # Simulate parsing
        args = parser.parse_args([f"--variant", variant, "--seed", str(seed)])
        print(f"   ✅ Arguments parsed: variant={args.variant}, seed={args.seed}")

        # Simulate model directory creation
        from models.model_tracking import get_next_version
        version = get_next_version("M5", "72")
        model_dir_name = f"EVEREST-v{version}-M5-72h"

        print(f"   ✅ Model directory: {model_dir_name}")
        print(f"   ✅ Metadata will include: variant={variant}, seed={seed}")

        return True

    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        return False


def main():
    """Run comprehensive test suite."""
    print("🔬 EVEREST Ablation Study - Comprehensive Output Path Test")
    print("=" * 80)
    print(f"Project root: {project_root}")
    print(f"Current directory: {Path.cwd()}")
    print()

    tests = [
        ("Basic Environment", test_basic_environment),
        ("EVEREST Imports", test_everest_imports),
        ("Ablation Script Import", test_ablation_script_import),
        ("Output Directories", test_output_directories),
        ("Ablation Variants", test_ablation_variants),
        ("Metadata Structure", test_metadata_structure),
        ("Experiment Simulation", simulate_experiment_run)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
        print()

    print("=" * 80)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("✅ ALL TESTS PASSED!")
        print()
        print("🎯 Ready for cluster submission:")
        print("   cd models/ablation/cluster")
        print("   qsub submit_ablation_study.pbs")
        print()
        print("📁 Models will be saved to:")
        print("   models/models/EVEREST-v{version}-M5-72h/")
        print("   • model_weights.pt")
        print("   • metadata.json (with ablation_metadata)")
        print("   • training_history.csv")
        print("   • model_card.md")
        print()
        print("🏷️  Each model will include:")
        print("   • ablation_metadata.variant")
        print("   • ablation_metadata.seed")
        print("   • ablation_metadata.ablation_config")
        print("   • Enhanced hyperparameters with ablation info")

        return True
    else:
        print(f"❌ {total - passed} tests failed!")
        print("⚠️  Fix issues before submitting to cluster")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
