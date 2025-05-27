#!/usr/bin/env python3
"""
Test script to verify EVEREST ablation study framework setup.
"""

import sys
import os

# Add models directory to path
sys.path.append('models')

def test_imports():
    """Test that all ablation modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        from ablation.config import (
            ABLATION_VARIANTS, SEQUENCE_LENGTH_VARIANTS, RANDOM_SEEDS,
            validate_config
        )
        print("✅ Config module imported successfully")
        
        from ablation.trainer import AblationTrainer, train_ablation_variant
        print("✅ Trainer module imported successfully")
        
        from ablation.analysis import AblationAnalyzer
        print("✅ Analysis module imported successfully")
        
        import ablation
        print("✅ Ablation package imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_configuration():
    """Test configuration validation."""
    print("\n🔧 Testing configuration...")
    
    try:
        from ablation.config import validate_config, ABLATION_VARIANTS, SEQUENCE_LENGTH_VARIANTS
        
        validate_config()
        print("✅ Configuration validation passed")
        
        print(f"   📊 {len(ABLATION_VARIANTS)} ablation variants configured")
        print(f"   📏 {len(SEQUENCE_LENGTH_VARIANTS)} sequence length variants configured")
        
        # Test variant configurations
        for variant_name, variant_info in ABLATION_VARIANTS.items():
            config = variant_info["config"]
            weights = config["loss_weights"]
            total_weight = sum(weights.values())
            
            if abs(total_weight - 1.0) > 1e-6:
                print(f"⚠️ {variant_name} loss weights sum to {total_weight:.6f}")
            else:
                print(f"   ✅ {variant_name}: weights normalized")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_directories():
    """Test that required directories can be created."""
    print("\n📁 Testing directory creation...")
    
    try:
        from ablation.config import OUTPUT_CONFIG, create_output_directories
        
        create_output_directories()
        print("✅ Output directories created successfully")
        
        # Check if directories exist
        for key, path in OUTPUT_CONFIG.items():
            if isinstance(path, str) and path.endswith(('results', 'plots', 'logs', 'models')):
                if os.path.exists(path):
                    print(f"   ✅ {key}: {path}")
                else:
                    print(f"   ⚠️ {key}: {path} (not created)")
        
        return True
        
    except Exception as e:
        print(f"❌ Directory test failed: {e}")
        return False

def test_trainer_creation():
    """Test that AblationTrainer can be created."""
    print("\n🏋️ Testing trainer creation...")
    
    try:
        from ablation.trainer import AblationTrainer
        
        # Test creating trainer for different variants
        test_cases = [
            ("full_model", 0, None),
            ("no_evidential", 1, None),
            ("full_model", 0, "seq_15")
        ]
        
        for variant, seed, seq_variant in test_cases:
            trainer = AblationTrainer(variant, seed, seq_variant)
            print(f"   ✅ Created trainer: {trainer.experiment_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Trainer creation failed: {e}")
        return False

def test_analyzer_creation():
    """Test that AblationAnalyzer can be created."""
    print("\n📊 Testing analyzer creation...")
    
    try:
        from ablation.analysis import AblationAnalyzer
        
        analyzer = AblationAnalyzer()
        print("✅ Analyzer created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Analyzer creation failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🔬 EVEREST Ablation Framework Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_configuration,
        test_directories,
        test_trainer_creation,
        test_analyzer_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n🏁 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("✅ All tests passed! Ablation framework is ready.")
        print("\n🚀 Next steps:")
        print("   1. Run a single experiment: python -m ablation.trainer --variant full_model --seed 0")
        print("   2. Run full study: python models/ablation/run_ablation_study.py")
        print("   3. Submit to cluster: cd models/ablation/cluster && ./submit_jobs.sh")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 