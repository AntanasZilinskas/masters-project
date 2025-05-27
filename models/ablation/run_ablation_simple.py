#!/usr/bin/env python3
"""
EVEREST Ablation Study Runner - Simple Version

This script follows the exact same pattern as the working HPO runner.
"""

import sys
import argparse
from pathlib import Path
import os

# Add project root to path (same as HPO)
project_root = Path(__file__).parent.parent.parent  # Go up to masters-project root
sys.path.insert(0, str(project_root))

# Change working directory to project root to ensure relative paths work (same as HPO)
os.chdir(project_root)

# Now import from models directory (same as HPO)
from models.utils import get_training_data, get_testing_data


def print_banner():
    """Print welcome banner."""
    print("=" * 80)
    print("🔬 EVEREST Ablation Study Framework")
    print("   Component and Sequence Length Ablations")
    print("=" * 80)


def validate_environment():
    """Validate environment like HPO does."""
    print("🧪 Validating environment...")
    
    # Test basic imports
    try:
        import torch
        print(f"   ✅ PyTorch: {torch.__version__}")
        
        import numpy as np
        print(f"   ✅ NumPy: {np.__version__}")
        
        # Test GPU
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"   ✅ GPU available: {gpu_name}")
        else:
            print("   ⚠️ GPU not available - using CPU")
            
    except Exception as e:
        print(f"   ❌ Basic imports failed: {e}")
        return False
    
    # Test EVEREST imports
    try:
        from models.solarknowledge_ret_plus import RETPlusWrapper
        print("   ✅ EVEREST model imported")
        
        from models.utils import get_training_data
        print("   ✅ Utils imported")
        
    except Exception as e:
        print(f"   ❌ EVEREST imports failed: {e}")
        return False
    
    return True


def validate_data(flare_class, time_window):
    """Validate data availability (same as HPO)."""
    print(f"   • Validating data for {flare_class}-class, {time_window}h...")
    
    try:
        X_train, y_train = get_training_data(time_window, flare_class)
        X_test, y_test = get_testing_data(time_window, flare_class)
        
        if X_train is None or y_train is None:
            print(f"   ❌ Training data not found for {flare_class}/{time_window}h")
            return False
            
        if X_test is None or y_test is None:
            print(f"   ❌ Testing data not found for {flare_class}/{time_window}h")
            return False
            
        print(f"   ✅ Data validated: {len(X_train)} train, {len(X_test)} test samples")
        return True
        
    except Exception as e:
        print(f"   ❌ Data validation failed: {e}")
        return False


def run_single_ablation(variant, seed, sequence=None):
    """Run a single ablation experiment."""
    print(f"\n🔬 Running ablation: {variant}, seed {seed}")
    if sequence:
        print(f"   Sequence variant: {sequence}")
    
    # Validate data first
    if not validate_data("M5", "72"):
        return False
    
    try:
        # Import ablation trainer (same pattern as HPO imports)
        sys.path.append('models/ablation')
        from trainer import train_ablation_variant
        
        # Run the training
        results = train_ablation_variant(variant, seed, sequence)
        
        print(f"   ✅ Completed successfully!")
        print(f"   • TSS: {results['final_metrics']['tss']:.4f}")
        print(f"   • F1: {results['final_metrics']['f1']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Ablation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="EVEREST Ablation Study Runner")
    
    parser.add_argument("--variant", required=True, 
                       choices=["full_model", "no_evidential", "no_evt", "mean_pool", 
                               "cross_entropy", "no_precursor", "fp32_training"],
                       help="Ablation variant to run")
    
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    
    parser.add_argument("--sequence", 
                       choices=["seq_5", "seq_7", "seq_10", "seq_15", "seq_20"],
                       help="Sequence length variant")
    
    args = parser.parse_args()
    
    print_banner()
    
    # Validate environment first (same as HPO)
    if not validate_environment():
        print("❌ Environment validation failed")
        return 1
    
    # Run the ablation
    success = run_single_ablation(args.variant, args.seed, args.sequence)
    
    if success:
        print("\n🎉 Ablation completed successfully!")
        return 0
    else:
        print("\n❌ Ablation failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 