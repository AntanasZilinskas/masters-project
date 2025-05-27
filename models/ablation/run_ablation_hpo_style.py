#!/usr/bin/env python3
"""
EVEREST Ablation Study Runner - HPO Style

This script follows the exact same pattern as the working HPO runner.
"""

import sys
import argparse
from pathlib import Path
import os

# Add project root to path (EXACT same as HPO)
project_root = Path(__file__).parent.parent.parent  # Go up to masters-project root
sys.path.insert(0, str(project_root))

# Change working directory to project root to ensure relative paths work (EXACT same as HPO)
os.chdir(project_root)

# Now import from models directory (EXACT same as HPO)
from models.utils import get_training_data, get_testing_data


def print_banner():
    """Print welcome banner."""
    print("=" * 80)
    print("🔬 EVEREST Ablation Study Framework")
    print("   Component and Sequence Length Ablations")
    print("=" * 80)


def validate_data(flare_class, time_window):
    """Validate data availability (EXACT same as HPO)."""
    print(f"   • Validating data availability...")
    
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


def validate_gpu():
    """Validate GPU configuration (EXACT same as HPO)."""
    try:
        import torch
        print(f"   • Validating GPU configuration...")
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            current_gpu = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_gpu)
            print(f"   ✅ GPU available: {gpu_name} (device {current_gpu}/{gpu_count})")
            return True
        else:
            print(f"   ❌ GPU not available - ablation requires GPU")
            return False
            
    except Exception as e:
        print(f"   ❌ GPU validation failed: {e}")
        return False


def run_single_ablation(variant, seed, sequence=None):
    """Run a single ablation experiment (following HPO pattern)."""
    print(f"\n🎯 Running ablation: {variant}, seed {seed}")
    if sequence:
        print(f"   Sequence variant: {sequence}")
    
    # Validate data first (EXACT same as HPO)
    if not validate_data("M5", "72"):
        return False
    
    # Validate GPU (EXACT same as HPO)
    if not validate_gpu():
        return False
    
    try:
        # Import ablation trainer (same pattern as HPO imports)
        from models.ablation.trainer import train_ablation_variant
        
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
    """Main function (EXACT same structure as HPO)."""
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