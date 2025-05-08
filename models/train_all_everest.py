#!/usr/bin/env python
"""
Run all EVEREST model trainings across different flare classes and time windows.

Usage:
    python models/train_all_everest.py [--simple] [--toy] [--fix]

This script automates training of EVEREST models for different combinations of 
flare classes and forecast windows. By default, it trains the advanced model
with evidential and EVT heads. Use the --simple flag to train the simpler
model without those advanced features.
"""

import subprocess
import time
import os
import argparse
from utils import supported_flare_class

# Define parameters
FLARE_CLASSES = ["C", "M", "M5"]  # All supported flare classes
TIME_WINDOWS = ["24", "48", "72"]  # Time windows in hours

def train_all(use_advanced=True, toy_mode=False, epochs=50, use_fixed_evt=True):
    """
    Train all combinations of models.
    
    Args:
        use_advanced: Whether to use the advanced model with evidential/EVT heads
        toy_mode: Whether to use toy mode (1% of data) for quick testing
        epochs: Number of epochs to train for
        use_fixed_evt: Whether to use the fixed EVT implementation
    """
    
    model_type = "advanced" if use_advanced else "standard"
    print(f"Training {model_type} EVEREST models for all flare classes and time windows")
    
    if toy_mode:
        print("Using TOY mode (fewer samples) for quick testing")
        
    if use_fixed_evt:
        print("Using fixed EVT head implementation for better TSS performance")
    
    for flare_class in FLARE_CLASSES:
        for time_window in TIME_WINDOWS:
            print("\n" + "="*80)
            print(f"Training {model_type} EVEREST model for {flare_class} flares with {time_window}h window")
            print("="*80 + "\n")
            
            # Choose the appropriate training script based on availability and preferences
            if use_fixed_evt and os.path.exists("models/fix_evt_simple.py"):
                # Use our custom simplified script with fixed EVT head
                cmd = f"python models/fix_evt_simple.py --flare {flare_class} --window {time_window} --epochs {epochs}"
                if toy_mode:
                    print("Note: Toy mode not available with fix_evt_simple.py")
            elif use_fixed_evt and os.path.exists("models/fix_evt_everest.py"):
                # Use the comprehensive fix script
                cmd = f"python models/fix_evt_everest.py --flare {flare_class} --window {time_window}"
                if toy_mode:
                    cmd += " --toy"
            elif os.path.exists("models/standalone_train.py"):
                # Use our standalone training script
                simple_flag = "--simple" if not use_advanced else ""
                cmd = f"python models/standalone_train.py --flare {flare_class} --window {time_window} --epochs {epochs} {simple_flag}"
                if toy_mode:
                    # Reduce epochs for toy mode
                    cmd = cmd.replace(f"--epochs {epochs}", "--epochs 5")
            elif os.path.exists("models/simple_train.py"):
                # Use the simple_train.py script
                simple_flag = "--simple" if not use_advanced else ""
                epochs_flag = f"--epochs {3 if toy_mode else epochs}"
                cmd = f"python models/simple_train.py --flare {flare_class} --window {time_window} {simple_flag} {epochs_flag}"
            else:
                # Fall back to original script
                cmd = f"python models/train_everest.py --specific-flare {flare_class} --specific-window {time_window}"
                if toy_mode:
                    cmd += " --toy 1"
                print("Warning: Using original train_everest.py script - results may not be optimal")
            
            try:
                print(f"Running command: {cmd}")
                subprocess.run(cmd, shell=True, check=True)
                print(f"Successfully trained {flare_class}-{time_window}h model")
            except subprocess.CalledProcessError as e:
                print(f"Error training {flare_class}-{time_window}h model: {e}")
            
            # Sleep briefly between runs to allow system to cool down if needed
            time.sleep(5)
            
    print("\nAll models trained!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EVEREST models for all flare classes and time windows")
    parser.add_argument("--simple", action="store_true", help="Train standard models without evidential/EVT heads")
    parser.add_argument("--toy", action="store_true", help="Use toy mode (fewer samples) for quick testing")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train for (default: 50)")
    parser.add_argument("--fix", action="store_true", help="Use fixed EVT implementation for better TSS")
    parser.add_argument("--no-fix", action="store_true", help="Do not use fixed EVT implementation")
    
    args = parser.parse_args()
    
    # Determine whether to use fixed EVT
    use_fixed_evt = not args.no_fix
    if args.fix:
        use_fixed_evt = True
    
    train_all(use_advanced=not args.simple, toy_mode=args.toy, epochs=args.epochs, use_fixed_evt=use_fixed_evt) 