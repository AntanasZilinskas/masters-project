#!/usr/bin/env python
"""
Run all EVEREST model trainings across different flare classes and time windows.

Usage:
    python models/train_all_everest.py [--simple] [--toy]

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

def train_all(use_advanced=True, toy_mode=False, epochs=50):
    """
    Train all combinations of models.
    
    Args:
        use_advanced: Whether to use the advanced model with evidential/EVT heads
        toy_mode: Whether to use toy mode (1% of data) for quick testing
        epochs: Number of epochs to train for
    """
    
    model_type = "advanced" if use_advanced else "standard"
    print(f"Training {model_type} EVEREST models for all flare classes and time windows")
    
    if toy_mode:
        print("Using TOY mode (few epochs) for quick testing")
    
    for flare_class in FLARE_CLASSES:
        for time_window in TIME_WINDOWS:
            print("\n" + "="*80)
            print(f"Training {model_type} EVEREST model for {flare_class} flares with {time_window}h window")
            print("="*80 + "\n")
            
            # Use simple_train.py with appropriate flags
            simple_flag = "--simple" if not use_advanced else ""
            epochs_flag = f"--epochs {3 if toy_mode else epochs}"
            cmd = f"python models/simple_train.py --flare {flare_class} --window {time_window} {simple_flag} {epochs_flag}"
            
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
    parser.add_argument("--toy", action="store_true", help="Use toy mode (fewer epochs) for quick testing")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train for (default: 50)")
    args = parser.parse_args()
    
    train_all(use_advanced=not args.simple, toy_mode=args.toy, epochs=args.epochs) 