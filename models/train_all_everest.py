#!/usr/bin/env python
"""
Run all EVEREST model trainings across different flare classes and time windows.

Usage:
    python train_all_everest.py [--standard]

This script automates training of EVEREST models for different combinations of 
flare classes and forecast windows. By default, it trains the advanced model
with evidential and EVT heads. Use the --standard flag to train the simpler
model without those advanced features.
"""

import subprocess
import time
import os
import argparse
from train_everest import train
from utils import supported_flare_class

# Define parameters
FLARE_CLASSES = ["C", "M", "M5"]  # All supported flare classes
TIME_WINDOWS = ["24", "48", "72"]  # Time windows in hours

def train_all(use_advanced=True, toy_mode=False):
    """
    Train all combinations of models.
    
    Args:
        use_advanced: Whether to use the advanced model with evidential/EVT heads
        toy_mode: Whether to use toy mode (1% of data) for quick testing
    """
    
    model_type = "advanced" if use_advanced else "standard"
    print(f"Training {model_type} EVEREST models for all flare classes and time windows")
    
    if toy_mode:
        print("Using TOY mode (1% of data) for quick testing")
    
    for flare_class in FLARE_CLASSES:
        for time_window in TIME_WINDOWS:
            print("\n" + "="*80)
            print(f"Training {model_type} EVEREST model for {flare_class} flares with {time_window}h window")
            print("="*80 + "\n")
            
            # Either call the train function directly (in-process)
            # model = train(time_window, flare_class, use_advanced_model=use_advanced, toy=toy_mode)
            
            # Or use subprocess to run in a separate process (more robust to failures)
            toy_flag = "--toy 1" if toy_mode else ""
            adv_flag = "--advanced 1" if use_advanced else "--advanced 0"
            cmd = f"python models/train_everest.py --specific-flare {flare_class} --specific-window {time_window} {adv_flag} {toy_flag}"
            try:
                subprocess.run(cmd, shell=True, check=True)
                print(f"Successfully trained {flare_class}-{time_window}h model")
            except subprocess.CalledProcessError:
                print(f"Error training {flare_class}-{time_window}h model")
            
            # Sleep briefly between runs to allow system to cool down if needed
            time.sleep(2)
            
    print("\nAll models trained!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EVEREST models for all flare classes and time windows")
    parser.add_argument("--standard", action="store_true", help="Train standard models without evidential/EVT heads")
    parser.add_argument("--toy", action="store_true", help="Use toy mode (1% of data) for quick testing")
    args = parser.parse_args()
    
    train_all(use_advanced=not args.standard, toy_mode=args.toy) 