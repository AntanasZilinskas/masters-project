#!/usr/bin/env python
"""
Run all EVEREST model trainings across different flare classes and time windows.

Usage:
    python train_all_everest.py

This script automates training of EVEREST models for different combinations of 
flare classes and forecast windows.
"""

import subprocess
import time
import os
from train_everest import train
from utils import supported_flare_class

# Define parameters
FLARE_CLASSES = ["C", "M", "M5"]  # All supported flare classes
TIME_WINDOWS = ["24", "48", "72"]  # Time windows in hours

def train_all():
    """Train all combinations of models."""
    
    for flare_class in FLARE_CLASSES:
        for time_window in TIME_WINDOWS:
            print("\n" + "="*80)
            print(f"Training EVEREST model for {flare_class} flares with {time_window}h window")
            print("="*80 + "\n")
            
            # Either call the train function directly (in-process)
            # model = train(time_window, flare_class)
            
            # Or use subprocess to run in a separate process (more robust to failures)
            cmd = f"python train_everest.py --specific-flare {flare_class} --specific-window {time_window}"
            try:
                subprocess.run(cmd, shell=True, check=True)
                print(f"Successfully trained {flare_class}-{time_window}h model")
            except subprocess.CalledProcessError:
                print(f"Error training {flare_class}-{time_window}h model")
            
            # Sleep briefly between runs to allow system to cool down if needed
            time.sleep(2)
            
    print("\nAll models trained!")

if __name__ == "__main__":
    train_all() 