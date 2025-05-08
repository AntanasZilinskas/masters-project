#!/usr/bin/env python
"""
Apply fixes to EVEREST model and run training for flare prediction.

This script:
1. Applies the TensorFlow tensor serialization fix
2. Patches the EVT and evidential head functions
3. Runs training for a specified flare class and time window

Usage: 
    python models/fix_evt_simple.py --flare M5 --window 24
"""

import os
import sys
import argparse
import tensorflow as tf
import numpy as np
from simple_fix import patch_everest_model

def run_training(flare_class, time_window, epochs=50, batch_size=256):
    """Run training with the fixed EVEREST model"""
    try:
        # Import the training function
        from train_everest import train

        # Run training with the patched model
        print(f"Starting training for {flare_class} flares with {time_window}h window...")
        model = train(time_window, flare_class, auto_increment=True, toy=False, use_advanced_model=True)
        
        return model
    except Exception as e:
        print(f"Error during training: {e}")
        
        # Try alternative approach using standalone_train.py if available
        try:
            print("Trying alternative training script...")
            cmd = f"python models/standalone_train.py --flare {flare_class} --window {time_window} --epochs {epochs} --batch-size {batch_size}"
            print(f"Running command: {cmd}")
            os.system(cmd)
            return True
        except Exception as e2:
            print(f"Error during alternative training: {e2}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Fix EVT head and run EVEREST training")
    parser.add_argument("--flare", default="M5", help="Flare class (M5, M, C)")
    parser.add_argument("--window", default="24", help="Time window in hours (24, 48, 72)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for training")
    
    args = parser.parse_args()
    
    # Apply the patches first
    success = patch_everest_model()
    
    if not success:
        print("Failed to apply patches. Exiting.")
        return
    
    # Check TensorFlow GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"SUCCESS: Found GPU: {gpus[0]}")
        # Set memory growth to avoid OOM errors
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except:
                pass
    else:
        print("WARNING: No GPU found, using CPU only (training will be slow)")
    
    # Run training
    run_training(args.flare, args.window, args.epochs, args.batch_size)

if __name__ == "__main__":
    main() 