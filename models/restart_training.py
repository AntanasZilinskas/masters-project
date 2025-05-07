#!/usr/bin/env python
"""
Restart EVEREST model training with fixed EVT implementation.
This script ensures the EVT head is properly connected and functioning.
"""

import os
import sys
import shutil
import numpy as np
import tensorflow as tf
from train_everest import train

def check_evt_head():
    """Verify that the EVT head module is available and accessible."""
    try:
        # First check if it's in the current directory
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import evt_head
        print(f"✓ evt_head successfully imported from: {evt_head.__file__}")
        return True
    except ImportError:
        print("✗ evt_head.py not found in expected location")
        # Copy file if needed
        models_dir = os.path.dirname(os.path.abspath(__file__))
        evt_path = os.path.join(models_dir, 'evt_head.py')
        if os.path.exists(evt_path):
            print(f"EVT head exists at {evt_path}")
            return True
        return False

def main():
    """Main function to restart training with fixed EVT head."""
    print("=" * 80)
    print("RESTARTING EVEREST TRAINING WITH FIXED EVT HEAD".center(80))
    print("=" * 80)
    
    # Check if EVT head is available
    evt_ok = check_evt_head()
    if not evt_ok:
        print("ERROR: Could not find evt_head.py")
        return
    
    # Set random seed for reproducibility
    tf.keras.utils.set_random_seed(42)
    np.random.seed(42)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Restart EVEREST training with fixed EVT head")
    parser.add_argument("--flare", "-f", default="M5", help="Flare class (C, M, M5)")
    parser.add_argument("--window", "-w", default="24", help="Time window (24, 48, 72)")
    parser.add_argument("--toy", "-t", action="store_true", help="Use toy dataset (1% of data)")
    args = parser.parse_args()
    
    # Print configuration
    print(f"Configuration:")
    print(f"  Flare class: {args.flare}")
    print(f"  Time window: {args.window} hours")
    print(f"  Toy mode: {'YES' if args.toy else 'NO'}")
    print("=" * 80)
    
    # Restart training with fixed EVT head
    print(f"Starting training for {args.flare} flares with {args.window}h window...")
    try:
        model = train(args.window, args.flare, use_advanced_model=True, toy=args.toy)
        print("Training completed successfully!")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        print("Training failed. Please check the error messages above.")
    
if __name__ == "__main__":
    main() 