#!/usr/bin/env python
"""
Test script to verify model saving to the new directory structure
"""

import os
import numpy as np
import tensorflow as tf
from everest_model import EVEREST
from model_tracking import save_model_with_metadata, get_next_version

def main():
    """Create a simple test model and save it to the new directory structure"""
    print("Testing model saving to models/trained_models directory...")
    
    # Create a simple test model
    print("Creating test model...")
    model = EVEREST(use_advanced_heads=True)
    
    # Generate some random training data
    seq_len, feat = 100, 14
    X = np.random.random((32, seq_len, feat)).astype("float32")
    y = tf.keras.utils.to_categorical(np.random.randint(0, 2, size=32), 2)
    
    # Build and compile the model
    model.build_base_model((seq_len, feat))
    model.compile()
    
    # Create mock history instead of training
    print("Creating mock training history...")
    mock_history = {
        'loss': [0.9, 0.8, 0.7],
        'accuracy': [0.6, 0.7, 0.8],
        'val_loss': [0.85, 0.75, 0.65],
        'val_accuracy': [0.65, 0.75, 0.85]
    }
    
    # Create some test metrics
    metrics = {
        "accuracy": 0.85,
        "precision": 0.75,
        "recall": 0.80,
        "tss": 0.70,
        "final_loss": 0.7,
        "final_val_loss": 0.65
    }
    
    # Create hyperparameters
    hyperparams = {
        "learning_rate": 1e-4,
        "batch_size": 32,
        "dropout": 0.3,
        "embed_dim": 128,
        "num_transformer_blocks": 4,
        "uses_evidential": True,
        "uses_evt": True
    }
    
    # Save the model to the new directory structure
    flare_class = "TEST"
    time_window = "24"
    version = get_next_version(flare_class, time_window)
    
    print(f"Saving model version v{version} to models/trained_models...")
    model_dir = save_model_with_metadata(
        model=model,
        metrics=metrics,
        hyperparams=hyperparams,
        history=mock_history,
        version=version,
        flare_class=flare_class,
        time_window=time_window,
        description="Test model for new directory structure"
    )
    
    # Verify model directory
    print(f"Model saved to: {model_dir}")
    if os.path.exists(model_dir):
        print("✅ Model directory exists")
        
        # Check files
        files_to_check = [
            "model_weights.weights.h5",
            "metadata.json",
            "history.npy"
        ]
        
        for file in files_to_check:
            path = os.path.join(model_dir, file)
            if os.path.exists(path):
                print(f"✅ {file} exists: {path}")
                # Print file size
                size = os.path.getsize(path)
                print(f"   Size: {size} bytes")
            else:
                print(f"❌ {file} not found: {path}")
    else:
        print(f"❌ Model directory not created: {model_dir}")
    
    # Test loading model
    print("\nTesting model loading...")
    load_model = EVEREST(use_advanced_heads=True)
    load_model.build_base_model((seq_len, feat))
    try:
        load_model.load_weights(w_dir=model_dir)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
    
    print("\nTest completed successfully!")
    return model_dir

if __name__ == "__main__":
    main() 