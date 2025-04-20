# -----------------------------
# File: test_everest.py
# -----------------------------
"""
Test script to verify that the modified EVEREST model with custom Performer works.
"""

import numpy as np
import tensorflow as tf
from everest_model import EVEREST

def test_everest_model():
    print("Testing EVEREST model with custom Performer implementation...")
    
    # Create dummy data
    seq_len, features = 100, 14  # Typical dimensions for SHARP data
    batch_size = 4
    
    # Random input data
    X = np.random.random((batch_size, seq_len, features)).astype("float32")
    y = tf.keras.utils.to_categorical(np.random.randint(0, 2, size=batch_size), 2)
    
    # Create the model
    print("Creating EVEREST model...")
    model = EVEREST()
    model.build_base_model((seq_len, features))
    model.compile()
    
    # Display model summary
    print("\nModel summary:")
    model.model.summary()
    
    # Run a single training epoch
    print("\nRunning a test training epoch...")
    model.fit(X, y, epochs=1)
    
    # Test Monte Carlo dropout prediction
    print("\nTesting Monte Carlo dropout prediction...")
    mean_preds, std_preds = model.mc_predict(X)
    
    print(f"MC prediction shapes: mean={mean_preds.shape}, std={std_preds.shape}")
    print(f"Average uncertainty (std): {std_preds.mean()}")
    
    print("\nTest completed successfully!")
    return True

if __name__ == "__main__":
    test_everest_model()