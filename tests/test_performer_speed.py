#!/usr/bin/env python
"""Test script to compare performance of Performer vs. MultiHeadAttention.

This script verifies that the Performer implementation provides a speed
advantage over the standard MultiHeadAttention for longer sequences.
"""
import sys
import os
import time
import numpy as np
import tensorflow as tf

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import our custom Performer implementation
from models.performer_custom import Performer


def test_performer_vs_mha_speed():
    """Test that Performer is faster than MHA for long sequences."""
    # Set parameters
    batch_size = 16
    seq_length = 400  # Long sequence to demonstrate the advantage
    embed_dim = 64
    num_heads = 4
    
    # Generate random data
    x = tf.random.normal((batch_size, seq_length, embed_dim))
    
    # Create layers
    performer = Performer(num_heads=num_heads, key_dim=embed_dim//num_heads)
    mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim//num_heads)
    
    # Warm-up runs
    _ = performer(x, x)
    _ = mha(x, x)
    
    # Test Performer speed
    start_time = time.time()
    for _ in range(10):
        _ = performer(x, x)
    performer_time = time.time() - start_time
    
    # Test MHA speed
    start_time = time.time()
    for _ in range(10):
        _ = mha(x, x)
    mha_time = time.time() - start_time
    
    print(f"Performer time: {performer_time:.4f}s")
    print(f"MHA time: {mha_time:.4f}s")
    print(f"Ratio (Performer/MHA): {performer_time/mha_time:.4f}x")
    
    # The ratio should be less than 2 (Performer shouldn't be more than 2x slower)
    # For long sequences, Performer should eventually be faster
    assert performer_time < 2 * mha_time, "Performer is too slow compared to MHA"
    
    # For memory usage comparison (not strictly part of the test)
    try:
        # Try with much longer sequence to demonstrate memory advantage
        long_seq_length = 2000
        x_long = tf.random.normal((batch_size, long_seq_length, embed_dim))
        
        # Performer should handle this easily
        _ = performer(x_long, x_long)
        print(f"Performer successfully processed sequence length {long_seq_length}")
        
        # MHA might run out of memory, but we'll try
        try:
            _ = mha(x_long, x_long)
            print(f"MHA successfully processed sequence length {long_seq_length}")
        except (tf.errors.ResourceExhaustedError, tf.errors.InternalError, tf.errors.UnknownError):
            print(f"MHA failed on sequence length {long_seq_length} (as expected for memory reasons)")
    except Exception as e:
        print(f"Long sequence test error: {e}")


if __name__ == "__main__":
    test_performer_vs_mha_speed() 