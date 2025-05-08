#!/usr/bin/env python
"""
Simple script to test TensorFlow installation on HPC
"""
import sys
import os

print("Python version:", sys.version)
print("Python executable:", sys.executable)
print("PYTHONPATH:", os.environ.get('PYTHONPATH', 'Not set'))
print("Sys path:", sys.path)

try:
    import tensorflow as tf
    print("\nTensorFlow successfully imported!")
    print("TensorFlow version:", tf.__version__)
    print("TensorFlow path:", tf.__file__)
    
    # Check for GPU availability
    print("\nGPU devices available:", tf.config.list_physical_devices('GPU'))
    
    # Try a simple TensorFlow operation
    a = tf.constant([[1, 2], [3, 4]])
    b = tf.constant([[5, 6], [7, 8]])
    c = tf.matmul(a, b)
    print("\nSimple matrix multiplication test:")
    print(c.numpy())
    print("\nTensorFlow test PASSED")
    
except ImportError as e:
    print("\nFailed to import TensorFlow:", e)
    print("\nChecking installed packages:")
    try:
        # Try to import pip and list packages
        import pip
        print("\nInstalled packages:")
        packages = [p.decode('utf-8') if isinstance(p, bytes) else p 
                   for p in os.popen('pip list').read().splitlines()]
        for p in packages:
            if 'tensorflow' in str(p).lower():
                print(p)
    except:
        print("Could not check installed packages") 