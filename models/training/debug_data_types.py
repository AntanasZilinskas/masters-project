#!/usr/bin/env python3
"""Debug script to check data types."""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.utils import get_testing_data

def main():
    print("Loading test data...")
    test_data = get_testing_data(72, 'M5')
    X_test = test_data[0]
    y_test = test_data[1]
    
    print(f"X_test type: {type(X_test)}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test type: {type(y_test)}")
    print(f"y_test shape: {y_test.shape if hasattr(y_test, 'shape') else len(y_test)}")
    
    # Test indexing
    try:
        indices = np.array([0, 1, 2])
        print(f"Test indexing X_test[indices]: {X_test[indices].shape}")
        print(f"Test indexing y_test[indices]: {y_test[indices] if hasattr(y_test, '__getitem__') else 'No indexing'}")
    except Exception as e:
        print(f"Indexing error: {e}")
    
    # Test if it's a pandas Series
    if hasattr(y_test, 'iloc'):
        print("y_test is pandas-like, using iloc")
    elif hasattr(y_test, '__getitem__'):
        print("y_test supports standard indexing")
    else:
        print("y_test indexing method unclear")

if __name__ == "__main__":
    main() 