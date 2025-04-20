"""
Data loading and processing utilities for the Solar Knowledge package.
"""

import os
import h5py
import numpy as np


def load_tiny_dataset(data_dir="datasets/tiny_sample", file_name="sharp_ci_sample.h5"):
    """
    Load the tiny SHARP dataset for CI testing.
    
    Parameters
    ----------
    data_dir : str
        Directory containing the dataset.
    file_name : str
        Name of the HDF5 file containing the dataset.
        
    Returns
    -------
    X : numpy.ndarray
        Array of shape (n_samples, sequence_length, n_features) containing the input data.
    y : numpy.ndarray
        Array of shape (n_samples,) containing the target labels (0 or 1).
    """
    file_path = os.path.join(data_dir, file_name)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset file not found at {file_path}. "
            f"Please run 'python scripts/generate_tiny_sample.py' to create it."
        )
    
    with h5py.File(file_path, "r") as f:
        X = f["X"][:]
        y = f["y"][:]
        
        # Print some info about the dataset
        print(f"Loaded SHARP-2023-CI-Sample dataset:")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  Class distribution: {np.bincount(y)}")
        
        if "created" in f.attrs:
            print(f"  Created: {f.attrs['created']}")
    
    return X, y


def create_train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Split the dataset into training and testing sets.
    
    Parameters
    ----------
    X : numpy.ndarray
        Input features.
    y : numpy.ndarray
        Target labels.
    test_size : float
        Fraction of the dataset to use for testing (0.0 to 1.0).
    random_state : int or None
        Random seed for reproducibility.
        
    Returns
    -------
    X_train, X_test, y_train, y_test : numpy.ndarray
        Split dataset.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Get indices for each class
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]
    
    # Shuffle indices
    np.random.shuffle(pos_indices)
    np.random.shuffle(neg_indices)
    
    # Split each class
    n_pos_test = max(1, int(len(pos_indices) * test_size))
    n_neg_test = max(1, int(len(neg_indices) * test_size))
    
    pos_train_idx = pos_indices[n_pos_test:]
    pos_test_idx = pos_indices[:n_pos_test]
    neg_train_idx = neg_indices[n_neg_test:]
    neg_test_idx = neg_indices[:n_neg_test]
    
    # Combine indices
    train_indices = np.concatenate([pos_train_idx, neg_train_idx])
    test_indices = np.concatenate([pos_test_idx, neg_test_idx])
    
    # Shuffle again
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    
    # Create splits
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Example usage
    X, y = load_tiny_dataset()
    X_train, X_test, y_train, y_test = create_train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Test shapes: X={X_test.shape}, y={y_test.shape}") 