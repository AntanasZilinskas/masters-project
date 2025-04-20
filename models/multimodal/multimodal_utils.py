"""
Utility functions for loading and preparing multimodal data (SHARP parameters + SDO images)
for the SolarKnowledge model.
"""

import os

import h5py
import numpy as np
import pandas as pd

from utils import (
    data_transform,
    get_class_num,
    load_data,
    mask_value,
    n_features,
    series_len,
    start_feature,
)


def get_multimodal_data(time_window, flare_class, is_training=True):
    """
    Load multimodal data (SHARP parameters + SDO images) for the given time window and flare class.

    Parameters:
    -----------
    time_window : str
        Time window (24, 48, 72)
    flare_class : str
        Flare class (C, M, M5)
    is_training : bool
        Whether to load training or testing data

    Returns:
    --------
    X_params : numpy.ndarray
        SHARP parameters time series data
    X_images : numpy.ndarray
        Preprocessed magnetogram images
    y : numpy.ndarray
        Labels (0 for negative, 1 for positive)
    """
    # Determine file paths
    data_type = (
        "testing_data" if is_training else "testing_data"
    )  # Adjust as needed
    csv_file = f"Nature_data/{data_type}_{flare_class}_{time_window}.csv"
    h5_file = f"Nature_data/multimodal_data_{flare_class}_{time_window}.h5"

    # Check if files exist
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    if not os.path.exists(h5_file):
        raise FileNotFoundError(f"H5 file not found: {h5_file}")

    # Load SHARP parameters from CSV
    X_params_all, y_all, _ = load_data(
        datafile=csv_file,
        flare_label=flare_class,
        series_len=series_len,
        start_feature=start_feature,
        n_features=n_features,
        mask_value=mask_value,
    )

    # Convert labels
    y_all = np.array([get_class_num(c) for c in y_all])

    # Load aligned images from H5 file
    with h5py.File(h5_file, "r") as hf:
        images = hf["images"][:]
        indices = hf["indices"][:]

    # Filter SHARP parameters to match available images
    X_params = X_params_all[indices]
    y = y_all[indices]
    X_images = images

    print(f"Loaded multimodal data for {flare_class}_{time_window}:")
    print(f"  SHARP parameters shape: {X_params.shape}")
    print(f"  Images shape: {X_images.shape}")
    print(f"  Labels shape: {y.shape}")
    print(f"  Positive samples: {np.sum(y)}")
    print(f"  Negative samples: {len(y) - np.sum(y)}")

    return X_params, X_images, y


def get_multimodal_training_data(time_window, flare_class):
    """Get multimodal training data."""
    return get_multimodal_data(time_window, flare_class, is_training=True)


def get_multimodal_testing_data(time_window, flare_class):
    """Get multimodal testing data."""
    return get_multimodal_data(time_window, flare_class, is_training=False)
