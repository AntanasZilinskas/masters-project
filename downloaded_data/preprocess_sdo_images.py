"""
Preprocess SDO/HMI magnetogram images for use in multimodal deep learning.
This script:
1. Loads magnetogram FITS files
2. Performs normalization and enhancement
3. Crops to active region
4. Resizes to consistent dimensions
5. Saves processed images in a format ready for the model
"""

import glob
import os
from datetime import datetime

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from astropy.io import fits
from astropy.time import Time
from astropy.visualization import AsinhStretch, ImageNormalize, SqrtStretch
from skimage.transform import resize
from sunpy.map import Map
from tqdm import tqdm


def parse_time(time_str):
    """Parse time string from CSV to datetime object."""
    time_str = str(time_str).strip()
    time_str = time_str.replace("T", " ").replace("Z", "")
    return datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f")


def find_closest_image(timestamp, image_files, max_minutes=15):
    """Find the closest image file to the given timestamp."""
    target_time = parse_time(timestamp)
    closest_file = None
    min_diff = float("inf")

    for img_file in image_files:
        # Extract time from filename (adjust pattern based on your filenames)
        file_time_str = os.path.basename(img_file).split(".")[2]
        try:
            file_time = datetime.strptime(file_time_str, "%Y%m%d_%H%M%S")
            time_diff = abs((file_time - target_time).total_seconds())
            if time_diff < min_diff and time_diff < max_minutes * 60:
                min_diff = time_diff
                closest_file = img_file
        except BaseException:
            continue

    return closest_file, min_diff / 60  # Return file and difference in minutes


def preprocess_magnetogram(fits_file, harp_number=None, crop_size=256):
    """
    Preprocess a magnetogram FITS file:
    1. Load the data
    2. Apply normalization
    3. Crop to active region if HARP number provided
    4. Resize to standard dimensions
    """
    try:
        # Load the FITS file
        hmi_map = Map(fits_file)
        data = hmi_map.data

        # Handle NaN values
        data = np.nan_to_num(data)

        # Apply normalization to enhance features
        # Use a symmetric logarithmic scale to handle positive and negative
        # values
        vmin, vmax = np.percentile(data[~np.isnan(data)], [1, 99])
        data_norm = np.clip(data, vmin, vmax)

        # If we have HARP info, try to crop to the active region
        if harp_number is not None:
            # This would require additional code to extract HARP region
            # For now, we'll use a simple center crop
            h, w = data.shape
            center_h, center_w = h // 2, w // 2
            start_h = max(0, center_h - crop_size // 2)
            start_w = max(0, center_w - crop_size // 2)
            data_norm = data_norm[
                start_h : start_h + crop_size, start_w : start_w + crop_size
            ]

        # Resize to standard dimensions
        data_resized = resize(data_norm, (128, 128), anti_aliasing=True)

        # Normalize to [-1, 1] range for neural network
        data_final = (
            2
            * (data_resized - np.min(data_resized))
            / (np.max(data_resized) - np.min(data_resized))
            - 1
        )

        return data_final

    except Exception as e:
        print(f"Error processing {fits_file}: {e}")
        return None


def create_aligned_dataset(
    csv_file, image_dir, output_file, time_window, flare_class
):
    """
    Create a dataset with aligned SHARP parameters and magnetogram images.

    Parameters:
    -----------
    csv_file : str
        Path to CSV file with SHARP parameters
    image_dir : str
        Directory containing magnetogram FITS files
    output_file : str
        Path to save the aligned dataset
    time_window : str
        Time window (24, 48, 72)
    flare_class : str
        Flare class (C, M, M5)
    """
    # Load CSV data
    df = pd.read_csv(csv_file)

    # Get list of all image files
    image_files = glob.glob(os.path.join(image_dir, "*.fits"))

    # Create arrays to store aligned data
    images = []
    indices = []

    print(f"Processing {len(df)} entries from {csv_file}...")

    # Process each row in the CSV
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        timestamp = row[1]  # Assuming timestamp is in column 1
        harp_number = row[3]  # Assuming HARP number is in column 3

        # Find the closest image
        closest_file, time_diff = find_closest_image(timestamp, image_files)

        if closest_file and time_diff <= 15:  # Only use if within 15 minutes
            # Preprocess the image
            processed_img = preprocess_magnetogram(closest_file, harp_number)

            if processed_img is not None:
                images.append(processed_img)
                indices.append(idx)

    # Convert to numpy arrays
    images = np.array(images)
    indices = np.array(indices)

    # Save the aligned dataset
    with h5py.File(output_file, "w") as hf:
        hf.create_dataset("images", data=images)
        hf.create_dataset("indices", data=indices)
        hf.attrs["csv_file"] = csv_file
        hf.attrs["time_window"] = time_window
        hf.attrs["flare_class"] = flare_class

    print(f"Created aligned dataset with {len(images)} entries")
    print(f"Saved to {output_file}")

    return len(images)


if __name__ == "__main__":
    # Example usage
    for time_window in ["24", "48", "72"]:
        for flare_class in ["C", "M", "M5"]:
            # Paths for training data
            csv_file = (
                f"Nature_data/testing_data_{flare_class}_{time_window}.csv"
            )
            image_dir = (
                f"SDO/data/hmi.m_45s"  # Adjust based on your download path
            )
            output_file = (
                f"Nature_data/multimodal_data_{flare_class}_{time_window}.h5"
            )

            if os.path.exists(csv_file) and os.path.exists(image_dir):
                create_aligned_dataset(
                    csv_file, image_dir, output_file, time_window, flare_class
                )
            else:
                print(f"Missing data for {flare_class}_{time_window}")
