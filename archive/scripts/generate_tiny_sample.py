#!/usr/bin/env python
"""
Generate a tiny sample dataset for CI testing.

This script creates a small synthetic dataset that mimics the structure
of the SHARP dataset used in Abduallah et al. (2023), specifically for CI testing.

Reference:
Abduallah, Y., Wang, J.T.L., Wang, H. et al. Operational prediction of solar flares
using a transformer-based framework. Sci Rep 13, 13665 (2023).
"""

import argparse
import os
from datetime import datetime

import h5py
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate tiny sample dataset for CI"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/tiny_sample",
        help="Directory to save the tiny sample dataset",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=30,
        help="Number of samples to generate (half positive, half negative)",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=100,
        help="Length of time sequence for each sample",
    )
    parser.add_argument(
        "--num-features",
        type=int,
        default=14,
        help="Number of features per timestep",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    return parser.parse_args()


def generate_sample(sequence_length, num_features, is_positive, seed=None):
    """Generate a single synthetic sample."""
    if seed is not None:
        np.random.seed(seed)

    # Base signal with some random noise
    base = np.sin(np.linspace(0, 4 * np.pi, sequence_length))[:, np.newaxis]
    noise = np.random.normal(0, 0.2, (sequence_length, num_features))

    # For positive samples (flares), add some characteristic patterns
    if is_positive:
        # Add a spike pattern characteristic of flares
        peak_idx = np.random.randint(
            sequence_length // 3, 2 * sequence_length // 3
        )
        spike = np.zeros((sequence_length, num_features))
        window = 10
        for i in range(
            max(0, peak_idx - window), min(sequence_length, peak_idx + window)
        ):
            dist = abs(i - peak_idx)
            magnitude = (1 - dist / window) * 2.0
            if magnitude > 0:
                spike[i, :] = magnitude * np.random.random(num_features)
    else:
        spike = 0

    # Combine components with different weights for different features
    feature_weights = np.random.random(num_features) * 2
    sample = (base * feature_weights + noise + spike) * 0.5

    return sample


def main():
    args = parse_args()
    np.random.seed(args.seed)

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Number of positive and negative samples
    num_pos = args.num_samples // 2
    num_neg = args.num_samples - num_pos

    # Generate dataset
    X = np.zeros((args.num_samples, args.sequence_length, args.num_features))
    y = np.zeros(args.num_samples, dtype=np.int32)

    print(f"Generating {num_pos} positive and {num_neg} negative samples...")

    # Generate positive samples
    for i in range(num_pos):
        X[i] = generate_sample(
            args.sequence_length,
            args.num_features,
            is_positive=True,
            seed=args.seed + i,
        )
        y[i] = 1

    # Generate negative samples
    for i in range(num_neg):
        X[i + num_pos] = generate_sample(
            args.sequence_length,
            args.num_features,
            is_positive=False,
            seed=args.seed + num_pos + i,
        )
        y[i + num_pos] = 0

    # Save dataset as HDF5 file
    output_file = os.path.join(args.output_dir, "sharp_ci_sample.h5")
    with h5py.File(output_file, "w") as f:
        f.create_dataset("X", data=X)
        f.create_dataset("y", data=y)
        f.attrs["created"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.attrs["num_samples"] = args.num_samples
        f.attrs["positive_samples"] = num_pos
        f.attrs["negative_samples"] = num_neg
        f.attrs["sequence_length"] = args.sequence_length
        f.attrs["num_features"] = args.num_features
        f.attrs["reference"] = "Abduallah et al. (2023)"

    print(f"Dataset saved to {output_file}")
    print(f"Shape: X={X.shape}, y={y.shape}")
    print(f"Class distribution: {np.bincount(y)}")

    # Also save a small metadata file
    metadata_file = os.path.join(args.output_dir, "metadata.txt")
    with open(metadata_file, "w") as f:
        f.write(f"SHARP-2023-CI-Sample Dataset\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(
            f"Samples: {args.num_samples} ({num_pos} positive, {num_neg} negative)\n"
        )
        f.write(f"Sequence length: {args.sequence_length}\n")
        f.write(f"Features: {args.num_features}\n")
        f.write(f"Reference: Abduallah et al. (2023)\n")

    return 0


if __name__ == "__main__":
    exit(main())
