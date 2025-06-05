#!/usr/bin/env python3
"""Create train/validation/test splits for SHARP ML dataset.

Uses chronological splits to avoid data leakage.
"""
import hashlib
import json
import os
import subprocess
from datetime import datetime

import pandas as pd

# Define paths
INPUT_FILE = "data/sharp_features_v1.csv.gz"
OUTPUT_DIR = "data/splits"

# Define date cutoffs for chronological splits
TRAIN_END = "2018-12-31"
VAL_END = "2021-12-31"
# Test: 2022-01-01 to present

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_git_hash():
    """Get the current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return "git_hash_not_available"
    except (subprocess.SubprocessError, FileNotFoundError):
        return "git_hash_not_available"


def calculate_sha256(file_path):
    """Calculate SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read and update hash in chunks for large files
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def main():
    """Create chronological dataset splits and save metadata."""
    print(f"Loading processed SHARP data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} observations.")

    # Convert timestamps to datetime for splitting
    df["datetime"] = pd.to_datetime(df["T_REC"])

    # Create chronological splits
    print("Creating chronological splits...")
    train_df = df[df["datetime"] <= TRAIN_END]
    val_df = df[(df["datetime"] > TRAIN_END) & (df["datetime"] <= VAL_END)]
    test_df = df[df["datetime"] > VAL_END]

    # Remove the helper datetime column
    train_df = train_df.drop("datetime", axis=1)
    val_df = val_df.drop("datetime", axis=1)
    test_df = test_df.drop("datetime", axis=1)

    # Save splits
    train_path = os.path.join(OUTPUT_DIR, "train.csv")
    val_path = os.path.join(OUTPUT_DIR, "val.csv")
    test_path = os.path.join(OUTPUT_DIR, "test.csv")

    print(f"Saving train split to {train_path} ({len(train_df)} rows)...")
    train_df.to_csv(train_path, index=False)

    print(f"Saving validation split to {val_path} ({len(val_df)} rows)...")
    val_df.to_csv(val_path, index=False)

    print(f"Saving test split to {test_path} ({len(test_df)} rows)...")
    test_df.to_csv(test_path, index=False)

    # Calculate SHA-256 hashes
    train_hash = calculate_sha256(train_path)
    val_hash = calculate_sha256(val_path)
    test_hash = calculate_sha256(test_path)

    # Get git commit hash
    git_hash = get_git_hash()

    # Create metadata
    meta = {
        "creation_date": datetime.now().isoformat(),
        "git_commit": git_hash,
        "data_counts": {
            "total": len(df),
            "train": len(train_df),
            "validation": len(val_df),
            "test": len(test_df),
        },
        "class_distribution": {
            "train": {
                "flare_C_24h": int(train_df["flare_C_24h"].sum()),
                "flare_M_24h": int(train_df["flare_M_24h"].sum()),
                "flare_M5_24h": int(train_df["flare_M5_24h"].sum()),
                "flare_X_24h": int(train_df["flare_X_24h"].sum()),
            },
            "validation": {
                "flare_C_24h": int(val_df["flare_C_24h"].sum()),
                "flare_M_24h": int(val_df["flare_M_24h"].sum()),
                "flare_M5_24h": int(val_df["flare_M5_24h"].sum()),
                "flare_X_24h": int(val_df["flare_X_24h"].sum()),
            },
            "test": {
                "flare_C_24h": int(test_df["flare_C_24h"].sum()),
                "flare_M_24h": int(test_df["flare_M_24h"].sum()),
                "flare_M5_24h": int(test_df["flare_M5_24h"].sum()),
                "flare_X_24h": int(test_df["flare_X_24h"].sum()),
            },
        },
        "file_hashes": {
            "train.csv": train_hash,
            "val.csv": val_hash,
            "test.csv": test_hash,
        },
        "split_criteria": {
            "method": "chronological",
            "train_cutoff": TRAIN_END,
            "val_cutoff": VAL_END,
        },
    }

    # Save metadata
    meta_path = os.path.join(OUTPUT_DIR, "meta.json")
    print(f"Saving metadata to {meta_path}...")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("\nSplit statistics:")
    print(
        f"  Train: {len(train_df)} observations ({100*len(train_df)/len(df):.1f}%)"
    )
    print(
        f"  Validation: {len(val_df)} observations ({100*len(val_df)/len(df):.1f}%)"
    )
    print(
        f"  Test: {len(test_df)} observations ({100*len(test_df)/len(df):.1f}%)"
    )

    print("\nClass distribution (flare_M_24h):")
    print(f"  Train: {train_df['flare_M_24h'].mean()*100:.2f}% positive")
    print(f"  Validation: {val_df['flare_M_24h'].mean()*100:.2f}% positive")
    print(f"  Test: {test_df['flare_M_24h'].mean()*100:.2f}% positive")


if __name__ == "__main__":
    main()
