#!/usr/bin/env python3
"""
Data validation and integrity checking for EVEREST datasets.
"""

import hashlib
import json
import os
from pathlib import Path


def compute_file_checksum(filepath: str) -> str:
    """Compute SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def save_dataset_manifest(data_dir: str, manifest_file: str = "dataset_manifest.json"):
    """Generate and save dataset manifest with checksums."""
    manifest = {
        "dataset_version": "1.0",
        "created_date": str(pd.Timestamp.now()),
        "files": {},
    }

    data_path = Path(data_dir)
    for file_path in data_path.rglob("*.npz"):
        relative_path = str(file_path.relative_to(data_path))
        checksum = compute_file_checksum(str(file_path))
        file_size = file_path.stat().st_size

        manifest["files"][relative_path] = {
            "checksum": checksum,
            "size_bytes": file_size,
            "modified_date": str(pd.Timestamp.fromtimestamp(file_path.stat().st_mtime)),
        }

    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"✅ Dataset manifest saved: {manifest_file}")
    return manifest


def verify_dataset_integrity(
    data_dir: str, manifest_file: str = "dataset_manifest.json"
):
    """Verify dataset integrity against manifest checksums."""
    if not os.path.exists(manifest_file):
        print(f"⚠️ No manifest found: {manifest_file}")
        return False

    with open(manifest_file, "r") as f:
        manifest = json.load(f)

    data_path = Path(data_dir)
    all_valid = True

    for relative_path, file_info in manifest["files"].items():
        file_path = data_path / relative_path

        if not file_path.exists():
            print(f"❌ Missing file: {relative_path}")
            all_valid = False
            continue

        actual_checksum = compute_file_checksum(str(file_path))
        expected_checksum = file_info["checksum"]

        if actual_checksum != expected_checksum:
            print(f"❌ Checksum mismatch: {relative_path}")
            print(f"   Expected: {expected_checksum}")
            print(f"   Actual:   {actual_checksum}")
            all_valid = False
        else:
            print(f"✅ Verified: {relative_path}")

    if all_valid:
        print("🔐 All datasets verified successfully!")
    else:
        print("⚠️ Dataset integrity check failed!")

    return all_valid


if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description="Dataset validation tool")
    parser.add_argument("--data-dir", default="data", help="Data directory to validate")
    parser.add_argument(
        "--create-manifest", action="store_true", help="Create new manifest"
    )
    parser.add_argument(
        "--verify", action="store_true", help="Verify against existing manifest"
    )

    args = parser.parse_args()

    if args.create_manifest:
        save_dataset_manifest(args.data_dir)

    if args.verify:
        verify_dataset_integrity(args.data_dir)
