#!/usr/bin/env python3
"""Clean and merge SHARP data files.

Parses TAI timestamps to UTC, removes invalid rows, and merges all SHARP patch files.
"""
import glob
import os
import tarfile
from datetime import datetime, timedelta

import pandas as pd

# Define the output file
OUTPUT_FILE = "data/all_sharp_10min.csv.gz"
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# Define the columns we want to keep
COLUMNS_TO_KEEP = [
    "HARPNUM",
    "NOAA_AR",
    "T_REC",  # Identifiers and timestamp
    "USFLUX",
    "MEANGAM",
    "MEANGBT",
    "MEANGBZ",
    "MEANGBH",  # Magnetic properties
    "MEANJZD",
    "TOTUSJZ",
    "MEANALP",
    "MEANJZH",
    "TOTUSJH",  # Helicity related
    "ABSNJZH",
    "SAVNCPP",
    "MEANPOT",
    "TOTPOT",  # Energy related
    "MEANSHR",
    "SHRGT45",
    "R_VALUE",  # Additional parameters
]


def tai_to_utc(tai_string):
    """Convert TAI timestamp to UTC ISO8601 format."""
    # Parse the TAI timestamp (format: YYYY.MM.DD_HH:MM:SS_TAI)
    dt = datetime.strptime(tai_string.split("_TAI")[0], "%Y.%m.%d_%H:%M:%S")
    # Adjust for leap seconds (approximate - proper conversion would need a leap second table)
    # As of 2021, TAI is ahead of UTC by 37 seconds
    utc = dt - timedelta(seconds=37)
    return utc.isoformat()


def process_tarball(tarball_path):
    """Extract and process files from a tarball."""
    dataframes = []
    with tarfile.open(tarball_path, "r") as tar:
        for member in tar.getmembers():
            if member.name.endswith(".txt"):
                f = tar.extractfile(member)
                if f is not None:
                    try:
                        # Skip the first line with metadata and read the data
                        lines = f.read().decode("utf-8").split("\n")
                        header = lines[1].split()  # Get column names
                        data = [
                            line.split() for line in lines[2:] if line.strip()
                        ]

                        if not data:  # Skip empty files
                            continue

                        df = pd.DataFrame(data, columns=header)

                        # Keep only the columns we're interested in if they exist
                        valid_cols = [
                            col for col in COLUMNS_TO_KEEP if col in df.columns
                        ]
                        df = df[valid_cols]

                        # Convert T_REC from TAI to UTC ISO8601
                        if "T_REC" in df.columns:
                            df["T_REC"] = df["T_REC"].apply(tai_to_utc)

                        dataframes.append(df)
                    except Exception as e:
                        print(f"Error processing {member.name}: {e}")

    return dataframes


def main():
    """Process all tarballs and create the merged CSV."""
    all_dataframes = []
    tarballs = glob.glob("data/raw_txt/sharp_*.tar")

    for tarball in sorted(tarballs):
        print(f"Processing {tarball}...")
        dfs = process_tarball(tarball)
        all_dataframes.extend(dfs)

    # Merge all dataframes
    if all_dataframes:
        merged_df = pd.concat(all_dataframes, ignore_index=True)

        # Convert numeric columns
        for col in merged_df.columns:
            if col not in ["T_REC", "HARPNUM", "NOAA_AR"]:
                merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce")

        # Drop rows with NaN or 0 in required features
        numeric_cols = [
            col
            for col in merged_df.columns
            if col not in ["T_REC", "HARPNUM", "NOAA_AR"]
        ]

        # Drop rows with NaN in any column
        merged_df = merged_df.dropna(subset=numeric_cols)

        # Drop rows with 0 in required features
        # Note: This differs from Abdullah's approach which fills with zeros
        for col in numeric_cols:
            merged_df = merged_df[merged_df[col] != 0]

        # Sort by timestamp and HARPNUM
        merged_df = merged_df.sort_values(["T_REC", "HARPNUM"])

        # Save to compressed CSV
        merged_df.to_csv(OUTPUT_FILE, index=False, compression="gzip")
        print(f"Merged data saved to {OUTPUT_FILE}")
        print(f"Total rows: {len(merged_df)}")
    else:
        print("No data found to process.")


if __name__ == "__main__":
    main()
