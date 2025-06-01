#!/usr/bin/env python3
"""Add flare labels to SHARP data by matching with NOAA SWPC event list."""
import argparse
import os
from datetime import datetime, timedelta

import pandas as pd
import requests

# Define paths
INPUT_FILE = "data/all_sharp_10min.csv.gz"
OUTPUT_FILE = "data/sharp_with_labels.csv.gz"
EVENTS_FILE = "data/events_2010-2024.txt"

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)


def download_noaa_events():
    """Download NOAA SWPC event list if it doesn't exist."""
    if not os.path.exists(EVENTS_FILE):
        print(f"Downloading NOAA event list to {EVENTS_FILE}...")
        os.makedirs(os.path.dirname(EVENTS_FILE), exist_ok=True)

        # NOAA's FTP server for event data
        url = "ftp://ftp.swpc.noaa.gov/pub/indices/events/events.txt"

        try:
            # Download the events file
            response = requests.get(url)
            response.raise_for_status()

            with open(EVENTS_FILE, "w") as f:
                f.write(response.text)

            print("Event list downloaded successfully.")
        except Exception as e:
            print(f"Error downloading event list: {e}")
            print(
                "Please download the events file manually from the NOAA website."
            )
            exit(1)


def parse_noaa_events():
    """Parse NOAA SWPC event list file."""
    print("Parsing NOAA event list...")

    flare_data = []
    with open(EVENTS_FILE, "r") as f:
        for line in f:
            # Skip comments and empty lines
            if line.startswith("#") or not line.strip():
                continue

            # Parse NOAA event line
            try:
                # Events file format:
                # Date       Start   Max      End     Type  Loc/Frq               Particulars       Reg#
                # YY/MM/DD   HHMM    HHMM     HHMM
                parts = line.split()

                # Check if this is a flare event (starts with 'XRA')
                if len(parts) >= 7 and parts[4] == "XRA":
                    date_str = parts[0]
                    peak_time_str = parts[2]

                    # Get flare class
                    particulars = parts[6]
                    if particulars[0] in ["A", "B", "C", "M", "X"]:
                        flare_class = particulars
                    else:
                        continue

                    # Get NOAA AR number (if available)
                    noaa_ar = None
                    if len(parts) >= 8:
                        try:
                            noaa_ar = int(parts[7])
                        except ValueError:
                            pass

                    # Parse date and time
                    year = 2000 + int(date_str[:2])  # Assuming 21st century
                    month = int(date_str[3:5])
                    day = int(date_str[6:8])

                    hour = int(peak_time_str[:2])
                    minute = int(peak_time_str[2:4])

                    peak_time = datetime(year, month, day, hour, minute)

                    flare_data.append(
                        {
                            "NOAA_AR": noaa_ar,
                            "flare_class": flare_class,
                            "peak_time": peak_time,
                            "flare_magnitude": flare_class[0],
                        }
                    )
            except Exception as e:
                print(f"Error parsing line: {line.strip()}")
                print(f"Error: {e}")
                continue

    return pd.DataFrame(flare_data)


def add_flare_labels(sharp_df, flares_df, horizon_hours=24):
    """Add flare labels to SHARP data."""
    print(f"Adding flare labels with {horizon_hours} hour horizon...")

    # Convert T_REC to datetime
    sharp_df["datetime"] = pd.to_datetime(sharp_df["T_REC"])

    # Initialize label columns
    sharp_df["flare_C_24h"] = False
    sharp_df["flare_M_24h"] = False
    sharp_df["flare_M5_24h"] = False
    sharp_df["flare_X_24h"] = False

    # Group flares by NOAA_AR
    grouped_flares = flares_df.dropna(subset=["NOAA_AR"]).groupby("NOAA_AR")

    # Process each SHARP row
    total_rows = len(sharp_df)

    for idx, row in sharp_df.iterrows():
        if idx % 10000 == 0:
            print(f"Processing row {idx}/{total_rows}...")

        # Get NOAA_AR
        noaa_ar = row.get("NOAA_AR")
        if pd.isna(noaa_ar) or noaa_ar == "":
            continue

        # Try to convert to integer (handle potential string types)
        try:
            noaa_ar = int(float(noaa_ar))
        except (ValueError, TypeError):
            continue

        # Skip if NOAA_AR not in flares
        if noaa_ar not in grouped_flares.groups:
            continue

        # Get observation time
        obs_time = row["datetime"]

        # Get prediction window end time
        window_end = obs_time + timedelta(hours=horizon_hours)

        # Get flares for this AR
        ar_flares = grouped_flares.get_group(noaa_ar)

        # Find flares within the prediction window
        future_flares = ar_flares[
            (ar_flares["peak_time"] > obs_time)
            & (ar_flares["peak_time"] <= window_end)
        ]

        if not future_flares.empty:
            # Set C class label (includes all flares C and above)
            sharp_df.loc[idx, "flare_C_24h"] = True

            # Check for specific flare classes
            if any(future_flares["flare_magnitude"] == "M"):
                sharp_df.loc[idx, "flare_M_24h"] = True

                # Check for M5+ flares
                m_flares = future_flares[
                    future_flares["flare_magnitude"] == "M"
                ]
                for _, flare in m_flares.iterrows():
                    magnitude_str = flare["flare_class"][1:]
                    try:
                        magnitude = float(magnitude_str)
                        if magnitude >= 5.0:
                            sharp_df.loc[idx, "flare_M5_24h"] = True
                            break
                    except ValueError:
                        pass

            # Check for X class flares
            if any(future_flares["flare_magnitude"] == "X"):
                sharp_df.loc[idx, "flare_X_24h"] = True
                # M5+ is also true for any X class
                sharp_df.loc[idx, "flare_M5_24h"] = True
                # M is also true for any X class
                sharp_df.loc[idx, "flare_M_24h"] = True

    # Remove the helper datetime column
    sharp_df = sharp_df.drop("datetime", axis=1)

    return sharp_df


def main():
    """Run the flare labeling process for SHARP data."""
    parser = argparse.ArgumentParser(
        description="Add flare labels to SHARP data."
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=24,
        help="Prediction horizon in hours (default: 24)",
    )
    args = parser.parse_args()

    # Download NOAA events if needed
    download_noaa_events()

    # Parse NOAA events
    flares_df = parse_noaa_events()
    print(f"Found {len(flares_df)} flare events.")

    # Load SHARP data
    print(f"Loading SHARP data from {INPUT_FILE}...")
    sharp_df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(sharp_df)} SHARP observations.")

    # Add flare labels
    labeled_df = add_flare_labels(
        sharp_df, flares_df, horizon_hours=args.horizon
    )

    # Save labeled data
    print(f"Saving labeled data to {OUTPUT_FILE}...")
    labeled_df.to_csv(OUTPUT_FILE, index=False, compression="gzip")

    # Print statistics
    print("\nLabel statistics:")
    print(f"Total observations: {len(labeled_df)}")
    print(
        f"C+ class flares: {labeled_df['flare_C_24h'].sum()} ({labeled_df['flare_C_24h'].mean()*100:.2f}%)"
    )
    print(
        f"M+ class flares: {labeled_df['flare_M_24h'].sum()} ({labeled_df['flare_M_24h'].mean()*100:.2f}%)"
    )
    print(
        f"M5+ class flares: {labeled_df['flare_M5_24h'].sum()} ({labeled_df['flare_M5_24h'].mean()*100:.2f}%)"
    )
    print(
        f"X class flares: {labeled_df['flare_X_24h'].sum()} ({labeled_df['flare_X_24h'].mean()*100:.2f}%)"
    )


if __name__ == "__main__":
    main()
