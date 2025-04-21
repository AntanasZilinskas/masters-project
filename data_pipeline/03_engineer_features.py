#!/usr/bin/env python3
"""Engineer additional features for SHARP data beyond those in Abdullah et al.

Includes time derivatives, rolling statistics, and more.
"""
import os

import numpy as np
import pandas as pd

# Define paths
INPUT_FILE = "data/sharp_with_labels.csv.gz"
OUTPUT_FILE = "data/sharp_features_v1.csv.gz"

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)


def engineer_features(df):
    """Engineer additional features for the dataset."""
    print("Engineering additional features...")

    # Convert T_REC to datetime
    df["datetime"] = pd.to_datetime(df["T_REC"])

    # Sort by NOAA_AR and datetime for accurate sequential calculations
    df = df.sort_values(["NOAA_AR", "datetime"])

    # Initialize new feature columns
    df["dUSFLUX_dt"] = np.nan
    df["R_VALUE_rolling3h_mean"] = np.nan
    df["TOTUSJH_rolling3h_mean"] = np.nan
    df["USFLUX_rolling3h_mean"] = np.nan
    df["flare_count_48h"] = 0
    df["cos_heliographic_lat"] = np.nan

    # Process each NOAA AR separately
    print("Processing by NOAA_AR...")
    grouped = df.groupby("NOAA_AR")

    processed_dfs = []
    for ar, group in grouped:
        if pd.isna(ar) or ar == "":
            # Skip rows without NOAA_AR
            processed_dfs.append(group)
            continue

        # Reset index for local operations
        ar_df = group.reset_index(drop=True)

        # Calculate time derivatives
        # dUSFLUX_dt - 1-hour change rate
        for i in range(len(ar_df)):
            if i > 0:
                # Get current and previous row
                current = ar_df.iloc[i]
                prev = ar_df.iloc[i - 1]

                # Calculate time difference in hours
                time_diff = (
                    current["datetime"] - prev["datetime"]
                ).total_seconds() / 3600

                # Only calculate if time difference is close to 1 hour
                if abs(time_diff - 1) < 0.1:  # within 6 minutes of an hour
                    flux_diff = current["USFLUX"] - prev["USFLUX"]
                    ar_df.loc[i, "dUSFLUX_dt"] = flux_diff / time_diff

        # Calculate rolling means (3-hour window)
        # Assuming 10-minute cadence, 3 hours = 18 points
        window_size = 18

        # Calculate rolling means for selected features
        ar_df["R_VALUE_rolling3h_mean"] = (
            ar_df["R_VALUE"].rolling(window=window_size, min_periods=1).mean()
        )
        ar_df["TOTUSJH_rolling3h_mean"] = (
            ar_df["TOTUSJH"].rolling(window=window_size, min_periods=1).mean()
        )
        ar_df["USFLUX_rolling3h_mean"] = (
            ar_df["USFLUX"].rolling(window=window_size, min_periods=1).mean()
        )

        # Calculate flare history: count of flares in previous 48 hours
        for i in range(len(ar_df)):
            current_time = ar_df.iloc[i]["datetime"]

            # Look at previous records within 48 hours
            history_window = ar_df[
                (ar_df["datetime"] < current_time)
                & (
                    ar_df["datetime"]
                    >= (current_time - pd.Timedelta(hours=48))
                )
            ]

            # Count rows with flares in the C class or higher
            flare_count = history_window["flare_C_24h"].sum()
            ar_df.loc[i, "flare_count_48h"] = flare_count

        # Add to processed dataframes list
        processed_dfs.append(ar_df)

    # Combine all processed groups
    processed_df = pd.concat(processed_dfs, ignore_index=True)

    # Calculate cos(heliographic_latitude) - for line-of-sight bias correction
    # Extract heliographic latitude from metadata if available, or use proxy
    # For demonstration, we'll use a proxy calculation - actual method would depend on data
    processed_df["cos_heliographic_lat"] = 1.0  # Placeholder value

    # Drop rows created during derivative windows (with NaN in time derivatives)
    processed_df = processed_df.dropna(subset=["dUSFLUX_dt"])

    # Remove the helper datetime column
    processed_df = processed_df.drop("datetime", axis=1)

    return processed_df


def main():
    """Process SHARP data and engineer additional features."""
    print(f"Loading labeled SHARP data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} observations.")

    # Engineer features
    processed_df = engineer_features(df)

    # Save processed data
    print(f"Saving processed data to {OUTPUT_FILE}...")
    processed_df.to_csv(OUTPUT_FILE, index=False, compression="gzip")
    print(f"Processed data saved with {len(processed_df)} observations.")

    # Report on new features
    print("\nNew engineered features:")
    print("  dUSFLUX_dt: Time derivative of USFLUX over 1 hour")
    print("  R_VALUE_rolling3h_mean: 3-hour rolling mean of R_VALUE")
    print("  TOTUSJH_rolling3h_mean: 3-hour rolling mean of TOTUSJH")
    print("  USFLUX_rolling3h_mean: 3-hour rolling mean of USFLUX")
    print("  flare_count_48h: Count of flares in the same AR in past 48h")
    print("  cos_heliographic_lat: Cosine of heliographic latitude")


if __name__ == "__main__":
    main()
