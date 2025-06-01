import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_flux_data(num_points=100000):
    """
    Generate sinusoidal flux data with some random noise.
    Each entry is spaced by 1 minute (60000 ms) in time.

    Returns:
        pd.DataFrame with columns ['time', 'flux', 'satellite'].
    """
    # Start time in milliseconds (example: 1278806400000)
    start_time = 1278806400000
    time_step_ms = 60000  # 1 minute in milliseconds

    # Set up some sinusoidal parameters:
    # Frequencies in cycles/sec, but our time steps are in ms. We'll convert accordingly.
    # i-th entry => t_s = i * 60 seconds
    f1 = 1.0 / (24 * 3600)  # 1-day period
    f2 = 1.0 / (12 * 3600)  # 12-hour period
    f3 = 1.0 / (1.5 * 3600)  # 1.5-hour period

    # Amplitudes and random phases for each sinusoid
    A1, A2, A3 = 1.0e-7, 5.0e-8, 2.0e-8
    phi1, phi2, phi3 = np.random.rand(3) * 2.0 * np.pi

    # Baseline flux around 2.0e-7
    baseline = 2.0e-7

    # Prepare arrays to store the data
    times = []
    fluxes = []
    satellites = []

    for i in range(num_points):
        # Compute the timestamp
        current_time = start_time + i * time_step_ms

        # Convert i to seconds for our sine function
        t_s = i * 60.0  # 60 seconds per step

        # Sum of sinusoids + some random noise
        flux_value = (
            baseline
            + A1 * np.sin(2.0 * np.pi * f1 * t_s + phi1)
            + A2 * np.sin(2.0 * np.pi * f2 * t_s + phi2)
            + A3 * np.sin(2.0 * np.pi * f3 * t_s + phi3)
            + np.random.normal(loc=0.0, scale=2.0e-8)
        )

        times.append(current_time)
        fluxes.append(flux_value)
        satellites.append("self-generated")

    df = pd.DataFrame({"time": times, "flux": fluxes, "satellite": satellites})
    return df


def plot_random_24_hour_window(parquet_file="synthetic_flux_data.parquet"):
    """
    Reads flux data from a Parquet file, picks a random 24-hour window,
    and plots the flux vs. time over that window.
    """
    df = pd.read_parquet(parquet_file)

    # Ensure data is sorted by time
    df = df.sort_values(by="time").reset_index(drop=True)

    # Convert time from ms to a pandas Timestamp
    df["datetime"] = pd.to_datetime(df["time"], unit="ms")

    # There are 1440 minutes in 24 hours
    samples_per_24h = 1440
    if len(df) < samples_per_24h:
        raise ValueError("Not enough data to extract a 24-hour window.")

    # Pick a random start index
    random_start = np.random.randint(0, len(df) - samples_per_24h)
    subset_df = df.iloc[random_start : random_start + samples_per_24h]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(subset_df["datetime"], subset_df["flux"], marker=".", linewidth=1)
    plt.xlabel("Time")
    plt.ylabel("Flux")
    plt.title("Random 24-Hour Flux Window")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Generate the data and write to Parquet
    df = generate_flux_data(num_points=100000)
    df.to_parquet("synthetic_flux_data.parquet", index=False)
    print("Parquet file 'synthetic_flux_data.parquet' has been generated.")

    # Plot a random 24-hour window from the generated data
    plot_random_24_hour_window("synthetic_flux_data.parquet")
