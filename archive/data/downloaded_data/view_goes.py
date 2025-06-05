"""
view_goes.py

This script demonstrates how to load and plot GOES-16 1-minute-averaged X-ray flux data
across an entire year (Jan 1 to Dec 31, 2022).

We assume you have daily netCDF files in your "GOES/data/test" directory, named like:
  sci_xrsf-l2-avg1m_g16_dYYYYMMDD_v2-2-0.nc

Usage:
------
    python view_goes.py

This will produce a plot showing flux (in the 1–8 Å and 0.5–4 Å channels) spanning the entire year,
plus horizontal lines indicating flare-class thresholds (B, C, M, X).
"""

from datetime import date, datetime, timedelta

import matplotlib.pyplot as plt
from netCDF4 import Dataset, num2date


def convert_cftime_to_datetime(cftime_times):
    """
    Convert an array/list of cftime.DatetimeGregorian objects into Python datetime objects.
    """
    py_datetimes = []
    for t in cftime_times:
        py_datetimes.append(
            datetime(t.year, t.month, t.day, t.hour, t.minute, int(t.second))
        )
    return py_datetimes


def load_goes_1min_files(file_list):
    """
    Loads and concatenates GOES-16 1-minute netCDF files for X-ray flux (1–8 Å, 0.5–4 Å).
    Returns three lists (times, flux_1_8, flux_05_4) spanning all files in chronological order.

    Parameters
    ----------
    file_list: list of str
        Paths to netCDF files, e.g.,
        ["GOES/data/test/sci_xrsf-l2-avg1m_g16_d20220101_v2-2-0.nc",
         "GOES/data/test/sci_xrsf-l2-avg1m_g16_d20220102_v2-2-0.nc", ...]

    Returns
    -------
    all_times, all_flux_1_8, all_flux_05_4: Lists of Python datetime, float, float
    """
    # Sort file_list to ensure chronological order (in case filenames are out
    # of order)
    file_list = sorted(file_list)

    all_times = []
    all_flux_1_8 = []
    all_flux_05_4 = []

    for fpath in file_list:
        try:
            with Dataset(fpath, "r") as ds:
                time_var = ds.variables["time"]
                cftime_vals = num2date(time_var[:], units=time_var.units)
                times = convert_cftime_to_datetime(cftime_vals)

                flux_1_8 = ds.variables["xrsb_flux"][:]  # 1–8 Å
                flux_05_4 = ds.variables["xrsa_flux"][:]  # 0.5–4 Å

                # Extend our master lists
                all_times.extend(times)
                all_flux_1_8.extend(flux_1_8)
                all_flux_05_4.extend(flux_05_4)

        except FileNotFoundError:
            # If a file is missing, we can either skip it or raise an error.
            print(f"Warning: File not found: {fpath}. Skipping.")
            continue

    return all_times, all_flux_1_8, all_flux_05_4


def plot_goes_1min(file_list):
    """
    Reads multiple GOES-16 1-minute netCDF files (provided via NOAA/NGDC or other),
    plots the 1–8 Å and 0.5–4 Å flux, plus horizontal lines for flare thresholds.
    """
    times, flux_1_8, flux_05_4 = load_goes_1min_files(file_list)

    plt.figure(figsize=(12, 6))
    plt.plot(times, flux_1_8, label="GOES-16 (1–8 Å)", color="blue")
    plt.plot(times, flux_05_4, label="GOES-16 (0.5–4 Å)", color="green")

    plt.yscale("log")

    # Add horizontal lines for flare-class thresholds
    flare_thresholds = [
        (1e-7, "B-class"),
        (1e-6, "C-class"),
        (1e-5, "M-class"),
        (1e-4, "X-class"),
    ]

    if len(times) > 0:
        x_label_align = times[-1]
    else:
        x_label_align = 0

    for flux_level, flare_label in flare_thresholds:
        plt.axhline(
            flux_level, color="gray", linestyle="--", linewidth=1, alpha=0.7
        )
        plt.text(
            x_label_align,
            flux_level * 1.1,
            flare_label,
            color="gray",
            fontsize=8,
            ha="left",
            va="bottom",
        )

    plt.xlabel("Time (UTC)")
    plt.ylabel("X-ray Flux (W/m^2)")
    plt.title(
        "GOES-16 (1-minute) X-ray Flux - Year-long Plot with Flare Thresholds"
    )
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Generate daily file paths for the entire 2022 year
    start_date = date(2022, 1, 1)
    end_date = date(2022, 12, 31)
    delta = timedelta(days=1)

    g16_files = []
    current = start_date
    while current <= end_date:
        date_str = current.strftime("%Y%m%d")
        # Adjust the path/filename template as needed:
        fpath = f"GOES/data/test/sci_xrsf-l2-avg1m_g16_d{date_str}_v2-2-0.nc"
        g16_files.append(fpath)
        current += delta

    plot_goes_1min(g16_files)
