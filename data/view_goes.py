"""
view_goes.py

This script is hard-coded to load and plot the GOES 1-minute-averaged X-ray flux data
from both GOES-16 and GOES-17 for 2022-01-01. The netCDF files should be in "GOES/data/test" folder.

The files:
 - sci_xrsf-l2-avg1m_g16_d20220101_v2-2-0.nc  (GOES-16, 1-min)
 - sci_xrsf-l2-avg1m_g17_d20220101_v2-2-0.nc  (GOES-17, 1-min)

Usage:
------
    python view_goes.py

This will produce a plot comparing both satellites' flux over time in the 1–8 Å channel and 0.5–4 Å channel.
"""

import matplotlib.pyplot as plt
from netCDF4 import Dataset, num2date
from datetime import datetime

def convert_cftime_to_datetime(cftime_times):
    """
    Convert an array/list of cftime.DatetimeGregorian objects into Python datetime objects.
    """
    py_datetimes = []
    for t in cftime_times:
        py_datetimes.append(datetime(t.year, t.month, t.day, t.hour, t.minute, int(t.second)))
    return py_datetimes

def plot_goes_1min(file_g16, file_g17):
    """
    Reads GOES-16 and GOES-17 1-minute netCDF files (provided via NOAA/NGDC or other sources)
    and plots the 1–8 Å and 0.5–4 Å flux from both satellites, along with horizontal lines 
    indicating flare classification levels.
    """
    #-----------------------------------------------------
    # Open GOES-16 1-minute data
    #-----------------------------------------------------
    with Dataset(file_g16, 'r') as ds16:
        time_var_16 = ds16.variables['time']
        cftime_16 = num2date(time_var_16[:], units=time_var_16.units)
        times_16 = convert_cftime_to_datetime(cftime_16)

        flux_1_8_16 = ds16.variables['xrsb_flux'][:]   # 1–8 Å
        flux_05_4_16 = ds16.variables['xrsa_flux'][:]  # 0.5–4 Å

    #-----------------------------------------------------
    # Open GOES-17 1-minute data
    #-----------------------------------------------------
    with Dataset(file_g17, 'r') as ds17:
        time_var_17 = ds17.variables['time']
        cftime_17 = num2date(time_var_17[:], units=time_var_17.units)
        times_17 = convert_cftime_to_datetime(cftime_17)

        flux_1_8_17 = ds17.variables['xrsb_flux'][:]   # 1–8 Å
        flux_05_4_17 = ds17.variables['xrsa_flux'][:]  # 0.5–4 Å

    #-----------------------------------------------------
    # Plot the flux time-series for both satellites
    #-----------------------------------------------------
    plt.figure(figsize=(10, 6))

    # Plot 1–8 Å flux
    plt.plot(times_16, flux_1_8_16, label='GOES-16 (1–8 Å)', color='blue')
    plt.plot(times_17, flux_1_8_17, label='GOES-17 (1–8 Å)', color='red', linestyle='--')

    # Plot 0.5–4 Å flux
    plt.plot(times_16, flux_05_4_16, label='GOES-16 (0.5–4 Å)', color='green')
    plt.plot(times_17, flux_05_4_17, label='GOES-17 (0.5–4 Å)', color='orange', linestyle='--')

    # Set log scale for Y-axis
    plt.yscale('log')

    #-----------------------------------------------------
    # Add horizontal lines for flare-class thresholds
    #-----------------------------------------------------
    # Typical NOAA thresholds for 1–8 Å flux classification.
    # Note: A-class starts at 1e-8, but you can add if desired.
    flare_thresholds = [
        (1e-7, 'B-class'),
        (1e-6, 'C-class'),
        (1e-5, 'M-class'),
        (1e-4, 'X-class')
    ]

    # If we have at least one time in times_16 for text placement:
    if len(times_16) > 0:
        # Position labels near the rightmost part of the data (e.g., times_16[-1]).
        x_label_align = times_16[-1]
    else:
        x_label_align = 0  # fallback if no data

    for flux_level, flare_label in flare_thresholds:
        plt.axhline(flux_level, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        # Slight vertical offset for text to avoid overlapping the line
        plt.text(x_label_align, flux_level * 1.1, flare_label,
                 color='gray', fontsize=8, ha='left', va='bottom')

    #-----------------------------------------------------
    # Final plotting details
    #-----------------------------------------------------
    plt.xlabel('Time (UTC)')
    plt.ylabel('X-ray Flux (W/m^2)')
    plt.title('GOES-16 & GOES-17 (1-minute) X-ray Flux with Flare Thresholds')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Hard-coded paths to the 1-minute netCDF files for GOES-16 and GOES-17
    file_goes16_1m = "GOES/data/test/sci_xrsf-l2-avg1m_g16_d20220101_v2-2-0.nc"
    file_goes17_1m = "GOES/data/test/sci_xrsf-l2-avg1m_g17_d20220101_v2-2-0.nc"

    plot_goes_1min(file_goes16_1m, file_goes17_1m)