"""
goes_download.py

This script demonstrates how to download GOES X-ray flux data (hosted via the VSO) using SunPy’s Fido module. 
The GOES X-ray Sensor (XRS) provides flux measurements in the 1–8 Å and 0.5–4 Å channels, commonly used 
for solar flare monitoring and classification (A, B, C, M, X classes). 

Note on GOES Data Cadence:
- GOES provides various cadences (e.g., 1-second (flx1s) and 1-minute (avg1m) netCDF files). 
  1) 1-second data (flx1s):
     - Offers finer temporal resolution, capturing rapid flare evolution.
     - Potentially noisier and yields larger file sizes.
     - More suitable if you need detailed event onset analysis (e.g., sub-minute flare detection).
  2) 1-minute data (avg1m):
     - Smooths short-term variability, often enough for most flare prediction tasks.
     - Smaller data sizes, easier to handle computationally.
     - Ideal for medium- to long-term forecasting where second-by-second variations aren’t critical.

Which cadence is “best” depends on your specific research goals. 
For typical flare forecasting models (with lead times of hours or more), 1-minute data is usually sufficient 
and less resource-intensive to handle. If you need second-level detail on flare onset or impulsive events, 
you might use the 1-second data.

Usage:
------
Call download_goes_data with a start_time and end_time in UTC (ISO format). 
This will search for and fetch the relevant GOES data in the specified time range.
"""

from sunpy.net import Fido, attrs as a

def download_goes_data(start_time="2014-11-01 00:00", end_time="2015-11-01 00:00"):
    """
    Downloads GOES data (X-ray flux) from the given time range using SunPy.
    
    Parameters:
    -----------
    start_time : str
        The start date/time (UTC) of the time range to download (YYYY-MM-DD HH:MM).
    end_time : str
        The end date/time (UTC) of the time range to download (YYYY-MM-DD HH:MM).

    Example:
    --------
    download_goes_data("2011-01-01 00:00", "2011-12-01 06:00")
    """
    # 1) Search for GOES data between start_time and end_time.
    #    a.Instrument("goes") includes GOES X-ray flux data.
    #    If you need a specific GOES satellite or product type (e.g., GOES-16 vs. GOES-17),
    #    you can add further constraints, but this simple example fetches the available data.
    result = Fido.search(
        a.Time(start_time, end_time),
        a.Instrument("goes")
    )
    
    # 2) Fetch (download) the resulting files into the "GOES/data/test" folder.
    #    The "{file}" token preserves original filenames for each downloaded file.
    downloaded_files = Fido.fetch(result, path="GOES/data/avg1m_2010_to_2024/{file}")
    
    # 3) Log or print out which files were downloaded.
    print(f"Downloaded GOES files for {start_time} to {end_time}:")
    for f in downloaded_files:
        print(f)

if __name__ == "__main__":
    download_goes_data()