"""
Download SDO magnetograms, GOES X-ray flux data, and SHARP parameters
into a local /data directory.
"""

import datetime
import os

import astropy.units as u
import drms  # pip install drms
import pandas as pd
import requests
from sunpy.net import Fido
from sunpy.net import attrs as a


def ensure_data_directory(data_path="./data"):
    """Ensure /data directory exists."""
    os.makedirs(data_path, exist_ok=True)


def download_sdo_magnetograms(start_date, end_date, data_path="./data"):
    """
    Download SDO HMI line-of-sight magnetograms (e.g., hmi.M_45s data)
    from the JSOC using the drms client.

    start_date, end_date: 'YYYY.MM.DD_HH:MM:SS' or similar
    """
    ensure_data_directory(data_path)

    # DRMS client configuration
    client = drms.Client(email="az2221@imperial.ac.uk")
    # Example series: "hmi.M_45s" for HMI line-of-sight magnetograms
    series = "hmi.M_45s"

    # Construct a DRMS query, see https://github.com/sunpy/drms for usage
    # Times can be specified in T AI range format
    qstr = f"{series}[{start_date}-{end_date}]"
    print(f"Querying JSOC for: {qstr}")

    try:
        result = client.query(qstr, key=["T_REC", "DATE"], seg="magnetogram")
        print(f"Found {len(result)} records. Downloading data...")

        # Download each magnetogram
        download_info = client.fetch(
            result, path=f"{data_path}/{{file}}", progress=True
        )
        print("SDO HMI magnetograms download complete.")
    except Exception as e:
        print(f"Error downloading SDO data: {e}")


def download_goes_xray_flux(start_date, end_date, data_path="./data"):
    """
    Download GOES X-ray flux data (1-min or 5-min) from NOAA SWPC.
    NOAA often provides text or netCDF; here we use an example text-based approach.
    """
    ensure_data_directory(data_path)

    # Example URL for GOES X-Ray flux from NOAA SWPC:
    #   https://services.swpc.noaa.gov/text/goes-xray-flux-primary/
    # This is an ongoing stream. Historical data can be found in separate archives.
    # For demonstration, we'll do a simple requests call:
    url = "https://services.swpc.noaa.gov/text/goes-xray-flux-primary.txt"
    filename = os.path.join(data_path, "goes-xray-flux-primary.txt")

    try:
        print("Downloading GOES X-ray flux data...")
        r = requests.get(url)
        r.raise_for_status()
        with open(filename, "wb") as f:
            f.write(r.content)
        print(f"GOES X-ray flux data saved to {filename}")
    except Exception as e:
        print(f"Error downloading GOES X-ray flux data: {e}")


def download_sharp_parameters(start_date, end_date, data_path="./data"):
    """
    Download SHARP (Spaceweather HMI Active Region Patches) tabular data from the JSOC.
    SHARP parameters can be found in series like 'hmi.sharp_720s',
    including summary parameters like TOTUSJH, TOTBSQ, etc.

    This is an example that queries the JSOC for NOAA AR data ranges.
    """
    ensure_data_directory(data_path)

    client = drms.Client(email="your_email@domain.com")
    series = "hmi.sharp_720s"  # the main SHARP series

    # Example of time-based query. In practice, you’d refine using HARPNUM or
    # NOAA AR numbers.
    qstr = f"{series}[{start_date}-{end_date}]"
    print(f"Querying JSOC for SHARP parameters: {qstr}")
    try:
        result = client.query(
            qstr,
            key=[
                "T_REC",
                "HARPNUM",
                "NOAA_ARS",
                "USFLUX",
                "TOTUSJH",
                "TOTPOT",
            ],
        )
        print(f"Found {len(result)} SHARP records. Saving to CSV...")

        csv_path = os.path.join(data_path, "sharp_parameters.csv")
        result.to_csv(csv_path, index=False)
        print(f"SHARP parameters saved to {csv_path}")
    except Exception as e:
        print(f"Error downloading SHARP parameters: {e}")


if __name__ == "__main__":
    # Example usage:
    # Note: The date format can vary. DRMS often uses
    # 'YYYY.MM.DD_HH:MM:SS_TAI'.
    sdate = "2023.01.01_TAI"
    edate = "2023.01.02_TAI"

    download_sdo_magnetograms(sdate, edate)
    download_goes_xray_flux(sdate, edate)
    download_sharp_parameters(sdate, edate)

    # Ensure you have a local data directory
    os.makedirs("data", exist_ok=True)

    # Example search: SDO/HMI Magnetograms for a short time range
    result = Fido.search(
        a.Time("2023-01-01 00:00", "2023-01-01 02:00"),  # time range
        a.Instrument("HMI"),  # instrument
        a.Physobs("LOS_magnetic_field"),  # LOS magnetogram
        a.Sample(3600 * u.second),  # 1-hour sampling
    )

    # Print found records
    print(result)

    # Download the data and save to local ./data folder
    downloaded_files = Fido.fetch(result, path="data/{file}")
    print(downloaded_files)
