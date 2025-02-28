"""
Download SDO/HMI data using SunPy's Fido interface.
Adjust the product (e.g., hmi.M_720s for line-of-sight, hmi.B_720s for vector) as needed.

Line-of-Sight Magnetograms: hmi.M_720s
Vector Magnetograms:       hmi.B_720s

Tip: For flare prediction research, many find line-of-sight magnetograms (hmi.M_720s)
easier to start with, but vector data theoretically holds more complete magnetic field info.
"""

from sunpy.net import Fido, attrs as a
import datetime
import astropy.units as u

def download_hmi_data(start_time=None, end_time=None, cadence=12*u.minute, product="hmi.M_45s"):
    """
    Download HMI data for a specific time range and cadence.
    
    Parameters:
    -----------
    start_time : str
        Start time in format "YYYY-MM-DD HH:MM"
    end_time : str
        End time in format "YYYY-MM-DD HH:MM"
    cadence : astropy.units.Quantity
        Time cadence for data (e.g., 12*u.minute, 45*u.second)
    product : str
        HMI data product (e.g., "hmi.M_45s", "hmi.M_720s", "hmi.B_720s")
    """
    # 1) Use the provided time range or default to the example range
    if start_time is None:
        start_time = "2010-06-01 00:00"  # Matching first entry in your CSV
    if end_time is None:
        end_time = "2010-12-01 00:00"    # Matching last entry in your CSV
    
    # 2) Determine which product and physical observable to use
    if "M_" in product:
        # For line-of-sight magnetograms:
        product_physobs = a.Physobs.los_magnetic_field
    elif "B_" in product:
        # For vector magnetograms:
        product_physobs = a.Physobs.vector_magnetic_field
    else:
        raise ValueError(f"Unknown product type: {product}")
    
    # 3) Perform the data search with specified cadence
    result = Fido.search(
        a.Time(start_time, end_time),
        a.Instrument.hmi,
        product_physobs,
        a.Sample(cadence)
    )
    
    print(f"Found {len(result)} files matching the search criteria")
    
    # 4) Download the files to a folder structure that includes the product name
    output_dir = f"SDO/data/{product.lower()}"
    files = Fido.fetch(result, path=f"{output_dir}/{{file}}")
    
    print(f"Downloaded {len(files)} files to '{output_dir}' folder")
    return files


if __name__ == "__main__":
    # Example: Download data matching the timestamps in your CSV file
    # Using 12-minute cadence to match the SHARP parameter cadence
    download_hmi_data(
        start_time="2010-05-01 00:00", 
        end_time="2015-12-01 00:00",
        cadence=12*u.minute,
        product="hmi.M_45s"  # Use 45-second line-of-sight data (can be changed to hmi.B_720s for vector)
    )
    
    # Alternative: Download data for a specific HARP/SHARP region
    # Uncomment and modify as needed
    """
    # The CSV shows HARP number 7890 and NOAA AR 12916
    download_hmi_data(
        start_time="2022-01-01 14:58", 
        end_time="2022-01-02 10:46",
        cadence=12*u.minute,
        product="hmi.M_45s"
    )
    """
