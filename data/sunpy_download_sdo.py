"""
Download SDO/HMI data using SunPy's Fido interface.
Adjust the product (e.g., hmi.M_720s for line-of-sight, hmi.B_720s for vector) as needed.

Line-of-Sight Magnetograms: hmi.M_720s
Vector Magnetograms:       hmi.B_720s

Tip: For flare prediction research, many find line-of-sight magnetograms (hmi.M_720s)
easier to start with, but vector data theoretically holds more complete magnetic field info.
"""

from sunpy.net import Fido, attrs as a

def download_hmi_data():
    # 1) Choose the time range you're interested in. This example is short to limit data volume.
    #    Increase it if you want more data.
    start_time = "2022-01-01 00:00" 
    end_time   = "2022-01-01 00:15"  # 1-hour window
    
    # 2) Specify which product to download by adjusting the attrs.
    #    - a.Instrument.hmi: The SDO/HMI instrument.
    #    - a.Physobs.los_magnetic_field: line-of-sight magnetograms
    #      If you want vector magnetograms, you can use a.Physobs.vector_magnetic_field
    #      or check the SunPy documentation for additional parameters.
    
    # For line-of-sight magnetograms:
    product_physobs = a.Physobs.los_magnetic_field
    
    # For vector magnetograms:
    # product_physobs = a.Physobs.vector_magnetic_field
    
    # 3) Perform the data search.
    result = Fido.search(
        a.Time(start_time, end_time),
        a.Instrument.hmi,
        product_physobs
    )
    
    # 4) Download the files to a new folder named "sunpy/" in your current directory.
    #    "{file}" preserves the original JSOC-style naming, e.g. hmi.M_720s.20220101...
    files = Fido.fetch(result, path="sunpy/SDO/data/{file}")
    
    print(f"Downloaded files to './sunpy' folder:\n{files}")


if __name__ == "__main__":
    download_hmi_data()
