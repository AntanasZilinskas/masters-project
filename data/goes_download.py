from sunpy.net import Fido, attrs as a

def download_goes_data(start_time="2022-01-01 00:00", end_time="2022-01-01 01:00"):
    """
    Downloads GOES data from the given time range using SunPy.
    
    Parameters:
    -----------
    start_time : str
        The start date/time (UTC) of the time range in ISO format (YYYY-MM-DD HH:MM).
    end_time : str
        The end date/time (UTC) of the time range in ISO format (YYYY-MM-DD HH:MM).

    Example:
    --------
    download_goes_data("2022-01-01 00:00", "2022-01-01 06:00")
    """
    # Search for GOES data in the specified time range
    result = Fido.search(
        a.Time(start_time, end_time),
        a.Instrument("goes")  # GOES flares or XRS data
    )
    
    # Download the data
    downloaded_files = Fido.fetch(result)
    print(f"Downloaded GOES files for {start_time} to {end_time}:")
    for f in downloaded_files:
        print(f)

# Run the download function if desired:
if __name__ == "__main__":
    download_goes_data() 