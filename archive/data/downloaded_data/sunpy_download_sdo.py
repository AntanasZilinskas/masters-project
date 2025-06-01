# ... existing code ...


def download_hmi_data(
    start_time,
    end_time,
    cadence,
    product="hmi.M_45s",
    email="antanas.zilinskas21@imperial.ac.uk",
):
    """
    Download HMI data using JSOC client instead of VSO

    Parameters:
    -----------
    start_time : str
        Start time in format 'YYYY-MM-DD HH:MM'
    end_time : str
        End time in format 'YYYY-MM-DD HH:MM'
    cadence : astropy.units.Quantity
        Time cadence for data
    product : str
        HMI data product (e.g., 'hmi.M_45s', 'hmi.B_720s')
    email : str
        Email address for JSOC export (required)
    """
    import astropy.units as u
    import sunpy.map
    from astropy.time import Time
    from sunpy.net import Fido
    from sunpy.net import attrs as a
    from sunpy.net import jsoc

    print(f"Searching for {product} data from {start_time} to {end_time}...")

    # Create JSOC client
    client = jsoc.JSOCClient()

    # Convert cadence to appropriate units
    if isinstance(cadence, u.Quantity):
        cadence_seconds = cadence.to(u.second).value
    else:
        # Assume minutes if not a Quantity
        cadence_seconds = cadence * 60

    # Format cadence for JSOC query
    cadence_str = f"{int(cadence_seconds)}s"

    # Create time range
    tstart = Time(start_time)
    tend = Time(end_time)

    # Query JSOC
    response = client.search(
        a.Time(tstart, tend),
        a.jsoc.Series(product),
        a.jsoc.Notify(email),
        a.jsoc.Segment("magnetogram"),
        a.Sample(cadence_str),
    )

    if len(response) == 0:
        print("No data found for the specified parameters.")
        return

    print(f"Found {len(response)} records. Downloading...")

    # Create download directory
    download_dir = f"downloaded_{product.replace('.', '_')}"
    os.makedirs(download_dir, exist_ok=True)

    # Request data export
    requests = client.request_data(response, method="url", protocol="fits")

    # Download data
    downloaded_files = client.get_request(requests, path=download_dir)

    print(f"Downloaded {len(downloaded_files)} files to {download_dir}/")
    return downloaded_files


# Example usage
if __name__ == "__main__":
    # Set your email address for JSOC requests
    email = "your.email@example.com"  # Replace with your email

    # Example: Download data matching the timestamps in your CSV files
    # Using 12-minute cadence to match the data in your CSV files
    download_hmi_data(
        start_time="2022-01-01 00:00",
        end_time="2022-09-16 04:10",
        cadence=12 * u.minute,
        product="hmi.sharp_cea_720s",  # SHARP data with vector magnetic field
        email=email,
    )

    # Alternative: Download magnetogram data for the same period
    download_hmi_data(
        start_time="2022-01-01 00:00",
        end_time="2022-09-16 04:10",
        cadence=12 * u.minute,
        product="hmi.M_45s",  # Line-of-sight magnetogram data
        email=email,
    )
