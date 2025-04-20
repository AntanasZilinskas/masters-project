def open_fit_files_in_data():
    """
    Opens all .fit files found in the data directory (adjust accordingly) using astropy.io.fits.
    Prints basic header and data details.
    """
    import os
    from astropy.io import fits

    # 1) Change "@data" to "data" if your folder name is valid
    directory = "data"

    # 2) If your "data" folder is in a different location, adjust the path accordingly
    # directory = "/Users/yourusername/path/to/data"

    # Ensure astropy is installed: pip install astropy
    for filename in os.listdir(directory):
        if filename.lower().endswith(".fit"):
            file_path = os.path.join(directory, filename)
            with fits.open(file_path) as hdul:
                print(f"Opening: {file_path}")
                # Print a summary of all HDUs
                hdul.info()

                # Optionally look at a specific HDU's header or data
                # e.g., the first extension (i.e., HDU index 0)
                # print("Header of HDU[0]:", hdul[0].header)
                # if hdul[0].data is not None:
                #     print("Data shape:", hdul[0].data.shape)
                print("---")


if __name__ == "__main__":
    open_fit_files_in_data()
