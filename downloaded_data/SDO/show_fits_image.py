import matplotlib.pyplot as plt
from astropy.io import fits


def show_fits_image(path_to_fits):
    """
    Opens a FITS file and uses matplotlib to display the image data.
    """
    with fits.open(path_to_fits) as hdul:
        hdul.info()
        for i, hdu in enumerate(hdul):
            print(
                f"HDU[{i}] = {type(hdu)} with shape {hdu.data.shape if hdu.data is not None else 'None'}")
        # For a simple FITS file, the primary HDU often contains the image data
        # Option 1: Hardcode the extension index:
        image_data = hdul[1].data

        # Option 2: Iterate over HDUs and pick the first non-empty data:
        # image_data = None
        # for hdu in hdul:
        #     if hdu.data is not None:
        #         image_data = hdu.data
        #         break

        if image_data is None:
            print("No image data found in the selected HDU.")
            return

        # Display the image using matplotlib
        plt.imshow(image_data, cmap='gray', origin='lower')
        plt.colorbar()
        plt.title("FITS Image")
        plt.show()


# Example usage, adjust path accordingly:
# show_fits_image("data/example_file.fits")
show_fits_image(
    "data/hmi.m_45s/hmi.m_45s.2022.01.01_00_01_30_TAI.magnetogram.fits")
