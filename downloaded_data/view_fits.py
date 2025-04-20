"""
This repository contains a script to view FITS (Flexible Image Transport System) files.
"""

from astropy.io import fits
import matplotlib.pyplot as plt

# Load the FITS file
file_path = "path_to_hmi_data.fits"
hdul = fits.open(file_path)

# Access data and metadata
data = hdul[0].data  # Image data (2D array)
header = hdul[0].header  # Metadata

# Display the magnetogram
plt.imshow(data, cmap='gray', origin='lower')
plt.colorbar(label="Magnetic Field Strength (Gauss)")
plt.title("HMI Magnetogram")
plt.show()
