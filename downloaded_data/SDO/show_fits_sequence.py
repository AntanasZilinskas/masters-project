"""
show_fits_sequence.py

Plays all .fits files from the folder "data/hmi.m_45s" in sequence as a
matplotlib animation. Each frame shows an HMI magnetogram so you can watch
sunspots evolve over time.

Dependencies:
- matplotlib
- astropy
- glob
- numpy

Usage:
------
    python show_fits_sequence.py

A matplotlib window will open, cycling through each .fits file in that folder.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from astropy.io import fits
import glob
import numpy as np
import os


def animate_fits_sequence(fits_files, interval_ms=200):
    """
    Creates an animation of SDO FITS image frames in a single matplotlib window.

    Parameters
    ----------
    fits_files : list of str
        Paths to FITS files, sorted by time or filename order.
    interval_ms : int
        Delay between frames in milliseconds (default 200 = 5 fps).
    """
    # Pre-load image data (faster than opening each file for every frame)
    data_list = []
    valid_files = []
    for f in fits_files:
        try:
            with fits.open(f) as hdul:
                # For HMI magnetograms, the image is often in HDU[1]
                img_data = hdul[1].data
                data_list.append(img_data)
                valid_files.append(f)
        except Exception as e:
            print(f"Could not open {f} due to error: {e}")
            data_list.append(None)
            valid_files.append(f)

    # Ensure we have at least one valid FITS file
    if not any(d is not None for d in data_list):
        print("No valid FITS data found. Exiting.")
        return

    # Set up figure and axes
    fig, ax = plt.subplots()
    first_valid_idx = next(i for i, d in enumerate(data_list) if d is not None)
    first_frame_data = data_list[first_valid_idx]

    # Initialize the first frame
    im = ax.imshow(first_frame_data, cmap='gray', origin='lower')
    ax.set_title(f"Frame 0: {os.path.basename(valid_files[first_valid_idx])}")
    cbar = plt.colorbar(im, ax=ax, orientation='vertical')
    cbar.set_label('Magnetogram Values')

    def init_animation():
        im.set_data(first_frame_data)
        ax.set_title(
            f"Frame 0: {os.path.basename(valid_files[first_valid_idx])}")
        return (im,)

    def update(frame_idx):
        frame_data = data_list[frame_idx]
        if frame_data is None:
            # Some files couldn't be opened or had errors
            im.set_data(np.zeros_like(first_frame_data))
            ax.set_title(f"Frame {frame_idx}: Invalid Data")
        else:
            im.set_data(frame_data)
            ax.set_title(
                f"Frame {frame_idx}: {os.path.basename(valid_files[frame_idx])}")
        return (im,)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=range(len(data_list)),
        init_func=init_animation,
        interval=interval_ms,
        blit=True
    )

    plt.show()


if __name__ == "__main__":
    # Change this path if your directory layout is different.
    # For example, if you're in the SDO folder, and there's a "data" subfolder
    # next to this script:
    folder_path = "data/hmi.m_45s"

    print("Loading from:", os.path.abspath(folder_path))

    fits_files = sorted(glob.glob(os.path.join(folder_path, "*.fits")))

    if len(fits_files) == 0:
        print(f"No FITS files found in {folder_path}")
    else:
        animate_fits_sequence(fits_files, interval_ms=200)
