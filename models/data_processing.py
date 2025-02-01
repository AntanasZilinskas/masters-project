import glob
import os
import re
import pandas as pd
import xarray as xr
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

# Helper to detect which GOES satellite the file belongs to
def _detect_satellite(filename: str) -> str:
    """
    Returns a short label for the satellite based on the filename
    (e.g. g13, g14, g15, etc.). Defaults to 'other' if no known satellite found.
    """
    for sat in ["g13", "g14", "g15", "g16", "g17"]:
        if sat in filename.lower():
            return sat
    return "other"

def _process_file(nc_file):
    """
    Helper function to open an individual .nc file, extract the flux data,
    detect the satellite, and return a DataFrame.
    Returns None if there's an error or no recognized flux variable.
    Uses engine="netcdf4" to avoid the h5netcdf error.
    """
    logging.info(f"Processing file: {nc_file}")
    try:
        ds = xr.open_dataset(nc_file, engine="netcdf4")
    except Exception as e:
        logging.error(f"  Error opening {nc_file} with engine='netcdf4': {e}")
        return None

    try:
        # Pick your flux variable
        if "xrsb_flux" in ds.variables:
            flux_var = "xrsb_flux"
        elif "b_flux" in ds.variables:
            flux_var = "b_flux"
        elif "a_flux" in ds.variables:
            flux_var = "a_flux"
        else:
            logging.warning(f"  No recognized flux variable in {nc_file}, skipping.")
            ds.close()
            return None

        # Convert to a pandas DataFrame
        df = ds[[flux_var]].to_dataframe().reset_index()
        df.rename(columns={flux_var: "flux"}, inplace=True)
        ds.close()

        # Detect which satellite this file comes from
        df["satellite"] = _detect_satellite(os.path.basename(nc_file))
        logging.info(f"  Successfully processed {nc_file}.")
        return df
    except Exception as e:
        logging.error(f"  Error processing {nc_file}: {e}")
        return None

def netcdf_to_parquet_parallel(data_dir, out_parquet="goes_avg1m_combined.parquet", max_workers=None):
    """
    Searches for all NetCDF files in data_dir matching '*avg1m*.nc', filters to those
    with a date in the filename between 2010 and 2024, extracts minute-level data in parallel,
    and combines them into a single DataFrame.
    
    For each minute, it prefers data from satellite 'g13'. If g13 is not available for a given minute,
    it falls back on g14, g15, etc. No minute will be missing if at least one satellite has data.
    
    The final DataFrame contains minute-by-minute flux data (log1p can be applied later
    if desired) extracted from whichever satellite is available.
    """
    pattern = os.path.join(data_dir, "*avg1m*.nc")
    all_files = sorted(glob.glob(pattern))
    if not all_files:
        raise FileNotFoundError(f"No files found matching {pattern}")

    # Filter files by date in name: look for a pattern of dYYYYMMDD and keep if 2010<=YYYY<=2024
    date_re = re.compile(r"d(\d{8})")
    filtered_files = []
    for f in all_files:
        match = date_re.search(os.path.basename(f))
        if match:
            date_str = match.group(1)  # e.g. "20160415"
            year = int(date_str[:4])
            if 2010 <= year <= 2024:
                filtered_files.append(f)
    if not filtered_files:
        raise ValueError("No files found within the date range 2010-2024.")

    logging.info(f"Found {len(filtered_files)} matching files in the date range 2010 to 2024.")
    logging.info(f"Using up to {max_workers or 'CPU_count'} processes for parallel I/O.\n")

    df_list = []
    # Process files in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(_process_file, f): f for f in filtered_files}
        logging.info(f"Processing {len(future_to_file)} files in parallel...")
        for future in as_completed(future_to_file):
            fpath = future_to_file[future]
            try:
                result_df = future.result()
                if result_df is not None:
                    df_list.append(result_df)
                    logging.info(f"File successfully added: {fpath}")
                else:
                    logging.warning(f"File returned no data: {fpath}")
            except Exception as exc:
                logging.error(f"{fpath} generated an exception: {exc}")

    if not df_list:
        raise ValueError("No valid data frames to save. All files might have been skipped or errored.")

    # Combine all processed DataFrames
    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df["time"] = pd.to_datetime(combined_df["time"])
    combined_df.sort_values(by="time", inplace=True)

    # Assign priority to each satellite (lower numbers mean higher priority)
    satellite_priority = {
        "g13": 1,
        "g14": 2,
        "g15": 3,
        "g16": 4,
        "g17": 5,
        "other": 6
    }
    combined_df["priority"] = combined_df["satellite"].map(satellite_priority).fillna(999)

    # For each minute, keep the row with the best priority.
    # This guarantees that for a given minute, if the primary satellite (g13) data exists, that will be chosen.
    grouped_df = combined_df.groupby("time", as_index=False).first()
    grouped_df.drop(columns=["priority"], inplace=True)
    grouped_df.sort_values(by="time", inplace=True)

    logging.info(f"Writing {len(grouped_df)} minute-level rows to {out_parquet} ...")
    grouped_df.to_parquet(out_parquet, index=False)
    logging.info("Done. Parquet file created.")

if __name__ == "__main__":
    DATA_DIR = "/Users/antanaszilinskas/Desktop/Imperial College London/D2P/Coursework/masters-project/data/GOES/data/avg1m_2010_to_2024"
    OUTPUT_NAME = "goes_avg1m_combined.parquet"
    netcdf_to_parquet_parallel(DATA_DIR, out_parquet=OUTPUT_NAME, max_workers=None)