import glob
import os
import xarray as xr
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

def _process_file(nc_file):
    """
    Helper function to open an individual .nc file, extract the flux data,
    and return a DataFrame. Returns None if there's an error or no recognized
    flux variable.
    """
    try:
        ds = xr.open_dataset(nc_file)
        # Pick your flux variable
        if "xrsb_flux" in ds.variables:
            flux_var = "xrsb_flux"
        elif "b_flux" in ds.variables:
            flux_var = "b_flux"
        elif "a_flux" in ds.variables:
            flux_var = "a_flux"
        else:
            print(f"  Warning: No recognized flux variable in {nc_file}, skipping.")
            ds.close()
            return None

        # Convert to DataFrame
        df = ds[[flux_var]].to_dataframe().reset_index()
        df.rename(columns={flux_var: "flux"}, inplace=True)
        df["filename"] = os.path.basename(nc_file)

        ds.close()
        return df

    except Exception as e:
        print(f"  Error reading {nc_file}: {e}")
        return None


def netcdf_to_parquet_parallel(data_dir, out_parquet="goes_avg1m_g13.parquet", max_workers=None):
    """
    Finds all netCDF files in data_dir matching '*avg1m_g13*.nc',
    extracts (time, flux) data in parallel, and saves to a single Parquet file.
    
    Parameters
    ----------
    data_dir : str
        Directory containing netCDF files (with 'avg1m_g13' in their names).
    out_parquet : str
        Filename for the output Parquet file.
    max_workers : int or None
        Number of worker processes. If None, defaults to the number of CPUs.
    """
    pattern = os.path.join(data_dir, "*avg1m_g13*.nc")
    all_files = sorted(glob.glob(pattern))
    if not all_files:
        raise FileNotFoundError(f"No files found matching {pattern}")

    print(f"Found {len(all_files)} matching files.")
    print(f"Using up to {max_workers or 'CPU_count'} processes for parallel I/O.\n")

    df_list = []

    # Process files in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(_process_file, f): f for f in all_files}
        print(f"Processing {len(future_to_file)} files in parallel...")
        for future in as_completed(future_to_file):
            print(f"Processing file {future_to_file[future]}...")
            fpath = future_to_file[future]
            try:
                result_df = future.result()
                print(f"Result: {result_df}")
                if result_df is not None:
                    df_list.append(result_df)
            except Exception as exc:
                print(f"  {fpath} generated an exception: {exc}")

    if not df_list:
        raise ValueError("No valid data frames to save. All files might have been skipped or errored.")

    # Concatenate all partial DataFrames
    final_df = pd.concat(df_list, ignore_index=True)

    # Write to Parquet
    print(f"Writing {len(final_df)} rows to {out_parquet} ...")
    final_df.to_parquet(out_parquet, index=False)
    print("Done. Parquet file created.")


if __name__ == "__main__":
    DATA_DIR = "/Users/antanaszilinskas/Desktop/Imperial College London/D2P/Coursework/masters-project/data/GOES/data/avg1m_2010_to_2024"
    OUTPUT_NAME = "goes_avg1m_g13.parquet"

    netcdf_to_parquet_parallel(DATA_DIR, out_parquet=OUTPUT_NAME, max_workers=None)