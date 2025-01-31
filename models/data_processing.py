import glob
import os
import xarray as xr
import pandas as pd

def netcdf_to_parquet(data_dir, out_parquet="goes_avg1m_g13.parquet"):
    """
    Finds all netCDF files in data_dir matching '*avg1m_g13*.nc',
    extracts (time, flux) data, and saves to a single Parquet file.
    """

    pattern = os.path.join(data_dir, "*avg1m_g13*.nc")
    all_files = sorted(glob.glob(pattern))
    if not all_files:
        raise FileNotFoundError(f"No files found matching {pattern}")

    print(f"Found {len(all_files)} matching files.")
    df_list = []

    for nc_file in all_files:
        print(f"Processing: {nc_file}")
        try:
            ds = xr.open_dataset(nc_file)
            # Pick which flux variable you want; adjust logic as needed
            if "xrsb_flux" in ds.variables:
                flux_var = "xrsb_flux"
            elif "b_flux" in ds.variables:
                flux_var = "b_flux"
            elif "a_flux" in ds.variables:
                flux_var = "a_flux"
            else:
                print(f"  Warning: No recognized flux variable in {nc_file}, skipping.")
                ds.close()
                continue

            # Convert time dimension to DataFrame. 
            # Note: Xarray automatically recognizes 'time' dimension if present.
            # If your dimension is named differently, adjust "ds['time']".
            df = ds[[flux_var]].to_dataframe().reset_index()

            # Optionally rename flux column for clarity
            df.rename(columns={flux_var: "flux"}, inplace=True)

            # Example: if you want separate columns for the dataset name or a note 
            # about the instrument, you can add them here:
            df["filename"] = os.path.basename(nc_file)

            ds.close()

            df_list.append(df)

        except Exception as e:
            print(f"  Error reading {nc_file}: {e}")
            continue

    # Concatenate all partial DataFrames
    if not df_list:
        raise ValueError("No valid data frames to save. All files might have been skipped.")

    final_df = pd.concat(df_list, ignore_index=True)

    # Write to Parquet
    print(f"Writing {len(final_df)} rows to {out_parquet} ...")
    final_df.to_parquet(out_parquet, index=False)
    print("Done. Parquet file created.")

if __name__ == "__main__":
    DATA_DIR = "/Users/antanaszilinskas/Desktop/Imperial College London/D2P/Coursework/masters-project/data/GOES/data/avg1m_2010_to_2024"
    netcdf_to_parquet(DATA_DIR, out_parquet="goes_avg1m_g13.parquet")