import intake
import xarray as xr
import pandas as pd
import numpy as np
import dask
import asyncio
import sys

# Configure dask to use multithreading for faster network I/O
dask.config.set(scheduler='threads')

if sys.version_info[0] == 3 and sys.version_info[1] >= 8 and sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

print("Connecting to the Google Cloud CMIP6 catalog...")
catalog_url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
col = intake.open_esm_datastore(catalog_url)

variables = ['tas', 'pr', 'psl', 'hursmax', 'hursmin', 'uas', 'vas']

def fetch_miroc6_data(experiment_name, start_date, end_date):
    """Helper function to fetch and slice data for a specific experiment"""
    query = dict(
        source_id='MIROC6',
        experiment_id=experiment_name,
        table_id='day',
        variable_id=variables,
        member_id='r1i1p1f1'
    )
    
    search_results = col.search(**query)
    dsets = search_results.to_dataset_dict(
        zarr_kwargs={'consolidated': True},
        xarray_combine_by_coords_kwargs={'compat': 'override'}
    )
    
    dataset_key = list(dsets.keys())[0]
    ds_global = dsets[dataset_key]
    
    # --- SPATIAL SLICING: The 3x3 UAE Bounding Box ---
    print("Slicing the global data down to the UAE 3x3 grid...")
    ds_uae = ds_global.sel(lat=slice(22.0, 26.2), lon=slice(52.0, 56.2))
    ds_timeframe = ds_uae.sel(time=slice(start_date, end_date)).compute()
    
    df = ds_timeframe.to_dataframe().reset_index()
    return df[['time', 'lat', 'lon', 'tas', 'pr', 'psl', 'hursmax', 'hursmin', 'uas', 'vas']]

print("Fetching Historical data (1950 - 2014)...")
df_miroc6 = fetch_miroc6_data('historical', '1950-01-01', '2014-12-31')


# --- Meteorological Conversions (To match ERA5 formatting) ---
print("Converting units to match ERA5 ground truth...")

df_miroc6['T_avg'] = df_miroc6['tas'] - 273.15 
df_miroc6['PCP'] = df_miroc6['pr'] * 86400
df_miroc6['AP'] = df_miroc6['psl'] / 100
df_miroc6['RH'] = (df_miroc6['hursmax'] + df_miroc6['hursmin']) / 2
df_miroc6['WS'] = np.sqrt(df_miroc6['uas']**2 + df_miroc6['vas']**2)

# --- THE GRID FIXES ---
# We MUST keep Lat and Lon in the final dataframe to preserve the 3x3 grid
df_final = df_miroc6[['time', 'lat', 'lon', 'T_avg', 'PCP', 'AP', 'RH', 'WS']]
df_final.rename(columns={'time': 'Date', 'lat': 'Lat', 'lon': 'Lon'}, inplace=True)

# Drop duplicates caused by hidden height dimensions, keeping the Date+Lat+Lon combo intact
df_final = df_final.drop_duplicates(subset=['Date', 'Lat', 'Lon'], keep='first')

print("\n--- Data successfully extracted and formatted! ---")
# Print 10 rows so you can verify that multiple pixels exist for the exact same date
print(df_final.head(10))

# Save to CSV (index=False prevents Pandas from writing arbitrary row numbers)
output_filename = "MIROC6_UAE_Spatial_Input_1950_2014.csv"
df_final.to_csv(output_filename, index=False)
print(f"\nSaved Gridded MIROC6 training data to {output_filename}")

# Clear fsspec instance cache to prevent asyncio loop errors at script exit
import fsspec
fsspec.clear_instance_cache()