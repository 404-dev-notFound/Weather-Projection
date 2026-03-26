import os
import cdsapi
import xarray as xr
import pandas as pd
import numpy as np

# 1. Define the Timeframe (Your Baseline: 1950 to 2025)
start_year = 1990
end_year = 2014

# 2. Define the Location (Dubai Bounding Box: North, West, South, East)
# ERA5 uses bounding boxes to slice data. This box strictly covers Dubai.
dubai_area = [25.35, 55.25, 25.15, 55.45] 

# Force the library to look in your current working directory
os.environ['CDSAPI_RC'] = os.path.join(os.getcwd(), '.cdsapirc')

print("Connecting to the Copernicus Climate Data Store (CDS)...")
c = cdsapi.Client()

# 3. Fetch the Hourly Data Month-by-Month
# The CDS servers will reject massive multi-variable yearly requests. 
# Looping month-by-month safely bypasses the "cost limits exceeded" error.
for year in range(start_year, end_year + 1):
    for month in range(1, 13):
        month_str = f"{month:02d}" # Formats 1 to '01', 2 to '02', etc.
        output_nc = f"ERA5_Dubai_{year}_{month_str}.nc"
        
        # Skip downloading if the file already exists on your hard drive
        if not os.path.exists(output_nc):
            print(f"Downloading ERA5 data for {year}-{month_str}...")
            c.retrieve(
                'reanalysis-era5-single-levels',
                {
                    'product_type': 'reanalysis',
                    'format': 'netcdf',
                    'variable': [
                        '2m_temperature',          
                        'total_precipitation',     
                        'surface_pressure',        
                        '2m_dewpoint_temperature', 
                        '10m_u_component_of_wind', 
                        '10m_v_component_of_wind', 
                    ],
                    'year': str(year),
                    'month': month_str, # Now we only request one month at a time
                    'day': [f"{d:02d}" for d in range(1, 32)],
                    'time': [f"{h:02d}:00" for h in range(24)],
                    'area': dubai_area,
                },
                output_nc
            )
            
# 4. Process and Clean the Data
print("Loading downloaded files and calculating daily statistics...")
# Load all yearly NetCDF files into one unified xarray dataset
ds = xr.open_mfdataset("ERA5_Dubai_*.nc", combine='by_coords')

# Average across the bounding box to get a single spatial point for Dubai
ds_point = ds.mean(dim=['latitude', 'longitude'])

# Convert to Pandas DataFrame for daily aggregation
df_hourly = ds_point.to_dataframe().reset_index()

# 5. Meteorological Conversions
# ERA5 provides raw physics units. We must convert them to standard weather formats.
T_kelvin = df_hourly['t2m']
Td_kelvin = df_hourly['d2m']
u_wind = df_hourly['u10']
v_wind = df_hourly['v10']

df_hourly['T_Celsius'] = T_kelvin - 273.15
# Relative Humidity (Magnus-Tetens approximation)
df_hourly['RH'] = 100 * (np.exp((17.625 * (Td_kelvin - 273.15)) / (243.04 + (Td_kelvin - 273.15))) / 
                         np.exp((17.625 * (T_kelvin - 273.15)) / (243.04 + (T_kelvin - 273.15))))
# Wind speed magnitude derived from U (East-West) and V (North-South) vectors
df_hourly['WS'] = np.sqrt(u_wind**2 + v_wind**2) 
df_hourly['PCP'] = df_hourly['tp'] * 1000 # Convert meters to mm
df_hourly['AP'] = df_hourly['sp'] / 100 # Convert Pascals to hPa (millibars)

# 6. Downsample Hourly Data to Daily Data 
print("Resampling hourly data to daily metrics for the LSTM...")
df_hourly.set_index('valid_time', inplace=True) # or 'time' depending on CDS API version

df_daily = pd.DataFrame()
df_daily['T_avg'] = df_hourly['T_Celsius'].resample('D').mean()
df_daily['T_min'] = df_hourly['T_Celsius'].resample('D').min()
df_daily['T_max'] = df_hourly['T_Celsius'].resample('D').max()
df_daily['PCP']   = df_hourly['PCP'].resample('D').sum() # We sum the hours for total daily rainfall
df_daily['RH']    = df_hourly['RH'].resample('D').mean()
df_daily['AP']    = df_hourly['AP'].resample('D').mean()
df_daily['WS']    = df_hourly['WS'].resample('D').mean()

# Handle any rare gaps smoothly
df_daily = df_daily.interpolate(method='linear', limit_direction='both')

# 7. View the Results and Save
print("\n--- ERA5 Data successfully loaded, converted, and cleaned! ---")
print(df_daily.head()) 
print("\n", df_daily.tail()) 

output_filename = "Dubai_ERA5_Baseline_1950_2025.csv"
df_daily.to_csv(output_filename)
print(f"\nSaved Gold-Standard ground truth data to {output_filename}")