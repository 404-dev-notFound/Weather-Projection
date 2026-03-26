import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os

class ClimateDataset(Dataset):
    def __init__(self, miroc6_path, era5_path=None, sequence_length=14):
        """
        Custom Dataset for Climate Downscaling using sliding windows.
        
        Args:
            miroc6_path (str): Path to the coarse input data (MIROC6 3x3).
            era5_path (str): Path to the target high-res data (ERA5 17x17).
                             If None, the dataset can be used for inference only.
            sequence_length (int): Number of days in the sliding window.
        """
        self.sequence_length = sequence_length
        self.features = ['T_avg', 'PCP', 'AP', 'RH', 'WS']
        
        print(f"Loading MIROC6 input data from {miroc6_path}...")
        self.df_x = pd.read_csv(miroc6_path)
        self.df_x['Date'] = pd.to_datetime(self.df_x['Date']).dt.date
        
        # Sort and group by date
        self.df_x = self.df_x.sort_values(by=['Date', 'Lat', 'Lon'])
        dates_x = self.df_x['Date'].unique()
        
        self.df_y = None
        if era5_path is not None and os.path.exists(era5_path):
            print(f"Loading ERA5 target data from {era5_path}...")
            self.df_y = pd.read_csv(era5_path)
            self.df_y['Date'] = pd.to_datetime(self.df_y['Date']).dt.date
            self.df_y = self.df_y.sort_values(by=['Date', 'Lat', 'Lon'])
            dates_y = self.df_y['Date'].unique()
            # Intersect dates to ensure strict alignment
            self.valid_dates = sorted(list(set(dates_x).intersection(set(dates_y))))
        else:
            if era5_path is not None:
                print(f"Warning: ERA5 dataset '{era5_path}' not found. Loading only continuous input data for inference.")
            self.valid_dates = sorted(dates_x)
            
        print("Reshaping spatial grids...")
        # Dictionary to store tensors by date for temporal slicing
        self.data_dict_x = self._build_spatial_dict(self.df_x, self.valid_dates, grid_size=3)
        self.data_dict_y = None
        if self.df_y is not None:
            self.data_dict_y = self._build_spatial_dict(self.df_y, self.valid_dates, grid_size=17)
            
        # We can only create sequences where we have enough historical data
        self.sequence_indices = []
        for i in range(len(self.valid_dates) - self.sequence_length + 1):
             self.sequence_indices.append(i)

    def _build_spatial_dict(self, df, valid_dates, grid_size):
        data_dict = {}
        # Make a fast lookup subset
        df_sub = df[df['Date'].isin(valid_dates)]
        grouped = df_sub.groupby('Date')
        
        for date, group in grouped:
            # group has size grid_size*grid_size x len(features)
            spatial_tensor = []
            for feat in self.features:
                mat = group[feat].values.reshape(grid_size, grid_size)
                spatial_tensor.append(mat)
            # shape: (Channels, Height, Width)
            spatial_tensor = np.stack(spatial_tensor, axis=0)
            data_dict[date] = torch.tensor(spatial_tensor, dtype=torch.float32)
        return data_dict

    def __len__(self):
        return len(self.sequence_indices)

    def __getitem__(self, idx):
        start_idx = self.sequence_indices[idx]
        seq_dates = self.valid_dates[start_idx : start_idx + self.sequence_length]
        
        # Build sequence tensor for X
        x_seq = [self.data_dict_x[d] for d in seq_dates]
        # X shape: (Time, Channels, Height, Width)
        x_tensor = torch.stack(x_seq, dim=0) 
        
        if self.data_dict_y is not None:
            y_seq = [self.data_dict_y[d] for d in seq_dates]
            y_tensor = torch.stack(y_seq, dim=0)
            return x_tensor, y_tensor
        
        return x_tensor

def get_dataloaders(miroc6_path, era5_path, batch_size=16, seq_length=14, split_ratio=0.8):
    dataset = ClimateDataset(miroc6_path, era5_path, sequence_length=seq_length)
    
    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
