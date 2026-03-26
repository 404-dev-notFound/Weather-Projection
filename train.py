import torch
import torch.nn as nn
import torch.optim as optim
from models import CNNLSTM_Downscaler
from data_loader import get_dataloaders
import os

def train_model(miroc6_path, era5_path, epochs=10, batch_size=4, lr=1e-3, seq_length=14):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")
    
    if not os.path.exists(era5_path):
        print(f"ERA5 target dataset at {era5_path} is missing.")
        print("Please ensure the data generation script has completed creating the spatial matrix before training.")
        return

    # 1. Load Data
    print("Initializing Data Loaders...")
    train_loader, val_loader = get_dataloaders(miroc6_path, era5_path, batch_size=batch_size, seq_length=seq_length)
    
    # 2. Instantiate Model
    print("Instantiating CNN-LSTM Model...")
    model = CNNLSTM_Downscaler(in_channels=5, hidden_channels=32, out_channels=5).to(device)
    
    # 3. Loss & Optimizer
    # HuberLoss is ideal for climate arrays as it penalizes extreme outliers less aggressively than MSE, 
    # preventing unstable gradients from sudden storm/extreme heat pixels.
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 4. Training Loop
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            # outputs shape: (Batch, Seq_Len, 5, 17, 17)
            outputs = model(x)
            
            # Compute loss over the entire spatiotemporal output sequence
            loss = criterion(outputs, y)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # 5. Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # 6. Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_downscaler_model.pth")
            print(">> Saved new improved model checkpoint!")

if __name__ == "__main__":
    MIROC6 = "MIROC6_UAE_Spatial_Input_1950_2014.csv"
    # Assuming ERA5 target spatial data will be generated later
    ERA5_TARGET = "ERA5_UAE_Spatial_Target_1950_2014.csv" 
    
    print("--- UAE Climate Downscaling Training Pipeline ---")
    train_model(MIROC6, ERA5_TARGET)
