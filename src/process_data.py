import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FILE = "Environmental_History_1985-2024.csv"
OUTPUT_TENSOR = "environmental_tensor.pt"

def process_data():
    print("1. Loading raw environmental data...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_FILE}. Run the generator first!")
        return
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    print("2. Aggregating and Reshaping...")
    
    # Instead of pivot(), we use pivot_table() with aggfunc='mean'
    # This averages the temperature across multiple transects for the same Reef ID
    pivot_sst = df.pivot_table(index='Date', columns='Site_ID', values='SST', aggfunc='mean')
    pivot_dhw = df.pivot_table(index='Date', columns='Site_ID', values='DHW', aggfunc='mean')
    
    # Fill any tiny missing gaps (Linear interpolation)
    pivot_sst = pivot_sst.interpolate(method='linear', limit_direction='both')
    pivot_dhw = pivot_dhw.interpolate(method='linear', limit_direction='both')

    # Get dimensions
    dates = pivot_sst.index
    sites = pivot_sst.columns
    num_days = len(dates)
    num_sites = len(sites)
    
    print(f"   Dimensions: {num_sites} Sites x {num_days} Days")

    # ==========================================
    # 3. NORMALIZATION (Crucial for Neural Nets)
    # ==========================================
    print("3. Normalizing data...")
    
    # Standardize SST (Z-score)
    scaler_sst = StandardScaler()
    sst_values = scaler_sst.fit_transform(pivot_sst.values) # Shape: (Days, Sites)
    
    # Standardize DHW
    scaler_dhw = StandardScaler()
    dhw_values = scaler_dhw.fit_transform(pivot_dhw.values)
    
    # Stack into 3D Array: (Sites, Days, Channels)
    # Channels = 2 (SST, DHW)
    # We transpose to get (Sites, Days)
    sst_T = sst_values.T 
    dhw_T = dhw_values.T
    
    # Combine
    data_np = np.stack([sst_T, dhw_T], axis=-1) # Shape: (Sites, Days, 2)
    
    # Convert to PyTorch Tensor
    data_tensor = torch.FloatTensor(data_np)
    
    print(f"   Final Tensor Shape: {data_tensor.shape}")

    # ==========================================
    # 4. VISUALIZATION
    # ==========================================
    print("4. Generating visualization...")
    plt.figure(figsize=(10, 5))
    # Plot the first site's data
    plt.plot(dates, sst_values[:, 0], label="Normalized SST", alpha=0.7)
    plt.plot(dates, dhw_values[:, 0], label="Normalized DHW", alpha=0.7, color='orange')
    plt.title(f"Processed Control Path for Site: {sites[0]}")
    plt.xlabel("Date")
    plt.ylabel("Normalized Value (Z-Score)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("processed_data_plot.png")
    print("   Saved plot to 'processed_data_plot.png'")

    # ==========================================
    # 5. SAVE TENSOR
    # ==========================================
    torch.save(data_tensor, OUTPUT_TENSOR)
    print(f"5. Saved processed tensor to {OUTPUT_TENSOR}")
    print("   Success! You are ready for the Neural CDE model.")

if __name__ == "__main__":
    process_data()