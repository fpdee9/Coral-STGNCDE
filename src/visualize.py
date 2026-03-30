import torch
import torchcde
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from coral_model import CoralSTGNCDE

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR = "data/processed/"
MODEL_PATH = "coral_model_best.pth" 
OUTPUT_DIR = "results/plots/"

# --- FIXED: Must match the new train_coral.py ---
HIDDEN_DIM = 24  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalize_tensor(tensor):
    mean = tensor.mean(dim=1, keepdim=True)
    std = tensor.std(dim=1, keepdim=True)
    std[std == 0] = 1.0 
    return (tensor - mean) / std

def main():
    print("--- GENERATING VISUALIZATIONS ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Data
    print("Loading Data...")
    if not os.path.exists(f"{DATA_DIR}X.pt"):
        print("Error: processed data not found.")
        return

    X_raw = torch.load(f"{DATA_DIR}X.pt").float()
    y = torch.load(f"{DATA_DIR}y.pt").float()
    mask = torch.load(f"{DATA_DIR}mask.pt").float()
    adj = torch.load(f"{DATA_DIR}adjacency_matrix.pt").float()
    site_list = pd.read_csv(f"{DATA_DIR}site_list.csv")
    
    num_sites, num_times, num_features = X_raw.shape
    
    # 2. Prepare Inputs
    X_normalized = normalize_tensor(X_raw)
    X_time_first = X_normalized.permute(1, 0, 2)
    X_flat = X_time_first.reshape(num_times, -1).to(DEVICE)
    
    coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(X_flat)
    
    # 3. Load Model
    print(f"Loading Model: {MODEL_PATH}")
    model = CoralSTGNCDE(
        num_sites=num_sites,
        input_features=num_features,
        hidden_dim=HIDDEN_DIM,
        output_features=1,
        adj_matrix=adj
    ).to(DEVICE)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print(f"Error: {MODEL_PATH} not found.")
        return
        
    model.eval()
    
    # 4. Run Prediction
    print("Running Inference...")
    with torch.no_grad():
        pred = model(coeffs) 
    
    y_np = y.squeeze(-1).numpy()
    pred_np = pred.permute(1, 0, 2).squeeze(-1).cpu().numpy()
    mask_np = mask.squeeze(-1).numpy()
    
    # --- FIXED TIMELINE: Changed 'W' (Weekly) to 'MS' (Month Start) ---
    time_dates = pd.date_range(start='1985-01-01', periods=num_times, freq='MS')
    split_date = time_dates[int(num_times * 0.8)] 
    
    # 5. Plotting Loop
    print(f"Saving plots to {OUTPUT_DIR}...")
    
    for i in range(num_sites):
        site_name = site_list.iloc[i]['Site_ID']
        
        site_y = y_np[i, :]
        site_pred = pred_np[i, :]
        site_mask = mask_np[i, :]
        
        valid_indices = np.where(site_mask > 0)[0]
        valid_time = time_dates[valid_indices] 
        valid_y = site_y[valid_indices]
        
        plt.figure(figsize=(12, 6))
        
        # Plot Model Prediction (Blue Line) against Datetimes
        plt.plot(time_dates, site_pred, color='blue', label='AI Prediction', linewidth=1.5, alpha=0.8)
        
        # Plot Ground Truth (Red Lines & Dots) against Datetimes
        plt.plot(valid_time, valid_y, color='red', linestyle='-', linewidth=1.0, alpha=0.5)
        plt.scatter(valid_time, valid_y, color='red', label='Observed Data', s=10, zorder=5)
        
        # Plot Split Line at the correct Date
        plt.axvline(x=split_date, color='black', linestyle='--', alpha=0.5, label='Train / Test Split')
        
        # Formatting
        plt.title(f"Site: {site_name}")
        plt.xlabel("Year") 
        plt.ylabel("Coral Cover (0.0 - 1.0)")
        plt.ylim(-0.05, 1.05)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save
        safe_name = str(site_name).replace("/", "_").replace(" ", "_")
        plt.savefig(f"{OUTPUT_DIR}{safe_name}.png")
        plt.close()
        
        if (i+1) % 10 == 0:
            print(f"   Plotting {i+1}/{num_sites}...")

    print("Done! Check the 'results/plots/' folder.")

if __name__ == "__main__":
    main()