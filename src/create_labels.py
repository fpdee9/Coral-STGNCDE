import pandas as pd
import numpy as np
import torch
import os

# ==========================================
# CONFIGURATION
# ==========================================
# File Paths
ENV_FILE = "Environmental_History_1985-2024.csv" # Needed to get the exact Site IDs and Dates
GBR_FILE = "Coral Cover.xlsx - GBR CC.csv"
SJVI_FILE = "Coral Cover.xlsx - SJVI CC.csv"
MOOREA_FILE = "Coral Cover.xlsx - Moorea CC average.csv"

OUTPUT_LABELS = "labels.pt"
OUTPUT_MASK = "mask.pt"

def create_labels():
    print("1. Loading Environmental Baseline...")
    # We need the exact order of Sites and Dates from your environmental file
    # to make sure everything lines up perfectly.
    env_df = pd.read_csv(ENV_FILE)
    
    # Extract unique sites and dates in order
    unique_sites = list(env_df.columns[1:]) if 'Site_ID' not in env_df.columns else env_df['Site_ID'].unique()
    unique_dates = pd.to_datetime(env_df['Date'].unique())
    unique_dates = np.sort(unique_dates)
    
    # Create lookups
    site_to_idx = {site: i for i, site in enumerate(unique_sites)}
    date_to_idx = {date: i for i, date in enumerate(unique_dates)}
    
    num_sites = len(unique_sites)
    num_days = len(unique_dates)
    print(f"   Baseline: {num_sites} Sites over {num_days} Days.")

    # Initialize Tensors with NaNs
    # Shape: (Sites, Days, 1)
    labels = torch.full((num_sites, num_days, 1), float('nan'))
    mask = torch.zeros((num_sites, num_days, 1)) # 0 = No Data, 1 = Data

    # ==========================================
    # 2. PROCESS GREAT BARRIER REEF (Daily Precision)
    # ==========================================
    print("2. Processing GBR Data...")
    try:
        gbr = pd.read_csv(GBR_FILE)
        # Filter for only Hard Coral cover
        gbr = gbr[gbr['GROUP_CODE'] == 'Hard Coral']
        gbr['Date'] = pd.to_datetime(gbr['SAMPLE_DATE'])
        
        count = 0
        for _, row in gbr.iterrows():
            site_id = str(row['REEF_ID'])
            date = row['Date']
            cover = row['COVER'] / 100.0 # Normalize 0-100 -> 0.0-1.0
            
            # Map to index
            if site_id in site_to_idx and date in date_to_idx:
                s_idx = site_to_idx[site_id]
                d_idx = date_to_idx[date]
                
                labels[s_idx, d_idx, 0] = cover
                mask[s_idx, d_idx, 0] = 1.0
                count += 1
        print(f"   Mapped {count} GBR surveys.")
        
    except FileNotFoundError:
        print("   WARNING: GBR file not found. Skipping.")

    # ==========================================
    # 3. PROCESS ST. JOHN (Annual -> Summer)
    # ==========================================
    print("3. Processing St. John Data...")
    try:
        sjvi = pd.read_csv(SJVI_FILE)
        # Methodology says surveys were July/August. We assign to July 15th.
        
        count = 0
        for _, row in sjvi.iterrows():
            site_name = str(row['Site'])
            year = int(row['Year'])
            cover = row['Coral_percent_cover'] / 100.0
            
            # Construct approximate date
            date = pd.Timestamp(year=year, month=7, day=15)
            
            if site_name in site_to_idx:
                s_idx = site_to_idx[site_name]
                # Find closest valid date index
                if date in date_to_idx:
                    d_idx = date_to_idx[date]
                    labels[s_idx, d_idx, 0] = cover
                    mask[s_idx, d_idx, 0] = 1.0
                    count += 1
        print(f"   Mapped {count} St. John surveys.")
        
    except FileNotFoundError:
        print("   WARNING: SJVI file not found. Skipping.")

    # ==========================================
    # 4. PROCESS MOOREA (Annual -> April)
    # ==========================================
    print("4. Processing Moorea Data...")
    try:
        moorea = pd.read_csv(MOOREA_FILE)
        # Methodology says surveys were in April. We assign to April 15th.
        
        count = 0
        for _, row in moorea.iterrows():
            # Create site string (e.g., "LTER 1")
            # Check column name (some files say 'Site', others 'Site_ID')
            site_val = row.get('Site') or row.get('Site_ID')
            site_name = f"LTER {int(site_val)}" 
            
            year = int(row['Year'])
            cover = row.get('Stony_coral_cover') or row.get('Percent_Cover')
            cover = cover / 100.0
            
            date = pd.Timestamp(year=year, month=4, day=15)
            
            if site_name in site_to_idx and date in date_to_idx:
                s_idx = site_to_idx[site_name]
                d_idx = date_to_idx[date]
                
                labels[s_idx, d_idx, 0] = cover
                mask[s_idx, d_idx, 0] = 1.0
                count += 1
                
        print(f"   Mapped {count} Moorea surveys.")

    except Exception as e:
        print(f"   WARNING: Moorea issue ({e}). Skipping.")

    # ==========================================
    # 5. SAVE
    # ==========================================
    torch.save(labels, OUTPUT_LABELS)
    torch.save(mask, OUTPUT_MASK)
    print(f"\nSaved {OUTPUT_LABELS} and {OUTPUT_MASK}")
    print("Final Check: Do these numbers look right?")
    print(f"Total Observations: {mask.sum().item()}")

if __name__ == "__main__":
    create_labels()