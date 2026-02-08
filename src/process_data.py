import pandas as pd
import numpy as np
import torch
import os

# ==========================================
# CONFIGURATION
# ==========================================
ENV_FILE = "data/raw/seawater temperature/Environmental_History_1985-2024.csv"
BIO_FILE = "data/processed/Biological_Observations_1985-2024.csv"
CO2_FILE = "data/raw/co2 concentration/co2_daily_mlo.csv"
OUT_DIR = "data/processed/"

def main():
    print("--- STARTING DATA PROCESSING(SST + DHW + CO2)---")
    
    # 1. Load Environmental Data
    print(f"Loading Environmental Data from {ENV_FILE}...")
    env_df = pd.read_csv(ENV_FILE)
    env_df['Date'] = pd.to_datetime(env_df['Date'])
    
    initial_len = len(env_df)
    env_df = env_df.drop_duplicates(subset=['Site_ID', 'Date'], keep='first')
    if len(env_df) < initial_len:
        print(f"   > Removed {initial_len - len(env_df)} duplicate rows from Environmental Data.")

    # 2. Load CO2 Data & Preprocess
    print(f"Loading Global CO2 Data from {CO2_FILE}...")
    # If 'Year' col doesn't exist, try reading without skip or check header row
    try:
        # Handles both raw NOAA files (with # comments) and clean CSVs
        co2_df = pd.read_csv(CO2_FILE, comment='#')
        # Cleanup column names (strip spaces)
        co2_df.columns = [c.strip() for c in co2_df.columns]
        # If 'Year' col doesn't exist, try reading without skip or check header row
        if 'Year' in co2_df.columns and 'Month' in co2_df.columns:
            # Construct Date column from Year, Month, Day
            co2_df['Date'] = pd.to_datetime(co2_df[['Year', 'Month', 'Day']])
            # Find CO2 column
            # We look for a column with "CO2" in the name
            co2_col = [c for c in co2_df.columns if 'CO2' in c and 'fraction' in c.lower()]
            if not co2_col:
                co2_col = [c for c in co2_df.columns if 'CO2' in c] # Fallback
            if co2_col:
                target_col = co2_col[0]
                co2_df = co2_df[['Date', target_col]].copy()
                co2_df.columns = ['Date', 'CO2']
                # Interpolate CO2 to daily (fill gaps)
                co2_df = co2_df.set_index('Date').resample('D').interpolate(method='linear').reset_index()
                # Merge CO2 into Environmental Data (Left Join on Date)
                # Since CO2 is global, it applies to all sites equally
                env_df = pd.merge(env_df, co2_df, on='Date', how='left')
                # Fill any remaining CO2 NaNs (forward fill then backward fill)
                env_df['CO2'] = env_df['CO2'].ffill().bfill()
                print("   > CO2 Data merged successfully.")
            else:
                print("   ! Warning: CO2 column not found in file. Using 0.0")
                env_df['CO2'] = 0.0
        else:
            print("   ! Warning: Year/Month/Day columns not found in CO2 file. Using 0.0")
            env_df['CO2'] = 0.0
    except Exception as e:
        print(f"! Warning: CO2 processing failed ({e}). Proceeding without CO2.")
        env_df['CO2'] = 0.0

    # 3. Pivot to Tensor Format (Sites x Time x Features)
    print("Pivoting Data...")
    # Get sorted list of sites and dates
    sites = env_df['Site_ID'].unique()
    dates = env_df['Date'].sort_values().unique()
    
    # Save these lists for later use
    site_map = {site: i for i, site in enumerate(sites)}
    date_map = {date: i for i, date in enumerate(dates)}
    
    num_sites = len(sites)
    num_days = len(dates)
    num_features = 3  # SST, DHW, CO2
    
    # Initialize Tensor
    X = np.zeros((num_sites, num_days, num_features), dtype=np.float32)
    
    # Fill Tensor (Vectorized approach is harder with irregular CSV, looping is safer for now)
    print("   > Creating slices...")
    
    # Efficient pivot using reindex to ensure alignment
    # If duplicates existed, .pivot() would crash. .drop_duplicates() above prevents this.
    pivot_sst = env_df.pivot(index='Site_ID', columns='Date', values='SST').reindex(sites)
    pivot_dhw = env_df.pivot(index='Site_ID', columns='Date', values='DHW').reindex(sites)
    pivot_co2 = env_df.pivot(index='Site_ID', columns='Date', values='CO2').reindex(sites)
    
    X[:, :, 0] = pivot_sst.values
    X[:, :, 1] = pivot_dhw.values
    X[:, :, 2] = pivot_co2.values
    
    # Normalize (Z-Score)
    print("Normalizing...")
    # Handle NaNs (if any sites missing dates)
    X = np.nan_to_num(X)
    mean = X.mean(axis=(0, 1))
    std = X.std(axis=(0, 1))
    X = (X - mean) / (std + 1e-6)

    # 4. Process Biological Labels (Y)
    print(f"Processing Labels (Y) from {BIO_FILE}...")
    try:
        bio_df = pd.read_csv(BIO_FILE)
        
        # Initialize Y tensor and Mask
        # Y shape: (Sites, Time, 1) -> Coral Cover
        y_tensor = np.zeros((num_sites, num_days, 1), dtype=np.float32)
        mask_tensor = np.zeros((num_sites, num_days, 1), dtype=np.float32)

        count_matched = 0

        # Map Years to indices (Assumes survey happened on July 1st of that year)
        # Or find the index of the date closest to July 1st
        for idx, row in bio_df.iterrows():
            s_id = row['Site_ID']
            year = int(row['Year'])
            cover = row['Coral_Cover']
            
            if s_id in site_map:
                site_idx = site_map[s_id]
                # Assume survey is mid-year (July 1st)
                target_date = pd.Timestamp(year=year, month=7, day=1)
                
                # Find closest date index
                if target_date in date_map:
                    time_idx = date_map[target_date]
                    y_tensor[site_idx, time_idx, 0] = cover
                    mask_tensor[site_idx, time_idx, 0] = 1.0
                    count_matched += 1
                else:
                    # Fallback: Find closest date index manually if exact date missing
                    # This handles edge cases where July 1st might be missing in Env data
                    closest_date = min(dates, key=lambda d: abs(d - target_date))
                    time_idx = date_map[closest_date]
                    y_tensor[site_idx, time_idx, 0] = cover
                    mask_tensor[site_idx, time_idx, 0] = 1.0
                    count_matched += 1
        print(f"   > Matched {count_matched} biological observations to the environmental timeline.")

        # 5. Save
        os.makedirs(OUT_DIR, exist_ok=True)
        torch.save(torch.tensor(X), f"{OUT_DIR}X.pt")
        torch.save(torch.tensor(y_tensor), f"{OUT_DIR}y.pt")
        torch.save(torch.tensor(mask_tensor), f"{OUT_DIR}mask.pt")
    
        # Save Metadata (Site IDs and Dates) for later reference
        pd.DataFrame(sites, columns=['Site_ID']).to_csv(f"{OUT_DIR}site_list.csv", index=False)
        
        print("DONE! Processed X.pt, y.pt, and mask.pt")

    except FileNotFoundError:
        print(f"Error: {BIO_FILE} not found. Run merge_biology.py first!")

        
if __name__ == "__main__":
    main()