import pandas as pd
import numpy as np
import torch
import os
from math import radians, cos, sin, asin, sqrt

# ==========================================
# CONFIGURATION
# ==========================================
PATH_AIMS = "data/raw/coral cover/GBR/[RAW] ltmp_hc_sc_a_by_site.csv"
PATH_REEFCLOUD = "data/raw/coral cover/EcoRRAP_Benthic_Data_2021to2024/reefcloudcover42_functionalgroup42.csv"

# Make sure these match your exact file paths
PATH_MOOREA = "data/raw/coral cover/Initial Set/moorea_avg_coral_cover.csv"
PATH_STJOHN = "data/raw/coral cover/Initial Set/st_john_avg_coral_cover.csv"

# This should point to your newly merged environmental file
PATH_ENV = "data/raw/seawater temperature/Environmental_History_1985-2024.csv" 

OUTPUT_DIR = "data/processed/"
GRAPH_THRESHOLD_KM = 50.0

def ingest_biology():
    print("--- STEP 1: INGESTING GLOBAL BIOLOGICAL DATA ---")
    
    # 1. AIMS (GBR)
    try:
        df_aims = pd.read_csv(PATH_AIMS)
        if 'GROUP_CODE' in df_aims.columns:
            df_aims = df_aims[df_aims['GROUP_CODE'].isin(['Hard Coral'])]
        aims_clean = df_aims.rename(columns={'REEF_ID': 'Site_ID', 'SAMPLE_DATE': 'Date', 'COVER': 'Coral_Cover', 'LATITUDE': 'Latitude', 'LONGITUDE': 'Longitude'})[['Site_ID', 'Date', 'Coral_Cover', 'Latitude', 'Longitude']].copy()
        aims_clean['Coral_Cover'] = pd.to_numeric(aims_clean['Coral_Cover'], errors='coerce') / 100.0
        print(f"   > AIMS (GBR): Loaded {len(aims_clean)} surveys.")
    except Exception as e: print(f"Error AIMS: {e}"); aims_clean = pd.DataFrame()

    # 2. ReefCloud (GBR)
    try:
        df_rc = pd.read_csv(PATH_REEFCLOUD)
        rc_clean = df_rc.rename(columns={'reef': 'Site_ID', 'date': 'Date', 'HC': 'Coral_Cover', 'site_latitude': 'Latitude', 'site_longitude': 'Longitude'})[['Site_ID', 'Date', 'Coral_Cover', 'Latitude', 'Longitude']].copy()
        rc_clean['Coral_Cover'] = pd.to_numeric(rc_clean['Coral_Cover'], errors='coerce') / 100.0
        rc_clean['Date'] = pd.to_datetime(rc_clean['Date'], format='%Y%m', errors='coerce').fillna(pd.to_datetime(rc_clean['Date'], errors='coerce'))
        print(f"   > ReefCloud (GBR): Loaded {len(rc_clean)} surveys.")
    except Exception as e: print(f"Error ReefCloud: {e}"); rc_clean = pd.DataFrame()

    # # 3. Moorea (French Polynesia)
    # try:
    #     df_moorea = pd.read_csv(PATH_MOOREA)
        
    #     # Safely map whatever column names exist in the CSV to our standard names
    #     moorea_map = {'Site': 'Site_ID', 'Year': 'Date', 'Stony_coral_cover': 'Coral_Cover'}
    #     df_moorea = df_moorea.rename(columns=moorea_map)
        
    #     # Only extract the columns we know exist
    #     moorea_clean = df_moorea[['Site_ID', 'Date', 'Coral_Cover']].copy()
        
    #     # Format Dates (if it's just a 4-digit year like '2005', make it '2005-06-01')
    #     if str(moorea_clean['Date'].iloc[0]).isdigit() and len(str(moorea_clean['Date'].iloc[0])) == 4:
    #         moorea_clean['Date'] = pd.to_datetime(moorea_clean['Date'].astype(str) + '-06-01')
    #     else:
    #         moorea_clean['Date'] = pd.to_datetime(moorea_clean['Date'])
            
    #     # Format Cover (Divide by 100 if it's a percentage)
    #     moorea_clean['Coral_Cover'] = pd.to_numeric(moorea_clean['Coral_Cover'], errors='coerce')
    #     if moorea_clean['Coral_Cover'].max() > 1.0:
    #         moorea_clean['Coral_Cover'] = moorea_clean['Coral_Cover'] / 100.0
            
    #     # Hardcode approximate Moorea GPS coordinates since they aren't in the CSV
    #     moorea_clean['Latitude'] = -17.53
    #     moorea_clean['Longitude'] = -149.83
        
    #     # Fix Moorea Names
    #     def fix_moorea_name(x):
    #         x_str = str(x).strip()
    #         if x_str.isdigit(): return f"Moorea LTER {x_str}"
    #         elif "LTER" in x_str and "Moorea" not in x_str: return x_str.replace("LTER", "Moorea LTER")
    #         return x_str
            
    #     moorea_clean['Site_ID'] = moorea_clean['Site_ID'].apply(fix_moorea_name)
    #     print(f"   > Moorea (LTER): Loaded {len(moorea_clean)} surveys.")
    # except Exception as e: print(f"Error Moorea: {e}"); moorea_clean = pd.DataFrame()

    # # 4. St. John (USVI)
    # try:
    #     df_stjohn = pd.read_csv(PATH_STJOHN)
        
    #     # Safely map St. John column names
    #     stjohn_map = {'Site': 'Site_ID', 'Year': 'Date', 'Stony_coral_cover': 'Coral_Cover'}
    #     df_stjohn = df_stjohn.rename(columns=stjohn_map)
        
    #     # Extract only the existing columns
    #     stjohn_clean = df_stjohn[['Site_ID', 'Date', 'Coral_Cover']].copy()
        
    #     # Format Dates and Cover
    #     stjohn_clean['Date'] = pd.to_datetime(stjohn_clean['Date'].astype(str) + '-06-01')
    #     stjohn_clean['Coral_Cover'] = pd.to_numeric(stjohn_clean['Coral_Cover'], errors='coerce')
    #     if stjohn_clean['Coral_Cover'].max() > 1.0:
    #         stjohn_clean['Coral_Cover'] = stjohn_clean['Coral_Cover'] / 100.0
        
    #     # Hardcode St. John GPS coordinates
    #     stjohn_clean['Latitude'] = 18.315
    #     stjohn_clean['Longitude'] = -64.725
    #     print(f"   > St. John (USVI): Loaded {len(stjohn_clean)} surveys.")
    # except Exception as e: print(f"Error St. John: {e}"); stjohn_clean = pd.DataFrame()

    # Merge All (ISOLATION TEST: GBR ONLY)
    full_df = pd.concat([aims_clean, rc_clean], ignore_index=True)
    # full_df = pd.concat([aims_clean, rc_clean, moorea_clean, stjohn_clean], ignore_index=True)
    
    final_bio = full_df.groupby(['Site_ID', 'Date']).agg({'Coral_Cover': 'mean', 'Latitude': 'first', 'Longitude': 'first'}).reset_index()
    print(f"   > Total Global Observations: {len(final_bio)}")
    return final_bio

def build_tensors_and_graph(bio_df):
    print("\n--- STEP 2: ALIGNING WITH SATELLITE DATA ---")
    
    bio_df['Date'] = pd.to_datetime(bio_df['Date'], errors='coerce')

    env_df = pd.read_csv(PATH_ENV)
    env_df['Date'] = pd.to_datetime(env_df['Date'])
    
    # Duplicate Cabritte Horn temperature data for the other St. John reefs
    # st_john_sites = ['East Tektite', 'Europa Bay', "Neptune's Table", 'West Little Lameshur', 'White Point']
    # cabritte_env = env_df[env_df['Site_ID'] == 'Cabritte Horn']
    
    # if not cabritte_env.empty:
    #     new_env_rows = []
    #     for site in st_john_sites:
    #         temp_df = cabritte_env.copy()
    #         temp_df['Site_ID'] = site
    #         new_env_rows.append(temp_df)
    #     env_df = pd.concat([env_df] + new_env_rows, ignore_index=True)
    
    env_df = env_df.groupby(['Site_ID', 'Date']).mean(numeric_only=True).reset_index()
    
    bio_sites = set(bio_df['Site_ID'].unique())
    env_sites = set(env_df['Site_ID'].unique())
    valid_sites = sorted(list(bio_sites.intersection(env_sites)))

    missing_sites = bio_sites - env_sites
    if missing_sites:
        print(f"   ! WARNING: {len(missing_sites)} biological sites have NO environmental data and will be DROPPED:")
        print(f"   ! Missing Sites: {list(missing_sites)[:10]}...") # Shows first 10
        
        # Count how many actual survey rows are being lost
        dropped_surveys = bio_df[bio_df['Site_ID'].isin(missing_sites)]
        print(f"   ! Total biological surveys lost due to missing env data: {len(dropped_surveys)}")
    else:
        print("   > Success: All biological sites matched with environmental data.")

    # Check for observations outside the 1985-2024 window
    env_start, env_end = env_df['Date'].min(), env_df['Date'].max()
    out_of_bounds = bio_df[(bio_df['Date'] < env_start) | (bio_df['Date'] > env_end)]
    if not out_of_bounds.empty:
        print(f"   ! WARNING: {len(out_of_bounds)} surveys are outside the environmental date range ({env_start.year}-{env_end.year})")

    
    print(f"   > Final Connected Sites: {len(valid_sites)}")
    if len(valid_sites) == 0: raise ValueError("No overlapping sites found!")

    bio_df = bio_df[bio_df['Site_ID'].isin(valid_sites)]
    env_df = env_df[env_df['Site_ID'].isin(valid_sites)]
    dates = sorted(env_df['Date'].unique())
    
    print("Constructing Tensors...")
    # 1. Pivot data (allow pandas to drop empty ones temporarily)
    sst_pivot = env_df.pivot_table(index='Date', columns='Site_ID', values='SST', aggfunc='mean').reindex(index=dates).ffill()
    dhw_pivot = env_df.pivot_table(index='Date', columns='Site_ID', values='DHW', aggfunc='mean').reindex(index=dates).ffill()
    
    # 2. Reindex to guarantee all valid_sites exist. If the satellite returned NaNs (land mask), this forces them back as 0.0
    sst_pivot = sst_pivot.reindex(columns=valid_sites).fillna(0.0)
    dhw_pivot = dhw_pivot.reindex(columns=valid_sites).fillna(0.0)
    
    print("Applying 12-Month Biological Smoothing...")
    # window=12 to disregard annual summer/winter zigzag
    sst_monthly = sst_pivot.resample('MS').mean().rolling(window=12, min_periods=1).mean().fillna(0.0)
    dhw_monthly = dhw_pivot.resample('MS').mean().rolling(window=12, min_periods=1).mean().fillna(0.0)
    monthly_dates = sst_monthly.index
    
    X_sst = torch.tensor(sst_monthly.values.T, dtype=torch.float32).unsqueeze(-1)
    X_dhw = torch.tensor(dhw_monthly.values.T, dtype=torch.float32).unsqueeze(-1)
    X = torch.cat([X_sst, X_dhw], dim=-1) 
    
    Y = torch.zeros((len(valid_sites), len(monthly_dates), 1))
    Mask = torch.zeros((len(valid_sites), len(monthly_dates), 1))
    
    date_to_idx = {d: i for i, d in enumerate(monthly_dates)}
    site_to_idx = {s: i for i, s in enumerate(valid_sites)}
    
    for _, row in bio_df.iterrows():
        # force date into Timestamp object right before the math, just in case it's a string
        bio_date = pd.to_datetime(row['Date'])
        
        closest_date = min(monthly_dates, key=lambda d: abs(d - bio_date))
        
        s_idx = site_to_idx[row['Site_ID']]
        t_idx = date_to_idx[closest_date]
        Y[s_idx, t_idx, 0] = row['Coral_Cover']
        Mask[s_idx, t_idx, 0] = 1.0

    print("Building Global Graph...")
    coords = bio_df.groupby('Site_ID')[['Latitude', 'Longitude']].first().reindex(valid_sites)
    lats = coords['Latitude'].values
    lons = coords['Longitude'].values
    num_sites = len(valid_sites)
    adj = torch.zeros((num_sites, num_sites))
    
    for i in range(num_sites):
        for j in range(num_sites):
            if i == j: 
                adj[i,j] = 1
                continue
            dlat = radians(lats[j] - lats[i])
            dlon = radians(lons[j] - lons[i])
            a = sin(dlat/2)**2 + cos(radians(lats[i])) * cos(radians(lats[j])) * sin(dlon/2)**2
            c = 2 * asin(sqrt(a))
            dist_km = 6371 * c
            if dist_km < GRAPH_THRESHOLD_KM:
                adj[i,j] = 1.0 / (dist_km + 1e-1) 

    return X, Y, Mask, adj, valid_sites

def main():
    bio_df = ingest_biology()
    if bio_df.empty: return
    X, Y, Mask, Adj, sites = build_tensors_and_graph(bio_df)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save(X, f"{OUTPUT_DIR}X.pt")
    torch.save(Y, f"{OUTPUT_DIR}y.pt")
    torch.save(Mask, f"{OUTPUT_DIR}mask.pt")
    torch.save(Adj, f"{OUTPUT_DIR}adjacency_matrix.pt")
    pd.DataFrame({'Site_ID': sites}).to_csv(f"{OUTPUT_DIR}site_list.csv", index=False)
    print("Success. Tensors generated.")

if __name__ == "__main__":
    main()