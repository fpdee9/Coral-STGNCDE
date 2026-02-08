import pandas as pd
import numpy as np
import os
from scipy.spatial import cKDTree

# ==========================================
# CONFIGURATION (MATCHING YOUR GIT STRUCTURE)
# ==========================================
# Path to your "Master" Environmental File (Source of Truth for Site IDs)
# (Note: This file is ignored by git)
ENV_FILE = "data/raw/seawater temperature/Environmental_History_1985-2024.csv"

# Paths to Biological Data Source Files
# Updated paths based on your commit log:
FILE_MANTA = "data/raw/coral cover/manta-tow-by-reef/manta-tow-by-reef.csv"
FILE_DRYAD = "data/raw/coral cover/doi_10_5061_dryad_ngf1vhj44__v20250107/coral_cover.csv"
FILE_REEFCLOUD = "data/raw/coral cover/EcoRRAP_Benthic_Data_2021to2024/reefcloudcover42_functionalgroup42.csv"

# Initial Sets
FILE_STJOHN = "data/raw/coral cover/Initial Set/st_john_avg_coral_cover.csv"
FILE_MOOREA = "data/raw/coral cover/Initial Set/moorea_avg_coral_cover.csv"

OUTPUT_FILE = "data/processed/Biological_Observations_1985-2024.csv"

def load_master_sites():
    # Extracts the 106 target sites and their lat/lon from the big environmental file.
    print(f"Loading Site List from {ENV_FILE}...")
    # Read only first chunk to get sites (assuming file is sorted/grouped)
    # or read unique IDs if file is huge.
    # Using a smarter strategy: Read specific columns
    # Based on your header: Site_ID,Latitude,Longitude,Date,SST,DHW
    try:
        df = pd.read_csv(ENV_FILE, usecols=['Site_ID', 'Latitude', 'Longitude'])
        sites = df.drop_duplicates(subset=['Site_ID']).copy()
        print(f"   > Found {len(sites)} unique target sites.")
        return sites
    except Exception as e:
        print(f"CRITICAL ERROR: Could not find {ENV_FILE}")
        print("Please check if the file is in 'data/raw/' or 'data/raw/seawater temperature/'")
        exit()

def process_manta_tow(target_ids):
    # Process AIMS Manta Tow Data.
    print("Processing AIMS Manta Tow...")
    try:
        df = pd.read_csv(FILE_MANTA)
        # Filter: Only keep rows where REEF_ID is in our target list
        df = df[df['REEF_ID'].isin(target_ids)]
        
        # Select & Rename
        # Manta Tow has 'MEAN_LIVE_CORAL' (0-100)
        df = df[['REEF_ID', 'REPORT_YEAR', 'MEAN_LIVE_CORAL']]
        df.columns = ['Site_ID', 'Year', 'Coral_Cover']
        
        # Convert % (0-100) to Float (0.0-1.0)
        df['Coral_Cover'] = df['Coral_Cover'] / 100.0
        
        # Handle formatting; Ensure Year is numeric
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        
        print(f"   > Extracted {len(df)} observations.")
        return df
    except Exception as e:
        print(f"   ! Warning: Could not process Manta Tow: {e}")
        return pd.DataFrame()

def process_dryad(target_ids, manta_mapping_df):
    # Process Dryad Data. Needs a Name->ID map because it uses Names.
    print("Processing Dryad Data...")
    try:
        df = pd.read_csv(FILE_DRYAD)
        
        # Create a Name->ID map from the Manta file (which has both)
        # This is a trick to link 'GANNETT CAY REEF' to '21556S'
        if not manta_mapping_df.empty:
            # Load Manta file again just to get names if needed, or rely on passed df if it had names.
            # Safer to load fresh to get all names, not just filtered ones.
            manta_full = pd.read_csv(FILE_MANTA)
            name_map = pd.read_csv(FILE_MANTA)[['REEF_NAME', 'REEF_ID']].drop_duplicates().set_index('REEF_NAME')['REEF_ID'].to_dict()
            df['Site_ID'] = df['REEF_NAME'].map(name_map)
        else:
             print("   ! Manta mapping missing, skipping Name->ID map.")

        # Filter for study sites (drop rows where mapping failed)
        df = df.dropna(subset=['Site_ID'])
        df = df[df['Site_ID'].isin(target_ids)]
        
        # YEAR_CODE (201415) -> Year (2015)
        df['Year'] = df['YEAR_CODE'].astype(str).str[:4].astype(int) + 1
        
        # Extract Hard Coral
        df = df[['Site_ID', 'Year', 'HARD_COVER']]
        df.columns = ['Site_ID', 'Year', 'Coral_Cover']
        df['Coral_Cover'] = df['Coral_Cover'] / 100.0
        
        print(f"   > Extracted {len(df)} observations.")
        return df
    except Exception as e:
        print(f"   ! Warning: Could not process Dryad: {e}")
        return pd.DataFrame()

def process_reefcloud(sites_df):
    # Process ReefCloud Data. Matches based on GPS distance.
    print("Processing ReefCloud (EcoRRAP)...")
    try:
        df = pd.read_csv(FILE_REEFCLOUD)
        
        # Create KDTree for nearest neighbor search from our Master Sites
        target_coords = sites_df[['Latitude', 'Longitude']].values
        tree = cKDTree(target_coords)
        
        # Query ReefCloud sites against Master Sites
        cloud_coords = df[['site_latitude', 'site_longitude']].values
        distances, indices = tree.query(cloud_coords)
        
        # Filter matches within 0.02 degrees (~2km)
        THRESHOLD = 0.02
        valid_matches = distances < THRESHOLD
        
        # Assign Site_ID to the ReefCloud rows that matched
        df['Site_ID'] = np.nan
        df['Site_ID'] = df['Site_ID'].astype(object)
        df.loc[valid_matches, 'Site_ID'] = sites_df.iloc[indices[valid_matches]]['Site_ID'].values
        
        # Drop non-matches
        df = df.dropna(subset=['Site_ID'])
        
        # Clean columns (HC = Hard Coral)
        df = df[['Site_ID', 'year', 'HC']]
        df.columns = ['Site_ID', 'Year', 'Coral_Cover']
        df['Coral_Cover'] = df['Coral_Cover'] / 100.0
        
        print(f"   > Extracted {len(df)} observations (matched via GPS).")
        return df
    except Exception as e:
        print(f"   ! Warning: Could not process ReefCloud: {e}")
        return pd.DataFrame()

def process_legacy():
    # Process St John and Moorea CSVs.
    print("Processing St John & Moorea...")
    buffer = []
    
    # St John
    try:
        sj = pd.read_csv(FILE_STJOHN)
        # Clean column names
        # Rename 'Site' to 'Site_ID' to match master list
        sj = sj.rename(columns={'Site': 'Site_ID', 'Stony_coral_cover': 'Coral_Cover'})
        sj['Coral_Cover'] = sj['Coral_Cover'] / 100.0
        buffer.append(sj[['Site_ID', 'Year', 'Coral_Cover']])
    except Exception as e: print(f"   ! St John Error: {e}")

    # Moorea
    try:
        mo = pd.read_csv(FILE_MOOREA)
        # Map LTER 1 -> LTER1 (if needed) or keep as is.
        # Assuming Environmental File has "LTER 1"
        mo = mo.rename(columns={'Site': 'Site_ID', 'Stony_coral_cover': 'Coral_Cover'})
        mo['Coral_Cover'] = mo['Coral_Cover'] / 100.0
        buffer.append(mo[['Site_ID', 'Year', 'Coral_Cover']])
    except Exception as e: print(f"   ! Moorea Error: {e}")
    
    if buffer:
        df = pd.concat(buffer)
        print(f"   > Extracted {len(df)} observations.")
        return df
    else:
        return pd.DataFrame()

def main():
    # 1. Get Master List
    sites_df = load_master_sites()
    target_ids = sites_df['Site_ID'].unique()
    
    # 2. Process All Sources
    # Pass 'manta' df to 'dryad' processing to help with ID mapping
    df_manta = process_manta_tow(target_ids)
    df_dryad = process_dryad(target_ids, df_manta) # Pass manta for ID mapping if needed
    df_cloud = process_reefcloud(sites_df)
    df_legacy = process_legacy()
    
    # 3. Merge
    all_data = pd.concat([df_manta, df_dryad, df_cloud, df_legacy], ignore_index=True)
    
    # 4. Clean Aggregation
    # If multiple surveys exist for 1 Site in 1 Year, Average them
    final_df = all_data.groupby(['Site_ID', 'Year'])['Coral_Cover'].mean().reset_index()
    
    # 5. Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    final_df.to_csv(OUTPUT_FILE, index=False)
    print("="*40)
    print(f"SUCCESS! Merged Biology Data saved to:\n{OUTPUT_FILE}")
    print(f"Total Unique Observations: {len(final_df)}")
    print("="*40)

if __name__ == "__main__":
    main()