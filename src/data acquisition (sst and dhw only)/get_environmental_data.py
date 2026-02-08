import pandas as pd
import requests
import io
import time
import os
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FILE = "C:\\Users\\ipedee\\Downloads\\Prototype\\LongLat.csv"
OUTPUT_FILE = "Environmental_History_1985-2024.csv"

# NOAA Endpoints
URL_SST = "https://coastwatch.noaa.gov/erddap/griddap/noaacrwsstDaily.csv"
URL_DHW = "https://coastwatch.noaa.gov/erddap/griddap/noaacrwdhwDaily.csv"

# Headers
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# ==========================================
# SETUP SESSION
# ==========================================
session = requests.Session()
retries = Retry(total=10, backoff_factor=2, status_forcelist=[500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))

def main():
    print("--- STARTING HIGH-VISIBILITY SCRIPT ---")
    print(f"Reading input file: {INPUT_FILE}")
    try:
        sites = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"CRITICAL ERROR: {INPUT_FILE} not found.")
        return

    unique_sites = sites.drop_duplicates(subset=['LATITUDE', 'LONGITUDE'])
    print(f"Found {len(unique_sites)} unique sites to process.")
    
    # --- AUTO-RESUME LOGIC ---
    finished_sites = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            # We force str to avoid type mismatch
            existing_data = pd.read_csv(OUTPUT_FILE, usecols=['Site_ID'], dtype={'Site_ID': str})
            finished_sites = set(existing_data['Site_ID'].str.strip().unique())
            print(f"RESUMING: Found {len(finished_sites)} sites already completed.")
        except Exception as e:
            print(f"WARNING: Output file exists but unreadable ({e}). Starting fresh.")
            pd.DataFrame(columns=['Site_ID', 'Latitude', 'Longitude', 'Date', 'SST', 'DHW']).to_csv(OUTPUT_FILE, index=False)
    else:
        print("Creating new output file...")
        pd.DataFrame(columns=['Site_ID', 'Latitude', 'Longitude', 'Date', 'SST', 'DHW']).to_csv(OUTPUT_FILE, index=False)

    total_sites = len(unique_sites)
    count = 0

    for index, row in unique_sites.iterrows():
        count += 1
        site_id = str(row['Site (Name or ID)']).strip()
        lat = row['LATITUDE']
        lon = row['LONGITUDE']

        if site_id in finished_sites:
            print(f"[{count}/{total_sites}] Skipping {site_id} (Already Done)")
            continue

        print(f"\n[{count}/{total_sites}] STARTING SITE: {site_id} ({lat}, {lon})")

        start_date = datetime(1985, 1, 1)
        end_date = datetime(2024, 12, 31)
        current_date = start_date

        while current_date <= end_date:
            month_start = current_date
            next_month = (current_date.replace(day=28) + timedelta(days=4)).replace(day=1)
            month_end = next_month - timedelta(days=1)
            if month_end > end_date: month_end = end_date

            s_str = month_start.strftime("%Y-%m-%dT12:00:00Z")
            e_str = month_end.strftime("%Y-%m-%dT12:00:00Z")
            month_display = month_start.strftime('%Y-%m')

            # --- STUBBORN RETRY LOOP ---
            success = False
            attempts = 0
            
            # Print start of request without newline so we can append status
            print(f"   > {month_display}: Requesting...", end='', flush=True)
            
            while not success:
                try:
                    # 1. Fetch SST
                    query_sst = f"?analysed_sst[({s_str}):1:({e_str})][({lat}):1:({lat})][({lon}):1:({lon})]"
                    r_sst = session.get(URL_SST + query_sst, headers=HEADERS, timeout=90)
                    
                    if r_sst.status_code == 200:
                        df_chunk = pd.read_csv(io.StringIO(r_sst.text), skiprows=[1])
                        df_chunk.rename(columns={'analysed_sst': 'SST', 'time': 'Date'}, inplace=True)
                        
                        # 2. Fetch DHW
                        query_dhw = f"?degree_heating_week[({s_str}):1:({e_str})][({lat}):1:({lat})][({lon}):1:({lon})]"
                        r_dhw = session.get(URL_DHW + query_dhw, headers=HEADERS, timeout=90)
                        
                        if r_dhw.status_code == 200:
                            df_dhw = pd.read_csv(io.StringIO(r_dhw.text), skiprows=[1])
                            if len(df_dhw) == len(df_chunk):
                                df_chunk['DHW'] = df_dhw['degree_heating_week']
                            else:
                                df_dhw.rename(columns={'time': 'Date', 'degree_heating_week': 'DHW'}, inplace=True)
                                df_chunk = pd.merge(df_chunk, df_dhw[['Date', 'DHW']], on='Date', how='left')
                        else:
                            df_chunk['DHW'] = float('nan')
                        
                        # === IMMEDIATE SAVE ===
                        df_chunk['Site_ID'] = site_id
                        df_chunk['Latitude'] = lat
                        df_chunk['Longitude'] = lon
                        df_chunk['Date'] = df_chunk['Date'].str.slice(0, 10)
                        
                        # Clean columns
                        df_chunk = df_chunk[['Site_ID', 'Latitude', 'Longitude', 'Date', 'SST', 'DHW']]
                        
                        # Write to disk NOW
                        df_chunk.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
                        
                        print(" OK. Saved.", flush=True) # Feedback
                        success = True 
                        
                    elif r_sst.status_code in [500, 502, 503, 504]:
                        attempts += 1
                        print(f"\n     [!] Server Busy ({r_sst.status_code}). Retrying in {2*attempts}s...", end='', flush=True)
                        time.sleep(2 * attempts)
                    else:
                        print(f"\n     [X] Permanent Error {r_sst.status_code}. Skipping.", flush=True)
                        success = True 

                except Exception as e:
                    attempts += 1
                    print(f"\n     [!] Connection Error. Retrying ({attempts})...", end='', flush=True)
                    time.sleep(5)

            current_date = next_month

    print("\nDONE! All sites processed.")

if __name__ == "__main__":
    main()