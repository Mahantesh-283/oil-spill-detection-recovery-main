import pandas as pd
import requests
from utils import latlon_to_dist

# --- 1. Fetch AIS Data from API ---
def fetch_ais_data():
    url = "https://data.aishub.net/ws.php"
    
    params = {
        "username": "YOUR_USERNAME",   # <-- Replace this
        "format": 1,                  # JSON
        "output": "json",
        "compress": 0,
        "latmin": 20,
        "latmax": 30,
        "lonmin": -100,
        "lonmax": -80
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print("API Error:", e)
        return None

    if not data:
        print("No AIS data received.")
        return None

    df = pd.DataFrame(data)

    # --- Normalize column names ---
    df.rename(columns={
        'lat': 'LAT',
        'lon': 'LON',
        'speed': 'SOG',
        'vessel_type': 'VesselType',
        'name': 'VesselName'
    }, inplace=True)

    # --- Handle missing data ---
    df = df.dropna(subset=['LAT', 'LON'])
    df['SOG'] = df['SOG'].fillna(12.0)

    # If VesselType missing → assign default
    if 'VesselType' not in df.columns:
        df['VesselType'] = 0

    if 'VesselName' not in df.columns:
        df['VesselName'] = "Unknown"

    return df


# --- 2. Vessel Tasking System ---
def task_recovery_system(spill_lat, spill_lon):
    df = fetch_ais_data()

    if df is None or len(df) == 0:
        print("No valid AIS data available.")
        return None

    # --- Filter responders ---
    responders = df[df['VesselType'].isin([52, 80, 82, 31, 32, 60])]

    # Fallback if filtering fails
    if len(responders) == 0:
        print("No filtered responders found, using all vessels.")
        responders = df.copy()

    # --- Distance Calculation ---
    responders['dist_km'] = responders.apply(
        lambda x: latlon_to_dist(spill_lat, spill_lon, x['LAT'], x['LON']), axis=1
    )

    # --- ETA Calculation ---
    responders['effective_speed'] = responders['SOG'].apply(lambda x: x if x > 0 else 12.0)
    responders['eta_h'] = responders['dist_km'] / (responders['effective_speed'] * 1.852)

    # --- Select Best Vessel ---
    best_ship = responders.sort_values('eta_h').iloc[0]

    print("\n--- EMERGENCY RESPONSE ASSIGNED ---")
    print(f"Vessel Name       : {best_ship['VesselName']}")
    print(f"Vessel Type       : {best_ship['VesselType']}")
    print(f"Distance to Spill : {best_ship['dist_km']:.2f} km")
    print(f"ETA               : {best_ship['eta_h']:.2f} hours")

    return best_ship


# --- 3. Run ---
if __name__ == "__main__":
    print("Starting Vessel Tasking System (AIS API)...")

    # Example spill location
    mission = task_recovery_system(27.54, -92.44)

    if mission is not None:
        print("Tasking Complete.")
    else:
        print("Tasking Failed.")