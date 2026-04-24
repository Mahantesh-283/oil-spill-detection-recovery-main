import pandas as pd
from utils import latlon_to_dist

def task_recovery_system(spill_lat, spill_lon):
    # Load the MarineCadastre AIS dataset
    try:
        df = pd.read_csv("vessels_data.csv")
    except FileNotFoundError:
        print("Error: vessels_data.csv not found. Please ensure your AIS dataset is in the folder.")
        return None
    
    # Filter for responders: 52 (Search/Rescue), 80-89 (Tankers/Response), 31 (Tugs)
    responders = df[df['VesselType'].isin([52, 80, 82, 31, 32, 60])].copy()
    
    # Calculate Distance using the fixed utils function
    responders['dist_km'] = responders.apply(
        lambda x: latlon_to_dist(spill_lat, spill_lon, x['LAT'], x['LON']), axis=1
    )
    
    # Calculate ETA: Distance / (Speed in Knots * 1.852 for km/h)
    # Default speed of 12 knots if ship is currently stationary
    responders['effective_speed'] = responders['SOG'].apply(lambda x: x if x > 0 else 12.0)
    responders['eta_h'] = responders['dist_km'] / (responders['effective_speed'] * 1.852)
    
    # Select the vessel with the lowest ETA
    best_ship = responders.sort_values('eta_h').iloc[0]
    
    print(f"\n--- EMERGENCY RESPONSE ASSIGNED ---")
    print(f"Vessel Name: {best_ship['VesselName']}")
    print(f"Vessel Type: {best_ship['VesselType']}")
    print(f"Distance to Spill: {best_ship['dist_km']:.2f} km")
    print(f"Estimated Time of Arrival: {best_ship['eta_h']:.2f} hours")
    
    return best_ship
if __name__ == "__main__":
    # Use coordinates from your Persian Gulf or Gulf of Mexico test area
    # Example: Lat 27.5, Lon -92.4
    print("Starting Vessel Tasking System...")
    mission = task_recovery_system(27.54, -92.44)
    
    if mission is not None:
        print("Tasking Complete.")
    else:
        print("Tasking Failed: Check if vessels_data.csv exists.")
# Example: Coordinate from a Persian Gulf detection
# task_recovery_system(27.54, -92.44)