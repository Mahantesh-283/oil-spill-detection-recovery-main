import torch
import numpy as np
import cv2
import os
import pandas as pd
from model import DualBranchDetector

# 1. Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "oil_spill_model_final.pth"
SENTINEL_DIR = r"E:/oil spill1/organized_train/sentinel/sat"
PALSAR_DIR = r"E:/oil spill1/organized_train/palsar/sat"

# Initialize Model
model = DualBranchDetector().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.eval()

# 2. Thickness Classification (µm)
THICKNESS_CLASSES = {
    "SHEEN": 0.1,        
    "RAINBOW": 5.0,      
    "THICK_SLICK": 50.0, 
    "EMULSION": 200.0    
}

def analyze_sensor_folder(folder_path, sensor_name):
    print(f"Analyzing {sensor_name} data...")
    all_images = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
    
    total_vol = 0
    count = 0

    for img_name in all_images:
        img_path = os.path.join(folder_path, img_name)
        raw_img = cv2.imread(img_path)
        if raw_img is None: continue

        img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        input_t = torch.from_numpy(cv2.resize(img_rgb, (256, 256))).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        
        with torch.no_grad():
            pred = model(input_t.to(DEVICE)).cpu().squeeze().numpy()
        
        mask = (pred > 0.5).astype(np.uint8)
        gray = cv2.resize(cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY), (256, 256))
        thickness_map = np.zeros_like(pred)
        
        # Thickness Mapping Logic
        thickness_map[(mask == 1) & (gray < 50)] = THICKNESS_CLASSES["SHEEN"]
        thickness_map[(mask == 1) & (gray >= 50) & (gray < 150)] = THICKNESS_CLASSES["RAINBOW"]
        thickness_map[(mask == 1) & (gray >= 150) & (gray < 220)] = THICKNESS_CLASSES["THICK_SLICK"]
        thickness_map[(mask == 1) & (gray >= 220)] = THICKNESS_CLASSES["EMULSION"]
        
        pixel_area_m2 = 100 
        total_vol += np.sum(thickness_map * 1e-6 * pixel_area_m2)
        count += 1

    return total_vol, count

def run_vessel_tasking(total_volume, total_images):
    vessels = pd.DataFrame({
        'MMSI': [413000123, 235010456, 311000789],
        'Name': ['Skimmer_Alpha', 'Response_Tug_One', 'Patrol_Vessel_X'],
        'Lat': [13.02, 12.92, 12.85],
        'Lon': [80.30, 80.20, 80.15],
        'Speed_knots': [14.5, 12.0, 24.0]
    })
    
    vessels['Dist_km'] = np.sqrt((vessels['Lat'] - 12.97)**2 + (vessels['Lon'] - 80.24)**2) * 111
    vessels['ETA_hrs'] = vessels['Dist_km'] / (vessels['Speed_knots'] * 1.852)
    best_ship = vessels.sort_values('ETA_hrs').iloc[0]
    
    print("\n" + "="*60)
    print("      MULTISENSOR OIL SPILL & RECOVERY REPORT")
    print("="*60)
    print(f"Total Images Processed   : {total_images}")
    print(f"Final Combined Volume    : {total_volume:.4f} m³")
    print(f"Recovery Priority        : {'CRITICAL' if total_volume > 10 else 'HIGH'}")
    print("-" * 60)
    print(f"PRIMARY RESPONDER        : {best_ship['Name']}")
    print(f"ETA TO SCENE             : {best_ship['ETA_hrs']:.2f} Hours")
    print("="*60)

if __name__ == "__main__":
    # 1. Individual Sensor Analysis
    vol_sentinel, count_s = analyze_sensor_folder(SENTINEL_DIR, "SENTINEL")
    vol_palsar, count_p = analyze_sensor_folder(PALSAR_DIR, "PALSAR")
    
    # 2. Metrics for Ablation Study
    final_vol = vol_sentinel + vol_palsar
    final_count = count_s + count_p
    
    # Calculate percentages for comparison
    p_sentinel = (vol_sentinel / final_vol * 100) if final_vol > 0 else 0
    p_palsar = (vol_palsar / final_vol * 100) if final_vol > 0 else 0

    print("\n" + "-"*60)
    print("              SENSOR VOLUME BREAKDOWN")
    print("-"*60)
    print(f"Sentinel-2 (Optical) : {vol_sentinel:12.2f} m³ ({p_sentinel:.1f}%)")
    print(f"PALSAR (SAR)         : {vol_palsar:12.2f} m³ ({p_palsar:.1f}%)")
    print("-" * 60)

    # 3. Final Tasking
    run_vessel_tasking(final_vol, final_count)