import torch
import numpy as np
import cv2
import os
from model import DualBranchDetector # Your CNN-ViT architecture

# 1. Device and Model Initialization
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DualBranchDetector().to(DEVICE)
model.load_state_dict(torch.load("oil_spill_model_final.pth"))
model.eval()

# 2. Thickness Classification (micrometers - µm)
THICKNESS_CLASSES = {
    "SHEEN": 0.1,        # Minimal impact
    "RAINBOW": 5.0,      # Moderate
    "THICK_SLICK": 50.0, # High priority
    "EMULSION": 200.0    # Very high priority
}

def calculate_volume(image_path):
    # Load and preprocess
    raw_img = cv2.imread(image_path)
    if raw_img is None: return None, 0
    
    img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    input_t = torch.from_numpy(cv2.resize(img_rgb, (256, 256))).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    
    with torch.no_grad():
        pred = model(input_t.to(DEVICE)).cpu().squeeze().numpy()
    
    mask = (pred > 0.5).astype(np.uint8)
    
    # Class-Based Thickness Assignment Logic
    # Simple spectral intensity proxy (Higher intensity ~ Thicker oil/Emulsion)
    gray = cv2.resize(cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY), (256, 256))
    thickness_map = np.zeros_like(pred)
    
    thickness_map[(mask == 1) & (gray < 50)] = THICKNESS_CLASSES["SHEEN"]
    thickness_map[(mask == 1) & (gray >= 50) & (gray < 150)] = THICKNESS_CLASSES["RAINBOW"]
    thickness_map[(mask == 1) & (gray >= 150) & (gray < 220)] = THICKNESS_CLASSES["THICK_SLICK"]
    thickness_map[(mask == 1) & (gray >= 220)] = THICKNESS_CLASSES["EMULSION"]
    
    # Volume calculation: Area * Thickness
    # Assuming Sentinel-2 resolution (10m x 10m per pixel)
    pixel_area_m2 = 100 
    total_volume_m3 = np.sum(thickness_map * 1e-6 * pixel_area_m2) # µm to m
    
    return thickness_map, total_volume_m3

if __name__ == "__main__":
    # Process first available satellite image
    test_path = r"E:/oil spill1/organized_train/sentinel/sat"
    sample = os.path.join(test_path, os.listdir(test_path)[0])
    _, vol = calculate_volume(sample)
    print(f"Total Spill Volume Estimated: {vol:.4f} m³")