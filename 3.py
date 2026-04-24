import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model import DualBranchDetector

# --- 1. Setup ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "oil_spill_model_final.pth"
IMAGE_PATH = r"E:\oil-spill-detection-recovery-main\1542735832746.png"

# --- 2. Load Model ---
model = DualBranchDetector().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# --- 3. Preprocess & Predict ---
raw_img = cv2.imread(IMAGE_PATH)
if raw_img is None:
    print("Error: Could not find the image!")
else:
    # Prepare for model (256x256)
    img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (256, 256))
    img_input = img_resized.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_input).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img_tensor)
        # Convert prediction to binary mask
        pred_mask = (output.cpu().squeeze().numpy() > 0.5).astype(np.uint8)

    # --- 4. Post-Process (The Booster) ---
    kernel = np.ones((3,3), np.uint8)
    refined_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel)
    
    # --- 5. Volume Estimation ---
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    thickness_map = np.zeros((256, 256))
    thickness_map[(refined_mask == 1) & (gray < 50)] = 0.1
    thickness_map[(refined_mask == 1) & (gray >= 50) & (gray < 150)] = 5.0
    thickness_map[(refined_mask == 1) & (gray >= 150)] = 50.0
    
    volume = np.sum(thickness_map * 1e-6 * 100) # m3

    # --- 6. Visualization ---
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.title("Original SAR Image")
    plt.imshow(img_resized)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Detected Oil Spill")
    plt.imshow(refined_mask, cmap='hot')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    # Overlay for better visualization
    overlay = img_resized.copy()
    overlay[refined_mask == 1] = [255, 0, 0] # Red color for spill
    plt.title(f"Overlay Result\nVol: {volume:.2f} m³")
    plt.imshow(overlay)
    plt.axis('off')

    plt.show()
    print(f"Analysis Complete. Estimated Volume: {volume:.4f} m³")