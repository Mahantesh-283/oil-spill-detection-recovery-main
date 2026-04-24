import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from model import DualBranchDetector

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "oil_spill_model_final.pth"

def visualize_specific_heatmap():
    # 1. Initialize and Load Model
    model = DualBranchDetector().to(DEVICE)
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found.")
        return
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 2. Hardcoded specific paths for Image 20839
    # Ensure these file names match your E: drive exactly
    img_path = r"E:\oil spill detection and recovery\test-20251120T124100Z-1-001\test\sentinel\sat\20001_sat.jpg"
    gt_path = r"E:\oil spill detection and recovery\test-20251120T124100Z-1-001\test\sentinel\gt\20001_mask.png"

    # 3. Read and Validate Files
    raw_img = cv2.imread(img_path)
    gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

    if raw_img is None or gt_mask is None:
        print(f"Error reading file. Check if {img_path} exists.")
        return

    # 4. Preprocessing
    img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    img_input = cv2.resize(img_rgb, (256, 256)).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_input).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    # 5. Inference - Capture Raw Probabilities
    with torch.no_grad():
        logits = model(img_tensor)
        # Sigmoid converts logits to a 0.0 to 1.0 probability range
        probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()

    # 6. Heatmap Presentation
    plt.figure(figsize=(15, 5))
    
    # Column 1: Original SAR Image
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title("SAR Input (20839)")
    plt.axis('off')

    # Column 2: Ground Truth
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.resize(gt_mask, (256, 256)), cmap='gray')
    plt.title("Ground Truth (Label)")
    plt.axis('off')

    # Column 3: AI Probability Heatmap
    # 'magma' or 'jet' helps visualize low-confidence detections
    plt.subplot(1, 3, 3)
    im = plt.imshow(probabilities, cmap='magma') 
    plt.title("AI Confidence Heatmap")
    plt.axis('off')
    
    # Add colorbar to show confidence scale (0.0 to 1.0)
    plt.colorbar(im, ax=plt.gca(), fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig("heatmap_20839.png")
    print("Success! Heatmap saved as 'heatmap_20839.png'")
    plt.show()

if __name__ == "__main__":
    visualize_specific_heatmap()