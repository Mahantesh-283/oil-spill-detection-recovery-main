import torch
import cv2
import numpy as np
import os
from model import DualBranchDetector

# 1. Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "oil_spill_model_final.pth"
OUTPUT_DIR = r"E:/oil spill1/test_visual_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

<<<<<<< HEAD


# 2. Paths for Testing Data
test_configs = {
    "SENTINEL": {
        "sat": r"E:\oil-spill-detection-recovery-main\test-20251120T124100Z-1-001\test\palsar\sat",
        "gt": r"E:\oil-spill-detection-recovery-main\test-20251120T124100Z-1-001\test\palsar\gt"
    },
    "PALSAR": {
        "sat": r"E:\oil-spill-detection-recovery-main\test-20251120T124100Z-1-001\test\sentinel\sat",
        "gt": r"E:\oil-spill-detection-recovery-main\test-20251120T124100Z-1-001\test\sentinel\gt"
=======
# 2. Paths for Testing Data
test_configs = {
    "SENTINEL": {
        "sat": r"E:/oil spill1/test-20251120T124100Z-1-001/test/sentinel/sat",
        "gt": r"E:/oil spill1/test-20251120T124100Z-1-001/test/sentinel/gt"
    },
    "PALSAR": {
        "sat": r"E:/oil spill1/test-20251120T124100Z-1-001/test/palsar/sat",
        "gt": r"E:/oil spill1/test-20251120T124100Z-1-001/test/palsar/gt"
>>>>>>> 410cc706c221a986098d3d1ecb1abe3ba625cc7d
    }
}

# 3. Load Trained Model
model = DualBranchDetector().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.eval()

# 4. Helper Functions
def calculate_metrics(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = intersection / union if union > 0 else 1.0
    return iou

def estimate_volume(mask, raw_img):
    """Calculates volume based on predicted mask and pixel intensity"""
    gray = cv2.resize(cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY), (256, 256))
    THICKNESS_CLASSES = {"SHEEN": 0.1, "RAINBOW": 5.0, "SLICK": 50.0, "EMULSION": 200.0}
    
    thickness_map = np.zeros((256, 256))
    thickness_map[(mask == 1) & (gray < 50)] = THICKNESS_CLASSES["SHEEN"]
    thickness_map[(mask == 1) & (gray >= 50) & (gray < 150)] = THICKNESS_CLASSES["RAINBOW"]
    thickness_map[(mask == 1) & (gray >= 150) & (gray < 220)] = THICKNESS_CLASSES["SLICK"]
    thickness_map[(mask == 1) & (gray >= 220)] = THICKNESS_CLASSES["EMULSION"]
    
    pixel_area_m2 = 100 # 10m x 10m resolution
    volume_m3 = np.sum(thickness_map * 1e-6 * pixel_area_m2)
    return volume_m3

# 5. Run Evaluation
final_report = []
<<<<<<< HEAD
def post_process(mask_np):
    # 1. Morphological Closing (Fill holes)
    kernel = np.ones((3,3), np.uint8)
    refined = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)
    
    # 2. Area Filtering (Remove small noise)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(refined)
    clean_mask = np.zeros_like(refined)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 50: # Minimum 50 pixels
            clean_mask[labels == i] = 1
            
    return clean_mask
=======

>>>>>>> 410cc706c221a986098d3d1ecb1abe3ba625cc7d
for sensor, paths in test_configs.items():
    print(f"\nProcessing {sensor} Test Set...")
    sat_files = [f for f in os.listdir(paths['sat']) if f.endswith('.jpg')]
    sensor_ious = []
    sensor_volume = 0

    for filename in sat_files:
        img_path = os.path.join(paths['sat'], filename)
        gt_filename = filename.replace('_sat.jpg', '_mask.png')
        gt_path = os.path.join(paths['gt'], gt_filename)

        if not os.path.exists(gt_path): continue

        # Model Inference
        raw_img = cv2.imread(img_path)
        img_input = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        img_input = cv2.resize(img_input, (256, 256)).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_input).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred_tensor = model(img_tensor)
            pred_mask = (pred_tensor.cpu().squeeze().numpy() > 0.5).astype(np.uint8)

        # Accuracy Metric (IoU)
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = (cv2.resize(gt_mask, (256, 256)) > 127).astype(np.uint8)
        iou = calculate_metrics(pred_mask, gt_mask)
        sensor_ious.append(iou)

        # Operational Metric (Volume)
        vol = estimate_volume(pred_mask, raw_img)
        sensor_volume += vol

    final_report.append({
        "Sensor": sensor, 
        "mIoU": np.mean(sensor_ious), 
        "Total_Vol": sensor_volume,
        "Img_Count": len(sensor_ious)
    })

# 6. Final Summary Display
print("\n" + "="*45)
print("       FINAL TESTING & VOLUME REPORT")
print("="*45)
for res in final_report:
    print(f"SENSOR: {res['Sensor']}")
    print(f"Images Tested   : {res['Img_Count']}")
    print(f"Mean IoU Accuracy: {res['mIoU']:.4f}")
    print(f"Estimated Volume: {res['Total_Vol']:.2f} m³")
    print("-" * 45)

total_test_vol = sum([r['Total_Vol'] for r in final_report])
print(f"GRAND TOTAL TEST VOLUME: {total_test_vol:.2f} m³")
print("="*45)