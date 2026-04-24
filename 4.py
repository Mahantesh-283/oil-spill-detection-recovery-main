import sys
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model import DualBranchDetector

# --- 0. Fix Encoding (NO MORE EMOJI CRASH) ---
sys.stdout.reconfigure(encoding='utf-8')

# --- 1. Setup ---
DEVICE = torch.device("cpu")
MODEL_PATH = "oil_spill_model_final.pth"

IMAGE_PATH = r"E:\oil-spill-detection-recovery-main\test-20251120T124100Z-1-001\test\sentinel\sat\20001_sat.jpg"
MASK_PATH  = r"E:\oil-spill-detection-recovery-main\test-20251120T124100Z-1-001\test\sentinel\gt\20001_mask.png"

THRESHOLD = 0.5   # 🔥 tune this later

# Sentinel-2 assumption → 10m resolution
PIXEL_AREA = 10 * 10  # m²


# --- 2. Load Model SAFELY ---
print("Loading model...")

model = DualBranchDetector().to(DEVICE)

state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True  # 🔥 safer loading
)

model.load_state_dict(state_dict, strict=True)
model.eval()

print("Model loaded successfully")


# --- 3. Metrics ---
def calculate_all_metrics(pred, gt):
    pred_f = pred.flatten()
    gt_f = gt.flatten()

    tp = np.logical_and(pred_f == 1, gt_f == 1).sum()
    fp = np.logical_and(pred_f == 1, gt_f == 0).sum()
    fn = np.logical_and(pred_f == 0, gt_f == 1).sum()

    eps = 1e-7
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)

    return iou, f1, recall, precision


# --- 4. Load Data ---
print("Loading images...")

raw_img = cv2.imread(IMAGE_PATH)
gt_img = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)

if raw_img is None:
    raise ValueError("Input image not found")

if gt_img is None:
    raise ValueError("Ground truth mask not found")

print("Raw:", raw_img.shape, "GT:", gt_img.shape)


# --- 5. Preprocess ---
img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (256, 256))

img_input = img_resized.astype(np.float32) / 255.0

# 🔥 OPTIONAL: match training normalization (UNCOMMENT if used during training)
# mean = np.array([0.485, 0.456, 0.406])
# std = np.array([0.229, 0.224, 0.225])
# img_input = (img_input - mean) / std

img_tensor = torch.from_numpy(img_input).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

gt_resized = (cv2.resize(gt_img, (256, 256)) > 127).astype(np.uint8)


# --- 6. Inference ---
print("Running inference...")

with torch.no_grad():
    output = model(img_tensor)

if len(output.shape) != 4:
    raise ValueError(f"Unexpected output shape: {output.shape}")

raw_mask = output.detach().cpu().numpy()

if raw_mask.shape[1] == 1:
    raw_mask = raw_mask[:, 0, :, :]

raw_mask = raw_mask.squeeze()

if raw_mask.shape != (256, 256):
    raise ValueError(f"Mask shape mismatch: {raw_mask.shape}")

# 🔥 threshold is now configurable
raw_mask = (raw_mask > THRESHOLD).astype(np.uint8)


# --- 7. Post-processing ---
kernel = np.ones((3, 3), np.uint8)
refined_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)


# --- 8. Metrics ---
miou, f1, recall, prec = calculate_all_metrics(refined_mask, gt_resized)


# --- 9. Improved Thickness Estimation ---
gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)

# 🔥 Normalize grayscale to reduce lighting bias
gray_norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

thickness_map = np.zeros((256, 256), dtype=np.float32)

thin = (gray_norm < 85)
medium = (gray_norm >= 85) & (gray_norm < 170)
thick = (gray_norm >= 170)

thickness_map[(refined_mask == 1) & thin] = 0.5
thickness_map[(refined_mask == 1) & medium] = 5.0
thickness_map[(refined_mask == 1) & thick] = 20.0

# 🔥 volume calculation
volume = np.sum(thickness_map * PIXEL_AREA * 1e-3)


# --- 10. Visualization ---
overlay = img_resized.copy()
overlay[refined_mask == 1] = [255, 0, 0]

plt.figure(figsize=(16, 5))

plt.subplot(1, 4, 1)
plt.title("Original")
plt.imshow(img_resized)
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title("Ground Truth")
plt.imshow(gt_resized, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title("Prediction")
plt.imshow(refined_mask, cmap='hot')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title(f"Overlay\nmIoU: {miou:.3f} | F1: {f1:.3f}")
plt.imshow(overlay)
plt.axis('off')

plt.savefig("result.png")
print("Saved visualization as result.png")


# --- 11. Results ---
print("\n--- FINAL RESULTS ---")
print(f"mIoU:      {miou:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Volume:    {volume:.4f} m³")