import torch
import cv2
import numpy as np
import os
from model import DualBranchDetector

# --- 1. Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "oil_spill_model_final.pth"

# --- 2. Correct Dataset Paths ---
test_configs = {
    "SENTINEL": {
        "sat": r"E:\oil-spill-detection-recovery-main\test-20251120T124100Z-1-001\test\sentinel\sat",
        "gt": r"E:\oil-spill-detection-recovery-main\test-20251120T124100Z-1-001\test\sentinel\gt"
    },
    "PALSAR": {
        "sat": r"E:\oil-spill-detection-recovery-main\test-20251120T124100Z-1-001\test\palsar\sat",
        "gt": r"E:\oil-spill-detection-recovery-main\test-20251120T124100Z-1-001\test\palsar\gt"
    }
}

# --- 3. Validate Paths ---
for sensor, paths in test_configs.items():
    if not os.path.exists(paths["sat"]):
        raise ValueError(f"SAT path not found for {sensor}: {paths['sat']}")
    if not os.path.exists(paths["gt"]):
        raise ValueError(f"GT path not found for {sensor}: {paths['gt']}")

# --- 4. Load Model ---
model = DualBranchDetector().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()

# --- 5. Confusion Matrix ---
def update_confusion_matrix(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    TP = np.logical_and(pred, gt).sum()
    FP = np.logical_and(pred, np.logical_not(gt)).sum()
    FN = np.logical_and(np.logical_not(pred), gt).sum()
    TN = np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum()

    return TP, FP, FN, TN

# --- 6. Final Metrics ---
def compute_metrics(TP, FP, FN, TN):
    eps = 1e-7

    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    iou = TP / (TP + FP + FN + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    accuracy = (TP + TN) / (TP + TN + FP + FN + eps)
    specificity = TN / (TN + FP + eps)

    return {
        "Precision": precision,
        "Recall": recall,
        "IoU": iou,
        "F1-Score": f1,
        "Accuracy": accuracy,
        "Specificity": specificity
    }

# --- 7. Volume Estimation ---
def estimate_volume(mask, raw_img):
    gray = cv2.resize(cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY), (256, 256))

    thickness_map = np.zeros((256, 256))
    thickness_map[(mask == 1) & (gray < 50)] = 0.1
    thickness_map[(mask == 1) & (gray >= 50) & (gray < 150)] = 5.0
    thickness_map[(mask == 1) & (gray >= 150) & (gray < 220)] = 50.0
    thickness_map[(mask == 1) & (gray >= 220)] = 200.0

    pixel_area_m2 = 100
    volume_m3 = np.sum(thickness_map * 1e-6 * pixel_area_m2)

    return volume_m3

# --- 8. Post-processing ---
def post_process(mask):
    kernel = np.ones((3, 3), np.uint8)
    refined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(refined)
    clean = np.zeros_like(refined)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 50:
            clean[labels == i] = 1

    return clean

# --- 9. Evaluation ---
final_report = []

for sensor, paths in test_configs.items():
    print(f"\nProcessing {sensor} Test Set...")

    files = [f for f in os.listdir(paths['sat']) if f.endswith('.jpg')]

    TP_total, FP_total, FN_total, TN_total = 0, 0, 0, 0
    total_volume = 0
    count = 0

    for file in files:
        img_path = os.path.join(paths['sat'], file)
        gt_path = os.path.join(paths['gt'], file.replace('_sat.jpg', '_mask.png'))

        if not os.path.exists(gt_path):
            continue

        raw_img = cv2.imread(img_path)

        # --- Preprocess ---
        img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256)).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        # --- Inference ---
        with torch.no_grad():
            pred = model(tensor)
            mask = (pred.cpu().squeeze().numpy() > 0.5).astype(np.uint8)

        mask = post_process(mask)

        # --- Ground Truth ---
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        gt = (cv2.resize(gt, (256, 256)) > 127).astype(np.uint8)

        # --- Metrics ---
        TP, FP, FN, TN = update_confusion_matrix(mask, gt)
        TP_total += TP
        FP_total += FP
        FN_total += FN
        TN_total += TN

        # --- Volume ---
        total_volume += estimate_volume(mask, raw_img)
        count += 1

    metrics = compute_metrics(TP_total, FP_total, FN_total, TN_total)

    final_report.append({
        "Sensor": sensor,
        "Metrics": metrics,
        "Volume": total_volume,
        "Count": count
    })

# --- 10. Final Output ---
print("\n" + "="*60)
print("     FINAL TESTING & PERFORMANCE REPORT")
print("="*60)

for res in final_report:
    print(f"\nSENSOR: {res['Sensor']}")
    print(f"Images Tested : {res['Count']}")

    for k, v in res["Metrics"].items():
        print(f"{k:15}: {v:.4f}")

    print(f"Estimated Volume : {res['Volume']:.2f} m³")
    print("-" * 60)

total_vol = sum([r["Volume"] for r in final_report])

print(f"\nGRAND TOTAL VOLUME: {total_vol:.2f} m³")
print("="*60)