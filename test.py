import torch
import cv2
import numpy as np
import os
from model import DualBranchDetector

<<<<<<< HEAD
# --- 1. Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "oil_spill_model_final.pth"
OUTPUT_DIR = r"E:/oil spill1/test_visual_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

# --- 3. Load Model ---
model = DualBranchDetector().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()


# --- 4. FULL METRICS FUNCTION ---
def calculate_metrics(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    TP = np.logical_and(pred, gt).sum()
    FP = np.logical_and(pred, np.logical_not(gt)).sum()
    FN = np.logical_and(np.logical_not(pred), gt).sum()
    TN = np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum()

    eps = 1e-7

    # Foreground
    precision_fg = TP / (TP + FP + eps)
    recall_fg = TP / (TP + FN + eps)
    iou_fg = TP / (TP + FP + FN + eps)

    # Background
    precision_bg = TN / (TN + FN + eps)
    recall_bg = TN / (TN + FP + eps)
    iou_bg = TN / (TN + FP + FN + eps)

    # Mean Metrics
    mean_precision = (precision_fg + precision_bg) / 2
    mean_recall = (recall_fg + recall_bg) / 2
    mean_iou = (iou_fg + iou_bg) / 2

    # Other Metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN + eps)
    error_rate = 1 - accuracy
    mean_pixel_accuracy = accuracy
    specificity = TN / (TN + FP + eps)

    # F1 (foreground)
    f1 = 2 * (precision_fg * recall_fg) / (precision_fg + recall_fg + eps)

    return {
        "Mean Precision": mean_precision,
        "Mean Recall": mean_recall,
        "Accuracy": accuracy,
        "Error Rate": error_rate,
        "Mean IoU": mean_iou,
        "Mean Pixel Accuracy": mean_pixel_accuracy,
        "Mean Specificity": specificity,
        "F1": f1
    }


# --- 5. Volume Estimation ---
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


# --- 6. Post-processing ---
def post_process(mask_np):
    kernel = np.ones((3, 3), np.uint8)
    refined = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(refined)
    clean_mask = np.zeros_like(refined)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 50:
            clean_mask[labels == i] = 1

    return clean_mask


# --- 7. Evaluation ---
final_report = []

for sensor, paths in test_configs.items():
    print(f"\nProcessing {sensor} Test Set...")

    sat_files = [f for f in os.listdir(paths['sat']) if f.endswith('.jpg')]

    metric_store = {
        "Mean Precision": [],
        "Mean Recall": [],
        "Accuracy": [],
        "Error Rate": [],
        "Mean IoU": [],
        "Mean Pixel Accuracy": [],
        "Mean Specificity": [],
        "F1": []
    }

    sensor_volume = 0

    for filename in sat_files:
        img_path = os.path.join(paths['sat'], filename)
        gt_filename = filename.replace('_sat.jpg', '_mask.png')
        gt_path = os.path.join(paths['gt'], gt_filename)

        if not os.path.exists(gt_path):
            continue

        # --- Inference ---
        raw_img = cv2.imread(img_path)
        img_input = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        img_input = cv2.resize(img_input, (256, 256)).astype(np.float32) / 255.0

        img_tensor = torch.from_numpy(img_input).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred_tensor = model(img_tensor)
            pred_mask = (pred_tensor.cpu().squeeze().numpy() > 0.5).astype(np.uint8)

        pred_mask = post_process(pred_mask)

        # --- Ground Truth ---
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = (cv2.resize(gt_mask, (256, 256)) > 127).astype(np.uint8)

        # --- Metrics ---
        metrics = calculate_metrics(pred_mask, gt_mask)
        for key in metric_store:
            metric_store[key].append(metrics[key])

        # --- Volume ---
        sensor_volume += estimate_volume(pred_mask, raw_img)

    # --- Aggregate ---
    final_report.append({
        "Sensor": sensor,
        "Metrics": {k: np.mean(v) for k, v in metric_store.items()},
        "Total_Vol": sensor_volume,
        "Img_Count": len(metric_store["F1"])
    })


# --- 8. Final Report ---
print("\n" + "="*60)
print("     FINAL TESTING & PERFORMANCE REPORT")
print("="*60)

for res in final_report:
    print(f"\nSENSOR: {res['Sensor']}")
    print(f"Images Tested   : {res['Img_Count']}")

    for key, val in res["Metrics"].items():
        print(f"{key:22}: {val:.4f}")

    print(f"Estimated Volume      : {res['Total_Vol']:.2f} m³")
    print("-" * 60)

total_test_vol = sum([r['Total_Vol'] for r in final_report])

print(f"\nGRAND TOTAL TEST VOLUME: {total_test_vol:.2f} m³")
print("="*60)
=======
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "oil_spill_model_final.pth"

# --- UPGRADE 3: Comprehensive Metrics Helper ---
def calculate_all_metrics(pred, gt):
    pred = pred.flatten()
    gt = gt.flatten()
    
    tp = np.logical_and(pred == 1, gt == 1).sum()
    tn = np.logical_and(pred == 0, gt == 0).sum()
    fp = np.logical_and(pred == 1, gt == 0).sum()
    fn = np.logical_and(pred == 0, gt == 1).sum()
    
    pixel_acc = (tp + tn) / (tp + tn + fp + fn)
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return iou, pixel_acc, precision, recall, f1

# (Keep your estimate_volume function exactly as it was)
def estimate_volume(mask, raw_img):
    gray = cv2.resize(cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY), (256, 256))
    THICKNESS_CLASSES = {"SHEEN": 0.1, "RAINBOW": 5.0, "SLICK": 50.0, "EMULSION": 200.0}
    thickness_map = np.zeros((256, 256))
    thickness_map[(mask == 1) & (gray < 50)] = THICKNESS_CLASSES["SHEEN"]
    thickness_map[(mask == 1) & (gray >= 50) & (gray < 150)] = THICKNESS_CLASSES["RAINBOW"]
    thickness_map[(mask == 1) & (gray >= 150) & (gray < 220)] = THICKNESS_CLASSES["SLICK"]
    thickness_map[(mask == 1) & (gray >= 220)] = THICKNESS_CLASSES["EMULSION"]
    return np.sum(thickness_map * 1e-6 * 100)

# Run Evaluation with new metrics
test_configs = {
    "SENTINEL": {"sat": r"E:/oil spill1/test-20251120T124100Z-1-001/test/sentinel/sat", "gt": r"E:/oil spill1/test-20251120T124100Z-1-001/test/sentinel/gt"},
    "PALSAR": {"sat": r"E:/oil spill1/test-20251120T124100Z-1-001/test/palsar/sat", "gt": r"E:/oil spill1/test-20251120T124100Z-1-001/test/palsar/gt"}
}

model = DualBranchDetector().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.eval()

for sensor, paths in test_configs.items():
    print(f"\nProcessing {sensor} Test Set...")
    files = [f for f in os.listdir(paths['sat']) if f.endswith('.jpg')]
    m_ious, m_accs, m_precs, m_recs, m_f1s = [], [], [], [], []
    total_vol = 0

    for filename in files:
        img_path = os.path.join(paths['sat'], filename)
        gt_path = os.path.join(paths['gt'], filename.replace('_sat.jpg', '_mask.png'))
        if not os.path.exists(gt_path): continue

        # Inference
        raw_img = cv2.imread(img_path)
        img_in = cv2.resize(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB), (256, 256)).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_in).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            pred = (model(img_tensor).cpu().squeeze().numpy() > 0.5).astype(np.uint8)

        gt = (cv2.resize(cv2.imread(gt_path, 0), (256, 256)) > 127).astype(np.uint8)

        # Advanced Metrics
        iou, acc, prec, rec, f1 = calculate_all_metrics(pred, gt)
        m_ious.append(iou); m_accs.append(acc); m_precs.append(prec); m_recs.append(rec); m_f1s.append(f1)
        total_vol += estimate_volume(pred, raw_img)

    print(f"--- {sensor} RESULTS ---")
    print(f"Pixel Accuracy: {np.mean(m_accs):.4f} | mIoU: {np.mean(m_ious):.4f}")
    print(f"Precision: {np.mean(m_precs):.4f} | Recall: {np.mean(m_recs):.4f} | F1: {np.mean(m_f1s):.4f}")
    print(f"Estimated Volume: {total_vol:.2f} m³")
>>>>>>> 410cc706c221a986098d3d1ecb1abe3ba625cc7d
