# Multisensor Oil Spill Detection and Operational Recovery System

A Deep Learning-based decision support system that detects oil spills using **Sentinel-2 (Optical)** and **PALSAR (SAR)** satellite imagery, estimates spill volume, and tasks the nearest response vessel via **AIS (Automatic Identification System)** data.



## 🚀 Key Features
- **Dual-Branch Architecture:** Combines CNN and Vision Transformers (ViT) for robust detection across different sensors.
- **Multisensor Fusion:** Processes both Sentinel (Optical) and PALSAR (SAR) data.
- **Thickness & Volume Retrieval:** Uses spectral intensity to map oil thickness and calculate total volume in cubic meters.
- **Automated Recovery Tasking:** Identifies the nearest response vessel and calculates ETA.

## 📊 Performance Metrics
Based on our boosted training (Hybrid Dice-BCE Loss):
<<<<<<< HEAD
- **Mean IoU (Accuracy):** 0.6315
- **Recall (Sensitivity):** 0.8119
=======
- **Mean IoU (Accuracy):** 0.7442
- **Recall (Sensitivity):** 0.9119
>>>>>>> 410cc706c221a986098d3d1ecb1abe3ba625cc7d
- **Pixel Accuracy:** 92.89%

## 🛠️ Technology Stack
- **Framework:** PyTorch
- **Architecture:** Dual-Branch CNN + Transformer
- **Tools:** OpenCV, Pandas, NumPy
- **Satellite Data:** Sentinel-2 & ALOS PALSAR

## 📋 How to Run
1. **Train the model:**
   ```bash
   python src/train.py
2.Evaluate on Test Set:
  python src/test.py
3.Run Operational Recovery Report:
  Bash
<<<<<<< HEAD
  python src/thickness_calc.py
=======
  python src/thickness_calc.py
>>>>>>> 410cc706c221a986098d3d1ecb1abe3ba625cc7d
