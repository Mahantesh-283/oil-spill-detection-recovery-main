import os
import cv2
import torch
import numpy as np
<<<<<<< HEAD
import random
from torch.utils.data import Dataset, ConcatDataset

class OilSpillDataset(Dataset):
    def __init__(self, sat_dir, mask_dir, target_size=(256, 256), augment=False):
        self.sat_dir = sat_dir
        self.mask_dir = mask_dir
        self.target_size = target_size
        self.augment = augment

        self.img_names = sorted([
            f for f in os.listdir(sat_dir) if f.endswith('_sat.jpg')
        ])
=======
from torch.utils.data import Dataset, ConcatDataset

class OilSpillDataset(Dataset):
    def __init__(self, sat_dir, mask_dir, target_size=(256, 256)):
        self.sat_dir = sat_dir
        self.mask_dir = mask_dir
        self.target_size = target_size
        # Filter for only satellite images to avoid hidden files
        self.img_names = sorted([f for f in os.listdir(sat_dir) if f.endswith('_sat.jpg')])
>>>>>>> 410cc706c221a986098d3d1ecb1abe3ba625cc7d

    def __len__(self):
        return len(self.img_names)

<<<<<<< HEAD
    def apply_augmentation(self, image, mask):
        # -------- Flip --------
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)

        if random.random() > 0.5:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)

        # -------- Rotation (safe 90° only) --------
        if random.random() > 0.5:
            k = random.randint(0, 3)
            image = np.rot90(image, k).copy()
            mask = np.rot90(mask, k).copy()

        return image, mask

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.sat_dir, img_name)

=======
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.sat_dir, img_name)
        
        # MAPPING: Converts '10777_sat.jpg' -> '10777_mask.png'
>>>>>>> 410cc706c221a986098d3d1ecb1abe3ba625cc7d
        mask_name = img_name.replace('_sat.jpg', '_mask.png')
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            raise FileNotFoundError(f"Error loading pair: {img_path} or {mask_path}")

<<<<<<< HEAD
        # Resize first
        image = cv2.resize(image, self.target_size)
        mask = cv2.resize(mask, self.target_size)

        # Apply augmentation ONLY for training
        if self.augment:
            image, mask = self.apply_augmentation(image, mask)

        

        # Normalize
        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0
        mask = (mask > 0.5).astype(np.float32)
        image = np.log1p(image)

        return (
            torch.from_numpy(image).permute(2, 0, 1),
            torch.from_numpy(mask).unsqueeze(0)
        )


# -------- Combined Dataset --------
def get_combined_loader():
    p1 = OilSpillDataset(
        r"E:/oil-spill-detection-recovery-main/organized_train/palsar/sat",
        r"E:/oil-spill-detection-recovery-main/organized_train/palsar/mask",
        augment=True   # ✅ ON for training
    )

    s1 = OilSpillDataset(
        r"E:/oil-spill-detection-recovery-main/organized_train/sentinel/sat",
        r"E:/oil-spill-detection-recovery-main/organized_train/sentinel/mask",
        augment=True   # ✅ ON for training
    )

=======
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.target_size).astype(np.float32) / 255.0
        mask = cv2.resize(mask, self.target_size).astype(np.float32) / 255.0
        mask = (mask > 0.5).astype(np.float32)

        return torch.from_numpy(image).permute(2, 0, 1), torch.from_numpy(mask).unsqueeze(0)

def get_combined_loader(batch_size=16):
    p1 = OilSpillDataset(r"E:/oil spill1/organized_train/palsar/sat", r"E:/oil spill1/organized_train/palsar/mask")
    s1 = OilSpillDataset(r"E:/oil spill1/organized_train/sentinel/sat", r"E:/oil spill1/organized_train/sentinel/mask")
>>>>>>> 410cc706c221a986098d3d1ecb1abe3ba625cc7d
    return ConcatDataset([p1, s1])