import torch
<<<<<<< HEAD
import torch.nn as nn
import torch.nn.functional as F
=======
>>>>>>> 410cc706c221a986098d3d1ecb1abe3ba625cc7d
from torch.utils.data import DataLoader
from data_loader import get_combined_loader
from model import DualBranchDetector

<<<<<<< HEAD
# --- UPGRADE 1: Hybrid Loss Function ---
class JointEdgeLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        # Laplacian kernel for edge detection (3x3)
        self.kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], 
                                   dtype=torch.float32).view(1, 1, 3, 3).to(device)

    def forward(self, outputs, targets):
        # 1. BCE Loss (Expects raw logits)
        bce = F.binary_cross_entropy_with_logits(outputs, targets)
        
        # 2. Dice Loss (Requires probabilities)
        probs = torch.sigmoid(outputs)
        inter = (probs * targets).sum()
        dice = 1 - (2. * inter + 1.) / (probs.sum() + targets.sum() + 1.)
        
        # 3. Edge Preservation (L_edge)
        # We detect edges in the Ground Truth to weight the loss
        with torch.no_grad():
            gt_edges = F.conv2d(targets, self.kernel, padding=1)
            # Dilate edges slightly to give the model some 'leeway'
            edge_mask = (torch.abs(gt_edges) > 0.1).float()
        
        # Final Formula: L = L_edge_weight * (L_BCE + L_Dice)
        # This forces the model to work harder on the pixels near the oil-water interface
        edge_weight = 1.0 + edge_mask 
        loss = edge_weight * (bce + dice)
        return loss.mean()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50
BATCH_SIZE = 8

def start_training():
    print(f"Starting BOOSTED Training on: {DEVICE}")
    model = DualBranchDetector().to(DEVICE)
    
    # --- UPGRADE 2: Improved Optimizer and Scheduler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = JointEdgeLoss()

    train_dataset = get_combined_loader()
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
=======
# 1. Hardware Check
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Training Parameters
EPOCHS = 50
BATCH_SIZE = 8 # Reduced for better stability on single GPU

def start_training():
    print(f"Starting Training Engine on: {DEVICE}")
    
    # Load Model (Proposed Dual-Branch CNN-ViT)
    model = DualBranchDetector().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.BCELoss()

    # Load Data (PALSAR + Sentinel)
    train_dataset = get_combined_loader()
    
    # WINDOWS FIX: num_workers=0 prevents the "silent hang"
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=True
    )

    print(f"Dataset Size: {len(train_dataset)} images found.")
>>>>>>> 410cc706c221a986098d3d1ecb1abe3ba625cc7d

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
<<<<<<< HEAD
=======
        
        # Batch Loop with Progress Tracking
>>>>>>> 410cc706c221a986098d3d1ecb1abe3ba625cc7d
        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
<<<<<<< HEAD
        
        scheduler.step() # Update learning rate
        avg_loss = running_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{EPOCHS}] Avg Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

    torch.save(model.state_dict(), "oil_spill_model_final.pth")
    print("Training Complete. Boosted weights saved.")

=======
            
            if (i + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        avg_loss = running_loss / len(train_loader)
        print(f"==> Epoch {epoch+1} Complete. Average Loss: {avg_loss:.4f}")

    # Save Model for Thickness Module
    torch.save(model.state_dict(), "oil_spill_model_final.pth")
    print("Project Training Successful. Model weights saved.")

# MANDATORY GUARD: Prevents infinite loops on Windows
>>>>>>> 410cc706c221a986098d3d1ecb1abe3ba625cc7d
if __name__ == "__main__":
    start_training()