import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data_loader import get_combined_loader
from model import DualBranchDetector

# --- UPGRADE 1: Hybrid Loss Function ---
<<<<<<< HEAD
class JointEdgeLoss(nn.Module):
    def __init__(self, device, alpha=4.0):
        super().__init__()
        self.alpha = alpha
        self.kernel = torch.tensor(
            [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]],
            dtype=torch.float32
        ).view(1,1,3,3).to(device)

    def forward(self, outputs, targets):
        # Edge detection
        with torch.no_grad():
            gt_edges = F.conv2d(targets, self.kernel, padding=1)
            edge_mask = torch.sigmoid(torch.abs(gt_edges))

        edge_weight = 1.0 + self.alpha * edge_mask

        # Pixel-wise BCE + Focal
        bce = F.binary_cross_entropy_with_logits(outputs, targets, reduction='none')
        pt = torch.exp(-bce)
        focal = (1 - pt) ** 2 * bce

        weighted_bce = (edge_weight * focal).mean()

        # Weighted Dice
        probs = torch.sigmoid(outputs)
        inter = (probs * targets * edge_weight).sum()
        union = (probs * edge_weight).sum() + (targets * edge_weight).sum()
        dice = 1 - (2. * inter + 1.) / (union + 1.)

        return weighted_bce + dice
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 70
=======
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Intersection over Union based loss
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        # Standard pixel-wise loss
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        
        return BCE + dice_loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50
>>>>>>> 410cc706c221a986098d3d1ecb1abe3ba625cc7d
BATCH_SIZE = 8

def start_training():
    print(f"Starting BOOSTED Training on: {DEVICE}")
    model = DualBranchDetector().to(DEVICE)
    
    # --- UPGRADE 2: Improved Optimizer and Scheduler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
<<<<<<< HEAD
    criterion = JointEdgeLoss(DEVICE)
=======
    criterion = DiceBCELoss()
>>>>>>> 410cc706c221a986098d3d1ecb1abe3ba625cc7d

    train_dataset = get_combined_loader()
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        scheduler.step() # Update learning rate
        avg_loss = running_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{EPOCHS}] Avg Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

<<<<<<< HEAD
    torch.save(model.state_dict(), "oil_spill_model_final1.pth")
=======
    torch.save(model.state_dict(), "oil_spill_model_final.pth")
>>>>>>> 410cc706c221a986098d3d1ecb1abe3ba625cc7d
    print("Training Complete. Boosted weights saved.")

if __name__ == "__main__":
    start_training()