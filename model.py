import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralCNNBranch(nn.Module):
    """Captures fine local spectral differences (oil vs. water) [cite: 106]"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)

class SpatialViTBranch(nn.Module):
    """Captures global spatial patterns like slick shapes and edges [cite: 107]"""
    def __init__(self, img_size=256, patch_size=16):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 128, kernel_size=patch_size, stride=patch_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=8, batch_first=True),
            num_layers=4
        )
        self.img_size = img_size
        self.patch_size = patch_size

    def forward(self, x):
        x = self.patch_embed(x) # (B, 128, 16, 16)
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2) # (B, 256, 128)
        x = self.transformer(x)
        x = x.transpose(1, 2).reshape(b, c, h, w)
        return F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear')

class DualBranchDetector(nn.Module):
    """The complete Proposed System for pixel-level accurate segmentation [cite: 108]"""
    def __init__(self):
        super().__init__()
        self.cnn = SpectralCNNBranch()
        self.vit = SpatialViTBranch()
        self.fusion = nn.Conv2d(256, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feat1 = self.cnn(x)
        feat2 = self.vit(x)
        combined = torch.cat([feat1, feat2], dim=1)
        return self.sigmoid(self.fusion(combined))