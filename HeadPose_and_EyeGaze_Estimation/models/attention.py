import torch
import torch.nn as nn

class SpatialAttention(nn.Module):
    """Spatial attention mechanism for focusing on facial features"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        attention = self.conv(x)
        attention = self.sigmoid(attention)
        return x * attention
