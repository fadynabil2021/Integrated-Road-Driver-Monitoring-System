import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    """Focal loss for handling difficult samples"""
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.mse = nn.MSELoss(reduction='none')
    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        pt = torch.exp(-mse_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * mse_loss
        return focal_loss.mean()
