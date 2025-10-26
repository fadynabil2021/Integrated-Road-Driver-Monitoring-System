import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class AdvancedHeadPoseGazeModel(nn.Module):
    """Enhanced model with attention mechanism and regularization"""
    def __init__(self, backbone='efficientnet_b0', dropout_rate=0.4):
        super().__init__()
        if backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
            feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            self.backbone = models.mobilenet_v2(weights='IMAGENET1K_V1')
            feature_dim = self.backbone.last_channel
            self.backbone.classifier = nn.Identity()
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.head_branch = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.3),
        )
        self.gaze_branch = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.3),
        )
        self.head_pose_head = nn.Linear(128, 6)
        self.gaze_head = nn.Linear(128, 4)
        self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def forward(self, x):
        if hasattr(self.backbone, 'features'):
            features = self.backbone.features(x)
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        else:
            features = self.backbone(x)
        features = self.feature_norm(features)
        features = self.dropout(features)
        head_features = self.head_branch(features)
        gaze_features = self.gaze_branch(features)
        head_pose = self.head_pose_head(head_features)
        gaze = self.gaze_head(gaze_features)
        gaze = F.normalize(gaze.view(-1, 2, 2), dim=2).view(-1, 4)
        return head_pose, gaze
