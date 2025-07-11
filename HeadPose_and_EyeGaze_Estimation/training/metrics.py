import numpy as np
import torch
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

class MetricsTracker:
    """Track and visualize training metrics"""
    def __init__(self):
        self.history = defaultdict(list)
        self.best_metrics = {}
    def update(self, epoch, **kwargs):
        for key, value in kwargs.items():
            self.history[key].append(value)
    def plot_training_curves(self, save_path='training_curves.png'):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Training Progress', fontsize=16)
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss', alpha=0.8)
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss', alpha=0.8)
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 1].plot(self.history['train_head_mae'], label='Train Head MAE', alpha=0.8)
        axes[0, 1].plot(self.history['val_head_mae'], label='Val Head MAE', alpha=0.8)
        axes[0, 1].set_title('Head Pose MAE (degrees)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 2].plot(self.history['train_gaze_mae'], label='Train Gaze MAE', alpha=0.8)
        axes[0, 2].plot(self.history['val_gaze_mae'], label='Val Gaze MAE', alpha=0.8)
        axes[0, 2].set_title('Gaze MAE (degrees)')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        if 'learning_rate' in self.history:
            axes[1, 0].plot(self.history['learning_rate'], alpha=0.8)
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        if 'val_head_yaw_mae' in self.history:
            axes[1, 1].plot(self.history['val_head_yaw_mae'], label='Yaw', alpha=0.8)
            axes[1, 1].plot(self.history['val_head_pitch_mae'], label='Pitch', alpha=0.8)
            axes[1, 1].plot(self.history['val_head_roll_mae'], label='Roll', alpha=0.8)
            axes[1, 1].set_title('Head Pose Components MAE')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        if 'val_gaze_yaw_mae' in self.history:
            axes[1, 2].plot(self.history['val_gaze_yaw_mae'], label='Yaw', alpha=0.8)
            axes[1, 2].plot(self.history['val_gaze_pitch_mae'], label='Pitch', alpha=0.8)
            axes[1, 2].set_title('Gaze Components MAE')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def compute_angle_from_sincos(sin_vals, cos_vals):
    """Convert sin/cos back to angles in degrees"""
    return torch.atan2(sin_vals, cos_vals) * 180 / np.pi

def compute_angular_error(pred_angles, gt_angles):
    """Compute angular error handling wraparound"""
    diff = torch.abs(pred_angles - gt_angles)
    return torch.min(diff, 360 - diff)
