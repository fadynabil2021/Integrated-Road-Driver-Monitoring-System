import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_angle_histograms(predictions, ground_truth, save_path='angle_histograms.png'):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Angle Distribution Analysis', fontsize=16)
    angles = ['Yaw', 'Pitch', 'Roll']
    colors = ['blue', 'green', 'red']
    for i, (angle, color) in enumerate(zip(angles, colors)):
        axes[0, i].hist(ground_truth[:, i], bins=50, alpha=0.7, color=color, label='Ground Truth', density=True)
        axes[0, i].hist(predictions[:, i], bins=50, alpha=0.7, color='orange', label='Predictions', density=True)
        axes[0, i].set_title(f'{angle} Distribution')
        axes[0, i].set_xlabel('Angle (degrees)')
        axes[0, i].set_ylabel('Density')
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)
        errors = predictions[:, i] - ground_truth[:, i]
        errors = np.where(errors > 180, errors - 360, errors)
        errors = np.where(errors < -180, errors + 360, errors)
        axes[1, i].hist(errors, bins=50, alpha=0.7, color=color, density=True)
        axes[1, i].set_title(f'{angle} Error Distribution')
        axes[1, i].set_xlabel('Error (degrees)')
        axes[1, i].set_ylabel('Density')
        axes[1, i].axvline(0, color='black', linestyle='--', alpha=0.5)
        axes[1, i].grid(True, alpha=0.3)
        mean_error = np.mean(np.abs(errors))
        std_error = np.std(errors)
        axes[1, i].text(0.05, 0.95, f'MAE: {mean_error:.2f}°\nSTD: {std_error:.2f}°', 
                       transform=axes[1, i].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_error_distributions(head_errors, gaze_errors, save_path='error_distributions.png'):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Error Distribution Analysis', fontsize=16)
    head_angles = ['yaw', 'pitch', 'roll']
    colors = ['blue', 'green', 'red']
    for i, (angle, color) in enumerate(zip(head_angles, colors)):
        errors = np.array(head_errors[angle])
        axes[0, i].hist(errors, bins=50, alpha=0.7, color=color, density=True)
        axes[0, i].set_title(f'Head {angle.capitalize()} Error Distribution')
        axes[0, i].set_xlabel('Error (degrees)')
        axes[0, i].set_ylabel('Density')
        axes[0, i].axvline(np.mean(errors), color='red', linestyle='--', label=f'Mean: {np.mean(errors):.2f}°')
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)
    gaze_angles = ['yaw', 'pitch']
    for i, angle in enumerate(gaze_angles):
        if len(gaze_errors[angle]) > 0:
            errors = np.array(gaze_errors[angle])
            axes[1, i].hist(errors, bins=50, alpha=0.7, color=colors[i], density=True)
            axes[1, i].set_title(f'Gaze {angle.capitalize()} Error Distribution')
            axes[1, i].set_xlabel('Error (degrees)')
            axes[1, i].set_ylabel('Density')
            axes[1, i].axvline(np.mean(errors), color='red', linestyle='--', label=f'Mean: {np.mean(errors):.2f}°')
            axes[1, i].legend()
            axes[1, i].grid(True, alpha=0.3)
        else:
            axes[1, i].text(0.5, 0.5, 'No valid gaze data', 
                           transform=axes[1, i].transAxes, ha='center', va='center')
            axes[1, i].set_title(f'Gaze {angle.capitalize()} Error Distribution')
    all_head_errors = np.concatenate([head_errors['yaw'], head_errors['pitch'], head_errors['roll']])
    axes[1, 2].hist(all_head_errors, bins=50, alpha=0.7, color='purple', density=True)
    axes[1, 2].set_title('Combined Head Pose Error Distribution')
    axes[1, 2].set_xlabel('Error (degrees)')
    axes[1, 2].set_ylabel('Density')
    axes[1, 2].axvline(np.mean(all_head_errors), color='red', linestyle='--', label=f'Mean: {np.mean(all_head_errors):.2f}°')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_accuracy_curves(head_errors, gaze_errors, save_path='accuracy_curves.png'):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Accuracy vs Error Threshold', fontsize=16)
    thresholds = np.arange(0, 31, 0.5)
    head_angles = ['yaw', 'pitch', 'roll']
    colors = ['blue', 'green', 'red']
    for angle, color in zip(head_angles, colors):
        errors = np.array(head_errors[angle])
        accuracies = [np.mean(errors <= t) * 100 for t in thresholds]
        axes[0].plot(thresholds, accuracies, label=f'Head {angle.capitalize()}', color=color, linewidth=2)
    axes[0].set_title('Head Pose Accuracy vs Threshold')
    axes[0].set_xlabel('Error Threshold (degrees)')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 30)
    axes[0].set_ylim(0, 100)
    gaze_thresholds = np.arange(0, 21, 0.2)
    gaze_angles = ['yaw', 'pitch']
    for i, angle in enumerate(gaze_angles):
        if len(gaze_errors[angle]) > 0:
            errors = np.array(gaze_errors[angle])
            accuracies = [np.mean(errors <= t) * 100 for t in gaze_thresholds]
            axes[1].plot(gaze_thresholds, accuracies, label=f'Gaze {angle.capitalize()}', color=colors[i], linewidth=2)
    axes[1].set_title('Gaze Accuracy vs Threshold')
    axes[1].set_xlabel('Error Threshold (degrees)')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 20)
    axes[1].set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
