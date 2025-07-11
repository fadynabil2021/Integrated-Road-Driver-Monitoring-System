import torch
import numpy as np
import time
from .plots import plot_angle_histograms, plot_error_distributions, plot_accuracy_curves

def comprehensive_evaluation(model, test_loader, save_plots=True):
    model.eval()
    all_head_angles_pred = []
    all_head_angles_gt = []
    all_head_trans_pred = []
    all_head_trans_gt = []
    all_gaze_pred = []
    all_gaze_gt = []
    all_gaze_masks = []
    head_errors = {'yaw': [], 'pitch': [], 'roll': []}
    gaze_errors = {'yaw': [], 'pitch': []}
    inference_times = []
    print("Running comprehensive evaluation...")
    with torch.no_grad():
        for batch_idx, (images, head_labels, gaze_labels, gaze_masks, confidence) in enumerate(test_loader):
            images = images.to(next(model.parameters()).device)
            start_time = time.time()
            head_pred, gaze_pred = model(images)
            inference_time = time.time() - start_time
            inference_times.append(inference_time / images.size(0))
            head_pred_np = head_pred.cpu().numpy()
            gaze_pred_np = gaze_pred.cpu().numpy()
            head_labels_np = head_labels.cpu().numpy()
            gaze_labels_np = gaze_labels.cpu().numpy()
            gaze_masks_np = gaze_masks.cpu().numpy()
            head_angles_pred = head_pred_np[:, :3]
            head_angles_gt = head_labels_np[:, :3]
            head_trans_pred = head_pred_np[:, 3:]
            head_trans_gt = head_labels_np[:, 3:]
            gaze_pred_angles = np.arctan2(gaze_pred_np[:, ::2], gaze_pred_np[:, 1::2]) * 180 / np.pi
            gaze_gt_angles = np.arctan2(gaze_labels_np[:, ::2], gaze_labels_np[:, 1::2]) * 180 / np.pi
            all_head_angles_pred.append(head_angles_pred)
            all_head_angles_gt.append(head_angles_gt)
            all_head_trans_pred.append(head_trans_pred)
            all_head_trans_gt.append(head_trans_gt)
            all_gaze_pred.append(gaze_pred_angles)
            all_gaze_gt.append(gaze_gt_angles)
            all_gaze_masks.extend(gaze_masks_np)
            for i in range(len(head_angles_pred)):
                for j, angle in enumerate(['yaw', 'pitch', 'roll']):
                    pred = head_angles_pred[i, j]
                    gt = head_angles_gt[i, j]
                    error = min(abs(pred - gt), 360 - abs(pred - gt))
                    head_errors[angle].append(error)
                if gaze_masks_np[i] == 1:
                    for j, angle in enumerate(['yaw', 'pitch']):
                        pred = gaze_pred_angles[i, j]
                        gt = gaze_gt_angles[i, j]
                        error = min(abs(pred - gt), 360 - abs(pred - gt))
                        gaze_errors[angle].append(error)
    all_head_angles_pred = np.vstack(all_head_angles_pred)
    all_head_angles_gt = np.vstack(all_head_angles_gt)
    all_head_trans_pred = np.vstack(all_head_trans_pred)
    all_head_trans_gt = np.vstack(all_head_trans_gt)
    all_gaze_pred = np.vstack(all_gaze_pred)
    all_gaze_gt = np.vstack(all_gaze_gt)
    metrics = {}
    print("\n=== EVALUATION RESULTS ===")
    print(f"Inference Time: {np.mean(inference_times)*1000:.2f} ± {np.std(inference_times)*1000:.2f} ms per image")
    print(f"Throughput: {1/np.mean(inference_times):.1f} FPS")
    print("\n--- HEAD POSE METRICS ---")
    for i, (angle, errors) in enumerate([('Yaw', head_errors['yaw']), ('Pitch', head_errors['pitch']), ('Roll', head_errors['roll'])]):
        if len(errors) > 0:
            errors = np.array(errors)
            mae = np.mean(errors)
            std = np.std(errors)
            rmse = np.sqrt(np.mean(errors**2))
            bins = np.arange(-180, 181, 10)
            bin_errors = []
            for b in range(len(bins)-1):
                bin_mask = (all_head_angles_gt[:, i] >= bins[b]) & (all_head_angles_gt[:, i] < bins[b+1])
                if bin_mask.sum() > 0:
                    bin_errors.append(np.mean(errors[bin_mask]))
            bmae = np.mean(bin_errors) if bin_errors else mae
            acc_5 = np.mean(errors <= 5) * 100
            acc_10 = np.mean(errors <= 10) * 100
            acc_15 = np.mean(errors <= 15) * 100
            metrics[f'Head {angle}'] = {
                'MAE': mae, 'STD': std, 'RMSE': rmse, 'BMAE': bmae,
                'Acc@5°': acc_5, 'Acc@10°': acc_10, 'Acc@15°': acc_15
            }
            print(f"{angle:>5} - MAE: {mae:5.2f}° | STD: {std:5.2f}° | RMSE: {rmse:5.2f}° | BMAE: {bmae:5.2f}°")
            print(f"       Acc@5°: {acc_5:5.1f}% | Acc@10°: {acc_10:5.1f}% | Acc@15°: {acc_15:5.1f}%")
    print("\n--- GAZE METRICS ---")
    gaze_valid_count = sum(all_gaze_masks)
    print(f"Valid gaze samples: {gaze_valid_count} / {len(all_gaze_masks)} ({gaze_valid_count/len(all_gaze_masks)*100:.1f}%)")
    for i, (angle, errors) in enumerate([('Yaw', gaze_errors['yaw']), ('Pitch', gaze_errors['pitch'])]):
        if len(errors) > 0:
            errors = np.array(errors)
            mae = np.mean(errors)
            std = np.std(errors)
            rmse = np.sqrt(np.mean(errors**2))
            valid_gaze_pred = all_gaze_pred[np.array(all_gaze_masks) == 1]
            valid_gaze_gt = all_gaze_gt[np.array(all_gaze_masks) == 1]
            if len(valid_gaze_gt) > 0:
                bins = np.arange(-90, 91, 10)
                bin_errors = []
                for b in range(len(bins)-1):
                    bin_mask = (valid_gaze_gt[:, i] >= bins[b]) & (valid_gaze_gt[:, i] < bins[b+1])
                    if bin_mask.sum() > 0:
                        bin_errors.append(np.mean(errors[bin_mask]))
                bmae = np.mean(bin_errors) if bin_errors else mae
            else:
                bmae = mae
            acc_3 = np.mean(errors <= 3) * 100
            acc_5 = np.mean(errors <= 5) * 100
            acc_10 = np.mean(errors <= 10) * 100
            metrics[f'Gaze {angle}'] = {
                'MAE': mae, 'STD': std, 'RMSE': rmse, 'BMAE': bmae,
                'Acc@3°': acc_3, 'Acc@5°': acc_5, 'Acc@10°': acc_10
            }
            print(f"{angle:>5} - MAE: {mae:5.2f}° | STD: {std:5.2f}° | RMSE: {rmse:5.2f}° | BMAE: {bmae:5.2f}°")
            print(f"       Acc@3°: {acc_3:5.1f}% | Acc@5°: {acc_5:5.1f}% | Acc@10°: {acc_10:5.1f}%")
    overall_head_mae = np.mean([head_errors['yaw'], head_errors['pitch'], head_errors['roll']])
    overall_gaze_mae = np.mean([gaze_errors['yaw'], gaze_errors['pitch']]) if gaze_errors['yaw'] else 0
    print(f"\n--- OVERALL PERFORMANCE ---")
    print(f"Head Pose MAE: {overall_head_mae:.2f}°")
    print(f"Gaze MAE: {overall_gaze_mae:.2f}°")
    print(f"Real-time capable: {'✅' if np.mean(inference_times) < 0.033 else '❌'} ({'30+ FPS' if np.mean(inference_times) < 0.033 else f'{1/np.mean(inference_times):.1f} FPS'})")
    if save_plots:
        plot_angle_histograms(all_head_angles_pred, all_head_angles_gt, 'head_pose_histograms.png')
        if gaze_valid_count > 100:
            valid_indices = np.array(all_gaze_masks) == 1
            plot_angle_histograms(all_gaze_pred[valid_indices, :2], all_gaze_gt[valid_indices, :2], 'gaze_histograms.png')
        plot_error_distributions(head_errors, gaze_errors, 'error_distributions.png')
        plot_accuracy_curves(head_errors, gaze_errors, 'accuracy_curves.png')
    return metrics, {
        'head_angles_pred': all_head_angles_pred, 'head_angles_gt': all_head_angles_gt,
        'head_trans_pred': all_head_trans_pred, 'head_trans_gt': all_head_trans_gt,
        'gaze_pred': all_gaze_pred, 'gaze_gt': all_gaze_gt,
        'gaze_masks': all_gaze_masks, 'inference_times': inference_times
    }
