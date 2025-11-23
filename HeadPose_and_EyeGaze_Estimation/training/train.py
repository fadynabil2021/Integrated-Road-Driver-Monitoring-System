import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from .losses import FocalLoss
from .metrics import MetricsTracker, compute_angle_from_sincos, compute_angular_error
import time
from collections import defaultdict
import numpy as np

def enhanced_train_model(model, train_loader, val_loader, epochs=100, 
                        checkpoint_path="best_model.pth", patience=15, 
                        min_lr=1e-7):
    """Enhanced training with advanced techniques"""
    
    # Optimizers and schedulers
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4, 
                           betas=(0.9, 0.999), eps=1e-8)
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=min_lr)
    
    # Loss functions
    focal_loss = FocalLoss(alpha=1, gamma=2)
    mse_loss = nn.MSELoss()
    
    # Mixed precision training
    scaler = GradScaler()
    
    # Metrics tracking
    tracker = MetricsTracker()
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_metrics = {}
    
    print("Starting enhanced training...")
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_metrics = defaultdict(float)
        train_counts = defaultdict(int)
        
        for batch_idx, (images, head_labels, gaze_labels, gaze_masks, confidence) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            head_labels = head_labels.to(device, non_blocking=True)
            gaze_labels = gaze_labels.to(device, non_blocking=True)
            gaze_masks = gaze_masks.to(device, non_blocking=True)
            confidence = confidence.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with autocast(device_type='cuda'):
            head_pred, gaze_pred = model(images)
                # Head pose loss: angles and translation
                head_angles_pred = head_pred[:, :3]
                head_trans_pred = head_pred[:, 3:]
                head_angles_gt = head_labels[:, :3]
                head_trans_gt = head_labels[:, 3:]
                angle_loss = F.mse_loss(head_angles_pred, head_angles_gt)
                trans_loss = F.mse_loss(head_trans_pred, head_trans_gt)
                head_loss = angle_loss + trans_loss
                # Gaze loss with confidence weighting
                gaze_loss_per_sample = mse_loss(gaze_pred, gaze_labels)
                valid_mask = gaze_masks.unsqueeze(1)
                confidence_weight = confidence.unsqueeze(1) * valid_mask
                
                if confidence_weight.sum() > 0:
                    weighted_gaze_loss = (gaze_loss_per_sample * confidence_weight).sum() / confidence_weight.sum()
                else:
                    weighted_gaze_loss = torch.tensor(0.0, device=device)
            
            # Combined loss with adaptive weighting
            total_loss = head_loss + 5.0 * weighted_gaze_loss
            
                # Add L2 regularization
                l2_reg = torch.tensor(0.0, device=device)
                for param in model.parameters():
                    l2_reg += torch.norm(param)
                total_loss += 1e-5 * l2_reg
            
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            # Compute detailed metrics
            with torch.no_grad():
                # Head pose metrics
                angle_errors = torch.abs(head_angles_pred - head_angles_gt)
                trans_errors = torch.abs(head_trans_pred - head_trans_gt)
                train_metrics['head_angle_mae'] += angle_errors.mean().item()
                train_metrics['head_trans_mae'] += trans_errors.mean().item()
                
            train_metrics['loss'] += total_loss.item()
                train_metrics['head_mae'] += angle_errors.mean().item()
                train_metrics['head_yaw_mae'] += angle_errors[:, 0].mean().item()
                train_metrics['head_pitch_mae'] += angle_errors[:, 1].mean().item()
                train_metrics['head_roll_mae'] += angle_errors[:, 2].mean().item()
        
                # Gaze metrics
                if gaze_masks.sum() > 0:
                    gaze_pred_angles = compute_angle_from_sincos(gaze_pred[:, ::2], gaze_pred[:, 1::2])
                    gaze_gt_angles = compute_angle_from_sincos(gaze_labels[:, ::2], gaze_labels[:, 1::2])
                    gaze_errors = compute_angular_error(gaze_pred_angles, gaze_gt_angles)
                    
                    valid_indices = gaze_masks == 1
                    if valid_indices.sum() > 0:
                        train_metrics['gaze_mae'] += gaze_errors[valid_indices].mean().item()
                        train_metrics['gaze_yaw_mae'] += gaze_errors[valid_indices, 0].mean().item()
                        train_metrics['gaze_pitch_mae'] += gaze_errors[valid_indices, 1].mean().item()
                        train_counts['gaze'] += valid_indices.sum().item()
                
                train_counts['total'] += images.size(0)
        
        # Validation phase
        model.eval()
        val_metrics = defaultdict(float)
        val_counts = defaultdict(int)
        
        with torch.no_grad():
            for images, head_labels, gaze_labels, gaze_masks, confidence in val_loader:
                images = images.to(device, non_blocking=True)
                head_labels = head_labels.to(device, non_blocking=True)
                gaze_labels = gaze_labels.to(device, non_blocking=True)
                gaze_masks = gaze_masks.to(device, non_blocking=True)
                confidence = confidence.to(device, non_blocking=True)
                
                head_pred, gaze_pred = model(images)
                head_angles_pred = head_pred[:, :3]
                head_trans_pred = head_pred[:, 3:]
                head_angles_gt = head_labels[:, :3]
                head_trans_gt = head_labels[:, 3:]
                angle_loss = F.mse_loss(head_angles_pred, head_angles_gt)
                trans_loss = F.mse_loss(head_trans_pred, head_trans_gt)
                head_loss = angle_loss + trans_loss
                gaze_loss_per_sample = mse_loss(gaze_pred, gaze_labels)
                valid_mask = gaze_masks.unsqueeze(1)
                confidence_weight = confidence.unsqueeze(1) * valid_mask
                
                if confidence_weight.sum() > 0:
                    weighted_gaze_loss = (gaze_loss_per_sample * confidence_weight).sum() / confidence_weight.sum()
                else:
                    weighted_gaze_loss = torch.tensor(0.0, device=device)
                
                total_loss = head_loss + 5.0 * weighted_gaze_loss
                
                # Metrics computation
                angle_errors = torch.abs(head_angles_pred - head_angles_gt)
                trans_errors = torch.abs(head_trans_pred - head_trans_gt)
                val_metrics['head_angle_mae'] += angle_errors.mean().item()
                val_metrics['head_trans_mae'] += trans_errors.mean().item()
                
                if gaze_masks.sum() > 0:
                    gaze_pred_angles = compute_angle_from_sincos(gaze_pred[:, ::2], gaze_pred[:, 1::2])
                    gaze_gt_angles = compute_angle_from_sincos(gaze_labels[:, ::2], gaze_labels[:, 1::2])
                    gaze_errors = compute_angular_error(gaze_pred_angles, gaze_gt_angles)
                    
                    valid_indices = gaze_masks == 1
                    if valid_indices.sum() > 0:
                        val_metrics['gaze_mae'] += gaze_errors[valid_indices].mean().item()
                        val_metrics['gaze_yaw_mae'] += gaze_errors[valid_indices, 0].mean().item()
                        val_metrics['gaze_pitch_mae'] += gaze_errors[valid_indices, 1].mean().item()
                        val_counts['gaze'] += valid_indices.sum().item()
                
                val_counts['total'] += images.size(0)
        
        # Average metrics
        for key in train_metrics:
            if key.endswith('_mae'):
                train_metrics[key] /= len(train_loader)
            else:
                train_metrics[key] /= len(train_loader)
        
        for key in val_metrics:
            if key.endswith('_mae'):
                val_metrics[key] /= len(val_loader)
            else:
                val_metrics[key] /= len(val_loader)
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Track metrics
        tracker.update(
            epoch=epoch,
            train_loss=train_metrics['loss'],
            val_loss=val_metrics['loss'],
            train_head_mae=train_metrics['head_mae'],
            val_head_mae=val_metrics['head_mae'],
            train_gaze_mae=train_metrics.get('gaze_mae', 0),
            val_gaze_mae=val_metrics.get('gaze_mae', 0),
            val_head_yaw_mae=val_metrics['head_yaw_mae'],
            val_head_pitch_mae=val_metrics['head_pitch_mae'],
            val_head_roll_mae=val_metrics['head_roll_mae'],
            val_gaze_yaw_mae=val_metrics.get('gaze_yaw_mae', 0),
            val_gaze_pitch_mae=val_metrics.get('gaze_pitch_mae', 0),
            learning_rate=current_lr
        )
        
        epoch_time = time.time() - start_time
            
            # Print progress
        print(f"Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s)")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Head MAE: {train_metrics['head_mae']:.2f}°, Gaze MAE: {train_metrics.get('gaze_mae', 0):.2f}°")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Head MAE: {val_metrics['head_mae']:.2f}°, Gaze MAE: {val_metrics.get('gaze_mae', 0):.2f}°")
        print(f"  LR: {current_lr:.2e}")
            
        # Save best model if validation loss improves
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
            best_metrics = val_metrics.copy()
            
            # Save checkpoint in a PyTorch 2.6 compatible way
            checkpoint_data = {
                    'epoch': epoch,
                'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                    'best_loss': best_val_loss,
                'metrics': best_metrics
            }
            
            # Use _use_new_zipfile_serialization=False for better compatibility
            torch.save(checkpoint_data, checkpoint_path, _use_new_zipfile_serialization=False)
            print(f" Best model saved!")
            patience_counter = 0
            else:
                patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement")
            break
    
        # Plot progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            tracker.plot_training_curves(f'training_curves_epoch_{epoch+1}.png')
    
    return tracker, best_metrics

