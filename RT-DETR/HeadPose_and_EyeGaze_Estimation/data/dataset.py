import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from scipy.spatial.transform import Rotation


def preprocess_subject_enhanced(subject_number):
    """Enhanced preprocessing with better outlier handling and data validation"""
    gt_path = f"/mnt/e/__GradProject/Fady/FadyTaskGradProject/data/raw/GroundTruth_AutoPOSE/Subject_{subject_number}_Groundtruth.csv"
    df = pd.read_csv(gt_path)
    df['image_path'] = df['FrameNumber'].apply(
        lambda x: f"/mnt/e/__GradProject/Fady/FadyTaskGradProject/data/raw/Dataset/Subject_{subject_number}_zip/Subject_{subject_number}/Subject_{subject_number}_{x}.png"
    )
    angle_cols = ['head_yaw', 'head_pitch', 'head_roll', 'gaze_yaw', 'gaze_pitch']
    for col in angle_cols:
        df[col] = np.nan
    df['gaze_mask'] = 0
    df['confidence'] = 0.0
    valid_count = 0
    for idx, row in df.iterrows():
        if row['Target_valid'] and row['Cam_valid'] and row['Car_valid']:
            try:
                R = np.array([row[f'R_Head2Cam_{i}'] for i in range(1, 10)]).reshape(3, 3)
                if not np.allclose(np.linalg.det(R), 1.0, atol=1e-2) or not np.allclose(R @ R.T, np.eye(3), atol=1e-2):
                    continue
                rot = Rotation.from_matrix(R)
                euler = rot.as_euler('zyx', degrees=True)
                if abs(euler[0]) > 90 or abs(euler[1]) > 60 or abs(euler[2]) > 45:
                    continue
                df.loc[idx, 'head_yaw'] = euler[0]
                df.loc[idx, 'head_pitch'] = euler[1]
                df.loc[idx, 'head_roll'] = euler[2]
                t_head = np.array([row[f't_Head_in_Cam_{i}'] for i in range(1, 4)])
                min_angle = float('inf')
                best_gaze = None
                target_distances = []
                for k in range(1, 7):
                    t_target = np.array([row[f't_Car_{k}_in_Cam_{i}'] for i in range(1, 4)])
                    distance = np.linalg.norm(t_target - t_head)
                    target_distances.append(distance)
                    if distance > 0.1:
                        dir_to_target = (t_target - t_head) / distance
                        forward_dir = R @ np.array([0, 0, 1])
                        angle = np.arccos(np.clip(np.dot(dir_to_target, forward_dir), -1.0, 1.0))
                        if angle < min_angle:
                            min_angle = angle
                            gaze_camera = dir_to_target
                            best_gaze = R.T @ gaze_camera
                if best_gaze is not None and min_angle < np.pi/3:
                    gaze_yaw = np.arctan2(best_gaze[0], best_gaze[2]) * 180 / np.pi
                    gaze_pitch = np.arcsin(-np.clip(best_gaze[1], -1, 1)) * 180 / np.pi
                    if abs(gaze_yaw) <= 90 and abs(gaze_pitch) <= 60:
                        df.loc[idx, 'gaze_yaw'] = gaze_yaw
                        df.loc[idx, 'gaze_pitch'] = gaze_pitch
                    df.loc[idx, 'gaze_mask'] = 1
                    df.loc[idx, 'confidence'] = max(0.1, 1.0 - min_angle / (np.pi/3))
                valid_count += 1
            except Exception as e:
                continue
    for angle in angle_cols:
        df[f'sin_{angle}'] = np.sin(np.deg2rad(df[angle].fillna(0)))
        df[f'cos_{angle}'] = np.cos(np.deg2rad(df[angle].fillna(0)))
    valid_mask = ~(df['head_yaw'].isna() & df['head_pitch'].isna() & df['head_roll'].isna())
    df = df[valid_mask].reset_index(drop=True)
    print(f"Subject {subject_number}: {len(df)} valid samples, {df['gaze_mask'].sum()} with gaze")
    return df

class EnhancedAutoPoseDataset(Dataset):
    """Enhanced dataset with better error handling and data validation"""
    def __init__(self, dataframe, transform=None, is_training=True):
        self.data = dataframe
        self.transform = transform
        self.is_training = is_training
        self._validate_data()
    def _validate_data(self):
        missing_images = []
        for idx, row in self.data.iterrows():
            if not os.path.exists(row['image_path']):
                missing_images.append(idx)
        if missing_images:
            print(f"Removing {len(missing_images)} samples with missing images")
            self.data = self.data.drop(missing_images).reset_index(drop=True)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        try:
            row = self.data.iloc[idx]
            image = Image.open(row['image_path']).convert('RGB')
            head_angles = [
                float(row['head_yaw']),
                float(row['head_pitch']),
                float(row['head_roll']),
                float(row['t_Head_in_Cam_1']),
                float(row['t_Head_in_Cam_2']),
                float(row['t_Head_in_Cam_3']),
            ]
            head_labels = torch.tensor(head_angles, dtype=torch.float32)
            gaze_mask = float(row['gaze_mask'])
            confidence = float(row.get('confidence', 1.0))
            if gaze_mask == 1:
                gaze_cols = ['sin_gaze_yaw', 'cos_gaze_yaw', 'sin_gaze_pitch', 'cos_gaze_pitch']
                gaze_values = row[gaze_cols].values.astype(np.float32)
                gaze_values = np.nan_to_num(gaze_values, nan=0.0)
                gaze_labels = torch.tensor(gaze_values, dtype=torch.float32)
            else:
                gaze_labels = torch.zeros(4, dtype=torch.float32)
            if self.transform:
                image = self.transform(image)
            return image, head_labels, gaze_labels, torch.tensor(gaze_mask), torch.tensor(confidence)
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            dummy_image = torch.zeros(3, 224, 224)
            dummy_head = torch.zeros(6)
            dummy_gaze = torch.zeros(4)
            return dummy_image, dummy_head, dummy_gaze, torch.tensor(0.0), torch.tensor(0.0)
