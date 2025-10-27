import os
import json
import numpy as np
from nuscenes.nuscenes import NuScenes
from PIL import Image
import shutil
try:
    from pypcd import pypcd
except ImportError:
    pypcd = None
    print("Warning: pypcd not installed. Use 'pip install pypcd' to save as .pcd.")

def download_sample_data(nusc, sample_token, output_dir, save_pcd=False, copy_original_pcd=False):
    """
    Download front camera image, radar point cloud, and calibration data for a given sample token.
    
    Args:
        nusc: NuScenes object instance
        sample_token: str, unique sample token
        output_dir: str, path to save the downloaded files
        save_pcd: bool, if True, save radar point cloud as .pcd (requires pypcd)
        copy_original_pcd: bool, if True, copy the original .pcd file directly
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the sample record
    try:
        sample = nusc.get('sample', sample_token)
    except KeyError:
        print(f"Error: Sample token {sample_token} not found in dataset.")
        return
    
    # --- Front Camera Data ---
    try:
        cam_front_token = sample['data']['CAM_FRONT']
        cam_front_data = nusc.get('sample_data', cam_front_token)
        
        # Copy the front camera image
        cam_image_path = os.path.join(nusc.dataroot, cam_front_data['filename'])
        cam_output_path = os.path.join(output_dir, f"front_camera_{sample_token}.jpg")
        shutil.copyfile(cam_image_path, cam_output_path)
        print(f"Front camera image saved to {cam_output_path}")
        
        # Save front camera calibration
        cam_calibrated_sensor = nusc.get('calibrated_sensor', cam_front_data['calibrated_sensor_token'])
        cam_calib_output_path = os.path.join(output_dir, f"front_camera_calibration_{sample_token}.json")
        with open(cam_calib_output_path, 'w') as f:
            json.dump(cam_calibrated_sensor, f, indent=4)
        print(f"Front camera calibration saved to {cam_calib_output_path}")
    except Exception as e:
        print(f"Error processing front camera data: {e}")
    
    # --- Radar Point Cloud Data ---
    try:
        radar_front_token = sample['data']['RADAR_FRONT']
        radar_front_data = nusc.get('sample_data', radar_front_token)
        
        # Load radar point cloud
        radar_pcd_path = os.path.join(nusc.dataroot, radar_front_data['filename'])
        if not os.path.exists(radar_pcd_path):
            print(f"Error: Radar file {radar_pcd_path} does not exist.")
            return
        
        # Option 1: Copy original .pcd file
        if copy_original_pcd:
            radar_output_path = os.path.join(output_dir, f"radar_pointcloud_{sample_token}.pcd")
            shutil.copyfile(radar_pcd_path, radar_output_path)
            print(f"Original radar point cloud copied to {radar_output_path}")
            return
        
        # Load raw data
        radar_data = np.fromfile(radar_pcd_path, dtype=np.float32)
        print(f"Radar file {radar_pcd_path}: {radar_data.size} float32 elements")
        
        # nuScenes radar points have 18 fields
        expected_fields = 18
        if radar_data.size % expected_fields != 0:
            print(f"Warning: Radar data size {radar_data.size} is not divisible by {expected_fields}. Saving empty array.")
            radar_points = np.array([])  # Save empty array
        else:
            radar_points = radar_data.reshape(-1, expected_fields)
            print(f"Loaded {radar_points.shape[0]} radar points with {expected_fields} fields each.")
        
        # Option 2: Save as .pcd (requires pypcd)
        if save_pcd and pypcd is not None:
            radar_output_path = os.path.join(output_dir, f"radar_pointcloud_{sample_token}.pcd")
            # Define fields for nuScenes radar data
            fields = ['x', 'y', 'z', 'dyn_prop', 'id', 'rcs', 'vx', 'vy', 'vx_comp', 'vy_comp',
                      'is_quality_valid', 'ambig_state', 'x_rms', 'y_rms', 'invalid_state', 'pdh0', 'vx_rms', 'vy_rms']
            if radar_points.size > 0:
                pc_dict = {f: radar_points[:, i] for i, f in enumerate(fields)}
                pc = pypcd.PointCloud.from_dict(pc_dict, np.float32)
                pc.save(radar_output_path)
                print(f"Radar point cloud saved as PCD to {radar_output_path}")
            else:
                print(f"Skipping PCD save: No valid radar points.")
        # Option 3: Save as .npy
        else:
            radar_output_path = os.path.join(output_dir, f"radar_pointcloud_{sample_token}.npy")
            np.save(radar_output_path, radar_points)
            print(f"Radar point cloud saved as NPY to {radar_output_path}")
        
        # Save radar calibration
        radar_calibrated_sensor = nusc.get('calibrated_sensor', radar_front_data['calibrated_sensor_token'])
        radar_calib_output_path = os.path.join(output_dir, f"radar_calibration_{sample_token}.json")
        with open(radar_calib_output_path, 'w') as f:
            json.dump(radar_calibrated_sensor, f, indent=4)
        print(f"Radar calibration saved to {radar_calib_output_path}")
    except Exception as e:
        print(f"Error processing radar data: {e}")

def main():
    # Initialize nuScenes dataset
    dataroot = r'D:\project\v1.0-mini'  # Update this to your nuScenes dataset path
    version = 'v1.0-mini'  # Update to your dataset version
    try:
        nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
    except Exception as e:
        print(f"Error initializing nuScenes dataset: {e}")
        return
    
    # Sample token
    sample_token = 'c4f9b75136384ec4a76c4adfc28b4259'
    output_dir = r'D:\project\calebration_m_2\python_data_prepare\image_pointcloud_ex'  # Update if needed
    
    # Download sample data (set save_pcd=True to save as .pcd, or copy_original_pcd=True to copy original)
    download_sample_data(nusc, sample_token, output_dir, save_pcd=True, copy_original_pcd=True)

if __name__ == '__main__':
    main()