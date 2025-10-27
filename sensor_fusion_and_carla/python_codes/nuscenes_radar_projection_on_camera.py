from nuscenes.nuscenes import NuScenes

# === USER INPUT ===
version = 'v1.0-mini'  # Change if using trainval
dataroot = r'D:\project\v1.0-mini'  # <-- Replace with your NuScenes path
sample_token = 'c4f9b75136384ec4a76c4adfc28b4259'  # <-- Replace with your sample token
output_path = r'D:\project\calebration_m_2\python_data_prepare\image_pointcloud_ex\cam_front_with_radar.jpg'  # Desired output path

# === INIT ===
nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)

# === GET CAM_FRONT TOKEN ===
sample = nusc.get('sample', sample_token)
cam_front_token = sample['data']['CAM_FRONT']

# === RENDER CAM_FRONT WITH RADAR ONLY ===
nusc.render_pointcloud_in_image(sample_token, pointsensor_channel='RADAR_FRONT', out_path=output_path)
print(f"Image saved at: {output_path}")
