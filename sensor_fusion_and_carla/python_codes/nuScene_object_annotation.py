from nuscenes.nuscenes import NuScenes

# Load nuScenes
nusc = NuScenes(version='v1.0-mini', dataroot=r'D:\project\v1.0-mini', verbose=True)

# Choose a sample token
sample = nusc.get('sample', 'c4f9b75136384ec4a76c4adfc28b4259')

# Choose a camera (e.g. CAM_FRONT)
cam_token = sample['data']['CAM_FRONT']

# Define the output path where the image will be saved
output_image_path = r'D:\project\calebration_m_2\python_data_prepare\image_pointcloud_ex\annotated_front_camera_image.jpg'

# Render and save the image
nusc.render_sample_data(cam_token, with_anns=True, out_path=output_image_path)
