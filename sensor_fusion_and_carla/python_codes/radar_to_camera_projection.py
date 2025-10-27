import cv2
import numpy as np
import pypcd4

# Function to read radar point cloud from a .pcd file and filter valid points
def read_pcd_points(pcd_path):
    pcd = pypcd4.PointCloud.from_path(pcd_path)

    # Extract required fields
    x = pcd.pc_data['x']
    y = pcd.pc_data['y']
    z = pcd.pc_data['z']
    rcs = pcd.pc_data['rcs']
    ambig_state = pcd.pc_data['ambig_state']
    invalid_state = pcd.pc_data['invalid_state']
    is_quality_valid = pcd.pc_data['is_quality_valid']
    dyn_prop = pcd.pc_data['dyn_prop']
    # Create a mask for valid radar points
    valid_mask = (
        (is_quality_valid == 1) &
        (invalid_state == 0)  )

       # (rcs > -1) &
        #(ambig_state == 3)  # unambiguous Doppler
   
    # Return only valid points as [N, 3] array
    valid_points = np.stack([x, y, z], axis=-1)[valid_mask]
    return valid_points

# Function to project radar points to image plane using camera intrinsics and extrinsics
def project_points_to_image(points_3d, T_radar_to_camera, K_camera):
    
    # Add homogeneous coordinate (1) to each point
    points_hom = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])  # [N, 4]

    # Apply transformation: radar -> camera coordinate frame
    points_cam = (T_radar_to_camera @ points_hom.T).T[:, :3]  # [N, 3]

    # Keep only points in front of the camera (z > 0)
    valid = points_cam[:, 2] > 0
    points_cam = points_cam[valid]

    # Save original x, y (from radar) of valid points
    radar_xy = points_3d[valid][:, :2]  # [N, 2]

    # Normalize points by depth (z) before projection
    points_norm = points_cam / points_cam[:, 2:3]

    # Project to 2D image using camera intrinsics
    image_points = (K_camera @ points_norm.T).T  # [N, 3]

    return image_points[:, :2], radar_xy  # return (u,v) and (x,y)

# Function to draw projected radar points and their original x,y on the image
def draw_points_on_image(image, image_points, radar_xy):
    for (u, v), (x, y) in zip(image_points, radar_xy):
        u, v = int(u), int(v)
        if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
            # Draw a green circle on the projected point
            cv2.circle(image, (u, v), 2, (0, 255, 0), -1)

            # Draw text with radar-based (x, y) coordinates in red
            cv2.putText(image, f"{x:.1f},{y:.1f}", (u + 5, v - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 255), 1)

    return image

# --- CONFIGURATION SECTION ---

# Input paths
image_path = r"D:\project\calebration_m_2\python_data_prepare\image_pointcloud_ex\front_camera_c4f9b75136384ec4a76c4adfc28b4259.jpg"  # <-- Change to your image path
pcd_path = r"D:\project\calebration_m_2\python_data_prepare\image_pointcloud_ex\radar_pointcloud_c4f9b75136384ec4a76c4adfc28b4259.pcd"  # Replace with your PCD file path

# Transformation matrix from radar to camera (extrinsic)
T_radar_to_camera = np.array([
    [0.0148, -0.9998, -0.0122, 0.0343],
    [0.0084, 0.0124, -0.9999, 1.0090],
    [0.9999, 0.0147, 0.0086, 1.6813],      
])

# Camera intrinsic matrix
K_camera = np.array([
    [1252.813, 0, 826.588],
    [0, 1252.813, 469.984],
    [0, 0, 1]
])

# --- MAIN EXECUTION ---

# Load camera image
image = cv2.imread(image_path)

# Read and filter radar points
radar_points = read_pcd_points(pcd_path)

# Project radar points to image space and get real radar x,y
uv_points, radar_xy = project_points_to_image(radar_points, T_radar_to_camera, K_camera)

# Draw the radar points and their real coordinates on the image
image_with_points = draw_points_on_image(image.copy(), uv_points, radar_xy)

# Show the final image
cv2.imshow("Radar Points on Image", image_with_points)
cv2.imwrite('D:\project\calebration_m_2\python_data_prepare\image_pointcloud_ex\output_image.jpg', image_with_points)
cv2.waitKey(0)
cv2.destroyAllWindows()
