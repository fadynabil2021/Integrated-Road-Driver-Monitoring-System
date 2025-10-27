function radar_to_camera_projection()
    %% --- Calibration Data ---

    % Camera rotation (quaternion) and translation (world to camera)
    q_cam = [0.5077241387638071, -0.4973392230703816, 0.49837167536166627, -0.4964832014373754];
    t_cam = [1.72200568478; 0.00475453292289; 1.49491291905];

    % Radar rotation (quaternion) and translation (world to radar)
    q_radar = [0.9999974259839071,0.0,0.0,-0.0022689260808398757];
    t_radar = [3.412,0.0,0.5];
    % Ensure column vectors
    t_cam = t_cam(:);
    t_radar = t_radar(:);

    % Camera intrinsic matrix
    K = [1252.8131021185304, 0, 826.588114781398;
         0, 1252.8131021185304, 469.9846626224581;
         0, 0, 1];

    %% --- Convert Quaternions to Rotation Matrices ---
    R_cam = quaternionToRotationMatrix(q_cam);
    R_radar = quaternionToRotationMatrix(q_radar);

    %% --- Build 4x4 Homogeneous Transformation Matrices ---
    T_cam = [R_cam, t_cam; 0 0 0 1];        % World to camera
    T_radar = [R_radar, t_radar; 0 0 0 1];  % World to radar

    % Radar to camera transformation: T_radar_to_cam = T_cam⁻¹ * T_radar
    T_radar_to_cam = inv(T_cam) * T_radar 

    %% --- Load Image and Radar Point Cloud ---
    img = imread('camera2.jpg'); % Change as needed
    ptCloud = pcread('radar2.pcd'); % Change as needed
    xyz = ptCloud.Location; % Radar points as Nx3 matrix
    xyz(:, 3) = 0;
    %% --- Transform Radar Points to Camera Frame ---
    nPts = size(xyz, 1);                          % Number of radar points
    pts_radar_h = [xyz'; ones(1, nPts)];          % Convert to homogeneous coordinates
    pts_cam_h = T_radar_to_cam * pts_radar_h;     % Apply transformation
    pts_cam = pts_cam_h(1:3, :);                  % Drop homogeneous coordinate

    %% --- Project to Image Plane using Camera Intrinsics ---
    pts_img_h = K * pts_cam;                      % Project 3D points to 2D homogeneous
    pts_img = pts_img_h(1:2, :) ./ pts_img_h(3, :); % Normalize to get pixel coordinates

    %% --- Display Image with Projected Radar Points ---
    figure;
    imshow(img);
    hold on;
    title('Radar Points Projected on Camera Image');

    % Draw projected points as red dots
    scatter(pts_img(1, :), pts_img(2, :), 10, 'r', 'filled');

    % Annotate each point with its original 3D radar coordinates
    for i = 1:nPts
        x_img = pts_img(1, i);
        y_img = pts_img(2, i);
        text(x_img, y_img, sprintf('(%0.1f,%0.1f,%0.1f)', xyz(i,1), xyz(i,2), xyz(i,3)), ...
            'Color', 'yellow', 'FontSize', 6);
    end

    hold off;
end

function R = quaternionToRotationMatrix(q)
    % Convert quaternion [w x y z] to rotation matrix
    w = q(1); x = q(2); y = q(3); z = q(4);
    R = [1 - 2*(y^2 + z^2),   2*(x*y - z*w),     2*(x*z + y*w);
         2*(x*y + z*w),       1 - 2*(x^2 + z^2), 2*(y*z - x*w);
         2*(x*z - y*w),       2*(y*z + x*w),     1 - 2*(x^2 + y^2)];
end
