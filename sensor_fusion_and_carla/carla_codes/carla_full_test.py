#====================MODEL_PART=============#
import onnxruntime as ort
import torchvision.transforms as T
import torch
import cv2

# === ONNX model loading ===
so = ort.SessionOptions()
so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# Use CUDA if you have a compatible GPU
onnx_model = ort.InferenceSession(r"D:\project\carla\model.onnx", so, providers=["CUDAExecutionProvider"])

# If you want to use CPU only, comment the line above and use this one instead:
# onnx_model = ort.InferenceSession("model.onnx", so, providers=["CPUExecutionProvider"])

# === List of class names for decoding the model's output ===
clss = [
    "barrier", "bicycle", "bus", "car", "construction_vehicle",
    "motorcycle", "pedestrian", "traffic_cone", "trailer", "truck"
]

# === Inference function: takes an image and returns model predictions ===
def infer(model, image):
    # Get original image size
    w, h = image.size
    orig_size = torch.tensor([w, h])[None]  # Shape: (1, 2)

    # Preprocessing: resize and convert to tensor (match model training)
    transforms = T.Compose([
        T.Resize((640, 640)),  # Resize to the model's expected input
        T.ToTensor(),
    ])
    im_data = transforms(image)[None]  # Add batch dimension: shape (1, 3, 640, 640)

    # Run the ONNX model
    output = model.run(
        output_names=None,
        input_feed={
            "images": im_data.data.numpy(),            # Input image tensor
            "orig_target_sizes": orig_size.data.numpy(),  # Original width and height
        },
    )

    # Unpack model outputs
    labels, boxes, scores = output
    return labels, boxes, scores

######################################################

#!/usr/bin/env python

# Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
radar projection on RGB camera example
"""

import glob
import os
import sys
import random

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import argparse
from queue import Queue
from queue import Empty
from matplotlib import cm

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    from PIL import Image
except ImportError:
    raise RuntimeError('cannot import PIL, make sure "Pillow" package is installed')

VIRIDIS = np.array(cm.get_cmap('viridis').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])

def sensor_callback(data, queue):
    """
    This simple callback just stores the data on a thread safe Python Queue
    to be retrieved from the "main thread".
    """
    queue.put(data)
def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)
    if generation.lower() == "all":
        return bps
    try:
        int_generation = int(generation)
        return [x for x in bps if int(x.get_attribute('generation')) == int_generation]
    except:
        return []


def spawn_traffic(client, world, tm_port=8000, num_vehicles=30, num_walkers=30, seed=0):
    traffic_manager = client.get_trafficmanager(tm_port)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_hybrid_physics_mode(True)
    traffic_manager.set_hybrid_physics_radius(50.0)
    traffic_manager.set_random_device_seed(seed)

    blueprints = get_actor_blueprints(world, 'vehicle.*', 'All')
    walkers_bp = get_actor_blueprints(world, 'walker.pedestrian.*', '2')

    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    vehicles_list = []
    batch = []
    for i in range(min(num_vehicles, len(spawn_points))):
        bp = random.choice(blueprints)
        if bp.has_attribute('color'):
            bp.set_attribute('color', random.choice(bp.get_attribute('color').recommended_values))
        bp.set_attribute('role_name', 'autopilot')
        batch.append(carla.command.SpawnActor(bp, spawn_points[i])
                     .then(carla.command.SetAutopilot(carla.command.FutureActor, True, tm_port)))
    for response in client.apply_batch_sync(batch, True):
        if not response.error:
            vehicles_list.append(response.actor_id)

    walkers_list = []
    spawn_points = []
    for _ in range(num_walkers):
        loc = world.get_random_location_from_navigation()
        if loc:
            spawn_points.append(carla.Transform(loc))

    walker_speeds = []
    batch = []
    for spawn_point in spawn_points:
        walker_bp = random.choice(walkers_bp)
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        if walker_bp.has_attribute('speed'):
            walker_speeds.append(float(walker_bp.get_attribute('speed').recommended_values[1]))
        else:
            walker_speeds.append(0.0)
        batch.append(carla.command.SpawnActor(walker_bp, spawn_point))

    results = client.apply_batch_sync(batch, True)
    walker_ids = []
    for i, res in enumerate(results):
        if not res.error:
            walkers_list.append({'id': res.actor_id})
            walker_ids.append(res.actor_id)

    controller_bp = world.get_blueprint_library().find('controller.ai.walker')
    batch = [carla.command.SpawnActor(controller_bp, carla.Transform(), w['id']) for w in walkers_list]
    results = client.apply_batch_sync(batch, True)
    controller_ids = []
    for i, res in enumerate(results):
        if not res.error:
            walkers_list[i]['controller'] = res.actor_id
            controller_ids.append(res.actor_id)

    all_id = walker_ids + controller_ids
    all_actors = world.get_actors(all_id)

    world.set_pedestrians_cross_factor(0.0)
    for i in range(0, len(all_id), 2):
        controller = all_actors.find(controller_ids[int(i / 2)])
        walker = all_actors.find(walker_ids[int(i / 2)])
        controller.start()
        controller.go_to_location(world.get_random_location_from_navigation())
        controller.set_max_speed(float(walker_speeds[int(i / 2)]))

    print(f"[✓] Spawned {len(vehicles_list)} vehicles and {len(walkers_list)} walkers")
    return vehicles_list, walkers_list, all_id


def tutorial(args):
    """
    This function is intended to be a tutorial on how to retrieve data in a
    synchronous way, project 3D points from radar to 2D camera, and spawn traffic.
    """
    # Connect to the server
    client = carla.Client(args.host, args.port)
    client.set_timeout(2000)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    # Apply synchronous mode and fixed delta seconds
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05  # Adjust to 0.05 for smoother simulation
    world.apply_settings(settings)

    # Hide unwanted map layers for less visual clutter
    world.unload_map_layer(carla.MapLayer.Buildings)
    world.unload_map_layer(carla.MapLayer.Foliage)
    world.unload_map_layer(carla.MapLayer.Particles)
    world.unload_map_layer(carla.MapLayer.Props)
    world.unload_map_layer(carla.MapLayer.StreetLights)
    world.unload_map_layer(carla.MapLayer.Walls)

    # Setup Traffic Manager
    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)
    traffic_manager.set_hybrid_physics_mode(True)
    traffic_manager.set_hybrid_physics_radius(50.0)
    traffic_manager.set_random_device_seed(0)

    # Spawn traffic (vehicles + pedestrians)
    vehicles_list, walkers_list, all_id = spawn_traffic(client, world, tm_port=8000, num_vehicles=40, num_walkers=40, seed=0)

    vehicle = None
    camera = None
    radar = None
    top_camera = None
    try:
        if not os.path.isdir('_out'):
            os.mkdir('_out')
        # Search the desired blueprints
        vehicle_bp = bp_lib.filter("vehicle.audi.a2")[0]
        camera_bp = bp_lib.filter("sensor.camera.rgb")[0]
        radar_bp = bp_lib.filter("sensor.other.radar")[0]


        # Configure the blueprints
        camera_bp.set_attribute("image_size_x", str(args.width))
        camera_bp.set_attribute("image_size_y", str(args.height))
        camera_bp.set_attribute("fov", str(args.fov))
        

        radar_bp = bp_lib.filter("sensor.other.radar")[0]
        radar_bp.set_attribute('horizontal_fov', '80')
        radar_bp.set_attribute('vertical_fov', '0')
        radar_bp.set_attribute('range', str(args.range))
        radar_bp.set_attribute('points_per_second', str(args.points_per_second))
        # Spawn the blueprints
        
        spawn_point = world.get_map().get_spawn_points()[25]

        vehicle = world.spawn_actor(
           blueprint=vehicle_bp,
           transform=spawn_point)
        vehicle.set_autopilot(False)
        camera = world.spawn_actor(
            blueprint=camera_bp,
            transform=carla.Transform(carla.Location(x=0.55, z=1.35)),
            attach_to=vehicle)
        radar = world.spawn_actor(
         blueprint=radar_bp,
         transform=carla.Transform(carla.Location(x=1.8, z=0.8)),
         attach_to=vehicle)
        top_camera_bp = bp_lib.find("sensor.camera.rgb")
        top_camera_bp.set_attribute("image_size_x", str(args.width))
        top_camera_bp.set_attribute("image_size_y", str(args.height))
        top_camera_bp.set_attribute("fov", "70") 
        vehicle_transform = vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        top_camera_transform = carla.Transform(
         carla.Location(x=-8.0, y=0.0, z=7.0), 
         carla.Rotation(pitch=-20, yaw=0, roll=0) 
              )

          

        top_camera = world.spawn_actor(
          blueprint=top_camera_bp,
          transform=top_camera_transform,
          attach_to=vehicle
           )
        


        # Build the K projection matrix:
        # K = [[Fx,  0, image_w/2],
        #      [ 0, Fy, image_h/2],
        #      [ 0,  0,         1]]
        image_w = camera_bp.get_attribute("image_size_x").as_int()
        image_h = camera_bp.get_attribute("image_size_y").as_int()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v', )  
        video_path = "_out/output_video.mp4"     
        video_path_top = "_out/top_camera_video.mp4"
        video_writer = cv2.VideoWriter(video_path, fourcc, 20, (image_w, image_h))
        video_writer_top = cv2.VideoWriter(video_path_top, fourcc, 20, (image_w, image_h))

        fov = camera_bp.get_attribute("fov").as_float()
        focal = image_w / (2.0 * np.tan(fov * np.pi / 360.0))

        # In this case Fx and Fy are the same since the pixel aspect
        # ratio is 1
        K = np.identity(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = image_w / 2.0
        K[1, 2] = image_h / 2.0

        # The sensor data will be saved in thread-safe Queues
        image_queue = Queue()
        radar_queue = Queue()
        top_camera_queue = Queue()

        camera.listen(lambda data: sensor_callback(data, image_queue))
        radar.listen(lambda data: sensor_callback(data, radar_queue))
        top_camera.listen(lambda data: sensor_callback(data, top_camera_queue))
        # === Custom transformation assuming radar is placed on the ground ===
        radar_ground_transform = carla.Transform(
          location=carla.Location(x=1.8, y=0.0, z=0.0),  # Radar at ground level
          rotation=carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
           )

        camera_transform = carla.Transform(
          location=carla.Location(x=0.55, y=0.0, z=1.35),
          rotation=carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
          )

        T_radar = np.array(radar_ground_transform.get_matrix())
        T_camera = np.array(camera_transform.get_matrix())

        # Transform radar points to camera frame (camera ← radar)
        T_camera_radar = np.dot(np.linalg.inv(T_camera), T_radar)
 
        emergency_stop = False   

        for frame in range(args.frames):
            world.tick()
            control = carla.VehicleControl()

            if emergency_stop:
             control.throttle = 0.0
             control.brake = 1.0
            else:
             control.throttle = 1.0
             control.brake = 0.0
             control.steer = 0.0
            vehicle.apply_control(control)
            world_frame = world.get_snapshot().frame

            try:
                # Get the data once it's received.
                image_data = image_queue.get(True, 1.0)
                radar_data = radar_queue.get(True, 1.0)
                top_camera_data = top_camera_queue.get(True, 1.0)
                top_im_array = np.frombuffer(top_camera_data.raw_data, dtype=np.uint8)
                top_im_array = top_im_array.reshape((top_camera_data.height, top_camera_data.width, 4))
                top_im_bgr = top_im_array[:, :, :3] # RGB -> BGR
                video_writer_top.write(top_im_bgr)
            except Empty:
                print("[Warning] Some sensor data has been missed")
                continue

           # assert image_data.frame == radar_data.frame == world_frame
            # At this point, we have the synchronized information from the 2 sensors.
            sys.stdout.write("\r(%d/%d) Simulation: %d Camera: %d radar: %d" %
                (frame, args.frames, world_frame, image_data.frame, radar_data.frame) + ' ')
            sys.stdout.flush()

            # Get the raw BGRA buffer and convert it to an array of RGB of
            # shape (image_data.height, image_data.width, 3).
            im_array = np.copy(np.frombuffer(image_data.raw_data, dtype=np.dtype("uint8")))
            im_array = np.reshape(im_array, (image_data.height, image_data.width, 4))
            im_array = im_array[:, :, :3][:, :, ::-1]

            # Get the radar data and convert it to a numpy array.
            points = np.frombuffer(radar_data.raw_data, dtype=np.float32)
            points = points.reshape((len(radar_data), 4))
            velocity = points[:, 0]       
            azimuth = points[:, 1]        
            altitude = points[:, 2]       
            depth = points[:, 3]          


            x = depth * np.cos(altitude) * np.cos(azimuth)
            y = depth * np.cos(altitude) * np.sin(azimuth)
            z = depth * np.sin(altitude)
            

            local_radar_points = np.vstack((x, y, z, np.ones_like(x)))

            
            sensor_points = np.dot(T_camera_radar, local_radar_points)
 

            # radar velocity array of shape (p_cloud_size,) but, for now, let's
            # focus on the 3D points.
            
              

            # Add an extra 1.0 at the end of each 3d point so it becomes of
            # shape (4, p_cloud_size) and it can be multiplied by a (4, 4) matrix.
            

            # This (4, 4) matrix transforms the points from radar space to world space.
            
            # New we must change from UE4's coordinate system to an "standard"
            # camera coordinate system (the same used by OpenCV):

            # ^ z                       . z
            # |                        /
            # |              to:      +-------> x
            # | . x                   |
            # |/                      |
            # +-------> y             v y

            # This can be achieved by multiplying by the following matrix:
            # [[ 0,  1,  0 ],
            #  [ 0,  0, -1 ],
            #  [ 1,  0,  0 ]]

            # Or, in this case, is the same as swapping:
            # (x, y ,z) -> (y, -z, x)
            point_in_camera_coords = np.array([
              sensor_points[1],
              sensor_points[2] * -1,
              sensor_points[0]])


            # Finally we can use our K matrix to do the actual 3D -> 2D.
            points_2d = np.dot(K, point_in_camera_coords)

            # Remember to normalize the x, y values by the 3rd value.
            points_2d = np.array([
                points_2d[0, :] / points_2d[2, :],
                points_2d[1, :] / points_2d[2, :],
                points_2d[2, :]])

            # At this point, points_2d[0, :] contains all the x and points_2d[1, :]
            # contains all the y values of our points. In order to properly
            # visualize everything on a screen, the points that are out of the screen
            # must be discarted, the same with points behind the camera projection plane.
            points_2d = points_2d.T
            velocity = velocity.T
            points_in_canvas_mask = \
                (points_2d[:, 0] > 0.0) & (points_2d[:, 0] < image_w) & \
                (points_2d[:, 1] > 0.0) & (points_2d[:, 1] < image_h) & \
                (points_2d[:, 2] > 0.0)
            points_2d = points_2d[points_in_canvas_mask]
            velocity = velocity[points_in_canvas_mask]

            # Extract the screen coords (uv) as integers.
            u_coord = points_2d[:, 0].astype(int)
            v_coord = points_2d[:, 1].astype(int)

            # Since at the time of the creation of this script, the velocity function
            # is returning high values, these are adjusted to be nicely visualized.
            

           

            # Save the image using Pillow module.
            image = Image.fromarray(im_array)
            # === Convert image to PIL
            image_pil = Image.fromarray(im_array)
            im_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
            # === Apply ONNX object detection inference
            labels, boxes, scores = infer(onnx_model, image_pil)

            # === Convert to OpenCV BGR image for drawing
            im_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

            # === Draw boxes from ONNX model
           
            for j in range(len(scores[0])):
             if scores[0][j] > 0.6:
              b = boxes[0][j]
              lab = labels[0][j]
              scr = scores[0][j]
              top_left = (int(b[0]), int(b[1]))
              bottom_right = (int(b[2]), int(b[3]))
              cv2.rectangle(im_cv, top_left, bottom_right, (0, 0, 255), 2)
              # text = f"{clss[lab]} {round(scr, 2)}"
              text = f"{clss[lab]}"
              cv2.putText(im_cv, text, (int(b[0]), int(b[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

             # === Find the closest radar point to the bottom edge of the box
              u1 = int(b[0])
              v1 = int(b[3])
              u2 = int(b[2])
              v2 = int(b[3])

              bottom_u_min = min(u1, u2)
              bottom_u_max = max(u1, u2)
              
              bottom_v = int((v1 + v2) / 2)

              tolerance = 20  # pixels allowed above or below the line

              # Filter radar points within this bottom edge range
              candidates = []
              for i in range(len(u_coord)):
               u = u_coord[i]
               v = v_coord[i]
               if bottom_u_min <= u <= bottom_u_max and abs(v - bottom_v) <= tolerance:
                candidates.append((u, v, i))
            
             idx = None
             if candidates:
             # Get candidate with smallest v (i.e. closest to top of image)
              closest_point = min(candidates, key=lambda x: x[1])
              u_closest, v_closest, idx = closest_point
 
             # Draw that point in yellow
              cv2.circle(im_cv, (u_closest, v_closest), radius=4, color=(0, 255, 255), thickness=-1)
              if idx is not None and idx < len(velocity):
                vx = velocity[idx]
                px_real = x[idx]
                py_real = y[idx]
                if (vx < 0) and (-1.0 < py_real < 1.0):
                    ttc = px_real / abs(vx) if vx != 0 else float('inf')
                    if ttc < 8:
                        cv2.putText(im_cv, "Warning", (int(b[0]), int(b[1]) + 20),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        cv2.putText(im_cv, clss[lab], (int(b[0]), int(b[1]) + 45),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                        obj_class_name = clss[labels[0][j]]
                        cv2.putText(im_cv, obj_class_name, (u_closest, v_closest - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                    if ttc < 2:
                     emergency_stop = True
    
            video_writer.write(im_cv) 
            # cv2.imwrite(f"_out/detected_{image_data.frame:08d}.png", im_cv)
        video_writer.release()
        video_writer_top.release()

    finally:
        # Apply the original settings when exiting.
        world.apply_settings(original_settings)

        # Destroy the actors in the scene.
        if camera:
            camera.destroy()
        if radar:
            radar.destroy()
        if vehicle:
            vehicle.destroy()
        if top_camera:
         top_camera.destroy()


def main():
    """Start function"""
    argparser = argparse.ArgumentParser(
        description='CARLA Sensor sync and projection tutorial')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='680x420',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--fov',
        metavar='F',
        default=70.0,
        type=float,
        help='Field of view for the camera (default: 90 degrees)')
    
    argparser.add_argument(
        '-f', '--frames',
        metavar='N',
        default=150,
        type=int,
        help='number of frames to record (default: 500)')
    argparser.add_argument(
        '-d', '--dot-extent',
        metavar='SIZE',
        default=3,
        type=int,
        help='visualization dot extent in pixels (Recomended [1-4]) (default: 2)')
    argparser.add_argument(
        '--no-noise',
        action='store_true',
        help='remove the drop off and noise from the normal (non-semantic) radar')
    argparser.add_argument(
        '--upper-fov',
        metavar='F',
        default=30.0,
        type=float,
        help='radar\'s upper field of view in degrees (default: 15.0)')
    argparser.add_argument(
        '--lower-fov',
        metavar='F',
        default=-25.0,
        type=float,
        help='radar\'s lower field of view in degrees (default: -25.0)')
    argparser.add_argument(
        '-c', '--channels',
        metavar='C',
        default=64.0,
        type=float,
        help='radar\'s channel count (default: 64)')
    argparser.add_argument(
        '-r', '--range',
        metavar='R',
        default=100.0,
        type=float,
        help='radar\'s maximum range in meters (default: 100.0)')
    argparser.add_argument(
        '--points-per-second',
        metavar='N',
        default='1000',
        type=int,
        help='radar points per second (default: 1000)')
    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]
    args.dot_extent -= 1

    try:
        tutorial(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
