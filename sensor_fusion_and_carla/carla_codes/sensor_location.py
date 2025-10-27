import carla
import cv2
import numpy as np
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--host', default='localhost', help='CARLA server IP')
parser.add_argument('--port', default=2000, type=int, help='CARLA server port')
args = parser.parse_args()

def main():
    try:
        # Connect to CARLA server
        client = carla.Client(args.host, args.port)
        client.set_timeout(15.0)
        world = client.get_world()
        print("Connected to world:", world.get_map().name)

        # Set synchronous mode
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        blueprint_library = world.get_blueprint_library()

        # Spawn vehicle
        vehicle_bp = blueprint_library.find('vehicle.audi.a2')
        spawn_points = world.get_map().get_spawn_points()
        vehicle = None
        for spawn_point in spawn_points:
            vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
            if vehicle is not None:
                print("Vehicle spawned at:", spawn_point.location)
                break
            else:
                print("Failed to spawn at:", spawn_point.location)

        if vehicle is None:
            print("Failed to spawn vehicle.")
            return

        # List of z positions to test
          
       
            
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '1600')
        camera_bp.set_attribute('image_size_y', '900')
        camera_bp.set_attribute('fov', '75')

        transform = carla.Transform(
                carla.Location(x=1.8, y=0, z=.8),
                carla.Rotation(pitch=0)
            )

        camera = world.spawn_actor(
                camera_bp,
                transform,
                attach_to=vehicle,
                attachment_type=carla.AttachmentType.Rigid
            )
        

            # Image capture
        image_captured = {'done': False, 'img': None}

        def process_img(image):
             if not image_captured['done']:
              array = np.frombuffer(image.raw_data, dtype=np.uint8)
              array = array.reshape((image.height, image.width, 4))  # BGRA format
              array = array[:, :, :3]  # Get only BGR channels
              image_captured['img'] = array.copy()  # Copy to avoid buffer issues
              image_captured['done'] = True
              

        camera.listen(process_img)
        

            # Wait for image
        start_time = time.time()
        while not image_captured['done'] and time.time() - start_time < 60:
                world.tick()
                time.sleep(0.1)

        camera.stop()

        if image_captured['done']:
                
                cv2.imshow(f'View ', image_captured['img'])
                cv2.imwrite(f'bev_z_.png', image_captured['img'])
                print(f"Saved image .png")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
                print(f"Timeout")

            # Clean up camera
        if camera.is_alive:
                camera.destroy()

    finally:
        # Revert to asynchronous mode
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        # Clean up
        if 'camera' in locals() and camera.is_alive:
            camera.destroy()
        if 'vehicle' in locals() and vehicle.is_alive:
            vehicle.destroy()

if __name__ == "__main__":
    main()