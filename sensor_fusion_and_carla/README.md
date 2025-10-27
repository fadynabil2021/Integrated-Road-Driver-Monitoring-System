# Radar-Camera Sensor Fusion in CARLA

## Overview
This project demonstrates **sensor fusion between a radar and an RGB camera** in the CARLA simulator, with object detection using an ONNX model. The main goal is to correctly associate radar points with their corresponding objects detected in the camera view.

---

## Important Notes

1. **Path Configuration**  
   - Please make sure to update all file paths in the Python and MATLAB codes before running.  
   - This includes paths to the ONNX model, saved outputs, and any other local resources.

2. **Radar-to-Body Association**  
   - To determine which radar point belongs to which object:  
     - We assume the radar is located on the **ground** by setting its Z-coordinate in the translation matrix (`Translation Matrix`) to `0` instead of using the radar's actual height.  
     - This modified translation matrix is used to compute the **radar-to-camera transformation**.  
     - As a result, points corresponding to objects will appear directly **below the detected boundary box**, close to the bottom edge of the box.  

3. **MATLAB-to-Python Consistency**  
   - The translation matrix calculated from the MATLAB code should be **directly used in the Python scripts** for correct radar-to-camera projection.  
   - Make sure to maintain the same coordinate conventions and units between MATLAB and Python.

