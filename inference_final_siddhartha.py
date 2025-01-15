import pyrealsense2 as rs
import numpy as np
import cv2
import os
import pyrealsense2 as rs
import numpy as np
import cv2
import os
from pathlib import Path
from typing import Tuple, Optional, List
from gemini_oop_object_detection import *
# from model_inference import  predict_and_save_trajectory
from inference import predict_trajectory

# Global transformation matrices - Replace these with your actual matrices
X = np.array([
      [ 0.068, -0.986,  0.152, -0.108],
      [ 0.998,  0.065, -0.023,  0.0 ],
      [ 0.013,  0.153,  0.988, -0.044],
      [ 0.0,    0.0,    0.0,    1.0  ]
    ])  # Camera to calibration target transformation
Y = np.array([
      [-0.47,   0.587,  -0.659,  0.73929],
      [ 0.877,  0.392,  -0.276, -0.16997],
      [ 0.096, -0.708,  -0.7,    0.86356],
      [ 0.0,    0.0,     0.0,    1.0    ]
    ])  # Calibration target to robot base transformation

def create_intrinsics():
    """Create RealSense intrinsics object with predefined values."""
    intrinsics = rs.intrinsics()
    intrinsics.width = 640
    intrinsics.height = 480
    intrinsics.ppx = 329.1317443847656
    intrinsics.ppy = 240.29669189453125
    intrinsics.fx = 611.0845947265625
    intrinsics.fy = 609.7639770507812
    intrinsics.model = rs.distortion.brown_conrady
    intrinsics.coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]
    return intrinsics

def get_valid_depth(depth_array, x, y):
    """Find the first non-zero depth value within a 10-pixel radius around the given point."""
    height, width = depth_array.shape
    if depth_array[y, x] > 0:
        return depth_array[y, x], x, y

    max_radius = 10
    for radius in range(1, max_radius + 1):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                new_x = x + dx
                new_y = y + dy
                if 0 <= new_x < width and 0 <= new_y < height:
                    depth = depth_array[new_y, new_x]
                    if depth > 0:
                        return depth, new_x, new_y

    return 0, x, y

def deproject_pixel_to_point(depth_array, pixel_coords, intrinsics):
    """Deproject pixel coordinates and depth to 3D point using RealSense intrinsics."""
    x, y = int(pixel_coords[0]), int(pixel_coords[1])
    if x < 0 or x >= depth_array.shape[1] or y < 0 or y >= depth_array.shape[0]:
        return np.array([0, 0, 0])
    
    depth, valid_x, valid_y = get_valid_depth(depth_array, x, y)
    if depth == 0:
        print(f"Warning: No valid depth found near pixel ({x}, {y})")
        return np.array([0, 0, 0])
    
    point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [valid_x, valid_y], depth)
    return np.array(point_3d)

def transform_coordinates(x, y, z):
    """Transforms coordinates from input space to cobot base."""
    B = np.eye(4)
    B[:3, 3] = [x / 1000, y / 1000, z / 1000]  # Convert to meters
    A = Y @ B @ np.linalg.inv(X)
    transformed_x, transformed_y, transformed_z = A[:3, 3] * 1000  # Convert back to mm
    return transformed_x, transformed_y, transformed_z


def call_gemini_api(rgb_path, depth_path, recording_dir: str, target_classes: list[str, str]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Placeholder for Gemini API call that returns coordinates for two objects."""
        object_detector = ObjectDetector(GEMINI_API_KEY, recording_dir=recording_dir)
        im = Image.open(rgb_path)
        object_1 , object_2 = target_classes
        center_object_1_x, center_object_1_y, _, _ = object_detector.get_object_center(im, object_1)
        center_object_2_x, center_object_2_y, _, _ = object_detector.get_object_center(im, object_2)
        return [center_object_1_x,center_object_1_y], [center_object_2_x, center_object_2_y]

def process_coordinates(pixel_coords: Tuple[float, float], depth_array: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """Process pixel coordinates to get real-world coordinates."""
        # Get 3D point in camera coordinates
        intrinsics = create_intrinsics()
        point_3d = deproject_pixel_to_point(depth_array, pixel_coords, intrinsics)
        
        # Transform to robot base coordinates if valid point found
        if not np.all(point_3d == 0):
            # Get real-world coordinates
            real_coords = transform_coordinates(point_3d[0], point_3d[1], point_3d[2])
            return real_coords
        return None

def main():
    # Create output directory if it doesn't exist
    if not os.path.exists('captured'):
        os.makedirs('captured')

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable color and depth streams
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # Start streaming
    pipeline.start(config)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            cv2.imshow('RealSense', color_image)
            
            key = cv2.waitKey(1)
            if key & 0xFF == ord('s'):
                cv2.imwrite('captured/captured_rgb.jpg', color_image)
                
                np.save('captured/captured_depth.npy', depth_image)

                print("Images saved successfully!")
                dir = "captured"
                rgb_path = f"{dir}/captured_rgb.jpg"
                depth_path = f"{dir}/captured_depth.npy"
                target_classes = ["soda can", "white mug"]
                
                coords1, coords2 = call_gemini_api(rgb_path,depth_path,recording_dir=dir,target_classes=target_classes)
                x1,y1,z1 = process_coordinates(coords1, depth_image)
                x2,y2,z2 = process_coordinates(coords2, depth_image)
                container_positions = [[
                    x1,y1,z1,x2,y2,z2
                ]]
                csv_path = predict_trajectory("pouring_trajectory_model.pth",container_positions)
                print(csv_path)
                
            # Break loop with 'q'
            elif key & 0xFF == ord('q'):
                break
                
    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()