import pyrealsense2 as rs
import numpy as np
import cv2
import os
from pathlib import Path
from typing import Tuple, Optional, List
from gemini_oop_object_detection import *
from model_inference import  predict_and_save_trajectory

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

class RealSenseInference:
    def __init__(self, class_names: List[str]):
        if len(class_names) != 2:
            raise ValueError("Exactly two class names must be provided")
            
        self.class_names = class_names
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.capture_count = 0
        self.capture_dir = Path("capture")
        self.capture_dir.mkdir(exist_ok=True)
        
        # Configure streams
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        # Start streaming
        self.pipeline.start(self.config)
        self.intrinsics = create_intrinsics()

    def capture_and_save_images(self):
        """Capture and save RGB and depth images."""
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            return None, None, None, None
        
        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # Save images
        rgb_path = self.capture_dir / f"capture_{self.capture_count}.jpg"
        depth_path = self.capture_dir / f"capture_{self.capture_count}.npy"
        
        cv2.imwrite(str(rgb_path), color_image)
        np.save(str(depth_path), depth_image)
        
        return str(rgb_path), str(depth_path), color_image, depth_image

    def call_gemini_api(self, rgb_path, depth_path, recording_dir: str, target_classes: list[str, str]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Placeholder for Gemini API call that returns coordinates for two objects."""
        # rgb_path = f'{recording_dir}/rgb_image/image_0.jpg'
        # depth_path = f'{recording_dir}/depth_image/image_0.npy'
        object_detector = ObjectDetector(GEMINI_API_KEY, recording_dir=recording_dir)
        im = Image.open(rgb_path)
        object_1 , object_2 = target_classes
        # centers_data = object_detector.get_object_centers(im, [object_1, object_2])
        # print(f'TYPE of the detection results is: {type(centers_data)}')
        # print(f'Centers for {object_1} are: {centers_data[object_1][0]}')
        # print(f'Centers for {object_2} are: {centers_data[object_2][0]}')
        # print(f"Calling Gemini API with {rgb_path} and {depth_path}...")
        # print(f"Looking for objects: {self.class_names[0]} and {self.class_names[1]}")
        # Simulated response: (x1, y1) for first class, (x2, y2) for second class
        # return  centers_data[object_1][0], centers_data[object_2][0]# Example coordinates
        center_object_1_x, center_object_1_y, _, _ = object_detector.get_object_center(im, object_1)
        center_object_2_x, center_object_2_y, _, _ = object_detector.get_object_center(im, object_2)
        return [center_object_1_x,center_object_1_y], [center_object_2_x, center_object_2_y]

    def process_coordinates(self, pixel_coords: Tuple[float, float], depth_array: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """Process pixel coordinates to get real-world coordinates."""
        # Get 3D point in camera coordinates
        point_3d = deproject_pixel_to_point(depth_array, pixel_coords, self.intrinsics)
        
        # Transform to robot base coordinates if valid point found
        if not np.all(point_3d == 0):
            # Get real-world coordinates
            real_coords = transform_coordinates(point_3d[0], point_3d[1], point_3d[2])
            return real_coords
        return None

    def run_inference(self):
        """Main inference loop."""
        try:
            while True:
                # Show live preview
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())

                depth_frame = frames.get_depth_frame()
                depth_image_old = np.asanyarray(depth_frame.get_data())
                np.save('depth.npy', depth_image_old)

                cv2.imshow('RealSense Preview', color_image)
                
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                elif key & 0xFF == ord('s'):
                    # Capture and save images
                    rgb_path, depth_path, color_image, depth_image = self.capture_and_save_images()
                    if rgb_path and depth_path:
                        print(f"\nImages saved:")
                        print(f"RGB: {rgb_path}")
                        print(f"Depth: {depth_path}")
                        
                        # Call Gemini API to get coordinates for both objects
                        coords1, coords2 = self.call_gemini_api(rgb_path, depth_path, self.capture_dir, self.class_names)
                        
                        # Process coordinates for both objects
                        display_img = color_image.copy()
                        
                        # Process first object
                        print(f"\nProcessing {self.class_names[1]}:")
                        print(f"Pixel coordinates: {coords1}")
                        real_coords1 = self.process_coordinates(coords1, depth_image_old)
                        if real_coords1:
                            x1,y1,z1 = real_coords1
                            print(f"Camera coordinates (mm): {real_coords1}")
                            cv2.circle(display_img, (int(coords1[0]), int(coords1[1])), 
                                     5, (0, 255, 0), -1)
                            cv2.putText(display_img, self.class_names[0], 
                                      (int(coords1[0])+10, int(coords1[1])-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Process second object
                        print(f"\nProcessing {self.class_names[0]}:")
                        print(f"Pixel coordinates: {coords2}")
                        real_coords2 = self.process_coordinates(coords2, depth_image_old)
                        if real_coords2:
                            x2,y2,z2 = real_coords2
                            print(f"Camera coordinates (mm): {real_coords2}")
                            cv2.circle(display_img, (int(coords2[0]), int(coords2[1])), 
                                     5, (0, 0, 255), -1)
                            cv2.putText(display_img, self.class_names[1], 
                                      (int(coords2[0])+10, int(coords2[1])-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        
                        # Show results
                        if real_coords1 or real_coords2:
                            csv_path = predict_and_save_trajectory(x1, y1, z1, x2, y2, z2,output_path="final_predictions.csv")
                            cv2.imshow('Detected Points', display_img)
                            cv2.waitKey(2000)  # Show marked image for 2 seconds
                        else:
                            print("Failed to process coordinates - no valid depth data")
                        
                        self.capture_count += 1
                        print("\nReady for next capture (press 's' to capture, 'q' to quit)")

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example class names - replace with your actual class names
    class_names = ["black bottle", "white mug"]
    
    print("Starting RealSense inference script...")
    print(f"Looking for objects: {class_names[0]} and {class_names[1]}")
    print("Press 's' to capture images and process coordinates")
    print("Press 'q' to quit")
    print("\nInitializing camera...")
    
    try:
        inference = RealSenseInference(class_names)
        print("Camera initialized successfully!")
        print("\nReady for capture...")
        inference.run_inference()
    except Exception as e:
        print(f"Error: {str(e)}")