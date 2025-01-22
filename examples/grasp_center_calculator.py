import cv2
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs
from ultralytics import YOLO
import json
from datetime import datetime
import os
import subprocess
import logging

class PositionCalculator:
    def __init__(self, target_class='bottle'):
        # Initialize YOLO World model
        self.yolo_model = YOLO('yolov8l-world.pt')
        self.data_dir = 'data'
        self.yolo_model.set_classes([target_class])
        os.makedirs(self.data_dir, exist_ok=True)
        self.calib_matrix_x = np.array([
            [ 0.068, -0.986,  0.152, -0.108],
            [ 0.998,  0.065, -0.023,  0.0 ],
            [ 0.013,  0.153,  0.988, -0.044],
            [ 0.0,    0.0,    0.0,    1.0  ]
            ])
        self.calib_matrix_y = np.array([
            [-0.47,   0.587,  -0.659,  0.73929],
            [ 0.877,  0.392,  -0.276, -0.16997],
            [ 0.096, -0.708,  -0.7,    0.86356],
            [ 0.0,    0.0,     0.0,    1.0    ]
            ])

        # Initialize the position calculation file
        self.result_file = os.path.join(self.data_dir, 'latest_position_calculation.json')
        if not os.path.exists(self.result_file):
            initial_data = {
                'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
                'object_center_xyz': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                'relative_pose': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                'final_position_xyz': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                'final_position_pixel': {'x': 0, 'y': 0}
            }
            with open(self.result_file, 'w') as f:
                json.dump(initial_data, f, indent=4)
        
        # Initialize RealSense
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        
        # Start RealSense pipeline
        self.pipeline.start(self.config)
        
        # Get camera intrinsics
        self.align = rs.align(rs.stream.color)
        profile = self.pipeline.get_active_profile()
        depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
        self.depth_intrinsics = depth_profile.get_intrinsics()
        
        self.running = True
        self.target_class = target_class  # Set target class for object detection
    def get_object_center(self, frame, target_class='bottle'):
        """Get center coordinates of detected object."""
        results = self.yolo_model(frame)
        
        for result in results[0].boxes.data:
            x1, y1, x2, y2, conf, class_id = result
            if self.yolo_model.names[int(class_id)] == target_class:
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                return int(center_x), int(center_y)
        return None

    def get_xyz_from_pixel(self, depth_frame, pixel_x, pixel_y):
        """Convert pixel coordinates to 3D world coordinates using RealSense depth."""
        try:
            width = depth_frame.get_width()
            height = depth_frame.get_height()
            
            pixel_x = max(0, min(int(pixel_x), width - 1))
            pixel_y = max(0, min(int(pixel_y), height - 1))
            
            depth = depth_frame.get_distance(pixel_x, pixel_y)
            
            if depth <= 0 or depth > 10:
                return np.array([0, 0, 0])
                
            point_3d = rs.rs2_deproject_pixel_to_point(
                self.depth_intrinsics, 
                [pixel_x, pixel_y], 
                depth
            )
            return np.array(point_3d)
            
        except Exception as e:
            print(f"Error in get_xyz_from_pixel: {e}")
            return np.array([0, 0, 0])

    def get_pixel_from_xyz(self, xyz_coords):
        """Convert 3D world coordinates back to pixel coordinates."""
        try:
            pixel = rs.rs2_project_point_to_pixel(self.depth_intrinsics, xyz_coords)
            return (int(pixel[0]), int(pixel[1]))
        except Exception as e:
            print(f"Error converting xyz to pixel: {e}")
            return None

    def get_latest_recording_file(self):
        """Get the most recent recording file from the data directory."""
        recording_files = [f for f in os.listdir(self.data_dir) if f.startswith('grasp_recording_')]
        if not recording_files:
            return None
        latest_file = max(recording_files, key=lambda x: os.path.getctime(os.path.join(self.data_dir, x)))
        return os.path.join(self.data_dir, latest_file)

    def calculate_final_position(self, object_xyz, relative_pose):
        """Calculate final position by adding object center coordinates with relative pose."""
        final_x = object_xyz[0] + relative_pose['position']['x']
        final_y = object_xyz[1] + relative_pose['position']['y'] 
        final_z = object_xyz[2] + relative_pose['position']['z']
        
        return {
            'x': float(final_x),
            'y': float(final_y),
            'z': float(final_z)
        }

    def process_frame(self, frame, depth_frame, target_class='bottle'):
        """Process a single frame and calculate positions."""
        object_center = self.get_object_center(frame, target_class)
        if not object_center:
            return None, "No object detected!"
            
        # Get object center in 3D coordinates
        object_xyz = self.get_xyz_from_pixel(depth_frame, object_center[0], object_center[1])
        
        # Get latest recording file
        latest_file = self.get_latest_recording_file()
        if not latest_file:
            return None, "No recording file found!"
            
        try:
            with open(latest_file, 'r') as f:
                recording_data = json.load(f)
                
            relative_pose = recording_data['relative_pose']
            final_position = self.calculate_final_position(object_xyz, relative_pose)
            
            # Convert final position to pixel coordinates for saving and visualization
            final_xyz = [final_position['x'], final_position['y'], final_position['z']]
            final_pixel = self.get_pixel_from_xyz(final_xyz)
            final_xyz_transformed = self.transform_coordinates(final_xyz)
            if not final_pixel:
                return None, "Could not convert final position to pixel coordinates"
            
            # Create result dictionary with pixel coordinates
            result = {
                'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
                'object_center_xyz': {  # Keep 3D coordinates for reference
                    'x': float(object_xyz[0]),
                    'y': float(object_xyz[1]),
                    'z': float(object_xyz[2])
                },
                'relative_pose': relative_pose['position'],
                'final_position_xyz': {  # Keep 3D coordinates for reference
                    'x': float(final_xyz[0]),
                    'y': float(final_xyz[1]),
                    'z': float(final_xyz[2])
                },
                'final_position_pixel': {  # Save pixel coordinates
                    'x': int(final_pixel[0]),
                    'y': int(final_pixel[1])
                },
                'final_xyz_transformed': {  
                    'x': float(final_xyz_transformed[0]),
                    'y': float(final_xyz_transformed[1]),
                    'z': float(final_xyz_transformed[2])
                }
            }
            
            # Draw visualizations
            cv2.circle(frame, object_center, 5, (255, 0, 0), -1)  # Blue dot for object center
            cv2.circle(frame, final_pixel, 5, (0, 255, 255), -1)  # Yellow dot for calculated position
            cv2.line(frame, object_center, final_pixel, (0, 255, 0), 2)  # Green line connecting them
            
            # Add text displays
            cv2.putText(frame, f"Object XYZ: ({object_xyz[0]:.3f}, {object_xyz[1]:.3f}, {object_xyz[2]:.3f})",
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(frame, f"Final Pixel: ({final_pixel[0]}, {final_pixel[1]})",
                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            return result, None
            
        except Exception as e:
            return None, f"Error processing calculation: {e}"

    def transform_coordinates(self, point):
            """Transform coordinates using X and Y matrices."""
            B = np.eye(4)
            B[:3, 3] = point
            A = self.calib_matrix_y @ B @ np.linalg.inv(self.calib_matrix_x)
            transformed_point = A[:3, 3] * 1000
            return transformed_point/1000


    def run(self):
        """Main loop for continuous position calculation."""
        try:
            while self.running:
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                if not depth_frame or not color_frame:
                    continue
                    
                frame = np.asanyarray(color_frame.get_data())
                
                result, error = self.process_frame(frame, depth_frame, self.target_class)  # Pass target class
                
                if error:
                    print(error)
                elif result:
                    # Add timestamp to the result
                    result['timestamp'] = datetime.now().strftime('%Y%m%d_%H%M%S')
                    
                    # Update the same position calculation file
                    result_file = os.path.join(self.data_dir, 'latest_position_calculation.json')
                    with open(result_file, 'w') as f:
                        json.dump(result, f, indent=4)
                    
                    # Print results with updated key names
                    print("\nCalculation Results:")
                    print(f"Object Center (XYZ): {result['object_center_xyz']}")
                    print(f"Relative Pose: {result['relative_pose']}")
                    print(f"Final Position (XYZ): {result['final_position_xyz']}")
                    print(f"Final Position (Pixel): {result['final_position_pixel']}")
                    print(f"Updated results saved to {result_file}")
                
                cv2.imshow('Position Calculation', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                    
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    target_object = 'bottle'  # You can change this to any target class you want
    calculator = PositionCalculator(target_class=target_object)
    calculator.run()