import asyncio
import json
import os
import cv2
import numpy as np
import pyrealsense2 as rs
from PIL import Image
from datetime import datetime
from pynput import keyboard

from llama_index.core.tools import FunctionTool
from gemini_constant_api_key import GEMINI_API_KEY
from gemini_oop_object_detection import ObjectDetector
from examples.openai_realtime_client.handlers.audio_handler import AudioHandler
from examples.openai_realtime_client.client.realtime_client import RealtimeClient
from utils import get_real_world_coordinates, transform_coordinates, create_zip_archive, process_images

class HybridGraspCapture:
    def __init__(self, target_classes=['bottle'], grasp_object='bottle'):
        self.target_classes = target_classes
        self.grasp_object = grasp_object
        self.pipeline = None
        self.current_folder = None
        self.processing_lock = asyncio.Lock()
        
    def initialize(self):
        """Initialize the RealSense camera"""
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)
        print("\nCamera initialized successfully!")

    async def handle_command(self, command: str, message_callback=None):
        """Handle both voice and text commands"""
        command = command.lower().strip()
        response = None

        if command in ['capture', 'take picture', 'take a picture']:
            response = await self.capture_frame()
        elif command in ['process', 'analyze']:
            async with self.processing_lock:
                target_classes = input("Enter the target classes to detect: ")
                grasp_object = input("Enter the Object to Grasp: ")
                response = await self.process_latest_frame(target_classes = target_classes, grasp_object = grasp_object)
        elif command in ['quit', 'exit', 'q']:
            response = "Exiting..."
        else:
            response = f"Unknown command: '{command}'. Use 'capture' to take a picture or 'process' to analyze it."

        # If there's a message callback (for voice responses), use it
        if message_callback:
            message_callback(response)
        
        # Always print the response for text input
        print(f"\n{response}")
        return response

    async def capture_frame(self):
        """Capture and save a frame"""
        try:
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                return "Failed to capture frames"

            # Create directories
            root_dir = "captured_frames"
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_folder = f"{root_dir}/captured_frame_{current_time}"
            
            os.makedirs(f'{self.current_folder}/rgb_image', exist_ok=True)
            os.makedirs(f'{self.current_folder}/depth_image', exist_ok=True)
            os.makedirs(f'{self.current_folder}/output', exist_ok=True)

            # Save frames
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            cv2.imwrite(f'{self.current_folder}/rgb_image/image_0.jpg', color_image)
            np.save(f'{self.current_folder}/depth_image/image_0.npy', depth_image)

            # Create zip archives
            rgb_zip_path = f"{self.current_folder}/rgb.zip"
            depth_zip_path = f"{self.current_folder}/depth.zip"
            create_zip_archive(f'{self.current_folder}/rgb_image', rgb_zip_path)
            create_zip_archive(f'{self.current_folder}/depth_image', depth_zip_path)

            return "Frame captured successfully. Use 'process' to analyze the grasp."

        except Exception as e:
            return f"Error capturing frame: {str(e)}"

    async def process_latest_frame(self,target_dir=None, target_classes=[], grasp_object=None):
        """Process the most recently captured frame"""
        try:
            if not self.current_folder:
                return "No frame captured yet. Use 'capture' to take a picture first."
            target_dir = target_dir if target_dir else self.current_folder
            # OBJECT DETECTION:
            object_detector = ObjectDetector(api_key = GEMINI_API_KEY, recording_dir=target_dir)
            rgb_im = Image.open(f'{target_dir}/rgb_image/image_0.jpg')
            depth_im = np.load(f"{target_dir}/depth_image/image_0.npy")
            object_centers= object_detector.get_object_centers(rgb_im,target_classes=target_classes)
            
            print(f"OBJECT CENTERS: {object_centers}")
            
            # Get paths
            rgb_zip_path = f"{target_dir}/rgb.zip"
            depth_zip_path = f"{target_dir}/depth.zip"
            output_dir = f"{target_dir}"

            decoded_csv_data, keypoints = process_images(rgb_zip_path, depth_zip_path, output_dir=output_dir)
            
            if not keypoints or len(keypoints) == 0:
                return "No grasp keypoints detected in the frame."

            keypoint_3, keypoint_7 = keypoints[0][3], keypoints[0][7]
            
            # Transform object center coordinates to real world coordinates
            for object_ in object_centers.keys():
                object_centers[object_]["real_coords"] = transform_coordinates(
                    *get_real_world_coordinates(target_dir, 
                                            object_centers[object_]["center"][0], 
                                            object_centers[object_]['center'][1])
                )

            # Get and transform coordinates
            real_world_point_3 = get_real_world_coordinates(target_dir, keypoint_3[0], keypoint_3[1])
            real_world_point_7 = get_real_world_coordinates(target_dir, keypoint_7[0], keypoint_7[1])
            transformed_point_3 = transform_coordinates(*real_world_point_3)
            transformed_point_7 = transform_coordinates(*real_world_point_7)

            # Calculate grasp center
            grasp_center_x = (keypoint_3[0] + keypoint_7[0]) // 2
            grasp_center_y = (keypoint_3[1] + keypoint_7[1]) // 2
            real_world_grasp = get_real_world_coordinates(target_dir, grasp_center_x, grasp_center_y)
            transformed_grasp = transform_coordinates(*real_world_grasp)

            # Calculate Z-distance between grasp center and target object
            z_distance = abs(transformed_grasp[2] - object_centers[grasp_object]['real_coords'][2])
            print(f"Z-distance between grasp center and {grasp_object}: {z_distance}")

            # Calculate angles
            angle_2d = self.calculate_2d_angle(
                (grasp_center_x, grasp_center_y),
                object_centers[grasp_object]['center']
            )
            theta, phi, radius = self.calculate_3d_angle(
                np.array(object_centers[grasp_object]['real_coords']),
                np.array(transformed_grasp),
            )
            robot_angle = self.remap_angle(theta)
            pickup_mode = self.determine_pickup_mode(angle_2d)

            print(f"2D Angle: {angle_2d} degrees")
            print(f"Theta (Azimuthal Angle): {theta} degrees")
            print(f"Phi (Polar Angle): {phi} degrees")
            print(f"Radius: {radius}")
            print(f"Robot Angle: {robot_angle}")
            print(f"Pickup Mode: {pickup_mode}")
            print(f'keypoint 3: {keypoints[0][3]} \n keypoint 7: {keypoints[0][7]}')
            print("Object centers:", object_centers)
            print("Center points:", [grasp_center_x, grasp_center_y])
            print(f"Z-distance between grasp center and {grasp_object}: {z_distance}")

            # Create visualization
            vis_img = cv2.imread(f'{target_dir}/rgb_image/image_0.jpg')
            cv2.circle(vis_img, (int(keypoint_3[0]), int(keypoint_3[1])), 5, (0, 0, 255), -1)
            cv2.circle(vis_img, (int(keypoint_7[0]), int(keypoint_7[1])), 5, (255, 0, 0), -1)
            cv2.circle(vis_img, (int(grasp_center_x), int(grasp_center_y)), 5, (0, 255, 255), -1)
            cv2.line(vis_img, (int(keypoint_3[0]), int(keypoint_3[1])), 
                    (int(keypoint_7[0]), int(keypoint_7[1])), (0, 255, 0), 2)

            cv2.putText(vis_img, f"Robot Angle: {robot_angle:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis_img, f"Mode: {pickup_mode}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Save results
            cv2.imwrite(f'{target_dir}/rgb_image/analyzed_grasp.jpg', vis_img)
            results = {
                "grasp_object": grasp_object,
                "z_distance": z_distance,
                "transformed_grasp_center": transformed_grasp,
                "object_centers": object_centers,
                "keypoints": keypoints,
                "angle_2d": angle_2d,
                "theta": theta,
                "phi": phi,
                "radius": radius,
                "robot_angle": robot_angle,
                "pickup_mode": pickup_mode
            }
            
            with open(f'{target_dir}/grasp_analysis.json', 'w') as f:
                json.dump(results, f, indent=4)

            # cv2.imshow('Grasp Analysis', vis_img)
            # cv2.waitKey(1)

            return results #f"Analysis complete! Robot angle is {robot_angle:.1f} degrees, pickup mode is {pickup_mode}"

        except Exception as e:
            return {e} #f"Error processing frame: {str(e)}"

    # Utility functions (calculate_2d_angle, calculate_3d_angle, etc.) remain the same
    def calculate_2d_angle(self, center_point, grasp_point):
        delta_x = grasp_point[0] - center_point[0]
        delta_y = center_point[1] - grasp_point[1]
        angle_rad = np.arctan2(delta_y, delta_x)
        angle_deg = np.degrees(angle_rad)
        return (angle_deg + 360) % 360

    def calculate_3d_angle(self, center_xyz, grasp_xyz):
        rel_pos = grasp_xyz - center_xyz
        r = np.linalg.norm(rel_pos)
        theta = np.degrees(np.arctan2(rel_pos[1], rel_pos[0]))
        theta = (theta + 360) % 360
        phi = np.degrees(np.arccos(rel_pos[2] / r))
        return theta, phi, r

    def determine_pickup_mode(self, angle_2d):
        normalized_angle = angle_2d if angle_2d <= 180 else angle_2d - 180
        threshold = 30
        if (normalized_angle <= threshold or normalized_angle >= 180 - threshold):
            return "HORIZONTAL PICKUP"
        elif (90 - threshold <= normalized_angle <= 90 + threshold):
            return "VERTICAL PICKUP"
        else:
            return "UNDEFINED"

    def remap_angle(self, theta, in_min=250, in_max=150, out_min=30, out_max=125):
        robot_angle = (theta - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
        return 180 - robot_angle

    def cleanup(self):
        if self.pipeline:
            self.pipeline.stop()
        cv2.destroyAllWindows()

async def main():
    
    # Initialize the capture system

    capture_system = HybridGraspCapture()
    capture_system.initialize()

    print("\nHybrid grasp capture system ready!")
    print("Commands:")
    print("- Type 'capture' or 'take picture' to capture a frame")
    print("- Type 'process' or 'analyze' to process the latest frame")
    print("- Type 'quit' or 'q' to exit")
    print("\nEnter command: ", end='', flush=True)

    try:
        while True:
            # Wait for commands from input handler
            # command_type, command_data = await input_handler.command_queue.get()
            
            # if command_data.lower() in ['quit', 'exit', 'q']:
            #     break
            command_data= input("Enter the command: ")
            # Process the command
            await capture_system.handle_command(command_data)
            print("\nEnter command: ", end='', flush=True)

            await asyncio.sleep(0.01)
            
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        capture_system.cleanup()

if __name__ == "__main__":
    asyncio.run(main())