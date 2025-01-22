import csv
import json
import time
from tkinter import Image
import cv2
from hi_robotics.vision_ai.cameras.intel_realsense_camera import IntelRealSenseCamera
import streamlit as st
import asyncio
import os
from pynput import keyboard
from examples.openai_api_key import OPENAI_API_KEY
from audio.voice_clients import OpenAIInteractiveVoiceClient, TurnDetectionMode
from audio.handlers import AudioHandler, KeyboardInputHandler
from llama_index.core.tools import FunctionTool
import logging
from datetime import datetime
import numpy as np
import queue
from threading import Thread

from gemini_constant_api_key import GEMINI_API_KEY
from utils import get_real_world_coordinates, transform_coordinates, create_zip_archive, process_images
from gemini_oop_object_detection import ObjectDetector

class GraspRnD:
    def __init__(self, camera_manager):
        self.camera_manager = camera_manager
        self.latest_capture_path = None
        self.processing_results = None
        self.command_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.voice_thread = None

    # ============================================================================================
    # GENERAL UTILITY FUNCTIONS
    # ============================================================================================

    def calculate_2d_angle(self, center_point, grasp_point):
        """Calculate 2D angle in degrees between grasp direction and positive x-axis."""
        delta_x = grasp_point[0] - center_point[0]
        delta_y = center_point[1] - grasp_point[1]  # Inverted because y increases downward
        angle_rad = np.arctan2(delta_y, delta_x)
        angle_deg = np.degrees(angle_rad)
        return (angle_deg + 360) % 360

    def calculate_3d_angle(self, center_xyz, grasp_xyz):
        """Calculate 3D angles (theta and phi) in spherical coordinates."""
        rel_pos = grasp_xyz - center_xyz
        r = np.linalg.norm(rel_pos)
        
        # Calculate theta (azimuthal angle in x-y plane from x-axis)
        theta = np.degrees(np.arctan2(rel_pos[1], rel_pos[0]))
        theta = (theta + 360) % 360
        
        # Calculate phi (polar angle from z-axis)
        phi = np.degrees(np.arccos(rel_pos[2] / r))
        
        return theta, phi, r

    def determine_pickup_mode(self, angle_2d):
        """Determine pickup mode based on 2D angle."""
        normalized_angle = angle_2d if angle_2d <= 180 else angle_2d - 180
        threshold = 30
        
        if (normalized_angle <= threshold or normalized_angle >= 180 - threshold):
            return "HORIZONTAL PICKUP"
        elif (90 - threshold <= normalized_angle <= 90 + threshold):
            return "VERTICAL PICKUP"
        else:
            return "UNDEFINED"

    def remap_angle(self, theta, in_min=250, in_max=150, out_min=30, out_max=125):
        """Remap theta angle to robot angle range."""
        robot_angle = (theta - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
        return 180 - robot_angle

    # ======================================================================================================================
    # CAMERA UTILITIES
    # ======================================================================================================================

    def capture_and_save_frames(self):
        """Capture and save frames from the camera"""
        root_dir = "captured_frames"
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{root_dir}/captured_frame_{current_time}"
        
        # Create necessary directories
        os.makedirs(root_dir, exist_ok=True)
        os.makedirs(folder_name, exist_ok=True)
        os.makedirs(f'{folder_name}/rgb_image', exist_ok=True)
        os.makedirs(f'{folder_name}/output', exist_ok=True)
        os.makedirs(f'{folder_name}/depth_image', exist_ok=True)
        
        try:
            frames = self.camera_manager.get_frames()
            color_frame, depth_frame = frames
            
            if not depth_frame or not color_frame:
                st.error("Error: Could not capture frames")
                return None
            
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # Save images
            color_path = os.path.join(f'{folder_name}/rgb_image', "image_0.jpg")
            cv2.imwrite(color_path, color_image)
            
            depth_raw_path = os.path.join(f'{folder_name}/depth_image', "image_0.npy")
            np.save(depth_raw_path, depth_image)
            
            # Create zip files
            create_zip_archive(f'{folder_name}/rgb_image', f"{folder_name}/rgb.zip")
            create_zip_archive(f'{folder_name}/depth_image', f"{folder_name}/depth.zip")
            
            # st.success(f"Frames captured and saved in {folder_name}")
            self.latest_capture_path = folder_name
            return folder_name
            
        except Exception as e:
            st.error(f"Error capturing frames: {str(e)}")
            return None

    def plot_grasp_points(self, image_path, points_list, output_path, radius=5):
        """Plot grasp points and save visualization"""
        img = cv2.imread(image_path)
        if img is None:
            img = np.ones((600, 600, 3), dtype=np.uint8) * 255
        
        colors = [
            (255, 0, 0),    # Blue
            (0, 255, 0),    # Green
            (0, 0, 255),    # Red
            (255, 0, 255),  # Magenta
            (0, 255, 255)   # Yellow
        ]
        
        center_x, center_y = None, None
        
        for group_idx, group in enumerate(points_list):
            color = colors[group_idx % len(colors)]
            
            for i, point in enumerate(group):
                x, y = int(point[0]), int(point[1])
                cv2.circle(img, (x, y), radius, color, -1)
                
                if i > 0:
                    prev_x, prev_y = int(group[i-1][0]), int(group[i-1][1])
                    cv2.line(img, (prev_x, prev_y), (x, y), (0, 255, 255), 2)
                    
                    center_x = (prev_x + x) // 2
                    center_y = (prev_y + y) // 2
                    cv2.circle(img, (center_x, center_y), radius, (0, 0, 255), -1)
        
        cv2.imwrite(output_path, img)
        return [center_x, center_y]

    # ======================================================================================================================
    # PROCESSING UTILITIES
    # ======================================================================================================================

    def save_grasp_measurements(self, measurements, csv_path, append=True):
        """
        Save grasp measurements to a CSV file.
        
        Args:
            measurements (dict): Dictionary containing measurement data
            csv_path (str): Path to save the CSV file
            append (bool): Whether to append to existing file or create new
        """
        file_exists = os.path.isfile(csv_path)
        mode = 'a' if append else 'w'
        
        with open(csv_path, mode=mode, newline='') as file:
            writer = csv.writer(file)
            if not file_exists or not append:
                writer.writerow(['object_name', 'z_distance'])
            writer.writerow([
                measurements['grasp_object'],
                measurements['z_distance']
            ])
        print(f"Measurements saved to {csv_path}")

    def save_detailed_data(self, measurements, json_path):
        """
        Save detailed grasp and object data to a JSON file.
        
        Args:
            measurements (dict): Dictionary containing measurement data
            json_path (str): Path to save the JSON file
        """
        data_to_save = {
            "grasp_object": measurements['grasp_object'],
            "z_distance": measurements['z_distance'],
            "transformed_grasp_center": measurements['transformed_grasp_center'],
            "object_centers": {k: {"real_coords": v["real_coords"]} 
                            for k, v in measurements['object_centers'].items()}
        }
        
        with open(json_path, 'w') as json_file:
            json.dump(data_to_save, json_file, indent=4)
        print(f"Detailed data saved to {json_path}")

    def get_all_directory_based_paths(self, latest_directory):
        folder_path = f'captured_frames/{latest_directory}'
        return {
            'folder_path': folder_path,
            'rgb_zip_path': f'{folder_path}/rgb.zip',
            'depth_zip_path': f'{folder_path}/depth.zip'
        }

    def process_captured_images(self, target_classes, grasp_object, api_key=GEMINI_API_KEY):
        """
        Main function to process captured images and save measurements.
        
        Args:
            target_classes (list): List of object classes to detect
            grasp_object (str): Name of the object being grasped
            api_key (str): API key for object detection
        """
        # Get Latest Directory
        latest_directory = self.get_latest_subdirectory('captured_frames')
        print(f"Processing directory: {latest_directory}")
        
        paths = self.get_all_directory_based_paths(latest_directory)
        folder_path = paths['folder_path']
        
        # Process frames and get measurements
        measurements = self.process_frames(
            folder_path=folder_path,
            rgb_zip_path=paths['rgb_zip_path'],
            depth_zip_path=paths['depth_zip_path'],
            target_classes=target_classes,
            grasp_object=grasp_object,
            api_key=api_key
        )
        
        # Save measurements to CSV
        csv_path = os.path.join('captured_frames', "grasp_distances.csv")
        self.save_grasp_measurements(measurements, csv_path)
        
        # Save detailed data to JSON
        json_path = os.path.join(folder_path, "grasp_data.json")
        self.save_detailed_data(measurements, json_path)

    def process_frames(self, folder_path, rgb_zip_path, depth_zip_path, target_classes, grasp_object, api_key, output_dir=None):
        """
        Process frames and analyze object distances from grasp points.
        
        Args:
            folder_path (str): Path to the folder containing captured frames
            rgb_zip_path (str): Path to the RGB images zip file
            depth_zip_path (str): Path to the depth images zip file
            target_classes (list): List of object classes to detect
            grasp_object (str): Name of the object being grasped
            api_key (str): API key for object detection
            output_dir (str, optional): Directory for output files. Defaults to folder_path/output
            
        Returns:
            dict: Dictionary containing processed data and measurements
        """
        if output_dir is None:
            output_dir = f"{folder_path}/output"

        # Process images and get keypoints
        decoded_csv_data, keypoints = process_images(rgb_zip_path, depth_zip_path, output_dir=output_dir)
        keypoint_3, keypoint_7 = keypoints[0][3], keypoints[0][7]
        print(f'keypoint 3: {keypoints[0][3]} \n keypoint 7: {keypoints[0][7]}')
        keypoints_to_print = [keypoint_3, keypoint_7]
        
        # Initialize object detector with specific target classes
        object_detector = ObjectDetector(api_key=api_key, recording_dir=folder_path)
        im = Image.open(f'{folder_path}/rgb_image/image_0.jpg')
        object_centers = object_detector.get_object_centers(im, target_classes=target_classes)
        print("Object centers:", object_centers)
        
        # Verify if grasp object is detected
        if grasp_object not in object_centers:
            raise ValueError(f"Grasp object '{grasp_object}' not found in detected objects: {list(object_centers.keys())}")
        
        # Transform object center coordinates to real world coordinates
        for object_ in object_centers.keys():
            object_centers[object_]["real_coords"] = transform_coordinates(
                *get_real_world_coordinates(folder_path, 
                                        object_centers[object_]["center"][0], 
                                        object_centers[object_]['center'][1])
            )
        
        # Get specific keypoints and plot them
        grasp_center = self.plot_grasp_points(f'{folder_path}/rgb_image/image_0.jpg', [keypoints_to_print], output_path= f'{folder_path}/output/plotted_grasp_image.jpg',radius=3)
        print("Center points:", grasp_center)

        # Get real world coordinates
        real_world_grasp_center = get_real_world_coordinates(folder_path, grasp_center[0], grasp_center[1])
        print("Real world grasp center:", real_world_grasp_center)
        transformed_grasp_center = transform_coordinates(*real_world_grasp_center)

        # Calculate Z-distance between grasp center and target object
        z_distance = abs(transformed_grasp_center[2] - object_centers[grasp_object]['real_coords'][2])
        print(f"Z-distance between grasp center and {grasp_object}: {z_distance}")

        # Calculate angles
        angle_2d = self.calculate_2d_angle(object_centers[grasp_object]['center'],grasp_center)
        theta, phi, radius = self.calculate_3d_angle(
            np.array(object_centers[grasp_object]['real_coords']),
            np.array(transformed_grasp_center)
        )
        robot_angle = self.remap_angle(theta)
        pickup_mode = self.determine_pickup_mode(angle_2d)

        print(f"2D Angle: {angle_2d} degrees")
        print(f"Theta (Azimuthal Angle): {theta} degrees")
        print(f"Phi (Polar Angle): {phi} degrees")
        print(f"Radius: {radius}")
        print(f"Robot Angle: {robot_angle}")
        print(f"Pickup Mode: {pickup_mode}")

        results = {
            "grasp_object": grasp_object,
            "z_distance": z_distance,
            "transformed_grasp_center": transformed_grasp_center,
            "object_centers": object_centers,
            "keypoints": keypoints,
            "angle_2d": angle_2d,
            "theta": theta,
            "phi": phi,
            "radius": radius,
            "robot_angle": robot_angle,
            "pickup_mode": pickup_mode
        }
        print(f"RESULTS: {results}")
        self.processing_results = results
        return results

    def save_measurements(self, measurements, folder_path):
        """Save measurements to CSV and JSON"""
        # Save to CSV
        csv_path = os.path.join('captured_frames', "grasp_distances.csv")
        file_exists = os.path.exists(csv_path)
        
        with open(csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['object_name', 'z_distance'])
            writer.writerow([
                measurements['grasp_object'],
                measurements['z_distance']
            ])
        
        # Save detailed data to JSON
        json_path = os.path.join(folder_path, "grasp_data.json")
        data_to_save = {
            "grasp_object": measurements['grasp_object'],
            "z_distance": measurements['z_distance'],
            "transformed_grasp_center": measurements['transformed_grasp_center'],
            "object_centers": {k: {"real_coords": v["real_coords"]} 
                            for k, v in measurements['object_centers'].items()},
            "angles": {
                "angle_2d": measurements['angle_2d'],
                "theta": measurements['theta'],
                "phi": measurements['phi'],
                "robot_angle": measurements['robot_angle']
            },
            "pickup_mode": measurements['pickup_mode']
        }
        
        with open(json_path, 'w') as json_file:
            json.dump(data_to_save, json_file, indent=4)

    def _normalize_depth_for_display(self, depth_image):
        """Convert depth image to colorized visualization"""
        normalized_depth = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        normalized_depth = normalized_depth.astype(np.uint8)
        colored_depth = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)
        return colored_depth

    def save_captured_frames(self, color_image, depth_image):
        """Save captured frames with enhanced organization"""
        try:
            root_dir = "captured_frames"
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_name = f"{root_dir}/captured_frame_{current_time}"
            
            # Create directories
            for subdir in ['rgb_image', 'output', 'depth_image']:
                os.makedirs(f'{folder_name}/{subdir}', exist_ok=True)
            
            # Save images
            color_path = os.path.join(f'{folder_name}/rgb_image', "image_0.jpg")
            cv2.imwrite(color_path, color_image)
            
            depth_raw_path = os.path.join(f'{folder_name}/depth_image', "image_0.npy")
            np.save(depth_raw_path, depth_image)
            
            # Create zip files
            create_zip_archive(f'{folder_name}/rgb_image', f"{folder_name}/rgb.zip")
            create_zip_archive(f'{folder_name}/depth_image', f"{folder_name}/depth.zip")
            
            self.latest_capture_path = folder_name
            return folder_name
        except Exception as e:
            st.error(f"Error saving frames: {str(e)}")
            return None

    def get_latest_subdirectory(self, directory):
        # Get all entries in the directory
        entries = [os.path.join(directory, d) for d in os.listdir(directory)]
        # Filter entries to only include directories
        subdirectories = [d for d in entries if os.path.isdir(d)]
        if not subdirectories:
            return None  # No subdirectories found
        
        # Find the subdirectory with the most recent modification time
        latest_subdirectory = max(subdirectories, key=os.path.getmtime)
        return os.path.basename(latest_subdirectory)

    # ======================================================================================================================
    # FINISH UTILS
    # ======================================================================================================================

    # Voice command tools
    def capture_image(self) -> str:
        """Captures and saves an image from the camera"""
        try:
            folder_path = self.capture_and_save_frames()
            if folder_path:
                self.latest_capture_path = folder_path
                return "Image captured successfully"
            return "Failed to capture image"
        except Exception as e:
            return f"Error capturing image: {str(e)}"

    def process_captured_image(self, target_classes: str = "cup,bottle,can", grasp_object: str = "can") -> str:
        """Processes the last captured image"""
        try:
            if not self.latest_capture_path:
                return "No image has been captured yet"
                
            target_classes_list = [cls.strip() for cls in target_classes.split(',')]
            results = self.process_frames(
                folder_path=self.latest_capture_path,
                rgb_zip_path=f"{self.latest_capture_path}/rgb.zip",
                depth_zip_path=f"{self.latest_capture_path}/depth.zip",
                target_classes=target_classes_list,
                grasp_object=grasp_object,
                api_key=GEMINI_API_KEY
            )
            
            # Save the results
            self.save_measurements(results, self.latest_capture_path)
            self.processing_results = results
            
            return "Image processed and results saved successfully"
        except Exception as e:
            return f"Error processing image: {str(e)}"

# Initialize voice client tools
if 'camera_manager' not in st.session_state:
    st.session_state.camera_manager = IntelRealSenseCamera()

tools = [
    FunctionTool.from_defaults(fn=GraspRnD(st.session_state.camera_manager).capture_and_save_frames),
    FunctionTool.from_defaults(fn=GraspRnD(st.session_state.camera_manager).process_captured_image)
]

class VoiceThread(Thread):
    """Dedicated thread for voice pipeline"""
    def __init__(self, command_queue, result_queue):
        super().__init__()
        self.command_queue = command_queue
        self.result_queue = result_queue
        self.running = True

    def run(self):
        """Main thread loop for voice pipeline"""
        async def voice_pipeline():
            audio_handler = AudioHandler()
            keyboard_input_handler = KeyboardInputHandler()
            keyboard_input_handler.loop = asyncio.get_event_loop()

            client = OpenAIInteractiveVoiceClient(
                api_key=OPENAI_API_KEY,
                instructions="""
                You are a helpful assistant controlling a camera system. You can:
                1. Capture images when asked ("take a picture", "capture image", etc.)
                2. Process captured images ("process image", "analyze image", etc.)
                Always confirm actions with clear responses.
                """,
                on_text_delta=lambda text: self.result_queue.put(("text", text)),
                on_audio_delta=lambda audio: audio_handler.play_audio(audio),
                on_interrupt=lambda: audio_handler.stop_playback_immediately(),
                turn_detection_mode=TurnDetectionMode.SERVER_VAD,
                tools=tools,
            )

            try:
                await client.connect()
                self.result_queue.put(("status", "Voice client connected!"))
                
                message_handler = asyncio.create_task(client.handle_messages())
                streaming_task = asyncio.create_task(audio_handler.start_streaming(client))
                
                while self.running:
                    try:
                        # Check for commands from main thread
                        cmd = self.command_queue.get_nowait()
                        if cmd == "stop":
                            break
                    except queue.Empty:
                        pass
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                self.result_queue.put(("error", f"Voice client error: {str(e)}"))
            finally:
                audio_handler.stop_streaming()
                audio_handler.cleanup()
                await client.close()

        asyncio.run(voice_pipeline())

    def stop(self):
        """Stop the voice thread"""
        self.running = False
        self.command_queue.put("stop")


def handle_live_feed_with_voice():
    """Main function for handling live feed with voice control"""
    st.title("Voice-Controlled Camera System 🎤")
    # Initialize session state attributes
    if 'camera_manager' not in st.session_state:
        st.session_state.camera_manager = IntelRealSenseCamera()
    if 'command_queue' not in st.session_state:
        st.session_state.command_queue = queue.Queue()
    if 'result_queue' not in st.session_state:
        st.session_state.result_queue = queue.Queue()
    if 'voice_thread' not in st.session_state:
        st.session_state.voice_thread = VoiceThread(
                    st.session_state.command_queue,
                    st.session_state.result_queue
            )
        st.session_state.voice_thread.start()
        # st.session_state.voice_thread = None
    if 'latest_capture_path' not in st.session_state:
        st.session_state.latest_capture_path = None
    if 'processing_results' not in st.session_state:
        st.session_state.processing_results = None
    # Voice control section
    st.subheader("Voice Control")
    print(st.session_state)
    col1, col2 = st.columns(2)
    
    with col1:
        if not st.session_state.voice_thread:
            if st.button("🎤 Start Voice System"):
                st.session_state.voice_thread = VoiceThread(
                    st.session_state.command_queue,
                    st.session_state.result_queue
                )
                st.session_state.voice_thread.start()
    with col2:
        if st.session_state.voice_thread:
            if st.button("⏹️ Stop Voice System"):
                if st.session_state.voice_thread:
                    st.session_state.voice_thread.stop()
                    st.session_state.voice_thread.join()
                    st.session_state.voice_thread = None
                st.rerun()
    
    # Camera initialization and preview
    if st.session_state.camera_manager is None:
        try:
            st.session_state.camera_manager = IntelRealSenseCamera()
            # st.session_state.camera_manager.initialize_camera()
        except Exception as e:
            st.error(f"Camera initialization failed: {str(e)}")
            return

    # Live preview
    st.subheader("📺 Live Preview")
    preview_placeholder = st.empty()

    # Message display area
    message_area = st.empty()
    
    # Background thread for checking voice system messages
    def check_messages():
        while st.session_state.voice_thread and st.session_state.voice_thread.is_alive():
            try:
                msg_type, msg = st.session_state.result_queue.get_nowait()
                if msg_type == "text":
                    message_area.markdown(f"Assistant: {msg}")
                elif msg_type == "status":
                    st.success(msg)
                elif msg_type == "error":
                    st.error(msg)
            except queue.Empty:
                pass
            time.sleep(0.1)

    # Start message checking thread
    if st.session_state.voice_thread and not hasattr(st.session_state, 'message_thread'):
        st.session_state.message_thread = Thread(target=check_messages)
        st.session_state.message_thread.daemon = True
        st.session_state.message_thread.start()

    # Main camera loop
    try:
        while True:
            frames = st.session_state.camera_manager.get_frames()
            if frames:
                color_frame, depth_frame = frames
                if color_frame and depth_frame:
                    color_image = np.asanyarray(color_frame.get_data())
                    depth_image = np.asanyarray(depth_frame.get_data())
                    depth_colormap = GraspRnD(st.session_state.camera_manager)._normalize_depth_for_display(depth_image)
                    display_image = np.hstack((color_image, depth_colormap))
                    preview_placeholder.image(display_image, channels="RGB", use_container_width=True)
            
            if not st.session_state.get("run", True):
                break
                
    except Exception as e:
        st.error(f"Error in camera feed: {str(e)}")
    finally:
        if st.session_state.camera_manager:
            st.session_state.camera_manager.stop_camera()

if __name__ == "__main__":
    # mp.set_start_method('spawn')  # Required for Windows compatibility
    handle_live_feed_with_voice()



