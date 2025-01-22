import time
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from datetime import datetime
import json
import csv
from hi_robotics.vision_ai.cameras.intel_realsense_camera import IntelRealSenseCamera
from gemini_constant_api_key import GEMINI_API_KEY
from utils import get_real_world_coordinates, transform_coordinates, create_zip_archive, process_images
from gemini_oop_object_detection import ObjectDetector

# ============================================================================================
# UTILITY FUNCTIONS
# ============================================================================================

def calculate_2d_angle(center_point, grasp_point):
    """Calculate 2D angle in degrees between grasp direction and positive x-axis."""
    delta_x = grasp_point[0] - center_point[0]
    delta_y = center_point[1] - grasp_point[1]  # Inverted because y increases downward
    angle_rad = np.arctan2(delta_y, delta_x)
    angle_deg = np.degrees(angle_rad)
    return (angle_deg + 360) % 360

def calculate_3d_angle(center_xyz, grasp_xyz):
    """Calculate 3D angles (theta and phi) in spherical coordinates."""
    rel_pos = grasp_xyz - center_xyz
    r = np.linalg.norm(rel_pos)
    
    # Calculate theta (azimuthal angle in x-y plane from x-axis)
    theta = np.degrees(np.arctan2(rel_pos[1], rel_pos[0]))
    theta = (theta + 360) % 360
    
    # Calculate phi (polar angle from z-axis)
    phi = np.degrees(np.arccos(rel_pos[2] / r))
    
    return theta, phi, r

def determine_pickup_mode(angle_2d):
    """Determine pickup mode based on 2D angle."""
    normalized_angle = angle_2d if angle_2d <= 180 else angle_2d - 180
    threshold = 30
    
    if (normalized_angle <= threshold or normalized_angle >= 180 - threshold):
        return "HORIZONTAL PICKUP"
    elif (90 - threshold <= normalized_angle <= 90 + threshold):
        return "VERTICAL PICKUP"
    else:
        return "UNDEFINED"

def remap_angle(theta, in_min=250, in_max=150, out_min=30, out_max=125):
    """Remap theta angle to robot angle range."""
    robot_angle = (theta - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    return 180 - robot_angle

# ======================================================================================================================
# CAMERA UTILITIES
# ======================================================================================================================

class CameraManager:
    def __init__(self):
        self.camera = None

    def initialize_camera(self):
        """Initialize the RealSense camera"""
        if self.camera is None:
            self.camera = IntelRealSenseCamera()
            st.success("Camera initialized successfully!")

    def stop_camera(self):
        """Stop the camera stream"""
        if self.camera is not None:
            self.camera = None
            st.warning("Camera stopped!")

    def get_frames(self):
        """Get frames from the camera"""
        if self.camera is not None:
            return self.camera.get_frames()
        return None, None

def capture_and_save_frames(camera_manager):
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
        frames = camera_manager.get_frames()
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
        return folder_name
        
    except Exception as e:
        st.error(f"Error capturing frames: {str(e)}")
        return None

def plot_grasp_points(image_path, points_list, output_path, radius=5):
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

def save_grasp_measurements(measurements, csv_path, append=True):
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


def save_detailed_data(measurements, json_path):
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



def get_all_directory_based_paths(latest_directory):
    folder_path = f'captured_frames/{latest_directory}'
    return {
        'folder_path': folder_path,
        'rgb_zip_path': f'{folder_path}/rgb.zip',
        'depth_zip_path': f'{folder_path}/depth.zip'
    }


def process_captured_images(target_classes, grasp_object, api_key=GEMINI_API_KEY):
    """
    Main function to process captured images and save measurements.
    
    Args:
        target_classes (list): List of object classes to detect
        grasp_object (str): Name of the object being grasped
        api_key (str): API key for object detection
    """
    # Get Latest Directory
    latest_directory = get_latest_subdirectory('captured_frames')
    print(f"Processing directory: {latest_directory}")
    
    paths = get_all_directory_based_paths(latest_directory)
    folder_path = paths['folder_path']
    
    # Process frames and get measurements
    measurements = process_frames(
        folder_path=folder_path,
        rgb_zip_path=paths['rgb_zip_path'],
        depth_zip_path=paths['depth_zip_path'],
        target_classes=target_classes,
        grasp_object=grasp_object,
        api_key=api_key
    )
    
    # Save measurements to CSV
    csv_path = os.path.join('captured_frames', "grasp_distances.csv")
    save_grasp_measurements(measurements, csv_path)
    
    # Save detailed data to JSON
    json_path = os.path.join(folder_path, "grasp_data.json")
    save_detailed_data(measurements, json_path)



def process_frames(folder_path, rgb_zip_path, depth_zip_path, target_classes, grasp_object, api_key, output_dir=None):
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
    grasp_center = plot_grasp_points(f'{folder_path}/rgb_image/image_0.jpg', [keypoints_to_print], output_path= f'{folder_path}/output/plotted_grasp_image.jpg',radius=3)
    print("Center points:", grasp_center)

    # Get real world coordinates
    real_world_grasp_center = get_real_world_coordinates(folder_path, grasp_center[0], grasp_center[1])
    print("Real world grasp center:", real_world_grasp_center)
    transformed_grasp_center = transform_coordinates(*real_world_grasp_center)

    # Calculate Z-distance between grasp center and target object
    z_distance = abs(transformed_grasp_center[2] - object_centers[grasp_object]['real_coords'][2])
    print(f"Z-distance between grasp center and {grasp_object}: {z_distance}")

    # Calculate angles
    angle_2d = calculate_2d_angle(object_centers[grasp_object]['center'],grasp_center)
    theta, phi, radius = calculate_3d_angle(
        np.array(object_centers[grasp_object]['real_coords']),
        np.array(transformed_grasp_center)
    )
    robot_angle = remap_angle(theta)
    pickup_mode = determine_pickup_mode(angle_2d)

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
    return results


def save_measurements(measurements, folder_path):
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

def _normalize_depth_for_display(depth_image):
        """Convert depth image to colorized visualization"""
        normalized_depth = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        normalized_depth = normalized_depth.astype(np.uint8)
        colored_depth = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)
        return colored_depth

def save_captured_frames(color_image, depth_image):
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
        
        return folder_name
    except Exception as e:
        st.error(f"Error saving frames: {str(e)}")
        return None


def get_latest_subdirectory(directory):
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



def __handle_live_feed():
    """Enhanced live feed handling with better UI feedback"""
    st.subheader("📹 Live Video Feed")
    
    # recorder = None
    camera = None
    
    try:
        with st.spinner("🎥 Initializing camera..."):
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    camera = IntelRealSenseCamera()
                    # recorder = RealSenseRecorder(camera=camera)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(2)
                    else:
                        raise e
        
        # if not recorder:
            # st.error("❌ Failed to initialize camera after multiple attempts")
            # return
        
        # Enhanced UI Controls
        st.markdown("### 🎮 Controls")
        controls_col1, controls_col2, controls_col3, controls_col4 = st.columns(4)
        
        with controls_col1:
            if st.button("🟢 Start Recording", use_container_width=True, 
                        disabled=st.session_state.get("recording_status", False)):
                st.session_state["recording_status"] = True
                # recorder.start_recording('Recorded_Demo')
                # st.success(f"📝 Recording to: {recorder.get_current_savepath()}")
                global output_path
                # output_path = recorder.get_current_savepath()
        
        with controls_col2:
            if st.button("🔴 Stop Recording", use_container_width=True,
                        disabled=not st.session_state.get("recording_status", False)):
                st.session_state["recording_status"] = False
                # recorder.stop_recording()
                st.info(f"✅ Captured frames")
        
        with controls_col3:
            if st.button("📸 Capture Frame", use_container_width=True):
                # Show waiting message
                wait_message = st.empty()
                wait_message.info("⏳ Waiting for camera to stabilize...")
                
                # Wait for 1 second
                time.sleep(1.0)
                
                # Capture frames
                rgb_frame, depth_frame = camera.get_frames()
                color_image = np.asanyarray(rgb_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                
                # Clear the waiting message
                wait_message.empty()
                folder_path = capture_and_save_frames(camera)
                st.success("✅ Frame captured and saved!")
        
        # Status indicators
        status_col1, status_col2 = st.columns(2)
        with status_col1:
            st.metric("Recording Status", 
                    "Active 🟢" if st.session_state.get("recording_status", False) else "")
        with status_col2:
            if st.session_state.get("recording_status", False):
                pass
        # Display frames with enhanced layout
        st.markdown("### 📺 Live Preview")
        frame_placeholder = st.empty()
        while st.session_state.get("run", True):
            try:
                rgb_frame, depth_frame = camera.get_frames()
                color_image = np.asanyarray(rgb_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                
                depth_colormap = _normalize_depth_for_display(depth_image)
                display_image = np.hstack((color_image, depth_colormap))
                display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
                
                frame_placeholder.image(display_image, channels="RGB", use_container_width=True,
                                    caption="Live Feed (Color + Depth)")
                
            except Exception as e:
                st.error(f"❌ Frame capture error: {str(e)}")
                break
                
    except Exception as e:
        st.error(f"❌ Camera initialization failed: {str(e)}")
        return
        
    finally:
        if camera:
            try:
                camera.release_camera()
            except Exception as e:
                st.error(f"❌ Camera release error: {str(e)}")
                
        st.session_state["run"] = False


def __handle_live_feed():
    """Enhanced live feed handling with better UI feedback"""
    st.subheader("📹 Live Video Feed")
    
    # Initialize session state for latest capture path if not exists
    if 'latest_capture_path' not in st.session_state:
        st.session_state.latest_capture_path = None
    
    camera = None
    
    try:
        with st.spinner("🎥 Initializing camera..."):
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    camera = IntelRealSenseCamera()
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(2)
                    else:
                        raise e
        
        st.markdown("### 🎮 Controls")
        controls_col1, controls_col2, controls_col3= st.columns(3)
        folder_path = None
        with controls_col1:
            if st.button("🟢 Start Recording", use_container_width=True, 
                        disabled=st.session_state.get("recording_status", False)):
                st.session_state["recording_status"] = True
        
        with controls_col2:
            if st.button("🔴 Stop Recording", use_container_width=True,
                        disabled=not st.session_state.get("recording_status", False)):
                st.session_state["recording_status"] = False
                st.info(f"✅ Captured frames")
        
        with controls_col3:
            if st.button("📸 Capture Frame", use_container_width=True):
                wait_message = st.empty()
                wait_message.info("⏳ Waiting for camera to stabilize...")
                time.sleep(1.0)
                
                rgb_frame, depth_frame = camera.get_frames()
                color_image = np.asanyarray(rgb_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                
                wait_message.empty()
                folder_path = capture_and_save_frames(camera)
                if folder_path:
                    st.session_state.latest_capture_path = folder_path
                    st.success("✅ Frame captured and saved!")



        status_col1, status_col2 = st.columns(2)
        with status_col1:
            st.metric("Recording Status", 
                    "Active 🟢" if st.session_state.get("recording_status", False) else "")
        
        st.markdown("### 📺 Live Preview")
        frame_placeholder = st.empty()
        while st.session_state.get("run", True):
            try:
                rgb_frame, depth_frame = camera.get_frames()
                color_image = np.asanyarray(rgb_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                
                depth_colormap = _normalize_depth_for_display(depth_image)
                display_image = np.hstack((color_image, depth_colormap))
                display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
                
                frame_placeholder.image(display_image, channels="RGB", use_container_width=True,
                                    caption="Live Feed (Color + Depth)")
                
            except Exception as e:
                st.error(f"❌ Frame capture error: {str(e)}")
                break
        
    except Exception as e:
        st.error(f"❌ Camera initialization failed: {str(e)}")
        return
        
    finally:
        if camera:
            try:
                camera.release_camera()
            except Exception as e:
                st.error(f"❌ Camera release error: {str(e)}")
                
        st.session_state["run"] = False


def handle_live_feed():
    """Enhanced live feed handling with better UI feedback"""
    st.subheader("📹 Live Video Feed")
    
    # Initialize session state for latest capture path and processing results
    if 'latest_capture_path' not in st.session_state:
        st.session_state.latest_capture_path = None
    if 'processing_results' not in st.session_state:
        st.session_state.processing_results = None
    
    camera = None
    
    try:
        with st.spinner("🎥 Initializing camera..."):
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    camera = IntelRealSenseCamera()
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(2)
                    else:
                        raise e
        
        st.markdown("### 🎮 Controls")
        controls_col1, controls_col2, controls_col3 = st.columns(3)
        
        with controls_col1:
            if st.button("🟢 Start Recording", use_container_width=True, 
                        disabled=st.session_state.get("recording_status", False)):
                st.session_state["recording_status"] = True
        
        with controls_col2:
            if st.button("🔴 Stop Recording", use_container_width=True,
                        disabled=not st.session_state.get("recording_status", False)):
                st.session_state["recording_status"] = False
                st.info(f"✅ Captured frames")
        
        with controls_col3:
            if st.button("📸 Capture Frame", use_container_width=True):
                wait_message = st.empty()
                wait_message.info("⏳ Waiting for camera to stabilize...")
                time.sleep(1.0)
                
                rgb_frame, depth_frame = camera.get_frames()
                color_image = np.asanyarray(rgb_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                
                wait_message.empty()
                folder_path = capture_and_save_frames(camera)
                if folder_path:
                    st.session_state.latest_capture_path = folder_path
                    st.success("✅ Frame captured and saved!")

        # Add image processing section
        st.markdown("### 🔄 Image Processing")
        process_col1, process_col2 = st.columns(2)
        
        with process_col1:
            target_classes = st.text_input("Target Classes (comma-separated)", 
                                         "cup,bottle,can", 
                                         help="Enter the object classes to detect, separated by commas")
            grasp_object = st.text_input("Grasp Object", 
                                        "can",
                                        help="Enter the name of the object being grasped")
        
        with process_col2:
            if st.button("🔍 Process Images", use_container_width=True):
                if st.session_state.latest_capture_path:
                    try:
                        with st.spinner("Processing captured images..."):
                            # Convert target_classes string to list
                            target_classes_list = [cls.strip() for cls in target_classes.split(',')]
                            
                            # Process the images
                            results = process_frames(
                                folder_path=st.session_state.latest_capture_path,
                                rgb_zip_path=f"{st.session_state.latest_capture_path}/rgb.zip",
                                depth_zip_path=f"{st.session_state.latest_capture_path}/depth.zip",
                                target_classes=target_classes_list,
                                grasp_object=grasp_object,
                                api_key=GEMINI_API_KEY
                            )
                            
                            # Save the results
                            save_measurements(results, st.session_state.latest_capture_path)
                            
                            # Store results in session state
                            st.session_state.processing_results = results
                            st.success("✅ Images processed successfully!")
                    except Exception as e:
                        st.error(f"❌ Error processing images: {str(e)}")
                else:
                    st.warning("Please capture frames first!")

        # Display processing results if available
        if st.session_state.processing_results:
            st.markdown("### 📊 Processing Results")
            st.json(st.session_state.processing_results)

        status_col1, status_col2 = st.columns(2)
        with status_col1:
            st.metric("Recording Status", 
                    "Active 🟢" if st.session_state.get("recording_status", False) else "")
        
        st.markdown("### 📺 Live Preview")
        frame_placeholder = st.empty()
        while st.session_state.get("run", True):
            try:
                rgb_frame, depth_frame = camera.get_frames()
                color_image = np.asanyarray(rgb_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                
                depth_colormap = _normalize_depth_for_display(depth_image)
                display_image = np.hstack((color_image, depth_colormap))
                display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
                
                frame_placeholder.image(display_image, channels="RGB", use_container_width=True,
                                    caption="Live Feed (Color + Depth)")
                
            except Exception as e:
                st.error(f"❌ Frame capture error: {str(e)}")
                break
        
    except Exception as e:
        st.error(f"❌ Camera initialization failed: {str(e)}")
        return
        
    finally:
        if camera:
            try:
                camera.release_camera()
            except Exception as e:
                st.error(f"❌ Camera release error: {str(e)}")
                
        st.session_state["run"] = False

if __name__ == "__main__":
    handle_live_feed()