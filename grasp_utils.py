import os
import cv2
from hi_robotics.vision_ai.cameras.intel_realsense_camera import IntelRealSenseCamera
import numpy as np
from PIL import Image
from collect2 import *
from gemini_oop_object_detection import *
from utils import get_real_world_coordinates, transform_coordinates, create_zip_archive, process_images
from datetime import datetime
import zipfile
import requests
import csv
import json

API_URL = "http://techolution.ddns.net:5000/process_pose"

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


# # ==========================================================================================
# # Capture an image
# # ==========================================================================================

def create_pipeline():
    # Configure depth and color streams
    # pipeline = rs.pipeline()
    # config = rs.config()
    
    # # Enable both streams
    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # # Start streaming
    # pipeline.start(config)
    camera = IntelRealSenseCamera()
    return camera


def capture_and_save_frames(camera:IntelRealSenseCamera):
    # Get current timestamp for folder name
    root_dir = "captured_frames"
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{root_dir}/captured_frame_{current_time}"
    
    # Create directory if it doesn't exist
    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(folder_name, exist_ok=True)
    os.makedirs(f'{folder_name}/rgb_image', exist_ok=True)
    os.makedirs(f'{folder_name}/output', exist_ok=True)
    os.makedirs(f'{folder_name}/depth_image', exist_ok=True)
    
    try:
        # Wait for a coherent pair of frames
        frames = camera.get_frames()
        color_frame, depth_frame = frames
        
        if not depth_frame or not color_frame:
            print("Error: Could not capture frames")
            return None
        
        # Convert frames to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # Save color image
        color_path = os.path.join(f'{folder_name}/rgb_image', "image_0.jpg")
        cv2.imwrite(color_path, color_image)
        
        # Save depth image
        # Normalize the depth data for better visualization
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        # depth_path = os.path.join(folder_name, "depth.png")
        # cv2.imwrite(depth_path, depth_'{folder_name}/depth_image', "image_0.npy")colormap)
        
        # Also save raw depth data
        depth_raw_path = os.path.join(f'{folder_name}/depth_image', "image_0.npy")
        np.save(depth_raw_path, depth_image)
        
        print(f"Frames captured and saved in {folder_name}")
        return folder_name
        
    except Exception as e:
        print(f"Error capturing frames: {str(e)}")
        return None


def upload_zip(zip_path, api_url):
    """Upload the zip file to the specified API"""
    try:
        with open(zip_path, 'rb') as zip_file:
            files = {'file': zip_file}
            response = requests.post(api_url, files=files)
            
            if response.status_code == 200:
                print("Successfully uploaded zip file")
                return True
            else:
                print(f"Failed to upload zip file. Status code: {response.status_code}")
                return False
                
    except Exception as e:
        print(f"Error uploading zip file: {str(e)}")
        return False


def capture_frames_to_send():
    # Initialize RealSense pipeline
    camera = create_pipeline()
    
    try:
        while True:
            # Show the frames
            frames = camera.get_frames()
            color_frame,depth_frame = frames
            
            if not depth_frame or not color_frame:
                continue
                
            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # Apply colormap on depth image
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            frames = np.hstack((color_image, depth_colormap))
            # Show frames
            cv2.imshow('RGB & Depth', frames)
            # cv2.imshow('Depth', depth_colormap)
            
            # Check for keyboard input
            key = cv2.waitKey(1)
            
            # Press 's' to save frames
            if key == ord('s'):
                folder_path = capture_and_save_frames(camera)
                if folder_path:
                    # Create zip files for RGB and depth_raw images
                    rgb_zip_path = f"{folder_path}/rgb.zip"
                    depth_zip_path = f"{folder_path}/depth.zip"
                    create_zip_archive(f'{folder_path}/rgb_image', rgb_zip_path)
                    create_zip_archive(f'{folder_path}/depth_image', depth_zip_path)
                    
                    # Upload the zip files to the API
                    # decoded_csv_data, keypoints = process_images(rgb_zip_path, depth_zip_path, output_dir=f"{folder_path}/output")
                    # print(keypoints)
                    return {
                        "rgb_zip_path": rgb_zip_path,
                        "depth_zip_path": depth_zip_path,
                        "folder_path": folder_path
                    }
            # Press 'q' to quit
            elif key == ord('q'):
                break
                
    finally:
        pass
        # Stop streaming
        # pipeline.stop()
        # cv2.destroyAllWindows()


# if __name__ == "__main__":
#     sample_keypoints_output_with_csv()

# import cv2
# import numpy as np


def plot_points_1(image_path, points_list, radius=5):
    """
    Plot points on an image using cv2
    
    Args:
        image_path (str): Path to the input image
        points_list (list): List of lists containing point coordinates
        radius (int): Radius of the dots to be plotted
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        # If no image is provided, create a blank white canvas
        img = np.ones((600, 600, 3), dtype=np.uint8) * 255
    
    # Colors for different point groups (in BGR format)
    colors = [
        (255, 0, 0),    # Blue
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 0, 255),  # Magenta
        (0, 255, 255)   # Yellow
    ]
    
    # Plot each group of points with different colors
    for group_idx, group in enumerate(points_list):
        color = colors[group_idx % len(colors)]
        
        # Plot each point in the group
        for point in group:
            x, y = int(point[0]), int(point[1])
            cv2.circle(img, (x, y), radius, color, -1)  # -1 means filled circle
            
    # Display the image
    return img
    # cv2.imshow('Points Visualization', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


# # ==========================================================================================
# # Process Captured Data
# # ==========================================================================================

# def euclidean_distance(point1, point2):
#     """
#     Calculate the 3D Euclidean distance between two points.
    
#     Args:
#         point1 (tuple or list): Coordinates of the first point (x1, y1, z1)
#         point2 (tuple or list): Coordinates of the second point (x2, y2, z2)
    
#     Returns:
#         float: The Euclidean distance between the two points
#     """
#     point1 = np.array(point1)
#     point2 = np.array(point2)
#     distance = np.linalg.norm(point1 - point2)
#     return distance




# def __deprecated_process_captured_images():
#     # Get Latest Directory
#     latest_directory = get_latest_subdirectory('captured_frames')
#     print(latest_directory)
#     paths = get_all_directory_based_paths(latest_directory)
#     folder_path = paths['folder_path']
#     rgb_zip_path = paths['rgb_zip_path']
#     depth_zip_path = paths['depth_zip_path']

#     # Process images and get keypoints
#     decoded_csv_data, keypoints = process_images(rgb_zip_path, depth_zip_path, output_dir=f"{folder_path}/output")
#     keypoint_3, keypoint_7 = keypoints[0][3], keypoints[0][7]
#     print(f'keypoint 3: {keypoints[0][3]} \n keypoint 7: {keypoints[0][7]}')
#     keypoints_to_print = [keypoint_3, keypoint_7]
    
#     # Initialize object detector and get object centers
#     object_detector = ObjectDetector(api_key=GEMINI_API_KEY, recording_dir=folder_path)
#     im = Image.open(f'{folder_path}/rgb_image/image_0.jpg')
#     object_centers = object_detector.get_object_centers(im, target_classes=None)
#     print("Object centers:", object_centers)
    
#     # Transform object center coordinates to real world coordinates
#     for object_ in object_centers.keys():
#         object_centers[object_]["real_coords"] = transform_coordinates(*get_real_world_coordinates(folder_path, object_centers[object_]["center"][0], object_centers[object_]['center'][1]))
    
#     # Get specific keypoints and plot them
#     center_points = plot_grasp_points(f'{folder_path}/rgb_image/image_0.jpg', [keypoints_to_print], radius=3)
#     print("Center points:", center_points)

#     # Get real world coordinates
#     real_world_grasp_center = get_real_world_coordinates(folder_path, center_points[0], center_points[1])
#     real_world_grasp_point_3 = get_real_world_coordinates(folder_path, keypoint_3[0], keypoint_3[1])
#     real_world_grasp_point_7 = get_real_world_coordinates(folder_path, keypoint_7[0], keypoint_7[1])
#     print("Real world grasp center:", real_world_grasp_center)

#     # Transform keypoint coordinates
#     transformed_point_3 = transform_coordinates(*real_world_grasp_point_3)
#     transformed_point_7 = transform_coordinates(*real_world_grasp_point_7)
#     transformed_grasp_center = transform_coordinates(*real_world_grasp_center)
#     # Identify the object closest to keypoint 3 and keypoint 7
#     closest_object_to_keypoint_3 = None
#     closest_object_to_keypoint_7 = None
#     min_distance_to_keypoint_3 = float('inf')
#     min_distance_to_keypoint_7 = float('inf')

#     # Calculate distances to each object for both keypoints
#     for object_name, object_data in object_centers.items():
#         # Calculate distance for keypoint 3
#         distance_to_keypoint_3 = euclidean_distance(
#             object_data['real_coords'],
#             transformed_point_3
#         )
#         if distance_to_keypoint_3 < min_distance_to_keypoint_3:
#             min_distance_to_keypoint_3 = distance_to_keypoint_3
#             closest_object_to_keypoint_3 = object_name

#         # Calculate distance for keypoint 7
#         distance_to_keypoint_7 = euclidean_distance(
#             object_data['real_coords'],
#             transformed_point_7
#         )
#         if distance_to_keypoint_7 < min_distance_to_keypoint_7:
#             min_distance_to_keypoint_7 = distance_to_keypoint_7
#             closest_object_to_keypoint_7 = object_name

#     # find the object that is closest to the keypoints
#     print(f"Closest object to keypoint 3: {closest_object_to_keypoint_3} with distance {min_distance_to_keypoint_3}")
#     print(f"Closest object to keypoint 7: {closest_object_to_keypoint_7} with distance {min_distance_to_keypoint_7}")

#     # Save the distances to closest objects for both keypoints to CSV
#     csv_file_path = os.path.join('captured_frames', "keypoint_distances.csv")
#     file_exists = os.path.isfile(csv_file_path)
    
#     closest_object = closest_object_to_keypoint_3 if closest_object_to_keypoint_3 else closest_object_to_keypoint_7
#     distance_from_center = euclidean_distance(
#         [0, 0, object_centers[closest_object]['real_coords'][2]], 
#         [0, 0, transformed_grasp_center[2]]
#     )

#     with open(csv_file_path, mode='a', newline='') as file:
#         writer = csv.writer(file)
#         # if not file_exists:
#         #     writer.writerow(["Keypoint", "Closest Object", "Distance"])
#         # writer.writerow([
#         #     "keypoint_3",
#         #     closest_object_to_keypoint_3,
#         #     min_distance_to_keypoint_3
#         # ])
#         # writer.writerow([
#         #     "keypoint_7",
#         #     closest_object_to_keypoint_7,
#         #     min_distance_to_keypoint_7
#         # ])
#         if not file_exists:
#             writer.writerow(['object_name', 'distance'])
#         print(f"DISTANCE FROM CENTER = {distance_from_center}")
#         writer.writerow([closest_object_to_keypoint_3 if closest_object_to_keypoint_3 else closest_object_to_keypoint_7, distance_from_center])
#     print(f"Keypoint distances saved to {csv_file_path}")

#     # Save the real world transformed points along with center points and object names into a JSON file
#     json_file_path = os.path.join(folder_path, "transformed_points.json")
#     data_to_save = {
#         "closest_object_to_keypoint_3": {
#             "object_name": closest_object_to_keypoint_3,
#             "distance": min_distance_to_keypoint_3
#         },
#         "closest_object_to_keypoint_7": {
#             "object_name": closest_object_to_keypoint_7,
#             "distance": min_distance_to_keypoint_7
#         },
#         "transformed_point_3": transformed_point_3,
#         "transformed_point_7": transformed_point_7,
#         "object_centers": {k: {"real_coords": v["real_coords"]} for k, v in object_centers.items()}
#     }

#     with open(json_file_path, 'w') as json_file:
#         json.dump(data_to_save, json_file, indent=4)

#     print(f"Transformed points and distances saved to {json_file_path}")


# if __name__ == "__main__":
#     # capture_frames_to_send()
#     process_captured_images()


def get_all_directory_based_paths(latest_directory):
    folder_path = f'captured_frames/{latest_directory}'
    return {
        'folder_path': folder_path,
        'rgb_zip_path': f'{folder_path}/rgb.zip',
        'depth_zip_path': f'{folder_path}/depth.zip'
    }


def plot_grasp_points(image_path, points_list, radius=5):
    """
    Plot points on an image using cv2 and draw lines between them
    
    Args:
        image_path (str): Path to the input image
        points_list (list): List of lists containing point coordinates
        radius (int): Radius of the dots to be plotted
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        # If no image is provided, create a blank white canvas
        img = np.ones((600, 600, 3), dtype=np.uint8) * 255
    
    # Colors for different point groups (in BGR format)
    colors = [
        (255, 0, 0),    # Blue
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 0, 255),  # Magenta
        (0, 255, 255)   # Yellow
    ]
    
    # Plot each group of points with different colors
    for group_idx, group in enumerate(points_list):
        color = colors[group_idx % len(colors)]
        
        # Plot each point in the group and draw lines between them
        for i, point in enumerate(group):
            x, y = int(point[0]), int(point[1])
            cv2.circle(img, (x, y), radius, color, -1)  # -1 means filled circle
            if i > 0:
                prev_x, prev_y = int(group[i-1][0]), int(group[i-1][1])
                cv2.line(img, (prev_x, prev_y), (x, y), (0, 255, 255), 2)  # Draw yellow line between points
                
                # Calculate the center point of the line
                center_x = (prev_x + x) // 2
                center_y = (prev_y + y) // 2
                cv2.circle(img, (center_x, center_y), radius, (0, 0, 255), -1)  # Plot center point in red

    # Display the image
    cv2.imshow('Points Visualization', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return [center_x, center_y]


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
    grasp_center = plot_grasp_points(f'{folder_path}/rgb_image/image_0.jpg', [keypoints_to_print], radius=3)
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


if __name__ == "__main__":
    # Example usage:
    target_classes = input("Enter target classes (comma-separated): ").split(",")
    grasp_object = input("Enter the object being grasped: ")
    capture_frames_to_send()
    process_captured_images(target_classes, grasp_object)


# import csv
# import json
# import os
# import cv2
# import numpy as np
# from PIL import Image
# from datetime import datetime
# from hi_robotics.vision_ai.cameras.intel_realsense_camera import IntelRealSenseCamera
# from gemini_constant_api_key import GEMINI_API_KEY
# from utils import get_real_world_coordinates, transform_coordinates, create_zip_archive, process_images
# from gemini_oop_object_detection import ObjectDetector

# def create_camera():
#     """Initialize and return the RealSense camera."""
#     return IntelRealSenseCamera()

# def create_capture_directories(root_dir="captured_frames"):
#     """Create directories for storing captured frames with timestamp."""
#     current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
#     folder_name = f"{root_dir}/captured_frame_{current_time}"
    
#     subdirs = ['rgb_image', 'depth_image', 'output']
#     os.makedirs(root_dir, exist_ok=True)
    
#     for subdir in subdirs:
#         os.makedirs(f'{folder_name}/{subdir}', exist_ok=True)
    
#     return folder_name

# def save_frames(color_image, depth_image, folder_path):
#     """Save color and depth frames to specified folder."""
#     # Save color image
#     color_path = os.path.join(f'{folder_path}/rgb_image', "image_0.jpg")
#     cv2.imwrite(color_path, color_image)
    
#     # Save depth data
#     depth_path = os.path.join(f'{folder_path}/depth_image', "image_0.npy")
#     np.save(depth_path, depth_image)
    
#     return {
#         'color_path': color_path,
#         'depth_path': depth_path
#     }

# def capture_frames():
#     """Capture frames from RealSense camera and return saved paths."""
#     camera = create_camera()
#     folder_path = None
#     saved_paths = None
    
#     try:
#         while True:
#             frames = camera.get_frames()
#             color_frame, depth_frame = frames
            
#             if not depth_frame or not color_frame:
#                 continue
            
#             # Convert to numpy arrays
#             depth_image = np.asanyarray(depth_frame.get_data())
#             color_image = np.asanyarray(color_frame.get_data())
            
#             # Display frames
#             depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
#             display_frames = np.hstack((color_image, depth_colormap))
#             cv2.imshow('RGB & Depth', display_frames)
            
#             key = cv2.waitKey(1)
            
#             if key == ord('s'):
#                 folder_path = create_capture_directories()
#                 saved_paths = save_frames(color_image, depth_image, folder_path)
#                 print(f"Frames captured and saved in {folder_path}")
#                 break
#             elif key == ord('q'):
#                 break
                
#     finally:
#         # cv2.destroyAllWindows()
#         pass
        
#     if folder_path:
#         # Create zip archives
#         rgb_zip_path = f"{folder_path}/rgb.zip"
#         depth_zip_path = f"{folder_path}/depth.zip"
#         create_zip_archive(f'{folder_path}/rgb_image', rgb_zip_path)
#         create_zip_archive(f'{folder_path}/depth_image', depth_zip_path)
        
#         return {
#             'folder_path': folder_path,
#             'rgb_zip_path': rgb_zip_path,
#             'depth_zip_path': depth_zip_path,
#             **saved_paths
#         }
#     return None

# def calculate_2d_angle(center_point, grasp_point):
#     """Calculate 2D angle in degrees between grasp direction and positive x-axis."""
#     delta_x = grasp_point[0] - center_point[0]
#     delta_y = center_point[1] - grasp_point[1]  # Inverted because y increases downward
#     angle_rad = np.arctan2(delta_y, delta_x)
#     angle_deg = np.degrees(angle_rad)
#     return (angle_deg + 360) % 360

# def calculate_3d_angle(center_xyz, grasp_xyz):
#     """Calculate 3D angles (theta and phi) in spherical coordinates."""
#     rel_pos = grasp_xyz - center_xyz
#     r = np.linalg.norm(rel_pos)
    
#     theta = np.degrees(np.arctan2(rel_pos[1], rel_pos[0]))
#     theta = (theta + 360) % 360
    
#     phi = np.degrees(np.arccos(rel_pos[2] / r))
    
#     return theta, phi, r

# def determine_pickup_mode(angle_2d):
#     """Determine pickup mode based on 2D angle."""
#     normalized_angle = angle_2d if angle_2d <= 180 else angle_2d - 180
#     threshold = 30
    
#     if (normalized_angle <= threshold or normalized_angle >= 180 - threshold):
#         return "HORIZONTAL PICKUP"
#     elif (90 - threshold <= normalized_angle <= 90 + threshold):
#         return "VERTICAL PICKUP"
#     else:
#         return "UNDEFINED"

# def remap_angle(theta, in_min=250, in_max=150, out_min=30, out_max=125):
#     """Remap theta angle to robot angle range."""
#     robot_angle = (theta - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
#     return 180 - robot_angle

# def process_frames(paths, target_classes, grasp_object, api_key):
#     """Process captured frames and analyze object distances from grasp points."""
#     # Process images and get keypoints
#     decoded_csv_data, keypoints = process_images(
#         paths['rgb_zip_path'],
#         paths['depth_zip_path'],
#         output_dir=f"{paths['folder_path']}/output"
#     )
    
#     keypoint_3, keypoint_7 = keypoints[0][3], keypoints[0][7]
#     keypoints_to_print = [keypoint_3, keypoint_7]
    
#     # Initialize object detector and detect objects
#     object_detector = ObjectDetector(api_key=api_key, recording_dir=paths['folder_path'])
#     im = Image.open(paths['color_path'])
#     object_centers = object_detector.get_object_centers(im, target_classes=target_classes)
    
#     if grasp_object not in object_centers:
#         raise ValueError(f"Grasp object '{grasp_object}' not found in detected objects: {list(object_centers.keys())}")
    
#     # Transform coordinates and calculate measurements
#     for obj in object_centers:
#         object_centers[obj]["real_coords"] = transform_coordinates(
#             *get_real_world_coordinates(
#                 paths['folder_path'],
#                 object_centers[obj]["center"][0],
#                 object_centers[obj]['center'][1]
#             )
#         )
    
#     # Plot grasp points and get center
#     grasp_center = plot_grasp_points(paths['color_path'], [keypoints_to_print], radius=3)
    
#     # Calculate real world coordinates and transformations
#     real_world_grasp_center = get_real_world_coordinates(
#         paths['folder_path'],
#         grasp_center[0],
#         grasp_center[1]
#     )
#     transformed_grasp_center = transform_coordinates(*real_world_grasp_center)
    
#     # Calculate measurements
#     z_distance = abs(
#         transformed_grasp_center[2] - 
#         object_centers[grasp_object]['real_coords'][2]
#     )
    
#     angle_2d = calculate_2d_angle(
#         object_centers[grasp_object]['center'],
#         grasp_center
#     )
    
#     theta, phi, radius = calculate_3d_angle(
#         np.array(object_centers[grasp_object]['real_coords']),
#         np.array(transformed_grasp_center)
#     )
    
#     robot_angle = remap_angle(theta)
#     pickup_mode = determine_pickup_mode(angle_2d)
    
#     return {
#         "grasp_object": grasp_object,
#         "z_distance": z_distance,
#         "transformed_grasp_center": transformed_grasp_center,
#         "object_centers": object_centers,
#         "keypoints": keypoints,
#         "angle_2d": angle_2d,
#         "theta": theta,
#         "phi": phi,
#         "radius": radius,
#         "robot_angle": robot_angle,
#         "pickup_mode": pickup_mode
#     }

# def save_measurements(measurements, folder_path):
#     """Save measurements to CSV and JSON files."""
#     # Save to CSV
#     csv_path = os.path.join('captured_frames', "grasp_distances.csv")
#     file_exists = os.path.isfile(csv_path)
    
#     with open(csv_path, mode='a', newline='') as file:
#         writer = csv.writer(file)
#         if not file_exists:
#             writer.writerow(['object_name', 'z_distance'])
#         writer.writerow([
#             measurements['grasp_object'],
#             measurements['z_distance']
#         ])
    
#     # Save to JSON
#     json_path = os.path.join(folder_path, "grasp_data.json")
#     data_to_save = {
#         "grasp_object": measurements['grasp_object'],
#         "z_distance": measurements['z_distance'],
#         "transformed_grasp_center": measurements['transformed_grasp_center'],
#         "object_centers": {
#             k: {"real_coords": v["real_coords"]} 
#             for k, v in measurements['object_centers'].items()
#         }
#     }
#     print(data_to_save)
    
#     with open(json_path, 'w') as json_file:
#         json.dump(data_to_save, json_file, indent=4)

# def main():
#     """Main function to run the capture and processing pipeline."""
#     target_classes = input("Enter target classes (comma-separated): ").split(",")
#     grasp_object = input("Enter the object being grasped: ")
    
#     # Capture frames
#     paths = capture_frames()
#     if not paths:
#         print("No frames captured")
#         return
    
#     # Process frames
#     try:
#         measurements = process_frames(paths, target_classes, grasp_object, GEMINI_API_KEY)
#         save_measurements(measurements, paths['folder_path'])
#         print("Processing completed successfully")
#     except Exception as e:
#         print(f"Error processing frames: {str(e)}")

# if __name__ == "__main__":
#     main()