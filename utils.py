import json
import os
import re
import shutil
import tempfile
import zipfile
import h5py
import json
import numpy as np
import pyrealsense2 as rs
import vertexai
from google.cloud import storage

import yaml
import requests


def create_zip_archive(source_dir, zip_name_with_path):
    """Create a zip file from a directory"""
    with zipfile.ZipFile(zip_name_with_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(source_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, source_dir)
                zipf.write(file_path, arcname)
    print(f"Created zip archive: {zip_name_with_path}")

def get_base64_encoded_hamer_response(rgb_zip_path, depth_zip_path, url_endpoint= "http://techolution.ddns.net:5000/process_pose",):
    url = url_endpoint if url_endpoint else "http://techolution.ddns.net:5000/process_pose"
   
    files = {
        'rgb_data': ('rgb.zip', open(rgb_zip_path, 'rb')),
        'depth_data': ('depth.zip', open(depth_zip_path, 'rb'))
    }
    response_encoded = None
    try:
        response = requests.post(url, files=files)
        response.raise_for_status()
        response_encoded =  json.loads(response.text)["base64_csv"]
        response_points = json.loads(response.text)["keypoints"]
        return {
            'csv_data':response_encoded, 
            'keypoints':response_points
            }
    
    except requests.exceptions.RequestException as e:
        print(f"Error: {str(e)}")
        return response_encoded

def process_images(rgb_zip_path, depth_zip_path, url_endpoint= "http://techolution.ddns.net:5000/process_pose", output_dir = "recordings/Recorded_Demo/hamer_output"):
       response_encoded = None
       response_encoded = get_base64_encoded_hamer_response(rgb_zip_path, depth_zip_path, url_endpoint)
       decoded_response = base64_to_csv(response_encoded['csv_data'], f'{output_dir}/predictions_hamer.csv')
       keypoints = response_encoded['keypoints']
       print("Response received successfully =====================", decoded_response)
       # Save CSV response
    #    with open(f'{output_dir}/predictions.csv', 'wb') as f:
    #        f.write(decoded_response)
       
       print("CSV saved as predictions.csv")
       return decoded_response, keypoints

# Example usage
# rgb_zip = "path/to/rgb.zip"
# depth_zip = "path/to/depth.zip" 
# process_images(rgb_zip, depth_zip)


def process_images_and_send_csv(rgb_zip_path, depth_zip_path, url_endpoint= "http://", output_dir = "hamer_output"):
    url = url_endpoint if url_endpoint else "http://34.28.22.203:5000/process_pose"

    files = {
        'rgb_data': ('rgb.zip', open(rgb_zip_path, 'rb')),
        'depth_data': ('depth.zip', open(depth_zip_path, 'rb'))
    }

    try:
        response = requests.post(url, files=files)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error: {str(e)}")
    finally:
        for _, f in files.values():
            f.close()


def load_config(config_path="config/config.yaml"):
    """
    Loads the configuration file (YAML format).

    Args:
        config_path (str): Path to the configuration YAML file. Defaults to "config.yaml".

    Returns:
        dict: A dictionary containing the configuration settings from the YAML file.
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


calib_matrix_x = np.array([
      [ 0.068, -0.986,  0.152, -0.108],
      [ 0.998,  0.065, -0.023,  0.0 ],
      [ 0.013,  0.153,  0.988, -0.044],
      [ 0.0,    0.0,    0.0,    1.0  ]
    ])

calib_matrix_y = np.array([
      [-0.47,   0.587,  -0.659,  0.73929],
      [ 0.877,  0.392,  -0.276, -0.16997],
      [ 0.096, -0.708,  -0.7,    0.86356],
      [ 0.0,    0.0,     0.0,    1.0    ]
    ])

# RLEF UTILS:
def convert_video(input_path, output_path):
    # Create a temporary file
    temp_output_path = tempfile.mktemp(suffix='.mp4')
    
    # Run the ffmpeg command to convert the video
    os.system(f"ffmpeg -i '{input_path}' -c:v libx264 '{temp_output_path}'")
    
    # Check if the conversion was successful
    if os.path.exists(temp_output_path):
        # Remove the original file
        os.remove(output_path)
        # Rename the temporary file to the original output path
        shutil.move(temp_output_path, output_path)
    else:
        print("Conversion failed.")


# Function to get the signed URL from the RLEF API
def get_signed_url(resource_id, hdf5_filename):
    url = "https://autoai-backend-exjsxe2nda-uc.a.run.app/resource/uploadHdf5File"
    form_data = {
        "resourceId": resource_id,
        "hdf5FileName": hdf5_filename
    }
    response = requests.put(url, data=form_data)
    
    if response.status_code == 200:
        try:
            response_dict = response.json()
            return response_dict.get("hdf5FileSignedUrlForUpload")
        except json.JSONDecodeError:
            print("Error: Response is not valid JSON.")
            return None
    else:
        print(f"Failed to get signed URL. Status code: {response.status_code}")
        return None

# Function to upload the HDF5 file to the signed URL
def upload_hdf5_file(signed_url, hdf5_filepath):
    headers = {"Content-Type": "text/csv"}
    
    with open(hdf5_filepath, 'rb') as file_data:
        response = requests.put(signed_url, headers=headers, data=file_data)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response from the server: {response.text}")



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


def get_intrinsics(metadata_filepath: str):
    config = load_config()
    camera_intrinsics = config['camera_config']['camera_intrinsics']
    color_intrinsics = camera_intrinsics['color_intrinsics']
    depth_intrinsics = camera_intrinsics['depth_intrinsics']
    return color_intrinsics, depth_intrinsics


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


def get_pixel_3d_coordinates(recording_dir, time_seconds, pixel_x, pixel_y):
    """
    Get the 3D coordinates (X, Y, Z) of a specific pixel at a specific time in the video
    
    Args:
        recording_dir (str): Path to the recording directory
        time_seconds (float): Time in seconds from the start of the video
        pixel_x (Optional[float]): X coordinate of the pixel
        pixel_y (Optional[float]): Y coordinate of the pixel
    
    Returns:
        tuple: ((X, Y, Z) coordinates in meters, actual_time in seconds)
               Returns (None, actual_time) if pixel coordinates are None
    """
    try:
        # Early return if pixel coordinates are None
        if pixel_x is None or pixel_y is None:
            print("Warning: Received None for pixel coordinates")
            return None, time_seconds
            
        with h5py.File(f"{recording_dir}/frames.h5", 'r') as h5_file:
            timestamps = h5_file['timestamps'][:]
            frame_times = timestamps[:, 0] - timestamps[0, 0]
            closest_frame = np.argmin(np.abs(frame_times - time_seconds))
            depth_frame = h5_file['depth_frames'][closest_frame]
            actual_time = frame_times[closest_frame]
            
            intrinsics_str = h5_file.attrs['camera_intrinsics']
            intrinsics_dict = json.loads(intrinsics_str)
            depth_scale = intrinsics_dict['depth_scale']
            
            color_intrinsics = rs.intrinsics()
            d_intr = intrinsics_dict['color_intrinsics']
            
            # Ensure pixel coordinates are within bounds
            pixel_x = min(max(0, float(pixel_x)), color_intrinsics.width - 1)
            pixel_y = min(max(0, float(pixel_y)), color_intrinsics.height - 1)
            
            # Get depth value and convert to meters
            depth_value = float(depth_frame[int(pixel_y), int(pixel_x)]) * depth_scale
            
            # Deproject pixel to 3D point
            point_3d = rs.rs2_deproject_pixel_to_point(
                color_intrinsics,
                [float(pixel_x), float(pixel_y)],
                depth_value
            )
            
            return point_3d, actual_time
            
    except Exception as e:
        print(f"Error in get_pixel_3d_coordinates: {e}")
        return None, time_seconds


def get_real_world_coordinates(image_dir, pixel_x, pixel_y):
    """
    Get the real world coordinates (X, Y, Z) of a specific pixel in a depth image.

    Args:
        image_dir (str): Path to the directory containing the depth image.
        pixel_x (int): X coordinate of the pixel.
        pixel_y (int): Y coordinate of the pixel.

    Returns:
        np.array: Real world coordinates (X, Y, Z) in meters.
    """
    # Load the depth image
    # depth_image_path = os.path.join(f'{image_dir}/depth_image', "depth_image.npy")
    depth_image_path = f'{image_dir}/depth_image/image_0.npy'
    depth_image = np.load(depth_image_path)

    # Load the camera intrinsics
    intrinsics = rs.intrinsics()
    color_intrinsics, depth_intrinsics = get_intrinsics("config/config.yaml")
    intrinsics.width = 640
    intrinsics.height = 480
    intrinsics.ppx = depth_intrinsics['ppx']
    intrinsics.ppy = depth_intrinsics['ppy']
    intrinsics.fx = depth_intrinsics['fx']
    intrinsics.fy = depth_intrinsics['fy']
    intrinsics.model = rs.distortion.inverse_brown_conrady
    intrinsics.coeffs = [0, 0, 0, 0, 0]

    # Deproject the pixel to 3D point
    point_3d = deproject_pixel_to_point(depth_image, (pixel_x, pixel_y), intrinsics)
    return point_3d

def _transform_coordinates(point_xyz, calib_matrix_x=calib_matrix_x, calib_matrix_y=calib_matrix_y):
    """
    Transform point through both calibration matrices
    
    Args:
        point (list): [x, y, z] coordinates
        calib_x (list): First calibration matrix (4x4)
        calib_y (list): Second calibration matrix (4x4)
    
    Returns:
        list: Final transformed coordinates as regular floats [x, y, z]
    """
    # Convert inputs to numpy arrays
    point_array = np.array([*point_xyz, 1.0])
    calib_x_array = np.array(calib_matrix_x)
    calib_y_array = np.array(calib_matrix_y)
    
    # First transformation (X calibration)
    transformed_x = calib_x_array @ point_array
    if transformed_x[3] != 1.0:
        transformed_x = transformed_x / transformed_x[3]
    
    # Second transformation (Y calibration)
    transformed_y = calib_y_array @ transformed_x
    if transformed_y[3] != 1.0:
        transformed_y = transformed_y / transformed_y[3]
    
    # Convert to regular floats and return as list
    return [float(transformed_y[0]), float(transformed_y[1]), float(transformed_y[2])]


# =================== PARSING UTILS ===================
def parse_to_json(response):
    pattern = r"```json\s*(\{.*\})"
    match = re.search(pattern, response, re.DOTALL)
    json_content = ''
    if match:
        json_content = match.group(1)  # Extract the JSON string
    else:
        print("No valid JSON found.")
    
    return json_content
    

def transform_coordinates(x, y, z):
    """Transforms coordinates from input space to cobot base."""
    B = np.eye(4)
    B[:3, 3] = [x / 1000, y / 1000, z / 1000]  # Convert to meters
    A = calib_matrix_y @ B @ np.linalg.inv(calib_matrix_x)
    transformed_x, transformed_y, transformed_z = A[:3, 3] * 1000  # Convert back to mm
    return transformed_x, transformed_y, transformed_z

def parse_list_boxes(text:str):
  result = []
  for line in text.strip().splitlines():
    # Extract the numbers from the line, remove brackets and split by comma
    try:
      numbers = line.split('[')[1].split(']')[0].split(',')
    except:
      numbers =  line.split('- ')[1].split(',')

    # Convert the numbers to integers and append to the result
    result.append([int(num.strip()) for num in numbers])

  return result


def parse_list_boxes_with_label(text:str):
  text = text.split("```\n")[0]
  return json.loads(text.strip("```").strip("python").strip("json").replace("'", '"').replace('\n', '').replace(',}', '}'))


def upload_to_bucket(destination_blob_name, file_path):
    """
    Upload a video file to a Google Cloud Storage bucket.

    Args:
        destination_blob_name (str): Name of the blob in the bucket.
        video_file_path (str): Path to the video file.

    Input Format: str, str
    Output Format: str (gs:// URL)
    """
    credentials_file="ai-hand-service-acc.json"
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_file
    storage_client = storage.Client()
    bucket = storage_client.bucket("video-analysing")
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(file_path)
    gs_url = f"gs://{bucket.name}/{destination_blob_name}"
    return gs_url


# PLOTTING UTILS ======================================================
# @title Plotting Utils
import json
import random
import io
from PIL import Image, ImageDraw
from PIL import ImageColor

additional_colors = [colorname for (colorname, colorcode) in ImageColor.colormap.items()]

def plot_bounding_boxes(im, noun_phrases_and_positions):
    """
    Plots bounding boxes on an image with markers for each noun phrase, using PIL, normalized coordinates, and different colors.

    Args:
        img_path: The path to the image file.
        noun_phrases_and_positions: A list of tuples containing the noun phrases
         and their positions in normalized [y1 x1 y2 x2] format.
    """

    # Load the image
    img = im
    width, height = img.size
    print(img.size)
    # Create a drawing object
    draw = ImageDraw.Draw(img)

    # Define a list of colors
    colors = [
    'red',
    'green',
    'blue',
    'yellow',
    'orange',
    'pink',
    'purple',
    'brown',
    'gray',
    'beige',
    'turquoise',
    'cyan',
    'magenta',
    'lime',
    'navy',
    'maroon',
    'teal',
    'olive',
    'coral',
    'lavender',
    'violet',
    'gold',
    'silver',
    ] + additional_colors

    # Iterate over the noun phrases and their positions
    for i, (noun_phrase, (y1, x1, y2, x2)) in enumerate(
        noun_phrases_and_positions):
        # Select a color from the list
        color = colors[i % len(colors)]

        # Convert normalized coordinates to absolute coordinates
        abs_x1 = int(x1/1000 * width)
        abs_y1 = int(y1/1000 * height)
        abs_x2 = int(x2/1000 * width)
        abs_y2 = int(y2/1000 * height)
        center_x, center_y = (abs_x1 + abs_x2) / 2, (abs_y1 + abs_y2) / 2

        # Draw the bounding box
        draw.rectangle(
            ((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4
        )
        draw.ellipse(
            (center_x - 1, center_y - 1, center_x + 1, center_y + 1),
            outline=colors[-14], width=5
        )


        # Draw the text
        draw.text((abs_x1 + 8, abs_y1 + 6), noun_phrase, fill=color)

    # Display the image
    img.show()


def normalize_box(box, width=640, height=480):
    """
    Normalize bounding boxes from pixel coordinates to [0, 1] range.

    Args:
        boxes (list): List of bounding boxes in [ymin, xmin, ymax, xmax] format.
        width (int): Image width.
        height (int): Image height.

    Returns:
        list: Normalized bounding boxes in [ymin, xmin, ymax, xmax] format.
    """

    ymin, xmin, ymax, xmax = box
    normalized_box = [ xmin / 1000*width, ymin / 1000*height, xmax / 1000*width, ymax / 1000*height]
    return normalized_box

# REGION SELECTOR UTILS ============================================================

def convert_time_to_seconds(time):
    time_parts = time.split(':')
    if len(time_parts) == 3:
        h, m, s = map(int, time_parts)
        return h * 3600 + m * 60 + s
    elif len(time_parts) == 2:
        m, s = map(int, time_parts)
        return m * 60 + s
    else:
        raise ValueError("Invalid time format")


import base64

# Convert CSV file to Base64
def get_csv_content(file_path):
    with open(file_path, "rb") as file:
        csv_content = file.read()
        return csv_content

def csv_to_base64(file_path):
    csv_content = get_csv_content(file_path)
    print(csv_content)
    base64_encoded = base64.b64encode(csv_content).decode('utf-8')
    
    return base64_encoded

# Convert Base64 string back to CSV file
def base64_to_csv(base64_string, output_file_path):
    csv_content = base64.b64decode(base64_string.encode('utf-8'))
    with open(output_file_path, "wb") as file:
        file.write(csv_content)
        return csv_content

# Example Usage
# csv_file_path = "example.csv"  # Input CSV file path
# base64_encoded_csv = csv_to_base64(csv_file_path)
# print("Base64 Encoded CSV:\n", base64_encoded_csv)

# output_csv_path = "decoded_example.csv"  # Output CSV file path
# base64_to_csv(base64_encoded_csv, output_csv_path)
# print(f"CSV file saved back to: {output_csv_path}")


if __name__ == "__main__":
    csv_file_path = "example.csv"  # Input CSV file path
    # base64_encoded_csv = csv_to_base64(csv_file_path)
    # # print("Base64 Encoded CSV:\n", base64_encoded_csv)

    # output_csv_path = "decoded_example.csv"  # Output CSV file path
    # base64_to_csv(base64_encoded_csv, output_csv_path)
    
    rgb_zip = "recordings/Recorded_Demo/archives/rgb_images_data_collection.zip"
    depth_zip = "recordings/Recorded_Demo/archives/depth_images_data_collection.zip"
    process_images(rgb_zip, depth_zip)


    # res = get_csv_content("hamer_output/predictions.csv")
    # res_csv =  base64_to_csv((json.loads(res)['base64_csv']), 'hamer_output/predictions_decoded.csv')
    # print(res_csv)














    # print(f"CSV file saved back to: {output_csv_path}")
# if __name__ == "__main__":
#     # Example parameters
#     recording_dir = "recordings/20241227_205319"
#     time_second = 5  # 5 seconds into the video
#     pixel_x = 320  # Center X (assuming 640x480 resolution)
#     pixel_y = 240  # Center Y (assuming 640x480 resolution)
    
#     try:
#         coords, actual_time = get_pixel_3d_coordinates(recording_dir, time_second, pixel_x, pixel_y)
#         print(f"3D coordinates at {actual_time:.3f} seconds (in meters):")
#         print(f"X: {coords[0]:.3f}")
#         print(f"Y: {coords[1]:.3f}")
#         print(f"Z: {coords[2]:.3f}")
#     except Exception as e:
#         print(f"Error getting coordinates: {str(e)}")


# if __name__== "__main__":
#     string= '''
#     ```json
#         {
#         "objects": [
#             "Coca-Cola can",
#             "Black mug",
#             "Blue water bottle"
#         ],
#         "Picking up": [
#             {
#             "start_time": "00:00",
#             "end_time": "00:03",
#             "object_name": "Coca-Cola can",
#             "notes": "Human hand picks up the Coca-Cola can."
#             },
#             {
#             "start_time": "00:03",
#             "end_time": "00:06",
#             "object_name": "Black mug",
#             "notes": "Human hand picks up the black mug after putting down the can."
#             },
#             {
#             "start_time": "00:08",
#             "end_time": "00:11",
#             "object_name": "Blue water bottle",
#             "notes": "Human hand picks up the blue water bottle."
#             }
#         ],
#         "Placing down": [
#             {
#             "start_time": "00:03",
#             "end_time": "00:04",
#             "object_name": "Coca-Cola can",
#             "notes": "Human hand places the Coca-Cola can back on the table."
#             },
#             {
#             "start_time": "00:06",
#             "end_time": "00:08",
#             "object_name": "Black mug",
#             "notes": "Human hand places the black mug back on the table."
#             }
#         ]
#         }
#         ```
#     '''

#     parse_to_json(string)