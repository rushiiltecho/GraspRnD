import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time
from datetime import datetime
import zipfile
import requests

from utils import create_zip_archive, process_images

def create_pipeline():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable both streams
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Start streaming
    pipeline.start(config)
    return pipeline

def capture_and_save_frames(pipeline):
    # Get current timestamp for folder name
    root_dir = "captured_frames"
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{root_dir}/captured_frames_{current_time}"
    
    # Create directory if it doesn't exist
    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(folder_name, exist_ok=True)
    os.makedirs(f'{folder_name}/rgb_image', exist_ok=True)
    os.makedirs(f'{folder_name}/output', exist_ok=True)
    os.makedirs(f'{folder_name}/depth_image', exist_ok=True)
    
    try:
        # Wait for a coherent pair of frames
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
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

def create_zip(folder_path):
    """Create a zip file of the captured frames"""
    zip_path = f"{folder_path}.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)
    return zip_path

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
    # Replace with your API endpoint
    API_URL = "http://techolution.ddns.net:5000/process_pose"
    
    # Initialize RealSense pipeline
    pipeline = create_pipeline()
    
    try:
        while True:
            # Show the frames
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
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
                folder_path = capture_and_save_frames(pipeline)
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
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()

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
    cv2.imshow('Points Visualization', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # Your image path (replace with actual image path)
    image_path = "/home/sai/openai_realtime_client_grasp/captured_frames/captured_frames_20250114_181808/rgb_image/image_0.jpg"  # If this doesn't exist, will create blank image
    
    # Parse and plot points
    points_data = [
        [
            [
                312.97021484375,
                383.9329833984375
            ],
            [
                342.99951171875,
                348.8988037109375
            ],
            [
                388.0435791015625,
                293.844970703125
            ],
            [
                403.0582275390625,
                273.825439453125
            ],
            [
                418.0728759765625,
                248.80096435546875
            ],
            [
                348.00439453125,
                268.82049560546875
            ],
            [
                368.02398681640625,
                248.80096435546875
            ],
            [
                398.0533447265625,
                243.79608154296875
            ],
            [
                423.0777587890625,
                248.80096435546875
            ],
            [
                353.00927734375,
                293.844970703125
            ],
            [
                378.03375244140625,
                278.830322265625
            ],
            [
                428.0826416015625,
                283.835205078125
            ],
            [
                453.10711669921875,
                293.844970703125
            ],
            [
                358.01416015625,
                318.869384765625
            ],
            [
                393.0484619140625,
                313.864501953125
            ],
            [
                428.0826416015625,
                318.869384765625
            ],
            [
                453.10711669921875,
                323.874267578125
            ],
            [
                363.01904296875,
                343.89385986328125
            ],
            [
                393.0484619140625,
                348.8988037109375
            ],
            [
                423.0777587890625,
                348.8988037109375
            ],
            [
                448.1021728515625,
                358.9085693359375
            ]
        ],
        [
            [
                510.36444091796875,
                202.0063018798828
            ],
            [
                510.09930419921875,
                199.09036254882812
            ],
            [
                503.7373046875,
                200.1507110595703
            ],
            [
                502.14678955078125,
                199.6205291748047
            ],
            [
                504.00238037109375,
                202.0063018798828
            ],
            [
                504.00238037109375,
                202.0063018798828
            ],
            [
                505.062744140625,
                202.0063018798828
            ],
            [
                505.062744140625,
                202.0063018798828
            ],
            [
                504.00238037109375,
                202.0063018798828
            ],
            [
                504.00238037109375,
                202.0063018798828
            ],
            [
                504.00238037109375,
                202.0063018798828
            ],
            [
                504.00238037109375,
                202.0063018798828
            ],
            [
                504.00238037109375,
                202.0063018798828
            ],
            [
                503.2071533203125,
                199.09036254882812
            ],
            [
                504.00238037109375,
                202.0063018798828
            ],
            [
                504.00238037109375,
                202.0063018798828
            ],
            [
                504.00238037109375,
                202.0063018798828
            ],
            [
                508.2437744140625,
                202.0063018798828
            ],
            [
                507.18341064453125,
                202.0063018798828
            ],
            [
                505.062744140625,
                202.0063018798828
            ],
            [
                505.062744140625,
                202.0063018798828
            ]
        ],
        [
            [
                523.2362060546875,
                163.39849853515625
            ],
            [
                524.1447143554688,
                158.12916564941406
            ],
            [
                523.7813110351562,
                156.4938507080078
            ],
            [
                523.0545043945312,
                155.0402374267578
            ],
            [
                525.2349243164062,
                161.036376953125
            ],
            [
                521.9642944335938,
                155.0402374267578
            ],
            [
                528.9597778320312,
                168.9403839111328
            ],
            [
                523.0545043945312,
                161.76318359375
            ],
            [
                525.2349243164062,
                161.94488525390625
            ],
            [
                528.9597778320312,
                168.9403839111328
            ],
            [
                522.3276977539062,
                161.21807861328125
            ],
            [
                523.2362060546875,
                161.94488525390625
            ],
            [
                525.2349243164062,
                161.94488525390625
            ],
            [
                524.6898193359375,
                155.94874572753906
            ],
            [
                524.6898193359375,
                155.94874572753906
            ],
            [
                525.4166259765625,
                159.7644805908203
            ],
            [
                523.599609375,
                162.1265869140625
            ],
            [
                524.6898193359375,
                155.94874572753906
            ],
            [
                524.8715209960938,
                156.1304473876953
            ],
            [
                525.2349243164062,
                159.21937561035156
            ],
            [
                525.2349243164062,
                160.6729736328125
            ]
        ],
        [
            [
                99.18907165527344,
                170.82298278808594
            ],
            [
                80.87284088134766,
                184.96966552734375
            ],
            [
                80.87284088134766,
                184.96966552734375
            ],
            [
                98.29559326171875,
                159.80345153808594
            ],
            [
                101.27384948730469,
                172.31210327148438
            ],
            [
                99.18907165527344,
                175.88600158691406
            ],
            [
                97.99777221679688,
                159.2078094482422
            ],
            [
                97.99777221679688,
                159.50563049316406
            ],
            [
                98.29559326171875,
                160.10128784179688
            ],
            [
                97.10429382324219,
                176.77947998046875
            ],
            [
                98.29559326171875,
                177.3751220703125
            ],
            [
                100.38037109375,
                161.5904083251953
            ],
            [
                99.48689270019531,
                179.16207885742188
            ],
            [
                97.10429382324219,
                176.77947998046875
            ],
            [
                97.10429382324219,
                177.3751220703125
            ],
            [
                99.78472137451172,
                163.07952880859375
            ],
            [
                99.18907165527344,
                179.45989990234375
            ],
            [
                96.80647277832031,
                178.56642150878906
            ],
            [
                99.78472137451172,
                163.07952880859375
            ],
            [
                100.38037109375,
                163.07952880859375
            ],
            [
                100.67819213867188,
                163.07952880859375
            ]
        ],
        [
            [
                29.840049743652344,
                472.2798156738281
            ],
            [
                31.058876037597656,
                471.0610046386719
            ],
            [
                12.776435852050781,
                390.6182556152344
            ],
            [
                12.776435852050781,
                425.9643249511719
            ],
            [
                16.43292236328125,
                425.9643249511719
            ],
            [
                16.43292236328125,
                433.2773132324219
            ],
            [
                80.42149353027344,
                421.69842529296875
            ],
            [
                80.42149353027344,
                421.69842529296875
            ],
            [
                80.42149353027344,
                414.38543701171875
            ],
            [
                31.058876037597656,
                414.9948425292969
            ],
            [
                34.715370178222656,
                425.9643249511719
            ],
            [
                80.42149353027344,
                421.69842529296875
            ],
            [
                5.4634552001953125,
                430.8396301269531
            ],
            [
                43.24717712402344,
                461.3103942871094
            ],
            [
                12.776435852050781,
                449.1221008300781
            ],
            [
                12.776435852050781,
                458.8727111816406
            ],
            [
                16.43292236328125,
                464.9668884277344
            ],
            [
                45.684837341308594,
                469.8421936035156
            ],
            [
                38.371856689453125,
                449.1221008300781
            ],
            [
                39.59068298339844,
                440.5903015136719
            ],
            [
                15.214096069335938,
                475.9363098144531
            ]
        ]
    ]


    plot_points_1(image_path, points_data)

# if __name__ == "__main__":
#     main()