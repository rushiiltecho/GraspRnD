import pyrealsense2 as rs
import numpy as np
import cv2
import os
import shutil
import requests
from datetime import datetime

def setup_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    return pipeline

def create_save_directories():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    rgb_dir = f'rgb_{timestamp}'
    depth_dir = f'depth_{timestamp}'
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    return rgb_dir, depth_dir

def create_zip_files(rgb_dir, depth_dir):
    rgb_zip = f'{rgb_dir}.zip'
    depth_zip = f'{depth_dir}.zip'
    
    # Create separate zip files for RGB and depth
    shutil.make_archive(rgb_dir, 'zip', rgb_dir)
    shutil.make_archive(depth_dir, 'zip', depth_dir)
    
    # Cleanup directories
    shutil.rmtree(rgb_dir)
    shutil.rmtree(depth_dir)
    
    return rgb_zip, depth_zip

def process_images_and_send_csv(rgb_zip_path, depth_zip_path, 
                              url_endpoint="http://techolution.ddns.net:5000/process_pose",
                              output_dir="hamer_output"):
    url = url_endpoint if url_endpoint else "http://34.28.22.203:5000/process_pose"
    
    files = {
        'rgb_data': ('rgb.zip', open(rgb_zip_path, 'rb')),
        'depth_data': ('depth.zip', open(depth_zip_path, 'rb'))
    }
    
    try:
        response = requests.post(url, files=files)
        response.raise_for_status()
        print("Successfully sent data to API")
        print(f"Response: {response.text}")
        
    except requests.exceptions.RequestException as e:
        print(f"Error: {str(e)}")
        
    finally:
        # Close file handles
        for _, f in files.values():
            f.close()
        # Clean up zip files
        os.remove(rgb_zip_path)
        os.remove(depth_zip_path)

def main():
    pipeline = setup_realsense()
    rgb_dir, depth_dir = create_save_directories()
    
    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue
            
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # Show images
            cv2.imshow('RealSense', color_image)
            key = cv2.waitKey(1)
            
            if key == ord('s'):
                # Save RGB as jpg
                rgb_filename = os.path.join(rgb_dir, 'rgb.jpg')
                cv2.imwrite(rgb_filename, color_image)
                
                # Save raw depth as npy
                depth_filename = os.path.join(depth_dir, 'depth.npy')
                np.save(depth_filename, depth_image)
                
                print("Images saved successfully")
                break  # Exit the capture loop after saving
                
            elif key == ord('q'):
                return  # Exit without saving
                
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        
        # Create zip files and send to API only if files were saved
        if os.path.exists(rgb_dir) and os.path.exists(depth_dir):
            print("Creating zip files...")
            rgb_zip, depth_zip = create_zip_files(rgb_dir, depth_dir)
            print("Sending to API...")
            process_images_and_send_csv(rgb_zip, depth_zip)
            print("Process completed")

if __name__ == "__main__":
    main()