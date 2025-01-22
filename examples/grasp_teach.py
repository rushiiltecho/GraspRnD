import cv2
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs
from ultralytics import YOLO
from collections import deque
import json
from datetime import datetime
import os 
import time
import argparse
import sys




class ObjectWristTracker:
    def __init__(self, target_class='bottle'):
        # Initialize YOLO World model
        self.yolo_model = YOLO('yolov8l-world.pt')
        self.data_dir = 'data'
        os.makedirs(self.data_dir, exist_ok=True)
        self.target_class = target_class  # Set target class
        self.yolo_model.set_classes([target_class])

        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
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

        # attributes for grasp point `recording`
        self.recording = False
        self.frames_to_record = 10
        self.recorded_frames = 0
        self.grasp_points = deque(maxlen=10)
        self.object_points = deque(maxlen=10)
        self.wrist_points = deque(maxlen=10)
        self.theta_angles = deque(maxlen=10)

        self.recording_complete = False
        self.pickup_modes = deque(maxlen=10)
        self.robot_angles = deque(maxlen=10)

    def check_grasp_status(self, frame, hand_landmarks, object_center, depth_frame, object_xyz):
        """
        Check if the object is being grasped and calculate grasp center point
        Returns: bool, str, tuple (is_grasped, status_message, grasp_center_xyz)
        """
        if not hand_landmarks or not object_center:
            return False, "NO GRASP", None

        # Get key finger landmarks
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        
        # Get palm center (average of several palm landmarks)
        palm_landmarks = [
            self.mp_hands.HandLandmark.WRIST,
            self.mp_hands.HandLandmark.THUMB_CMC,
            self.mp_hands.HandLandmark.INDEX_FINGER_MCP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP
        ]
        palm_x = sum(hand_landmarks.landmark[lm].x for lm in palm_landmarks) / len(palm_landmarks)
        palm_y = sum(hand_landmarks.landmark[lm].y for lm in palm_landmarks) / len(palm_landmarks)
        
        h, w, _ = frame.shape
        
        # Convert landmark coordinates to pixel coordinates
        thumb_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))
        index_pos = (int(index_tip.x * w), int(index_tip.y * h))
        middle_pos = (int(middle_tip.x * w), int(middle_tip.y * h))
        palm_pos = (int(palm_x * w), int(palm_y * h))
        
        # Calculate grasp center point in pixel coordinates (between thumb and index)
        grasp_center_pixel = (
            (thumb_pos[0] + index_pos[0]) // 2,
            (thumb_pos[1] + index_pos[1]) // 2
        )
        
        # Get 3D coordinates for all points
        thumb_xyz = self.get_xyz_from_pixel(depth_frame, thumb_pos[0], thumb_pos[1])
        index_xyz = self.get_xyz_from_pixel(depth_frame, index_pos[0], index_pos[1])
        middle_xyz = self.get_xyz_from_pixel(depth_frame, middle_pos[0], middle_pos[1])
        palm_xyz = self.get_xyz_from_pixel(depth_frame, palm_pos[0], palm_pos[1])
        
        # Get grasp center point in 3D coordinates
        grasp_center_xyz = self.get_xyz_from_pixel(depth_frame, grasp_center_pixel[0], grasp_center_pixel[1])
        
        # Calculate 3D distances to object center
        thumb_dist_3d = np.linalg.norm(thumb_xyz - object_xyz)
        index_dist_3d = np.linalg.norm(index_xyz - object_xyz)
        middle_dist_3d = np.linalg.norm(middle_xyz - object_xyz)
        palm_dist_3d = np.linalg.norm(palm_xyz - object_xyz)
        
        # Define thresholds
        grasp_threshold_3d = 0.1  # 10cm in meters
        min_palm_dist = 0.05     # Minimum 5cm palm distance to ensure hand is not too close
        max_palm_dist = 0.15     # Maximum 15cm palm distance to ensure hand is not too far
        
        # Check grasp configuration
        fingers_close = sum([
            thumb_dist_3d < grasp_threshold_3d,
            index_dist_3d < grasp_threshold_3d,
            middle_dist_3d < grasp_threshold_3d
        ])
        
        # Calculate angles between fingers to check grasp configuration
        def calculate_angle(p1, p2, p3):
            v1 = p1 - p2
            v2 = p3 - p2
            cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        
        # Calculate angle between thumb and index finger
        grasp_angle = calculate_angle(thumb_xyz, palm_xyz, index_xyz)
        
        # Conditions for grasp detection 
        is_fingers_close = fingers_close >= 2  # At least 2 fingers need to be close
        is_palm_distance_good = min_palm_dist < palm_dist_3d < max_palm_dist
        is_grasp_angle_good = 20 < grasp_angle < 90  # Typical grasp angle range
        
        # Draw grasp center point and line between thumb and index
        if is_fingers_close and is_palm_distance_good and is_grasp_angle_good:
            cv2.line(frame, thumb_pos, index_pos, (0, 255, 255), 2)  # Yellow line
            cv2.circle(frame, grasp_center_pixel, 5, (0, 255, 255), -1)  # Yellow dot
        
        # Combined conditions for grasp detection
        if is_fingers_close and is_palm_distance_good and is_grasp_angle_good:
            return True, "GRASPED", grasp_center_xyz
        elif is_fingers_close and not is_palm_distance_good:
            return False, "VERY CLOSE", None
        elif is_fingers_close and not is_grasp_angle_good:
            return False, "BAD GRASP ANGLE", None
        else:
            return False, "NO GRASP", None
        
    def get_xyz_from_pixel(self, depth_frame, pixel_x, pixel_y):
        """Convert pixel coordinates to 3D world coordinates using RealSense depth."""
        try:
            # Check if pixels are within frame boundaries
            width = depth_frame.get_width()
            height = depth_frame.get_height()
            
            # Ensure pixels are within valid range
            pixel_x = max(0, min(int(pixel_x), width - 1))
            pixel_y = max(0, min(int(pixel_y), height - 1))
            
            # Get depth value
            depth = depth_frame.get_distance(pixel_x, pixel_y)
            
            # Check if depth is valid
            if depth <= 0 or depth > 10:  # assume max depth of 10 meters
                return np.array([0, 0, 0])
                
            point_3d = rs.rs2_deproject_pixel_to_point(
                self.depth_intrinsics, 
                [pixel_x, pixel_y], 
                depth
            )
            return np.array(point_3d)
            
        except RuntimeError as e:
            print(f"Runtime Error in get_xyz_from_pixel: {e}")
            print(f"Attempted coordinates: x={pixel_x}, y={pixel_y}")
            print(f"Frame dimensions: {width}x{height}")
            return np.array([0, 0, 0])
            
        except Exception as e:
            print(f"Error in get_xyz_from_pixel: {e}")
            return np.array([0, 0, 0])

    def calculate_2d_angle(self, center_point, wrist_point):
        """Calculate 2D angle in degrees between wrist and positive x-axis."""
        delta_x = wrist_point[0] - center_point[0]
        delta_y = center_point[1] - wrist_point[1]  # Inverted because y increases downward in image
        angle_rad = np.arctan2(delta_y, delta_x)
        angle_deg = np.degrees(angle_rad)
        return (angle_deg + 360) % 360  # Normalize to [0, 360)

    def calculate_3d_angle(self, center_xyz, wrist_xyz):
        """Calculate 3D angles (theta and phi) in spherical coordinates."""
        # Convert to relative coordinates (wrist relative to center)
        rel_pos = wrist_xyz - center_xyz
        
        # Calculate radius (distance)
        r = np.linalg.norm(rel_pos)
        
        # Calculate theta (azimuthal angle in x-y plane from x-axis)
        theta = np.degrees(np.arctan2(rel_pos[1], rel_pos[0]))
        theta = (theta + 360) % 360  # Normalize to [0, 360)
        
        # Calculate phi (polar angle from z-axis)
        phi = np.degrees(np.arccos(rel_pos[2] / r))
        
        return theta, phi, r

    def determine_pickup_mode(self, angle_2d):
        """
        Determine if the pickup mode is horizontal or vertical based on 2D angle.
        Horizontal: when angle is near 0° or 180° (±30°)
        Vertical: when angle is near 90° or 270° (±30°)
        """
        # Normalize angle to 0-180 range since opposite directions are equivalent
        normalized_angle = angle_2d if angle_2d <= 180 else angle_2d - 180
        
        # Define angle thresholds
        threshold = 30
        
        if (normalized_angle <= threshold or normalized_angle >= 180 - threshold):
            return "HORIZONTAL PICKUP"
        elif (90 - threshold <= normalized_angle <= 90 + threshold):
            return "VERTICAL PICKUP"
        else:
            return "UNDEFINED"

    def get_object_center(self, frame):
        """Get center coordinates of detected object based on target_class."""
        results = self.yolo_model(frame)
        
        for result in results[0].boxes.data:
            x1, y1, x2, y2, conf, class_id = result
            if self.yolo_model.names[int(class_id)] == self.target_class:
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                return int(center_x), int(center_y)
        return None

    
    def get_wrist_position(self, frame):
        """Get wrist coordinates using MediaPipe."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            wrist = results.multi_hand_landmarks[0].landmark[self.mp_hands.HandLandmark.WRIST]
            # Return as tuple for position and the landmarks object
            return (int(wrist.x * frame.shape[1]), int(wrist.y * frame.shape[0])), results.multi_hand_landmarks[0]
        return None, None  # Return None for both position and landmarks if not found
    
    def draw_line(self, frame, start_point, end_point):
        """Draw a line between two points."""
        cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
    def remap_angle(self, theta, in_min=250, in_max=150, out_min=30, out_max=125):
        """
        Remap theta angle to robot angle range.
        Input theta range: 250° to 150°
        Output robot range: 30° to 125°
        """
        # return (theta - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
        robot_angle = (theta - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
        return 180 - robot_angle 
    def calculate_relative_pose(self, grasp_pose, object_pose):
        """Calculate relative position between grasp point and object center."""
        relative_position = {
            'x': float(grasp_pose[0] - object_pose[0]),
            'y': float(grasp_pose[1] - object_pose[1]),
            'z': float(grasp_pose[2] - object_pose[2])
        }
        distance = np.sqrt(
            relative_position['x']**2 + 
            relative_position['y']**2 + 
            relative_position['z']**2
        )
        return relative_position, float(distance)

    def process_frame(self):
        """Process a single frame and draw annotations."""
        # Get frames from RealSense
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            return None
        
        # Convert frames to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        
        # Get object center based on target_class
        object_center = self.get_object_center(color_image)
        
        # Get wrist position and hand landmarks
        wrist_pos, hand_landmarks = self.get_wrist_position(color_image)
        
        if object_center and wrist_pos:
            # Get 3D coordinates
            object_xyz = self.get_xyz_from_pixel(depth_frame, object_center[0], object_center[1])
            wrist_xyz = self.get_xyz_from_pixel(depth_frame, wrist_pos[0], wrist_pos[1])
            
            # Check grasp status with 3D coordinates
            is_grasped, grasp_status, grasp_center_xyz = self.check_grasp_status(
                color_image, 
                hand_landmarks, 
                object_center,
                depth_frame,
                object_xyz
            )
            
            # Draw points and line
            cv2.circle(color_image, object_center, 5, (255, 0, 0), -1)
            cv2.circle(color_image, wrist_pos, 5, (0, 0, 255), -1)
            self.draw_line(color_image, object_center, wrist_pos)
            
            # Calculate angles
            angle_2d = self.calculate_2d_angle(object_center, wrist_pos)
            theta, phi, radius = self.calculate_3d_angle(object_xyz, wrist_xyz)
            
            # Calculate robot angle from theta
            robot_angle = self.remap_angle(theta)

            # Determine pickup mode
            pickup_mode = self.determine_pickup_mode(angle_2d)

            # If recording and there's a valid grasp
            if self.recording and is_grasped and grasp_center_xyz is not None:
                self.recorded_frames += 1
                self.grasp_points.append(grasp_center_xyz)
                self.object_points.append(object_xyz)
                self.wrist_points.append(wrist_xyz)
                self.theta_angles.append(theta)
                self.pickup_modes.append(pickup_mode)
                self.robot_angles.append(robot_angle)

                # Draw recording status
                cv2.putText(color_image, f"Recording: {self.recorded_frames}/{self.frames_to_record}", 
                           (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                if self.recorded_frames >= self.frames_to_record:
                    self.recording = False
                    self.recorded_frames = 0
                    self.save_recorded_points()

            # Draw 3D coordinates and angles
            cv2.putText(color_image, f"Object xyz: {object_xyz}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(color_image, f"Wrist xyz: {wrist_xyz}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Draw grasp center xyz when grasped
            if is_grasped and grasp_center_xyz is not None:
                cv2.putText(color_image, f"Grasp xyz: {grasp_center_xyz}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Draw theta and robot angle with background
            angle_text = f"3D Theta: {theta:.1f}° | Robot: {robot_angle:.1f}°"
            text_size = cv2.getTextSize(angle_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Draw background rectangle for angles
            cv2.rectangle(color_image, 
                        (10, 110), 
                        (10 + text_size[0], 110 + text_size[1] + 10),
                        (0, 0, 0),
                        -1)
            
            # Draw pickup mode with background
            mode_text = f"Mode: {pickup_mode}"
            text_size = cv2.getTextSize(mode_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            
            # Draw background rectangle for pickup mode
            cv2.rectangle(color_image, 
                         (10, 200), 
                         (10 + text_size[0], 200 + text_size[1] + 10),
                         (0, 0, 0),
                         -1)
            
            # Draw pickup mode text
            cv2.putText(color_image, mode_text, (10, 225),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                       (255, 255, 255), 2)
            # Draw angle text
            cv2.putText(color_image, angle_text, (10, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # Draw grasp status with background
            grasp_text = f"Grasp: {grasp_status}"
            grasp_text_size = cv2.getTextSize(grasp_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            
            # Draw background rectangle for grasp status
            cv2.rectangle(color_image, 
                         (10, 240), 
                         (10 + grasp_text_size[0], 240 + grasp_text_size[1] + 10),
                         (0, 0, 0),
                         -1)
            
            # Draw grasp status text with color based on status
            text_color = (0, 255, 0) if is_grasped else (0, 165, 255)  # Green if grasped, orange if not
            cv2.putText(color_image, grasp_text, (10, 265),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                       text_color, 2)
                       
            # Draw coordinate system at object center
            axis_length = 50
            cv2.arrowedLine(color_image, object_center, 
                          (object_center[0] + axis_length, object_center[1]), 
                          (255, 0, 0), 2)  # X-axis (red)
            cv2.arrowedLine(color_image, object_center, 
                          (object_center[0], object_center[1] - axis_length), 
                          (0, 255, 0), 2)  # Y-axis (green)
        
        return color_image
    def save_recorded_points(self):
        """Save the averaged recording data to a JSON file."""
        if len(self.grasp_points) == 0:
            print("No points recorded!")
            return

        # Calculate averages
        avg_grasp = np.mean(self.grasp_points, axis=0)
        avg_object = np.mean(self.object_points, axis=0)
        avg_wrist = np.mean(self.wrist_points, axis=0)
        avg_theta = np.mean(self.theta_angles)
        avg_robot_angle = np.mean(self.robot_angles)

        # pickup mode 
        pickup_mode_list = list(self.pickup_modes)
        most_common_mode = max(set(pickup_mode_list), key=pickup_mode_list.count)
        relative_pose, grasp_distance = self.calculate_relative_pose(avg_grasp, avg_object)

        # Prepare data for saving
        data = {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'grasp_point': {
                'x': float(avg_grasp[0]), 
                'y': float(avg_grasp[1]), 
                'z': float(avg_grasp[2])
            },
            'object_point': {
                'x': float(avg_object[0]), 
                'y': float(avg_object[1]), 
                'z': float(avg_object[2])
            },
            'wrist_point': {
                'x': float(avg_wrist[0]), 
                'y': float(avg_wrist[1]), 
                'z': float(avg_wrist[2])
            },
            'angles': {
                'theta': float(avg_theta),
                'robot_angle': float(avg_robot_angle)
            },
            'relative_pose': {
                'position': relative_pose,
                'distance': grasp_distance
            },
            'pickup_mode': most_common_mode
        }
        # Save to file in data directory
        filename = os.path.join(self.data_dir, f'grasp_recording_{data["timestamp"]}.json')
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        
        print(f"Recording saved to {filename}")
        # stop running the code         
        # Clear the recording buffers
        self.grasp_points.clear()
        self.object_points.clear()
        self.wrist_points.clear()
        self.theta_angles.clear()

        # set flag to indicate recording has stopped
        self.recording_complete = True
    def run(self):
        """Modified main loop for automatic recording after buffer."""
        try:
            print("Starting 3-second buffer before recording...")
            buffer_time = 3  # seconds
            start_time = time.time()
            
            # Buffer period - just display frames
            while time.time() - start_time < buffer_time:
                frame = self.process_frame()
                if frame is not None:
                    cv2.imshow('Object and Wrist Tracking', frame)
                    cv2.waitKey(1)
            
            print("Buffer complete. Starting recording...")
            self.recording = True
            self.recorded_frames = 0
            
            # Record 10 frames
            while self.recorded_frames < self.frames_to_record:
                frame = self.process_frame()
                if frame is not None:
                    cv2.imshow('Object and Wrist Tracking', frame)
                    cv2.waitKey(1)
                    
                    # Save frame data regardless of grasp status
                    object_center = self.get_object_center(frame)
                    wrist_pos, hand_landmarks = self.get_wrist_position(frame)
                    
                    if object_center and wrist_pos:
                        # Get frames from RealSense
                        frames = self.pipeline.wait_for_frames()
                        aligned_frames = self.align.process(frames)
                        depth_frame = aligned_frames.get_depth_frame()
                        
                        # Get 3D coordinates
                        object_xyz = self.get_xyz_from_pixel(depth_frame, object_center[0], object_center[1])
                        wrist_xyz = self.get_xyz_from_pixel(depth_frame, wrist_pos[0], wrist_pos[1])
                        
                        # Calculate grasp center (midpoint between thumb and index if available)
                        if hand_landmarks:
                            thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                            index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                            grasp_x = (thumb_tip.x + index_tip.x) / 2 * frame.shape[1]
                            grasp_y = (thumb_tip.y + index_tip.y) / 2 * frame.shape[0]
                            grasp_center_xyz = self.get_xyz_from_pixel(depth_frame, int(grasp_x), int(grasp_y))
                        else:
                            grasp_center_xyz = wrist_xyz  # fallback to wrist position
                        
                        # Calculate angles
                        theta, phi, radius = self.calculate_3d_angle(object_xyz, wrist_xyz)
                        robot_angle = self.remap_angle(theta)
                        pickup_mode = self.determine_pickup_mode(self.calculate_2d_angle(object_center, wrist_pos))
                        
                        # Store the data
                        self.grasp_points.append(grasp_center_xyz)
                        self.object_points.append(object_xyz)
                        self.wrist_points.append(wrist_xyz)
                        self.theta_angles.append(theta)
                        self.pickup_modes.append(pickup_mode)
                        self.robot_angles.append(robot_angle)
                        self.recorded_frames += 1
                        
            # Cleanup
            self.pipeline.stop()
            cv2.destroyAllWindows()
            self.hands.close()
            
            return {
                'success': self.recording_complete,
                'frames_recorded': self.recorded_frames,
                'data_dir': self.data_dir
            }
            
        except Exception as e:
            print(f"Error during recording: {e}")
            self.pipeline.stop()
            cv2.destroyAllWindows()
            self.hands.close()
            return {
                'success': False,
                'error': str(e)
            }

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Object and Wrist Tracking with automatic recording')
    parser.add_argument('--target', type=str, default='bottle',
                      help='Target object class to track (default: bottle)')
    parser.add_argument('--frames', type=int, default=10,
                      help='Number of frames to record (default: 10)')
    parser.add_argument('--buffer', type=int, default=3,
                      help='Buffer time in seconds before recording (default: 3)')

    args = parser.parse_args()

    try:
        # Initialize tracker with command line arguments
        tracker = ObjectWristTracker(target_class=args.target)
        
        # Optionally set custom frames to record
        tracker.frames_to_record = args.frames
        
        print(f"Starting tracking for '{args.target}' with {args.frames} frames...")
        print("Make sure the object and your hand are visible to the camera")
        
        # Run the tracker
        result = tracker.run()
        
        # Print results
        if result['success']:
            print(f"\nRecording completed successfully!")
            print(f"Frames recorded: {result['frames_recorded']}")
            print(f"Data saved in: {result['data_dir']}")
        else:
            print(f"\nRecording failed!")
            if 'error' in result:
                print(f"Error: {result['error']}")
                
    except KeyboardInterrupt:
        print("\nRecording interrupted by user")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        print("\nRecording session ended")

if __name__ == "__main__":
    main()