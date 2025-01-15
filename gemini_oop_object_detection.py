import google.generativeai as genai
import json
from pathlib import Path
import cv2
import json_repair
import numpy as np
from PIL import Image
from gemini_constant_api_key import GEMINI_API_KEY
from typing import Dict, List, Tuple, Optional
import os
import torch 

from utils import convert_time_to_seconds, get_pixel_3d_coordinates, normalize_box, plot_bounding_boxes, transform_coordinates

class ObjectDetector:
    """A class to handle object detection using Google's Gemini model."""
    
    def __init__(self, api_key: str, model_name: str = 'gemini-1.5-flash-002', recording_dir = None):
        """
        Initialize the ObjectDetector.
        
        Args:
            api_key (str): Gemini API key
            model_name (str): Name of the Gemini model to use
        """
        self.boxes = None
        self.configure_gemini(api_key)
        self.model = genai.GenerativeModel(model_name=model_name)
        self.default_prompt = (
            "Return bounding boxes for object in the"
            " following format as a list. \n ```json{'<object_name>' : [xmin, ymin, xmax, ymax]"
            " ...}``` \n If there are more than one instance of an object, add"
            " them to the dictionary as '<object_name>', '<object_name>', etc."
        )
        self.recording_dir = Path(recording_dir)
        # self.video_path = self.recording_dir / "color.mp4"

        # # Open video to get properties
        # self.cap = cv2.VideoCapture(str(self.video_path))
        # if not self.cap.isOpened():
        #     raise RuntimeError(f"Could not open video file: {self.video_path}")
            
        # self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        # self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # self.duration = self.frame_count / self.fps
        
        # print(f"Video properties:")
        # print(f"- Duration: {self.duration:.2f} seconds")
        # print(f"- Frame count: {self.frame_count}")
        # print(f"- FPS: {self.fps}")
        
        # self.cap.release()

        
        # if not self.video_path.exists():
        #     raise FileNotFoundError(f"Video file not found: {self.video_path}")
        
    # TODO: return a PIL image
    def get_frame_at_time(self, time_seconds) -> Image.Image:
        """Extract frame from video at specific time and return as PIL image"""
        # Check if requested time is valid
        if time_seconds < 0 or time_seconds >= self.duration:
            raise ValueError(
                f"Requested time {time_seconds:.2f} seconds is outside video duration "
                f"(0 to {self.duration:.2f} seconds)"
            )
            
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {self.video_path}")
            
        # Convert time to frame number
        frame_number = int(time_seconds * self.fps)
        
        # Ensure frame number is valid
        if frame_number >= self.frame_count:
            frame_number = self.frame_count - 1
            
        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(
                f"Could not extract frame at {time_seconds} seconds "
                f"(frame {frame_number} of {self.frame_count})"
            )
        
        # Convert frame (numpy array) to PIL image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)

    @staticmethod
    def configure_gemini(api_key: str) -> None:
        """Configure Gemini with the provided API key."""
        genai.configure(api_key=api_key)

    def detect_objects_1(self, image_path: Optional[str]= None, image: Optional[Image.Image] = None,) -> List[Dict]:
        """
        Detect objects in the given image using Gemini model.
        
        Args:
            image_path (str): Path to the image file
            prompt (str, optional): Custom prompt for the model
            
        Returns:
            Dict: Dictionary containing detected objects and their bounding boxes
        """
        try:
            im = image if image else Image.open(image_path)
            prompt_text = (
            "Return bounding boxes for object in the"
            " following format as a list. \n ```json{'<object_name>' : [xmin, ymin, xmax, ymax]"
            " ...}``` \n If there are more than one instance of an object, add"
            " them to the dictionary as '<object_name>', '<object_name>', etc."
        )
            
            response = self.model.generate_content([im, prompt_text])
            print(response.text)
            boxes = json.loads(json_repair.repair_json(self._parse_to_json(response.text)))
            self.boxes= boxes
            return boxes
        except Exception as e:
            print(f"EXCEPTION during detect_objects: {e}")    

    def detect_objects(self, image_path: Optional[str]= None, image: Optional[Image.Image] = None, target_class: str= None) -> List[Dict]:
        """
        Detect objects in the given image using Gemini model.
        
        Args:
            image_path (str): Path to the image file
            prompt (str, optional): Custom prompt for the model
            
        Returns:
            Dict: Dictionary containing detected objects and their bounding boxes
        """
        try:
            im = image if image else Image.open(image_path)
            prompt_text = (
            "Return bounding boxes for object in the"
            f" following format as a list. \n ```json{'{'}'{target_class}' : [xmin, ymin, xmax, ymax]"
            " ...}``` \n If there are more than one instance of an object, add"
            " them to the dictionary as '{target_class}', '<object_name>', etc."
        )
            
            response = self.model.generate_content([im, prompt_text])
            print(response.text)
            boxes = json.loads(json_repair.repair_json(self._parse_to_json(response.text)))
            self.boxes= boxes
            return boxes
        except Exception as e:
            print(f"EXCEPTION during detect_objects: {e}")    


    def visualize_detections(self, 
                           image: Image.Image, 
                           boxes: Dict, 
                           output_dir: str,
                           filename: str = 'detection_visualization.jpg') -> Tuple[int, int]:
        """
        Visualize detected objects with bounding boxes and save the result.
        
        Args:
            image_path (str): Path to the original image
            boxes (Dict): Dictionary of detected objects and their bounding boxes
            output_dir (str): Directory to save the visualization
            filename (str): Name of the output file
            
        Returns:
            Tuple[int, int]: Image dimensions (width, height)
        """
        im = image # Image.open(image_path)
        self._plot_bounding_boxes(im, list(boxes.items()))
        
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        im.save(output_path)
        
        return im.width, im.height
    
    @staticmethod
    def _parse_to_json(text: str) -> str:
        """Extract JSON from the model's response text."""
        start_idx = text.find('{')
        end_idx = text.rfind('}') + 1
        if start_idx == -1 or end_idx == 0:
            raise ValueError("No JSON object found in the response")
        return text[start_idx:end_idx]
    
    @staticmethod
    def _plot_bounding_boxes(image: Image.Image, 
                           noun_phrases_and_positions: List[Tuple]) -> None:
        """Plot bounding boxes on the image."""
        plot_bounding_boxes(image, noun_phrases_and_positions)

    def get_real_boxes(self):
        if self.boxes is None:
            return None
        return {i: normalize_box(j) for i, j in self.boxes.items()}
    
    def get_object_center(self, im:Image, target_class:str):
        """
        Get the center of the detected object.
        
        Args:
            im: PIL Image
            target_class (str): Object class to detect
                
        Returns:
            Tuple[Optional[int], Optional[int], Optional[np.ndarray], Optional[float]]: 
                Center coordinates, bounding box, confidence score. All None if detection fails.
        """
        # Detect object
        unscaled_boxes = self.detect_objects(image=im, target_class=target_class) 
        if not unscaled_boxes:  # If detection fails
            print(f"No objects detected for class {target_class}")
            return None, None, None, None
            
        boxes = self.get_real_boxes()
        # self.visualize_detections(im, unscaled_boxes, self.recording_dir)
        
        if target_class not in boxes:
            print(f"Target class {target_class} not found in detected boxes")
            return None, None, None, None
        
        # Get bounding box and confidence score
        box = boxes[target_class]
        confidence = 100
        
        # Calculate center coordinates
        center_x = int((box[0] + box[2]) / 2)
        center_y = int((box[1] + box[3]) / 2)
        
        return center_x, center_y, box, confidence


    def get_object_centers(self, im: Image, target_classes: List[str]) -> Dict[str, Tuple[Optional[int], Optional[int], Optional[np.ndarray], Optional[float]]]:
        """
        Get the centers of the detected objects for the given target classes.
        
        Args:
            im: PIL Image
            target_classes (List[str]): List of object classes to detect
                
        Returns:
            Dict[str, Tuple[Optional[int], Optional[int], Optional[np.ndarray], Optional[float]]]: 
                Dictionary with target class as key and tuple of center coordinates, bounding box, confidence score as value. 
                All None if detection fails for a class.
        """
        centers = {}
        for target_class in target_classes:
            # Detect object
            unscaled_boxes = self.detect_objects(image=im, target_class=target_class) 
            if not unscaled_boxes:  # If detection fails
                print(f"No objects detected for class {target_class}")
                centers[target_class] = (None, None, None, None)
                continue
                
            boxes = self.get_real_boxes()
            # self.visualize_detections(im, unscaled_boxes, self.recording_dir)
            
            if target_class not in boxes:
                print(f"Target class {target_class} not found in detected boxes")
                centers[target_class] = (None, None, None, None)
                continue
            
            # Get bounding box and confidence score
            box = boxes[target_class]
            confidence = 100
            
            # Calculate center coordinates
            center_x = int((box[0] + box[2]) / 2)
            center_y = int((box[1] + box[3]) / 2)
            
            centers[target_class] = ([center_x, center_y], box, confidence)
        
        return centers


    def get_object_3d_coordinates(self, time_seconds, target_class):
        """
        Get 3D coordinates of object center at specific time
        
        Args:
            time_seconds (float): Time in video to analyze
            target_class (str): Object class to detect (must be in COCO classes)
            
        Returns:
            dict: Dictionary containing:
                - coordinates: (Z , Y , X) coordinates in meters
                - center_pixel: (u, v) pixel coordinates
                - actual_time: actual timestamp used
                - confidence: detection confidence
                - box: detection bounding box
        """
        # Get frame at specified time
        frame = self.get_frame_at_time(time_seconds)
        
        # Detect object and get center point
        center_x, center_y, box, confidence = self.get_object_center(im=frame, target_class=target_class)
        
        # Get 3D coordinates of center point
        coords, actual_time = get_pixel_3d_coordinates(
            self.recording_dir,
            time_seconds,
            center_x,
            center_y
        )
        
        return {
            "coordinates": coords,
            "center_pixel": (center_x, center_y),
            "actual_time": actual_time,
            "confidence": confidence,
            "box": box.cpu().numpy() if isinstance(box, torch.Tensor) else box
        }
    

    def get_real_world_coordinates(self, response: Dict):
        classes_to_detect = response['objects']
        actions = list(response.keys())[2:]
        ret_response = {}
        
        # Get 3D coordinates of everything and store in a dictionary
        for action in actions:
            result = {}
            center_x, center_y, box, confidence = None, None, None, None

            for i in range(len(response[action])):
                try:
                    object_name = response[action][i]['object_name']
                    start_time = response[action][i]['start_time']
                    end_time = response[action][i]['end_time']
                    time_seconds = convert_time_to_seconds(end_time if 'plac' in action.lower() else start_time)
                    
                    # Get frame and detect object
                    frame = self.get_frame_at_time(time_seconds)
                    center_x, center_y, box, confidence = self.get_object_center(im=frame, target_class=object_name)
                    
                    if center_x is None or center_y is None:
                        print(f"Warning: Could not detect {object_name} at time {start_time}")
                        continue
                    
                    # Convert pixel coordinates to floats for rs2_deproject_pixel_to_point
                    pixel = [float(center_x), float(center_y)]
                    # print(f"Pixel coordinates: {pixel}")
                    try:
                        coords, _ = get_pixel_3d_coordinates(
                            self.recording_dir,
                            time_seconds,
                            pixel[0],
                            pixel[1]
                        )
                    except Exception as e:
                        print(f"Error getting 3D coordinates: {e}")
                        coords = None
                        _ = time_seconds
                    
                    # Create the key using f-string
                    key = f"{action.replace(' ','_')}_{object_name}"
                    
                    # Merge dictionaries
                    value = {
                        **response[action][i],
                        "coordinates": transform_coordinates(coords),
                        "center_pixel": tuple(pixel),
                        "actual_time": time_seconds,
                        "confidence": confidence,
                        "box": box.cpu().numpy() if isinstance(box, torch.Tensor) else box
                    }
                    
                    result[key] = value
                    # print(f"Processed object {result}")
                except Exception as e:
                    print(f"Error processing object {object_name} at time {start_time}: {e}")
                    continue
                    
            ret_response.update(result)
        
        return ret_response



def demo_flow(recording_dir, response_annotations):
    print(f"Recording directory: {recording_dir}")
    print(f"Response annotations: {response_annotations}")
    detector = ObjectDetector(api_key=GEMINI_API_KEY, recording_dir= recording_dir)
    response = detector.get_real_world_coordinates(response_annotations)
    print(f'RESPONSE FOR WHOLE VIDEO:\n================ \n{ response } \n================')

    return response

# Example usage:
if __name__ == "__main__":
    recording_dir = 'recordings/Demo_Recording'
    # print(response)
    # classes_to_detect = response['objects']
    # TODO: Modify Prompt: get actions in agent response as a separate field to use differently, just like objects
    # actions = list(response.keys())[2:]
    # print(actions)
    # Initialize detector
    detector = ObjectDetector(api_key=GEMINI_API_KEY, recording_dir= recording_dir)
    # print(f"DETECTOR RESULTS: {response}")

# =======================================================
    # d1 = {'start_time': '00:04', 'end_time': '00:03', 'object_name': 'can'}

    # # Get 3D coordinates of everything and store in a dictionary
    # time_seconds = convert_time_to_seconds(d1['start_time'])
    # frame = detector.get_frame_at_time(time_seconds=time_seconds)
    # center_x, center_y, box, confidence = detector.get_object_center(im=frame, target_class=d1['object_name'])
    # coords, _ = get_pixel_3d_coordinates(
    #     recording_dir,
    #     time_seconds,
    #     center_x,
    #     center_y
    #             )

    # print(coords)
# =======================================================



# =======================================================

    # Get 3D coordinates
    # time_seconds = 1
    # target_class = 'soda_can_red'
    # result = detector.get_object_3d_coordinates(time_seconds, target_class)
    # print(f"3D coordinates at {time_seconds} seconds:", result)

    # print(f'\n\n\n CORRECTED COORDINATES: {transform_coordinates(result["coordinates"])}')