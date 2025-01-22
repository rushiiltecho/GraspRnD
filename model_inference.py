# import torch
# import numpy as np
# import pandas as pd
# import os

# from model import PouringTrajectoryNet

# def load_model(model_path):
#     """
#     Load the trained model and associated scalers.
    
#     Args:
#         model_path (str): Path to the saved model checkpoint
        
#     Returns:
#         tuple: (model, input_scaler, output_scaler)
#     """
#     # Load the checkpoint
#     checkpoint = torch.load(model_path)
    
#     # Initialize model with the same architecture
#     model = PouringTrajectoryNet()
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.eval()
    
#     return model, checkpoint['input_scaler'], checkpoint['output_scaler']

# def predict_trajectory(model, input_scaler, output_scaler, container_positions):
#     """
#     Predict pouring trajectory given container positions.
    
#     Args:
#         model: Trained PouringTrajectoryNet model
#         input_scaler: StandardScaler for input normalization
#         output_scaler: StandardScaler for output denormalization
#         container_positions (array-like): Array of 6 values [x1, y1, z1, x2, y2, z2]
        
#     Returns:
#         numpy.ndarray: Predicted trajectory points (80 points × 4 dimensions)
#     """
#     # Validate input
#     if len(container_positions) != 6:
#         raise ValueError("Container positions must be an array of 6 values")
    
#     # Convert input to numpy array and reshape
#     inputs = np.array(container_positions).reshape(1, -1)
    
#     # Scale inputs
#     inputs_scaled = input_scaler.transform(inputs)
    
#     # Convert to tensor
#     inputs_tensor = torch.FloatTensor(inputs_scaled)
    
#     # Make prediction
#     with torch.no_grad():
#         outputs_scaled = model(inputs_tensor)
    
#     # Convert to numpy and denormalize
#     outputs = output_scaler.inverse_transform(outputs_scaled.numpy())
    
#     # Reshape to 80 points × 4 dimensions (x, y, z, confidence)
#     trajectory = outputs.reshape(-1, 4)
    
#     return trajectory

# def format_trajectory(trajectory):
#     """
#     Format the trajectory into a more readable structure.
    
#     Args:
#         trajectory (numpy.ndarray): Array of shape (80, 4)
        
#     Returns:
#         list: List of dictionaries containing point information
#     """
#     formatted_trajectory = []
#     for i, point in enumerate(trajectory):
#         formatted_trajectory.append({
#             'point_id': i + 1,
#             'x': float(point[0]),
#             'y': float(point[1]),
#             'z': float(point[2]),
#             'confidence': float(point[3])
#         })
#     return formatted_trajectory

# def generate_trajectory_csv(x1, y1, z1, x2, y2, z2, model_path='pouring_trajectory_model.pth'):
#     """
#     Generate trajectory prediction CSV for given container positions.
    
#     Args:
#         x1, y1, z1: Position coordinates of first container
#         x2, y2, z2: Position coordinates of second container
#         model_path: Path to the model checkpoint file
    
#     Returns:
#         str: Path to the generated CSV file
#     """
#     # Combine inputs
#     container_positions = [x1, y1, z1, x2, y2, z2]
    
#     # Load model and scalers
#     model, input_scaler, output_scaler = load_model(model_path)
    
#     # Make prediction
#     trajectory = predict_trajectory(model, input_scaler, output_scaler, container_positions)
    
#     # Format results
#     formatted_trajectory = format_trajectory(trajectory)
    
#     # Convert to DataFrame
#     df = pd.DataFrame(formatted_trajectory)
    
#     # Create output filename with timestamp
#     output_dir = 'predictions'
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Create output filename based on input positions
#     positions_str = '_'.join(f"{p:.3f}" for p in container_positions)
#     output_file = os.path.join(output_dir, f'trajectory_prediction_{positions_str}.csv')
    
#     # Save to CSV
#     df.to_csv(output_file, index=False)
    
#     return output_file

# if __name__ == "__main__":
#     # Example usage
#     x1, y1, z1 = 0.5, 0.3, 0.2  # First container position
#     x2, y2, z2 = 0.8, 0.4, 0.1  # Second container position
    
#     csv_path = generate_trajectory_csv(x1, y1, z1, x2, y2, z2)
#     print(f"Trajectory saved to: {csv_path}")

# import numpy as np
# import tensorflow as tf
# import joblib

# def load_model_and_scalers(model_path="models/dense_model.keras",
#                           input_scaler_path="models/input_scaler.pkl",
#                           output_scaler_path="models/output_scaler.pkl"):
#     """Load the trained model and scalers."""
#     model = tf.keras.models.load_model(model_path)
#     input_scaler = joblib.load(input_scaler_path)
#     output_scaler = joblib.load(output_scaler_path)
#     return model, input_scaler, output_scaler

# def prepare_input(input_data):
#     """Prepare input data for inference."""
#     input_array = np.array(input_data)
#     if input_array.ndim == 1:
#         input_array = input_array.reshape(1, -1)
#     return input_array

# def predict_and_save_trajectory(input1_x, input1_y, input1_z, input2_x, input2_y, input2_z, 
#                               output_path="predicted_trajectory.csv"):
#     """
#     Predict trajectory from 6 input coordinates and save to CSV.
    
#     Args:
#         input1_x, input1_y, input1_z: Coordinates of first input point
#         input2_x, input2_y, input2_z: Coordinates of second input point
#         output_path: Path to save the output CSV file
        
#     Returns:
#         str: Path to the saved CSV file
#     """
#     try:
#         # Load model and scalers
#         model, input_scaler, output_scaler = load_model_and_scalers()
        
#         # Prepare input
#         input_data = [input1_x, input1_y, input1_z, input2_x, input2_y, input2_z]
#         X = prepare_input(input_data)
        
#         # Scale input and predict
#         X_scaled = input_scaler.transform(X)
#         y_scaled = model.predict(X_scaled)
        
#         # Inverse transform the prediction
#         y_pred = output_scaler.inverse_transform(y_scaled)
        
#         # Reshape into points (80 points × 4 values per point)
#         num_points = 80
#         output_data = y_pred.reshape(-1, 4)  # reshape to (80, 4)
        
#         # Save to CSV with headers
#         header = 'x,y,z,confidence'
#         np.savetxt(output_path, output_data, delimiter=',', header=header, comments='')
        
#         return output_path
        
#     except Exception as e:
#         raise Exception(f"Error in prediction: {str(e)}")

# if __name__ == "__main__":
#     # Example usage
#     try:
#         output_file = predict_and_save_trajectory(
#             input1_x=0.5, input1_y=0.3, input1_z=0.2,
#             input2_x=0.1, input2_y=0.4, input2_z=0.6
#         )
#         print(f"Prediction saved to: {output_file}")
#     except Exception as e:
#         print(f"Error: {str(e)}")



import numpy as np
import tensorflow as tf
import joblib

def load_model_and_scalers(model_path="models/dense_model.keras",
                           input_scaler_path="models/input_scaler.pkl",
                           output_scaler_path="models/output_scaler.pkl"):
    """Load the trained model and scalers."""
    model = tf.keras.models.load_model(model_path)
    input_scaler = joblib.load(input_scaler_path)
    output_scaler = joblib.load(output_scaler_path)
    return model, input_scaler, output_scaler

def prepare_input(input_data):
    """Prepare input data for inference."""
    input_array = np.array(input_data)
    if input_array.ndim == 1:
        input_array = input_array.reshape(1, -1)
    return input_array

def predict_and_save_trajectory(input1_x, input1_y, input1_z, 
                                input2_x, input2_y, input2_z, 
                                output_path="predicted_trajectory.csv"):
    """
    Predict trajectory from 6 input coordinates and save to CSV.
    
    Args:
        input1_x, input1_y, input1_z: Coordinates of first input point
        input2_x, input2_y, input2_z: Coordinates of second input point
        output_path: Path to save the output CSV file
        
    Returns:
        str: Path to the saved CSV file
    """
    try:
        # Load model and scalers
        model, input_scaler, output_scaler = load_model_and_scalers()
        
        # Prepare input data as expected by the model
        input_data = [input1_x, input1_y, input1_z, input2_x, input2_y, input2_z]
        X = prepare_input(input_data)
        
        # Scale input and perform prediction
        X_scaled = input_scaler.transform(X)
        y_scaled = model.predict(X_scaled)
        
        # Inverse transform the prediction
        y_pred = output_scaler.inverse_transform(y_scaled)
        
        # Reshape into points (assumes 80 points × 4 values per point)
        output_data = y_pred.reshape(-1, 4)  # shape becomes (80, 4)
        
        # Save to CSV with headers
        header = 'X,Y,Z,C'
        np.savetxt(output_path, output_data, delimiter=',', header=header, comments='')
        
        return output_path
        
    except Exception as e:
        raise Exception(f"Error in prediction: {str(e)}")

if __name__ == "__main__":
    # Example usage
    try:
        output_file = predict_and_save_trajectory(
            input1_x=0.5, input1_y=0.3, input1_z=0.2,
            input2_x=0.1, input2_y=0.4, input2_z=0.6
        )
        print(f"Prediction saved to: {output_file}")
    except Exception as e:
        print(f"Error: {str(e)}")
