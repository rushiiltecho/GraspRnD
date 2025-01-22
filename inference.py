import torch
import pandas as pd
import numpy as np
from pathlib import Path

class PouringTrajectoryNet(torch.nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(PouringTrajectoryNet, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(6, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            
            torch.nn.Linear(64, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            
            torch.nn.Linear(128, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            
            torch.nn.Linear(256, 160)
        )
    
    def forward(self, x):
        return self.network(x)

def load_model(model_path):
    """
    Load the trained model and scalers from the saved checkpoint
    """
    checkpoint = torch.load(model_path)
    
    model = PouringTrajectoryNet()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint['input_scaler'], checkpoint['output_scaler']

def predict_trajectory(model_path, container_positions):
    """
    Predict pouring trajectory from container positions
    
    Args:
        model_path: Path to the trained model checkpoint
        container_positions: List of lists, where each inner list contains 
                           [input1_x, input1_y, input1_z, input2_x, input2_y, input2_z]
    
    Returns:
        List of dictionaries, where each dictionary contains trajectory points
        {
            'input': [input1_x, input1_y, input1_z, input2_x, input2_y, input2_z],
            'trajectory': [
                {'x': x1, 'y': y1, 'z': z1, 'c': c1},
                {'x': x2, 'y': y2, 'z': z2, 'c': c2},
                ...
            ]
        }
    """
    # Convert input to numpy array
    container_positions = np.array(container_positions)
    
    # Validate input shape
    if container_positions.ndim == 1:
        container_positions = container_positions.reshape(1, -1)
    if container_positions.shape[1] != 6:
        raise ValueError("Each input must contain exactly 6 values: [input1_x, input1_y, input1_z, input2_x, input2_y, input2_z]")
    
    # Load model and scalers
    model, input_scaler, output_scaler = load_model(model_path)
    
    # Scale inputs
    scaled_inputs = input_scaler.transform(container_positions)
    
    # Convert to tensor and get predictions
    with torch.no_grad():
        input_tensor = torch.FloatTensor(scaled_inputs)
        scaled_predictions = model(input_tensor)
    
    # Convert predictions back to original scale
    predictions = output_scaler.inverse_transform(scaled_predictions.numpy())
    
    # Format results
    results = []
    for input_pos, pred in zip(container_positions, predictions):
        # Reshape predictions into points (x, y, z, c)
        points = pred.reshape(-1, 4)
        
        # Create trajectory points
        trajectory = []
        for x, y, z, c in points:
            trajectory.append({
                'X': float(x),
                'Y': float(y),
                'Z': float(z),
                'C': float(c)
            })
        
        results.append({
            'input': input_pos.tolist(),
            'trajectory': trajectory
        })
    
    return results

def save_predictions_to_csv(predictions, output_file='predicted_trajectories.csv'):
    """
    Save predictions to a CSV file in 80x4 format (80 rows of x,y,z,c columns)
    For multiple predictions, saves each one as output_1.csv, output_2.csv, etc.
    """
    base_name = output_file.rsplit('.', 1)[0]
    ext = output_file.rsplit('.', 1)[1]
    
    saved_files = []
    for i, pred in enumerate(predictions, 1):
        # Create 80x4 matrix from trajectory points
        rows = []
        for point in pred['trajectory']:
            rows.append([
                point['x'],
                point['y'],
                point['z'],
                point['c']
            ])
        
        # Save as DataFrame with x,y,z,c columns
        df = pd.DataFrame(rows, columns=['X', 'Y', 'Z', 'C'])
        
        # Generate filename for multiple predictions
        if len(predictions) > 1:
            file_name = f"{base_name}_{i}.{ext}"
        else:
            file_name = output_file
            
        df.to_csv(file_name, index=False)
        saved_files.append(file_name)
    
    return saved_files


# Example usage
if __name__ == "__main__":
    # Example input: List of container positions
    container_positions = [
        [-502,-689,50,-210,-652,102]  # First set of positions
  # Second set of positions
    ]
    
    # Get predictions
    predictions = predict_trajectory('pouring_trajectory_model.pth', container_positions)
    
    # Print first prediction
    print("\nFirst prediction:")
    print("Input positions:", predictions[0]['input'])
    print("First trajectory point:", predictions[0]['trajectory'][0])
    print("Last trajectory point:", predictions[0]['trajectory'][-1])
    
    # Optionally save to CSV
    csv_file = save_predictions_to_csv(predictions)
    print(f"\nPredictions saved to {csv_file}")