# import torch
# import torch.nn as nn
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# import matplotlib.pyplot as plt

# class EarlyStopping:
#     def __init__(self, patience=20, min_delta=0.001):
#         self.patience = patience
#         self.min_delta = min_delta
#         self.counter = 0
#         self.best_loss = None
#         self.early_stop = False
        
#     def __call__(self, val_loss):
#         if self.best_loss is None:
#             self.best_loss = val_loss
#         elif val_loss > self.best_loss - self.min_delta:
#             self.counter += 1
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_loss = val_loss
#             self.counter = 0
#         return self.early_stop

# class PouringTrajectoryNet(nn.Module):
#     def __init__(self, dropout_rate=0.2):
#         super(PouringTrajectoryNet, self).__init__()
#         self.network = nn.Sequential(
#             nn.Linear(6, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
            
#             nn.Linear(64, 128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
            
#             nn.Linear(128, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
            
#             nn.Linear(256, 320)
#         )
    
#     def forward(self, x):
#         return self.network(x)

# def prepare_data(csv_path):
#     """
#     Prepare the data for training by loading, scaling, and splitting the dataset.
#     """
#     # Read the CSV file
#     df = pd.read_csv(csv_path)
    
#     # Extract input features (container positions)
#     input_columns = ['input1_x', 'input1_y', 'input1_z', 
#                     'input2_x', 'input2_y', 'input2_z']
#     inputs = df[input_columns].values
    
#     # Extract output features (trajectory points)
#     trajectory_columns = []
#     for i in range(1, 81):  # 80 points
#         trajectory_columns.extend([f'p{i}_x', f'p{i}_y', f'p{i}_z', f'p{i}_c'])
#     outputs = df[trajectory_columns].values
    
#     # Initialize scalers
#     input_scaler = StandardScaler()
#     output_scaler = StandardScaler()
    
#     # Scale the data
#     inputs_scaled = input_scaler.fit_transform(inputs)
#     outputs_scaled = output_scaler.fit_transform(outputs)
    
#     # Split into train and validation sets
#     X_train, X_val, y_train, y_val = train_test_split(
#         inputs_scaled, outputs_scaled, 
#         test_size=0.2, 
#         random_state=42
#     )
    
#     # Convert to PyTorch tensors
#     X_train = torch.FloatTensor(X_train)
#     X_val = torch.FloatTensor(X_val)
#     y_train = torch.FloatTensor(y_train)
#     y_val = torch.FloatTensor(y_val)
    
#     return X_train, X_val, y_train, y_val, input_scaler, output_scaler

# def calculate_metrics(y_true, y_pred, output_scaler):
#     """Calculate detailed metrics for each coordinate type"""
#     # Inverse transform predictions and true values
#     y_true_unscaled = output_scaler.inverse_transform(y_true.cpu().numpy())
#     y_pred_unscaled = output_scaler.inverse_transform(y_pred.cpu().numpy())
    
#     metrics = {}
#     coord_names = ['x', 'y', 'z', 'c']
    
#     # Overall metrics
#     metrics['overall'] = {
#         'mse': mean_squared_error(y_true_unscaled, y_pred_unscaled),
#         'mae': mean_absolute_error(y_true_unscaled, y_pred_unscaled),
#         'r2': r2_score(y_true_unscaled, y_pred_unscaled)
#     }
    
#     # Metrics for each coordinate type
#     for coord_idx, coord in enumerate(coord_names):
#         coord_indices = [i for i in range(len(y_true_unscaled[0])) if i % 4 == coord_idx]
        
#         y_true_coord = y_true_unscaled[:, coord_indices]
#         y_pred_coord = y_pred_unscaled[:, coord_indices]
        
#         metrics[coord] = {
#             'mse': mean_squared_error(y_true_coord, y_pred_coord),
#             'mae': mean_absolute_error(y_true_coord, y_pred_coord),
#             'r2': r2_score(y_true_coord, y_pred_coord)
#         }
    
#     return metrics

# def print_metrics(metrics, set_name=""):
#     print(f"\n{set_name} Metrics:")
#     print("=" * 50)
    
#     # Overall metrics
#     print("\nOverall Metrics:")
#     print(f"MSE: {metrics['overall']['mse']:.6f}")
#     print(f"MAE: {metrics['overall']['mae']:.6f}")
#     print(f"R²:  {metrics['overall']['r2']:.6f}")
    
#     # Coordinate-specific metrics
#     for coord in ['x', 'y', 'z', 'c']:
#         print(f"\n{coord.upper()} Coordinate Metrics:")
#         print(f"MSE: {metrics[coord]['mse']:.6f}")
#         print(f"MAE: {metrics[coord]['mae']:.6f}")
#         print(f"R²:  {metrics[coord]['r2']:.6f}")

# def train_model(model, X_train, X_val, y_train, y_val, output_scaler, epochs=1000, batch_size=32):
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode='min', factor=0.5, patience=10, verbose=False
#     )
#     early_stopping = EarlyStopping(patience=30, min_delta=0.001)
    
#     train_losses = []
#     val_losses = []
#     epoch_metrics = []
    
#     train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
#     for epoch in range(epochs):
#         model.train()
#         epoch_train_losses = []
        
#         for batch_X, batch_y in train_loader:
#             optimizer.zero_grad()
#             outputs = model(batch_X)
#             loss = criterion(outputs, batch_y)
#             loss.backward()
#             optimizer.step()
#             epoch_train_losses.append(loss.item())
        
#         avg_train_loss = np.mean(epoch_train_losses)
#         train_losses.append(avg_train_loss)
        
#         # Validation
#         model.eval()
#         with torch.no_grad():
#             val_outputs = model(X_val)
#             val_loss = criterion(val_outputs, y_val)
#             val_losses.append(val_loss.item())
            
#             # Calculate metrics every 50 epochs or on the last epoch
#             if (epoch + 1) % 50 == 0 or epoch == epochs - 1:
#                 train_outputs = model(X_train)
#                 train_metrics = calculate_metrics(y_train, train_outputs, output_scaler)
#                 val_metrics = calculate_metrics(y_val, val_outputs, output_scaler)
                
#                 epoch_metrics.append({
#                     'epoch': epoch + 1,
#                     'train_metrics': train_metrics,
#                     'val_metrics': val_metrics
#                 })
                
#                 print(f'\nEpoch [{epoch+1}/{epochs}]')
#                 print_metrics(train_metrics, "Training")
#                 print_metrics(val_metrics, "Validation")
#                 print('-' * 50)
        
#         scheduler.step(val_loss)
        
#         if early_stopping(val_loss):
#             print(f"Early stopping triggered at epoch {epoch+1}")
#             break
    
#     return train_losses, val_losses, epoch_metrics

# if __name__ == "__main__":
#     # Data preparation and model training
#     X_train, X_val, y_train, y_val, input_scaler, output_scaler = prepare_data('dataset.csv')
#     model = PouringTrajectoryNet()
    
#     # Train the model
#     train_losses, val_losses, epoch_metrics = train_model(
#         model, X_train, X_val, y_train, y_val, output_scaler
#     )
    
#     # Plot learning curves
#     plt.figure(figsize=(12, 8))
#     plt.subplot(2, 1, 1)
#     plt.plot(train_losses, label='Training Loss')
#     plt.plot(val_losses, label='Validation Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title('Training and Validation Learning Curves')
#     plt.legend()
#     plt.grid(True)
    
#     # Plot R² scores over time
#     plt.subplot(2, 1, 2)
#     epochs = [m['epoch'] for m in epoch_metrics]
#     train_r2 = [m['train_metrics']['overall']['r2'] for m in epoch_metrics]
#     val_r2 = [m['val_metrics']['overall']['r2'] for m in epoch_metrics]
    
#     plt.plot(epochs, train_r2, label='Training R²')
#     plt.plot(epochs, val_r2, label='Validation R²')
#     plt.xlabel('Epoch')
#     plt.ylabel('R² Score')
#     plt.title('R² Score Evolution')
#     plt.legend()
#     plt.grid(True)
    
#     plt.tight_layout()
#     plt.show()

#     # Save everything
#     torch.save({
#         'model_state_dict': model.state_dict(),
#         'input_scaler': input_scaler,
#         'output_scaler': output_scaler,
#         'train_losses': train_losses,
#         'val_losses': val_losses,
#         'metrics_history': epoch_metrics
#     }, 'pouring_trajectory_model.pth')





import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

def apply_moving_average(df, window_size=10):
    """Apply moving average separately for x, y, z, c coordinates."""
    smoothed_df = df.copy()
    
    # Get all column names for each coordinate type
    x_cols = [col for col in df.columns if col.endswith('_x')]
    y_cols = [col for col in df.columns if col.endswith('_y')]
    z_cols = [col for col in df.columns if col.endswith('_z')]
    c_cols = [col for col in df.columns if col.endswith('_c')]
    
    # Apply moving average to each coordinate type
    for cols in [x_cols, y_cols, z_cols, c_cols]:
        smoothed_df[cols] = df[cols].rolling(window=window_size, min_periods=1).mean()
    
    return smoothed_df

def load_and_preprocess_data(csv_path, window_size=10):
    """Load, smooth, and prepare input/output data from CSV."""
    df = pd.read_csv(csv_path)
    
    # Extract input features (6 columns) - don't smooth these
    input_cols = ['input1_x', 'input1_y', 'input1_z',
                 'input2_x', 'input2_y', 'input2_z']
    X = df[input_cols].values
    
    # Get trajectory columns
    trajectory_cols = []
    for i in range(1, 81):
        cols = [f'p{i}_x', f'p{i}_y', f'p{i}_z', f'p{i}_c']
        trajectory_cols.extend(cols)
    
    # Apply moving average to trajectory data
    trajectory_df = df[trajectory_cols]
    smoothed_df = apply_moving_average(trajectory_df, window_size)
    
    # Reshape trajectory data to (n_samples, 320)
    Y = smoothed_df.values
    
    # Plot original vs smoothed for first sample
    plt.figure(figsize=(15, 10))
    
    # Plot X coordinates
    plt.subplot(2, 2, 1)
    x_cols = [col for col in trajectory_cols if col.endswith('_x')]
    plt.plot(df[x_cols].iloc[0], label='Original', alpha=0.5)
    plt.plot(smoothed_df[x_cols].iloc[0], label='Smoothed')
    plt.title('X Coordinate Smoothing')
    plt.legend()
    
    # Plot Y coordinates
    plt.subplot(2, 2, 2)
    y_cols = [col for col in trajectory_cols if col.endswith('_y')]
    plt.plot(df[y_cols].iloc[0], label='Original', alpha=0.5)
    plt.plot(smoothed_df[y_cols].iloc[0], label='Smoothed')
    plt.title('Y Coordinate Smoothing')
    plt.legend()
    
    # Plot Z coordinates
    plt.subplot(2, 2, 3)
    z_cols = [col for col in trajectory_cols if col.endswith('_z')]
    plt.plot(df[z_cols].iloc[0], label='Original', alpha=0.5)
    plt.plot(smoothed_df[z_cols].iloc[0], label='Smoothed')
    plt.title('Z Coordinate Smoothing')
    plt.legend()
    
    # Plot C values
    plt.subplot(2, 2, 4)
    c_cols = [col for col in trajectory_cols if col.endswith('_c')]
    plt.plot(df[c_cols].iloc[0], label='Original', alpha=0.5)
    plt.plot(smoothed_df[c_cols].iloc[0], label='Smoothed')
    plt.title('C Value Smoothing')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('smoothing_visualization.png')
    plt.close()
    
    return X, Y

def create_model():
    """Create a smaller dense model appropriate for 72 samples."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(6,)),
        tf.keras.layers.Dense(32, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(320, activation=None)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def train_and_evaluate():
    # Load and preprocess data
    print("Loading and smoothing data...")
    X, Y = load_and_preprocess_data("dataset.csv", window_size=10)
    
    # Split data (80-20 split)
    X_train, X_val, y_train, y_val = train_test_split(
        X, Y, test_size=0.2, random_state=SEED
    )
    
    # Scale the data
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    X_train_scaled = X_scaler.fit_transform(X_train)
    X_val_scaled = X_scaler.transform(X_val)
    
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_val_scaled = y_scaler.transform(y_val)
    
    # Create and train model
    print("\nTraining model...")
    model = create_model()
    
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=30,
        restore_best_weights=True
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6
    )
    
    history = model.fit(
        X_train_scaled, y_train_scaled,
        validation_data=(X_val_scaled, y_val_scaled),
        epochs=1000,
        batch_size=4,  # Small batch size for 72 samples
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # Save model and scalers
    os.makedirs("models", exist_ok=True)
    model.save("models/dense_model.keras")
    joblib.dump(X_scaler, "models/input_scaler.pkl")
    joblib.dump(y_scaler, "models/output_scaler.pkl")
    print("\nModel and scalers saved to 'models' directory")
    
    # Visualize training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    # Print final metrics
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    print(f"\nFinal Training Loss: {final_train_loss:.6f}")
    print(f"Final Validation Loss: {final_val_loss:.6f}")
    
    return model, X_scaler, y_scaler, history

def main():
    model, X_scaler, y_scaler, history = train_and_evaluate()
    print("\nTraining visualization saved as 'training_history.png'")
    print("Smoothing visualization saved as 'smoothing_visualization.png'")

if __name__ == "__main__":
    main()