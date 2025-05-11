import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import logging
from typing import Tuple, List, Dict
import os
import joblib

logger = logging.getLogger(__name__)

class AirQualityLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, prediction_horizon: int, dropout_rate: float = 0.3):
        super(AirQualityLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.prediction_horizon = prediction_horizon
        self.output_size = output_size
        self.dropout_rate = dropout_rate

        # LSTM layers with dropout (non-bidirectional)
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Single batch normalization layer
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        
        # Fully connected layers with dropout
        self.fc1 = nn.Linear(hidden_size, 128)  # Changed to match saved model
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 144)  # Changed to match saved model
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(144, output_size * prediction_horizon)  # Added fc3 layer
        
        # Activation functions
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))  # (batch_size, seq_len, hidden_size)
        
        # Take the last output of the sequence
        out = out[:, -1, :]  # (batch_size, hidden_size)
        
        # Apply batch normalization
        out = self.batch_norm(out)
        
        # Additional fully connected layers
        out1 = self.leaky_relu(self.fc1(out))
        out1 = self.dropout1(out1)
        
        out2 = self.leaky_relu(self.fc2(out1))
        out2 = self.dropout2(out2)
        
        out3 = self.fc3(out2)
        
        # Reshape to [batch_size, prediction_horizon, output_size]
        out = out3.view(out3.size(0), self.prediction_horizon, self.output_size)
        return out

class AirQualityPredictor:
    def __init__(self, sequence_length: int = 24, prediction_horizon: int = 24):
        """
        Initialize the air quality predictor.
        
        Args:
            sequence_length (int): Number of past hours to use for prediction
            prediction_horizon (int): Number of hours to predict into the future
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.scaler = MinMaxScaler()
        self.model = None
        self.feature_columns = ['co', 'no2', 'o3', 'so2', 'pm25', 'pm10']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def prepare_data(self, data: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare data for model training with improved preprocessing"""
        # Sort by datetime
        data = data.sort_values('datetime')
        
        # Handle missing values
        for col in self.feature_columns:
            # Forward fill
            data[col] = data[col].fillna(method='ffill')
            # Backward fill for any remaining NaNs
            data[col] = data[col].fillna(method='bfill')
            # If still any NaNs, fill with median
            data[col] = data[col].fillna(data[col].median())
        
        # Remove outliers using IQR method
        for col in self.feature_columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
        
        # Scale the features
        scaled_data = self.scaler.fit_transform(data[self.feature_columns])
        
        # Create sequences with overlapping windows
        X, y = [], []
        for i in range(len(scaled_data) - self.sequence_length - self.prediction_horizon + 1):
            X.append(scaled_data[i:(i + self.sequence_length)])
            y.append(scaled_data[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon])
            
        # Convert to PyTorch tensors
        X = torch.FloatTensor(np.array(X)).to(self.device)
        y = torch.FloatTensor(np.array(y)).to(self.device)
        
        return X, y
    
    def build_model(self) -> None:
        """Build the enhanced LSTM model"""
        input_size = 6  # Number of features
        hidden_size = 256
        num_layers = 3
        output_size = 6
        prediction_horizon = 24
        
        self.model = AirQualityLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            prediction_horizon=prediction_horizon,
            dropout_rate=0.3
        ).to(self.device)
        
        logger.info("Enhanced model architecture created")
    
    def train(self, X: torch.Tensor, y: torch.Tensor, epochs: int = 200, batch_size: int = 32) -> None:
        """Train the model with improved training process"""
        if self.model is None:
            self.build_model()
            
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Early stopping variables
        best_loss = float('inf')
        patience = 20
        patience_counter = 0
        best_model_state = None
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            scheduler.step(avg_loss)
            
            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                logger.info(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
            
            if patience_counter >= patience:
                logger.info(f'Early stopping triggered at epoch {epoch+1}')
                break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        logger.info("Model training completed")
    
    def predict(self, data: pd.DataFrame, hours_ahead: int = 24) -> pd.DataFrame:
        """
        Make predictions for future air quality.
        
        Args:
            data (pd.DataFrame): Recent air quality measurements
            hours_ahead (int): Number of hours to predict into the future
            
        Returns:
            pd.DataFrame: Predictions for future air quality
        """
        logger.info(f"Input data columns: {data.columns.tolist()}")
        logger.info(f"Input data index: {data.index.name}")
        logger.info(f"Input data shape: {data.shape}")
        
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        if hours_ahead <= 0:
            raise ValueError("hours_ahead must be positive")
            
        # Check if we have enough data for prediction
        if len(data) < self.sequence_length:
            raise ValueError(f"Not enough data for prediction. Need at least {self.sequence_length} records, but got {len(data)}.")
            
        # Store datetime column if it exists
        datetime_col = None
        if 'datetime' in data.columns:
            datetime_col = data['datetime']
            logger.info("Stored datetime column for later use")
            
        # Ensure all required columns are present
        for col in self.feature_columns:
            if col not in data.columns:
                logger.warning(f"Missing feature column: {col}, adding with default value 0.0")
                data[col] = 0.0
                
        # Reorder columns to match the scaler's expected order
        data = data[self.feature_columns]
        logger.info(f"Data columns after reordering: {data.columns.tolist()}")
        
        # Scale the features
        scaled_data = self.scaler.transform(data)
        
        # Initialize predictions list
        all_predictions = []
        current_sequence = scaled_data[-self.sequence_length:].copy()
        
        # Make predictions recursively
        self.model.eval()
        with torch.no_grad():
            for _ in range(hours_ahead):
                # Prepare input tensor
                input_tensor = torch.FloatTensor(current_sequence).unsqueeze(0).to(self.device)
                
                # Make prediction
                output = self.model(input_tensor)
                # Get the prediction for next hour (6 features)
                next_pred = output.cpu().numpy()[0, 0, :]  # Take first prediction and all features
                
                # Add prediction to results
                all_predictions.append(next_pred)
                
                # Update sequence for next prediction
                current_sequence = np.roll(current_sequence, -1, axis=0)
                current_sequence[-1] = next_pred
        
        # Convert predictions to numpy array
        predictions = np.array(all_predictions)  # Shape: (hours_ahead, 6)
        
        # Inverse transform predictions
        predictions = self.scaler.inverse_transform(predictions)
        
        # Get the last timestamp from the input data
        logger.info("Attempting to find timestamp in input data...")
        
        last_timestamp = None
        if datetime_col is not None:
            last_timestamp = datetime_col.iloc[-1]
            logger.info("Using stored datetime column")
        elif 'datetime' in data.index.names:
            last_timestamp = data.index[-1]
            logger.info("Found timestamp in index names")
        elif 'datetime' in data.columns:
            last_timestamp = data['datetime'].iloc[-1]
            logger.info("Found timestamp in datetime column")
        elif 'timestamp' in data.columns:
            last_timestamp = data['timestamp'].iloc[-1]
            logger.info("Found timestamp in timestamp column")
        else:
            logger.error("No datetime or timestamp column found in input data")
            raise ValueError("No datetime or timestamp column found in input data")
            
        logger.info(f"Last timestamp found: {last_timestamp}")
            
        # Ensure last_timestamp is a datetime object
        if not isinstance(last_timestamp, pd.Timestamp):
            logger.info(f"Converting timestamp to datetime: {last_timestamp}")
            last_timestamp = pd.to_datetime(last_timestamp)
            
        # Create future timestamps
        future_times = [last_timestamp + pd.Timedelta(hours=i+1) for i in range(len(predictions))]
        logger.info(f"Created {len(future_times)} future timestamps, from {future_times[0]} to {future_times[-1]}")
        
        # Create DataFrame with predictions
        result = pd.DataFrame(
            predictions,
            index=future_times,
            columns=self.feature_columns
        )
        
        logger.info(f"Created result DataFrame with shape: {result.shape}")
        return result
    
    def save_model(self, path: str) -> None:
        """
        Save the trained model and scaler.
        
        Args:
            path (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
            
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), f"{path}/model.pth")
        joblib.dump(self.scaler, f"{path}/scaler.joblib")
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load_model(cls, path: str) -> 'AirQualityPredictor':
        """
        Load a trained model and scaler.
        
        Args:
            path (str): Path to the saved model
            
        Returns:
            AirQualityPredictor: Loaded model instance
        """
        instance = cls()
        
        # Build model with dimensions matching the saved model
        input_size = 6  # Number of features: co, no2, o3, so2, pm25, pm10
        hidden_size = 256  # Hidden size from saved model
        num_layers = 3
        output_size = 6  # Output size from saved model
        prediction_horizon = 24  # Prediction horizon from saved model
        
        instance.model = AirQualityLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            prediction_horizon=prediction_horizon,
            dropout_rate=0.3
        ).to(instance.device)
        
        # Load the saved state dict
        instance.model.load_state_dict(torch.load(path))
        
        # Load the original scaler
        scaler_path = path.replace('_model.pth', '_scaler.pkl')
        instance.scaler = joblib.load(scaler_path)
        
        logger.info(f"Model loaded from {path}")
        
        return instance