import torch
import torch.nn as nn
import numpy as np
from database.db_manager import get_db_session, AirQualityMeasurement
from config import SEQUENCE_LENGTH, PREDICTION_HORIZON
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import logging
import os
import joblib

# Налаштування логування
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AirQualityDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.FloatTensor(self.targets[idx])

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, prediction_horizon, dropout=0.2):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.prediction_horizon = prediction_horizon
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size * prediction_horizon)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate GRU
        out, _ = self.gru(x, h0)
        
        # Apply dropout
        out = self.dropout(out)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        
        # Reshape to [batch_size, prediction_horizon, output_size]
        out = out.view(out.size(0), self.prediction_horizon, -1)
        return out

def prepare_data(city):
    """Prepare data for GRU training"""
    session = get_db_session()
    
    try:
        # Get data for the city
        query = session.query(
            AirQualityMeasurement.timestamp,
            AirQualityMeasurement.co,
            AirQualityMeasurement.no2,
            AirQualityMeasurement.o3,
            AirQualityMeasurement.so2,
            AirQualityMeasurement.pm25,
            AirQualityMeasurement.pm10
        ).filter(
            AirQualityMeasurement.city == city
        ).order_by(
            AirQualityMeasurement.timestamp
        )
        
        df = pd.read_sql(query.statement, session.bind)
        
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        # Handle missing values for numeric columns only
        air_quality_columns = ['co', 'no2', 'o3', 'so2', 'pm25', 'pm10']
        df[air_quality_columns] = df[air_quality_columns].interpolate(method='time')
        
        # Normalize the data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[air_quality_columns])
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_data) - SEQUENCE_LENGTH - PREDICTION_HORIZON + 1):
            X.append(scaled_data[i:(i + SEQUENCE_LENGTH)])
            y.append(scaled_data[i + SEQUENCE_LENGTH:i + SEQUENCE_LENGTH + PREDICTION_HORIZON])

        # Convert to numpy arrays first
        X = np.array(X)
        y = np.array(y)
        # Remove NO2 from X and y (columns: 0-co, 1-no2, 2-o3, 3-so2, 4-pm25, 5-pm10)
        X = X[:, :, [0, 2, 3, 4, 5]]
        y = y[:, :, [0, 2, 3, 4, 5]]
        
        return X, y, scaler, df[air_quality_columns]
    
    finally:
        session.close()

def train_model(city, num_epochs=100, batch_size=32, learning_rate=0.001):
    """Train GRU model for a specific city"""
    # Prepare data
    X, y, scaler, original_data = prepare_data(city)
    
    # Split into train and test sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Create datasets and dataloaders
    train_dataset = AirQualityDataset(X_train, y_train)
    test_dataset = AirQualityDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    input_size = 5  # co, o3, so2, pm25, pm10
    hidden_size = 256
    num_layers = 2
    output_size = 5  # Predicting co, o3, so2, pm25, pm10
    
    model = GRUModel(input_size, hidden_size, num_layers, output_size, PREDICTION_HORIZON)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y).item()
        
        # Update learning rate
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        if old_lr != new_lr:
            logger.info(f'Learning rate changed from {old_lr:.6f} to {new_lr:.6f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'models/{city}_gru_best.pth')
        
        if (epoch + 1) % 10 == 0:
            logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(test_loader):.4f}')
    
    # Load best model
    model.load_state_dict(torch.load(f'models/{city}_gru_best.pth'))
    return model, scaler, original_data

def predict(model, scaler, data):
    """Make predictions for all parameters"""
    model.eval()
    with torch.no_grad():
        # Prepare input sequence
        input_data = data.values[-SEQUENCE_LENGTH:]
        input_data = scaler.transform(input_data)
        input_tensor = torch.FloatTensor(input_data).unsqueeze(0)
        
        # Make prediction
        output = model(input_tensor)
        predictions = output.squeeze().numpy()
        
        # Inverse transform predictions
        predictions = scaler.inverse_transform(predictions)
        return predictions

def save_model(model, scaler, city):
    """Save trained model and scaler"""
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), f'models/{city}_gru_model.pth')
    joblib.dump(scaler, f'models/{city}_gru_scaler.pkl')

def predict_future(model, scaler, initial_sequence, steps=PREDICTION_HORIZON):
    """Make predictions for future time steps"""
    model.eval()
    predictions = []
    current_sequence = initial_sequence.copy()
    
    with torch.no_grad():
        # Get initial prediction
        input_tensor = torch.FloatTensor(current_sequence).unsqueeze(0)
        output = model(input_tensor)
        predictions = output.squeeze().numpy()
    
    # Inverse transform predictions
    predictions = scaler.inverse_transform(predictions)
    return predictions

def train_gru_models(city_data, city):
    """Train GRU models for a specific city"""
    logger.info(f"Training GRU models for {city}")
    
    # Get the number of records available for training
    num_records = len(city_data)
    logger.info(f"Number of records available for GRU training: {num_records}")
    
    try:
        # Train the model
        trained_model, scaler, original_data = train_model(city)
        
        # Get predictions for all parameters
        predictions = predict(trained_model, scaler, original_data)
        
        # Calculate and log RMSE for each parameter
        param_indices = {'co': 0, 'o3': 1, 'so2': 2, 'pm25': 3, 'pm10': 4}
        for param, index in param_indices.items():
            # Get actual values
            actual_values = original_data[param].values[-len(predictions):]
            
            # Calculate RMSE
            rmse = np.sqrt(np.mean((predictions[:, index] - actual_values) ** 2))
            logger.info(f"RMSE for {param} in {city}: {rmse:.4f}")
        
        # Save the model
        save_model(trained_model, scaler, city)
        logger.info(f"GRU model for {city} trained and saved successfully")
        
    except Exception as e:
        logger.error(f"Error training GRU model for {city}: {str(e)}")
        raise

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    
    city = 'Kyiv'
    logger.info(f"Training model for {city}")
    model, scaler, original_data = train_model(city)
    save_model(model, scaler, city)
    logger.info(f"Model for {city} saved") 