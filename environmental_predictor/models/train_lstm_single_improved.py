import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import logging
import os
import joblib
from database.db_manager import get_db_session, AirQualityMeasurement
from config import SEQUENCE_LENGTH, PREDICTION_HORIZON
from datetime import datetime, timedelta

# Set up logging
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

class ImprovedSingleParameterLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, prediction_horizon, dropout_rate=0.3):
        super(ImprovedSingleParameterLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.prediction_horizon = prediction_horizon
        
        # Bidirectional LSTM layers with increased dropout
        self.lstm1 = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Second LSTM layer for better sequence processing
        self.lstm2 = nn.LSTM(
            hidden_size * 2,  # *2 for bidirectional
            hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention mechanism with improved architecture
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Fully connected layers with residual connections
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.fc3 = nn.Linear(hidden_size * 2, prediction_horizon)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_size * 2)
        self.layer_norm2 = nn.LayerNorm(hidden_size * 2)
        self.layer_norm3 = nn.LayerNorm(hidden_size * 2)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)
        
    def attention_net(self, lstm_output):
        attn_weights = self.attention(lstm_output)
        soft_attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.bmm(soft_attn_weights.transpose(1, 2), lstm_output)
        return context.squeeze(1)
        
    def forward(self, x):
        # First LSTM layer
        h0_1 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0_1 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        out1, _ = self.lstm1(x, (h0_1, c0_1))
        
        # Second LSTM layer
        h0_2 = torch.zeros(2, x.size(0), self.hidden_size).to(x.device)
        c0_2 = torch.zeros(2, x.size(0), self.hidden_size).to(x.device)
        out2, _ = self.lstm2(out1, (h0_2, c0_2))
        
        # Apply attention
        out = self.attention_net(out2)
        out = self.layer_norm1(out)
        
        # Fully connected layers with residual connections
        residual = out
        out = self.leaky_relu(self.fc1(out))
        out = self.layer_norm2(out)
        out = self.dropout1(out)
        
        out = self.leaky_relu(self.fc2(out))
        out = self.layer_norm3(out)
        out = self.dropout2(out)
        
        # Add residual connection
        out = out + residual
        
        # Final layer
        out = self.fc3(out)
        
        return out.view(out.size(0), self.prediction_horizon, 1)

def add_time_features(df):
    """Add comprehensive time-based features"""
    # Basic time features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['day_of_year'] = df.index.dayofyear
    df['week_of_year'] = df.index.isocalendar().week
    
    # Cyclical encoding of time features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    
    # Weekend flag
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Rush hour flags
    df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
    df['is_evening_rush'] = ((df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
    
    return df

def prepare_data(city, parameter):
    """Prepare data for LSTM training with improved features"""
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
        df.set_index('timestamp', inplace=True)
        
        # Add comprehensive time features
        df = add_time_features(df)
        
        # Handle missing values
        df[parameter] = df[parameter].interpolate(method='time')
        
        # Add rolling statistics with multiple windows
        windows = [6, 12, 24, 48]  # 6h, 12h, 24h, 48h
        for window in windows:
            df[f'{parameter}_rolling_mean_{window}h'] = df[parameter].rolling(window=window).mean()
            df[f'{parameter}_rolling_std_{window}h'] = df[parameter].rolling(window=window).std()
            df[f'{parameter}_rolling_min_{window}h'] = df[parameter].rolling(window=window).min()
            df[f'{parameter}_rolling_max_{window}h'] = df[parameter].rolling(window=window).max()
        
        # Add lag features
        for lag in [1, 2, 3, 6, 12, 24]:
            df[f'{parameter}_lag_{lag}h'] = df[parameter].shift(lag)
        
        # Fill NaN values
        df = df.bfill()
        
        # Select features for scaling
        feature_columns = [col for col in df.columns if parameter in col or col in [
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 
            'month_sin', 'month_cos', 'is_weekend',
            'is_morning_rush', 'is_evening_rush'
        ]]
        
        # Normalize the data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[feature_columns])
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_data) - SEQUENCE_LENGTH - PREDICTION_HORIZON + 1):
            X.append(scaled_data[i:(i + SEQUENCE_LENGTH)])
            y.append(scaled_data[i + SEQUENCE_LENGTH:i + SEQUENCE_LENGTH + PREDICTION_HORIZON, 0:1])
        
        return np.array(X), np.array(y), scaler, df[parameter]
    
    finally:
        session.close()

def train_model(city, parameter, num_epochs=100, batch_size=32, learning_rate=0.001):
    """Train improved LSTM model for a specific parameter"""
    logger.info(f"Starting data preparation for {city} - {parameter}")
    X, y, scaler, original_data = prepare_data(city, parameter)
    logger.info(f"Data prepared. Training set size: {len(X)}")
    
    # Split into train and test sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Create datasets and dataloaders
    train_dataset = AirQualityDataset(X_train, y_train)
    test_dataset = AirQualityDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model with larger hidden size
    input_size = X.shape[2]
    hidden_size = 128  # Increased hidden size
    num_layers = 2
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model = ImprovedSingleParameterLSTM(input_size, hidden_size, num_layers, PREDICTION_HORIZON).to(device)
    logger.info(f"Model initialized for {parameter}")
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Training loop with early stopping
    best_loss = float('inf')
    patience = 20  # Increased patience
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y).item()
        
        val_loss /= len(test_loader)
        
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        if patience_counter >= patience:
            logger.info(f'Early stopping triggered at epoch {epoch+1}')
            break
    
    # Load best model state
    model.load_state_dict(best_model_state)
    
    # Save model and scaler
    save_model(model, scaler, city, parameter)
    
    return model, scaler

def save_model(model, scaler, city, parameter):
    """Save model and scaler for a specific parameter"""
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save model
    model_path = f'models/lstm_improved_{city}_{parameter}.pth'
    torch.save(model.state_dict(), model_path)
    
    # Save scaler
    scaler_path = f'models/scaler_improved_{city}_{parameter}.pkl'
    joblib.dump(scaler, scaler_path)
    
    logger.info(f"Model and scaler saved for {city} - {parameter}")

def main():
    city = "Kyiv"
    parameters = ['co', 'no2', 'o3', 'so2', 'pm25', 'pm10']
    
    for parameter in parameters:
        logger.info(f"Training model for {parameter}")
        model, scaler = train_model(city, parameter)
        logger.info(f"Completed training for {parameter}")

if __name__ == "__main__":
    main() 