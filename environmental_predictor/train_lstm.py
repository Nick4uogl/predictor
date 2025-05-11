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

class ImprovedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, prediction_horizon, dropout_rate=0.2):
        super(ImprovedLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.prediction_horizon = prediction_horizon
        
        # LSTM layers with dropout
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True  # Using bidirectional LSTM
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Fully connected layers with dropout
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, output_size * prediction_horizon)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_size * 2)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)
        
    def attention_net(self, lstm_output):
        # Calculate attention weights
        attn_weights = self.attention(lstm_output)
        soft_attn_weights = torch.softmax(attn_weights, dim=1)
        
        # Apply attention weights
        context = torch.bmm(soft_attn_weights.transpose(1, 2), lstm_output)
        return context.squeeze(1)
        
    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # *2 for bidirectional
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Apply attention
        out = self.attention_net(out)
        
        # Apply layer normalization
        out = self.layer_norm1(out)
        
        # Fully connected layers
        out = self.leaky_relu(self.fc1(out))
        out = self.layer_norm2(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        
        # Reshape to [batch_size, prediction_horizon, output_size]
        out = out.view(out.size(0), self.prediction_horizon, -1)
        return out

def prepare_data(city):
    """Prepare data for LSTM training with improved preprocessing"""
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
        
        # Add time-based features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        
        # Handle missing values for numeric columns
        air_quality_columns = ['co', 'no2', 'o3', 'so2', 'pm25', 'pm10']
        df[air_quality_columns] = df[air_quality_columns].interpolate(method='time')
        
        # Add rolling statistics
        for col in air_quality_columns:
            df[f'{col}_rolling_mean_24h'] = df[col].rolling(window=24).mean()
            df[f'{col}_rolling_std_24h'] = df[col].rolling(window=24).std()
        
        # Fill NaN values in rolling statistics
        df = df.fillna(method='bfill')
        
        # Normalize the data
        scaler = MinMaxScaler()
        feature_columns = air_quality_columns + [col for col in df.columns if col not in ['hour', 'day_of_week', 'month']]
        scaled_data = scaler.fit_transform(df[feature_columns])
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_data) - SEQUENCE_LENGTH - PREDICTION_HORIZON + 1):
            X.append(scaled_data[i:(i + SEQUENCE_LENGTH)])
            y.append(scaled_data[i + SEQUENCE_LENGTH:i + SEQUENCE_LENGTH + PREDICTION_HORIZON, :len(air_quality_columns)])
        
        return np.array(X), np.array(y), scaler, df[air_quality_columns]
    
    finally:
        session.close()

def train_model(city, num_epochs=100, batch_size=64, learning_rate=0.0005):
    """Train improved LSTM model for a specific city"""
    logger.info(f"Starting data preparation for {city}")
    # Prepare data
    X, y, scaler, original_data = prepare_data(city)
    logger.info(f"Data prepared. Training set size: {len(X)}")
    
    # Split into train and test sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    logger.info(f"Data split into train ({len(X_train)}) and test ({len(X_test)}) sets")
    
    # Create datasets and dataloaders
    train_dataset = AirQualityDataset(X_train, y_train)
    test_dataset = AirQualityDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    input_size = X.shape[2]  # Updated input size with new features
    hidden_size = 128  # Reduced hidden size
    num_layers = 2  # Reduced number of layers
    output_size = 6  # Predicting all air quality parameters
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model = ImprovedLSTMModel(input_size, hidden_size, num_layers, output_size, PREDICTION_HORIZON).to(device)
    logger.info("Model initialized with architecture:")
    logger.info(f"- Input size: {input_size}")
    logger.info(f"- Hidden size: {hidden_size}")
    logger.info(f"- Number of LSTM layers: {num_layers}")
    logger.info(f"- Output size: {output_size}")
    logger.info(f"- Prediction horizon: {PREDICTION_HORIZON}")
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    logger.info("Starting training...")
    
    # Training loop with early stopping
    best_loss = float('inf')
    patience = 15
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
        
        # Early stopping check
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
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, scaler, original_data

def save_model(model, scaler, city):
    """Save trained model and scaler"""
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), f'models/{city}_lstm_model.pth')
    joblib.dump(scaler, f'models/{city}_lstm_scaler.pkl')
    logger.info(f"Model and scaler for {city} saved successfully")

def main():
    """Main function to train LSTM model for Kyiv"""
    city = 'Kyiv'
    
    logger.info(f"Training LSTM model for {city}")
    try:
        model, scaler, original_data = train_model(city)
        save_model(model, scaler, city)
        logger.info(f"Successfully trained and saved model for {city}")
    except Exception as e:
        logger.error(f"Error training model for {city}: {str(e)}")

if __name__ == "__main__":
    main() 