import torch
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import logging
import os
import joblib
from train_lstm import ImprovedLSTMModel, prepare_data
from config import SEQUENCE_LENGTH, PREDICTION_HORIZON

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_model(city='Kyiv'):
    """Evaluate LSTM model with R² and MSE metrics"""
    logger.info(f"Evaluating LSTM model for {city}")
    
    # Load model and scaler
    model_path = f'models/{city}_lstm_model.pth'
    scaler_path = f'models/{city}_lstm_scaler.pkl'
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Model or scaler files not found for {city}")
    
    # Prepare data
    X, y, scaler, original_data = prepare_data(city)
    
    # Initialize model
    input_size = X.shape[2]  # Updated input size with new features
    hidden_size = 128  # Reduced hidden size
    num_layers = 2  # Reduced number of layers
    output_size = 6  # Predicting all air quality parameters
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImprovedLSTMModel(input_size, hidden_size, num_layers, output_size, PREDICTION_HORIZON).to(device)
    
    # Load model weights
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Make predictions
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        predictions = model(X_tensor)
        predictions = predictions.cpu().numpy()
    
    # Reshape predictions and actual values for evaluation
    predictions = predictions.reshape(-1, output_size)
    actual_values = y.reshape(-1, output_size)
    
    # Inverse transform predictions and actual values
    predictions = scaler.inverse_transform(predictions)
    actual_values = scaler.inverse_transform(actual_values)
    
    # Calculate metrics for each parameter
    parameters = ['co', 'no2', 'o3', 'so2', 'pm25', 'pm10']
    results = {}
    
    for i, param in enumerate(parameters):
        param_predictions = predictions[:, i]
        param_actual = actual_values[:, i]
        
        # Calculate R²
        r2 = r2_score(param_actual, param_predictions)
        
        # Calculate MSE
        mse = mean_squared_error(param_actual, param_predictions)
        
        # Calculate RMSE
        rmse = np.sqrt(mse)
        
        results[param] = {
            'R²': r2,
            'MSE': mse,
            'RMSE': rmse
        }
        
        logger.info(f"\nMetrics for {param}:")
        logger.info(f"R² Score: {r2:.4f}")
        logger.info(f"MSE: {mse:.4f}")
        logger.info(f"RMSE: {rmse:.4f}")
    
    # Calculate overall metrics
    overall_r2 = r2_score(actual_values.flatten(), predictions.flatten())
    overall_mse = mean_squared_error(actual_values.flatten(), predictions.flatten())
    overall_rmse = np.sqrt(overall_mse)
    
    logger.info("\nOverall Metrics:")
    logger.info(f"Overall R² Score: {overall_r2:.4f}")
    logger.info(f"Overall MSE: {overall_mse:.4f}")
    logger.info(f"Overall RMSE: {overall_rmse:.4f}")
    
    return results

if __name__ == "__main__":
    evaluate_model() 