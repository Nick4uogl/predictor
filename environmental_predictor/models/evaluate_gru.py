import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gru_model import GRUModel, prepare_data
from config import SEQUENCE_LENGTH, PREDICTION_HORIZON
import logging
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns

# Налаштування логування
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Створюємо директорію для графіків
os.makedirs('plots_gru', exist_ok=True)

def load_model(city):
    """Load trained GRU model and scaler"""
    model_path = f'models/{city}_gru_model.pth'
    scaler_path = f'models/{city}_gru_scaler.pkl'
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Model files for {city} not found")
    
    # Initialize model
    input_size = 6  # co, no2, o3, so2, pm25, pm10
    hidden_size = 256
    num_layers = 2
    output_size = 6
    
    model = GRUModel(input_size, hidden_size, num_layers, output_size, PREDICTION_HORIZON)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Load scaler
    import joblib
    scaler = joblib.load(scaler_path)
    
    return model, scaler

def evaluate_model(city):
    """Evaluate GRU model performance"""
    logger.info(f"Evaluating GRU model for {city}")
    
    # Load model and prepare data
    model, scaler = load_model(city)
    X, y, _, original_data = prepare_data(city)
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X)
        predictions = model(X_tensor).numpy()
    
    # Reshape predictions and actual values
    predictions = predictions.reshape(-1, 6)  # 6 параметрів
    actual_values = y.reshape(-1, 6)
    
    # Inverse transform predictions and actual values
    predictions = scaler.inverse_transform(predictions)
    actual_values = scaler.inverse_transform(actual_values)
    
    # Calculate metrics for each parameter
    param_names = ['co', 'o3', 'so2', 'pm25', 'pm10']
    metrics = {}
    
    for i, param in enumerate(param_names):
        metrics[param] = {
            'rmse': np.sqrt(mean_squared_error(actual_values[:, i], predictions[:, i])),
            'mae': mean_absolute_error(actual_values[:, i], predictions[:, i]),
            'r2': r2_score(actual_values[:, i], predictions[:, i])
        }
        logger.info(f"\nMetrics for {param}:")
        logger.info(f"RMSE: {metrics[param]['rmse']:.4f}")
        logger.info(f"MAE: {metrics[param]['mae']:.4f}")
        logger.info(f"R2: {metrics[param]['r2']:.4f}")
    
    return metrics, predictions, actual_values, param_names

def plot_predictions(predictions, actual_values, param_names, city):
    """Plot predictions vs actual values for each parameter"""
    n_params = len(param_names)
    fig, axes = plt.subplots(n_params, 1, figsize=(15, 4*n_params))
    fig.suptitle(f'Predictions vs Actual Values for {city}', fontsize=16)
    
    for i, param in enumerate(param_names):
        ax = axes[i] if n_params > 1 else axes
        ax.plot(actual_values[:, i], label='Actual', alpha=0.7)
        ax.plot(predictions[:, i], label='Predicted', alpha=0.7)
        ax.set_title(f'{param.upper()}')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'plots_gru/{city}_predictions.png')
    plt.close()

def plot_error_distribution(predictions, actual_values, param_names, city):
    """Plot error distribution for each parameter"""
    n_params = len(param_names)
    fig, axes = plt.subplots(n_params, 1, figsize=(15, 4*n_params))
    fig.suptitle(f'Error Distribution for {city}', fontsize=16)
    
    for i, param in enumerate(param_names):
        ax = axes[i] if n_params > 1 else axes
        errors = predictions[:, i] - actual_values[:, i]
        sns.histplot(errors, kde=True, ax=ax)
        ax.set_title(f'{param.upper()} Error Distribution')
        ax.set_xlabel('Error')
        ax.set_ylabel('Count')
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'plots_gru/{city}_error_distribution.png')
    plt.close()

def plot_error_over_time(predictions, actual_values, param_names, city):
    """Plot error over time for each parameter"""
    n_params = len(param_names)
    fig, axes = plt.subplots(n_params, 1, figsize=(15, 4*n_params))
    fig.suptitle(f'Error Over Time for {city}', fontsize=16)
    
    for i, param in enumerate(param_names):
        ax = axes[i] if n_params > 1 else axes
        errors = predictions[:, i] - actual_values[:, i]
        ax.plot(errors)
        ax.set_title(f'{param.upper()} Error Over Time')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Error')
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'plots_gru/{city}_error_over_time.png')
    plt.close()

def calculate_accuracy_probability(predictions, actual_values, param_names, city):
    """Calculate probability of prediction accuracy for different time horizons"""
    n_params = len(param_names)
    horizons = range(1, PREDICTION_HORIZON + 1)
    accuracy_thresholds = [0.1, 0.2, 0.3]  # 10%, 20%, 30% error thresholds
    
    # Create figure for accuracy plots
    fig, axes = plt.subplots(n_params, 1, figsize=(15, 4*n_params))
    fig.suptitle(f'Prediction Accuracy Probability Over Time for {city}', fontsize=16)
    
    for i, param in enumerate(param_names):
        ax = axes[i] if n_params > 1 else axes
        
        # Calculate relative errors for each horizon
        relative_errors = np.abs(predictions[:, i] - actual_values[:, i]) / (actual_values[:, i] + 1e-10)
        
        # Calculate accuracy probabilities for each threshold
        for threshold in accuracy_thresholds:
            accuracies = []
            for h in horizons:
                # Get predictions and actual values for this horizon
                pred_h = predictions[h-1::PREDICTION_HORIZON, i]
                actual_h = actual_values[h-1::PREDICTION_HORIZON, i]
                
                # Calculate relative error for this horizon
                rel_error = np.abs(pred_h - actual_h) / (actual_h + 1e-10)
                
                # Calculate percentage of predictions within threshold
                accuracy = np.mean(rel_error <= threshold) * 100
                accuracies.append(accuracy)
            
            # Plot accuracy curve
            ax.plot(horizons, accuracies, 
                   label=f'Error ≤ {threshold*100}%',
                   marker='o')
        
        ax.set_title(f'{param.upper()}')
        ax.set_xlabel('Prediction Horizon (hours)')
        ax.set_ylabel('Accuracy Probability (%)')
        ax.legend()
        ax.grid(True)
        
        # Set y-axis limits
        ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(f'plots_gru/{city}_accuracy_probability.png')
    plt.close()
    
    # Log accuracy statistics
    logger.info(f"\nPrediction Accuracy Statistics for {city}:")
    for i, param in enumerate(param_names):
        logger.info(f"\n{param.upper()}:")
        for threshold in accuracy_thresholds:
            accuracies = []
            for h in horizons:
                pred_h = predictions[h-1::PREDICTION_HORIZON, i]
                actual_h = actual_values[h-1::PREDICTION_HORIZON, i]
                rel_error = np.abs(pred_h - actual_h) / (actual_h + 1e-10)
                accuracy = np.mean(rel_error <= threshold) * 100
                accuracies.append(accuracy)
            
            logger.info(f"Error ≤ {threshold*100}%:")
            logger.info(f"  Average accuracy: {np.mean(accuracies):.2f}%")
            logger.info(f"  Best accuracy: {np.max(accuracies):.2f}% (at {horizons[np.argmax(accuracies)]} hours)")
            logger.info(f"  Worst accuracy: {np.min(accuracies):.2f}% (at {horizons[np.argmin(accuracies)]} hours)")

def main():
    city = 'Kyiv'
    
    # Evaluate model
    metrics, predictions, actual_values, param_names = evaluate_model(city)
    
    # Plot results
    plot_predictions(predictions, actual_values, param_names, city)
    plot_error_distribution(predictions, actual_values, param_names, city)
    plot_error_over_time(predictions, actual_values, param_names, city)
    calculate_accuracy_probability(predictions, actual_values, param_names, city)
    
    logger.info(f"\nEvaluation completed. Plots saved in plots_gru directory.")

if __name__ == "__main__":
    main() 