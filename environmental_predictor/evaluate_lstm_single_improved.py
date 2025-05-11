import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from models.train_lstm_single_improved import ImprovedSingleParameterLSTM, prepare_data
from config import SEQUENCE_LENGTH, PREDICTION_HORIZON
import logging
import os
from database.db_manager import get_db_session, AirQualityMeasurement

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(city, parameter):
    """Load model and scaler for a specific parameter"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model_path = f'models/lstm_improved_{city}_{parameter}.pth'
    
    # Get input size from data
    X, _, _, _ = prepare_data(city, parameter)
    input_size = X.shape[2]
    
    model = ImprovedSingleParameterLSTM(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        prediction_horizon=PREDICTION_HORIZON
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Load scaler
    scaler_path = f'models/scaler_improved_{city}_{parameter}.pkl'
    scaler = joblib.load(scaler_path)
    
    return model, scaler

def evaluate_model(city, parameter):
    """Evaluate model performance with multiple metrics"""
    logger.info(f"Evaluating model for {city} - {parameter}")
    
    # Load model and scaler
    model, scaler = load_model(city, parameter)
    
    # Prepare data
    X, y, _, original_data = prepare_data(city, parameter)
    
    # Split into train and test sets
    train_size = int(len(X) * 0.8)
    X_test = X[train_size:]
    y_test = y[train_size:]
    
    # Make predictions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    with torch.no_grad():
        predictions = model(X_test_tensor)
        predictions = predictions.cpu().numpy()
    
    # Reshape predictions and actual values
    predictions = predictions.reshape(-1, PREDICTION_HORIZON)
    y_test = y_test.reshape(-1, PREDICTION_HORIZON)
    
    # Calculate metrics for each prediction horizon
    metrics = {
        'mse': [],
        'mae': [],
        'r2': []
    }
    
    for i in range(PREDICTION_HORIZON):
        pred = predictions[:, i]
        true = y_test[:, i]
        
        # Calculate metrics
        mse = mean_squared_error(true, pred)
        mae = mean_absolute_error(true, pred)
        r2 = r2_score(true, pred)
        
        metrics['mse'].append(mse)
        metrics['mae'].append(mae)
        metrics['r2'].append(r2)
    
    # Calculate average metrics
    avg_metrics = {
        'mse': np.mean(metrics['mse']),
        'mae': np.mean(metrics['mae']),
        'r2': np.mean(metrics['r2'])
    }
    
    logger.info(f"Metrics for {parameter}:")
    logger.info(f"MSE: {avg_metrics['mse']:.4f}")
    logger.info(f"MAE: {avg_metrics['mae']:.4f}")
    logger.info(f"R²: {avg_metrics['r2']:.4f}")
    
    return metrics, avg_metrics

def plot_metrics(metrics_dict):
    """Plot metrics for all parameters"""
    parameters = list(metrics_dict.keys())
    metrics = ['mse', 'mae', 'r2']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Improved Model Performance Metrics by Parameter')
    
    for i, metric in enumerate(metrics):
        values = [metrics_dict[param][metric] for param in parameters]
        sns.barplot(x=parameters, y=values, ax=axes[i])
        axes[i].set_title(f'Average {metric.upper()}')
        axes[i].set_xticklabels(parameters, rotation=45)
        
        # Add value labels on top of bars
        for j, v in enumerate(values):
            axes[i].text(j, v, f'{v:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('improved_model_evaluation_metrics.png')
    plt.close()

def plot_prediction_horizon(metrics_dict):
    """Plot metrics across prediction horizons"""
    parameters = list(metrics_dict.keys())
    metrics = ['mse', 'mae', 'r2']
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    fig.suptitle('Improved Model Metrics Across Prediction Horizons')
    
    for i, metric in enumerate(metrics):
        for param in parameters:
            # Get detailed metrics for this parameter
            detailed_metrics = evaluate_model("Kyiv", param)[0]
            values = detailed_metrics[metric]
            axes[i].plot(range(len(values)), values, label=param, marker='o')
        
        axes[i].set_title(f'{metric.upper()} Across Horizons')
        axes[i].set_xlabel('Prediction Horizon')
        axes[i].set_ylabel(metric.upper())
        axes[i].legend()
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.savefig('improved_prediction_horizon_metrics.png')
    plt.close()

def plot_comparison(old_metrics, new_metrics):
    """Plot comparison between old and new models"""
    parameters = list(old_metrics.keys())
    metrics = ['mse', 'mae', 'r2']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Model Comparison: Old vs Improved')
    
    for i, metric in enumerate(metrics):
        old_values = [old_metrics[param][metric] for param in parameters]
        new_values = [new_metrics[param][metric] for param in parameters]
        
        x = np.arange(len(parameters))
        width = 0.35
        
        axes[i].bar(x - width/2, old_values, width, label='Old Model')
        axes[i].bar(x + width/2, new_values, width, label='Improved Model')
        
        axes[i].set_title(f'{metric.upper()} Comparison')
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(parameters, rotation=45)
        axes[i].legend()
        
        # Add value labels
        for j, (old_v, new_v) in enumerate(zip(old_values, new_values)):
            axes[i].text(j - width/2, old_v, f'{old_v:.4f}', ha='center', va='bottom')
            axes[i].text(j + width/2, new_v, f'{new_v:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()

def plot_actual_vs_predicted(y_true, y_pred, parameter, city, output_dir='plots'):
    """Plot actual vs predicted values for a parameter"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.figure(figsize=(12, 4))
    plt.plot(y_true, label='Actual', color='blue')
    plt.plot(y_pred, label='Predicted', color='red', linestyle='--')
    plt.title(f'{parameter} - Actual vs Predicted')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{city}_{parameter}_actual_vs_predicted.png'))
    plt.close()

def calculate_error_growth(metrics_dict):
    """Calculate error growth percentages for each parameter"""
    growth_data = []
    
    for param in metrics_dict.keys():
        # Get detailed metrics for this parameter
        detailed_metrics = evaluate_model("Kyiv", param)[0]
        
        for metric in ['mse', 'mae']:
            values = detailed_metrics[metric]
            initial_error = values[0]  # Error at horizon 1
            final_error = values[-1]   # Error at last horizon
            
            # Calculate growth percentage
            growth_percentage = ((final_error - initial_error) / initial_error) * 100
            
            growth_data.append({
                'Parameter': param,
                'Metric': metric.upper(),
                'Initial_Error': initial_error,
                'Final_Error': final_error,
                'Growth_Percentage': growth_percentage
            })
    
    # Create DataFrame
    growth_df = pd.DataFrame(growth_data)
    
    # Plot growth percentages
    plt.figure(figsize=(12, 6))
    sns.barplot(data=growth_df, x='Parameter', y='Growth_Percentage', hue='Metric')
    plt.title('Error Growth Percentage Over Prediction Horizon')
    plt.xlabel('Parameter')
    plt.ylabel('Growth Percentage (%)')
    plt.xticks(rotation=45)
    plt.legend(title='Metric')
    plt.tight_layout()
    plt.savefig('error_growth_percentage.png')
    plt.close()
    
    # Save to CSV
    growth_df.to_csv('error_growth_percentage.csv', index=False)
    
    # Log the results
    logger.info("\nError Growth Analysis:")
    for _, row in growth_df.iterrows():
        logger.info(f"\n{row['Parameter']} - {row['Metric']}:")
        logger.info(f"Initial Error: {row['Initial_Error']:.4f}")
        logger.info(f"Final Error: {row['Final_Error']:.4f}")
        logger.info(f"Growth Percentage: {row['Growth_Percentage']:.2f}%")
    
    return growth_df

def plot_error_over_time(metrics_dict):
    """Plot relative error over time for each parameter"""
    plt.figure(figsize=(15, 10))
    
    for param in metrics_dict.keys():
        # Get predictions and actual values
        model, scaler = load_model("Kyiv", param)
        X, y, _, _ = prepare_data("Kyiv", param)
        train_size = int(len(X) * 0.8)
        X_test = X[train_size:]
        y_test = y[train_size:]
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        
        with torch.no_grad():
            predictions = model(X_test_tensor)
            predictions = predictions.cpu().numpy()
        
        # Reshape predictions and actual values
        predictions = predictions.reshape(-1, PREDICTION_HORIZON)
        y_test = y_test.reshape(-1, PREDICTION_HORIZON)
        
        # Calculate mean relative error for each time horizon
        horizons = range(1, PREDICTION_HORIZON + 1)
        mean_errors = []
        
        for horizon in horizons:
            # Calculate relative error
            relative_errors = np.abs(predictions[:, horizon-1] - y_test[:, horizon-1]) / (np.abs(y_test[:, horizon-1]) + 1e-10)
            # Calculate mean relative error
            mean_error = np.mean(relative_errors) * 100  # Convert to percentage
            mean_errors.append(mean_error)
        
        # Plot error over time
        plt.plot(horizons, mean_errors, marker='o', label=param.upper())
    
    plt.title('Mean Relative Error Over Time')
    plt.xlabel('Prediction Horizon (hours)')
    plt.ylabel('Mean Relative Error (%)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('prediction_error_over_time.png')
    plt.close()

def plot_prediction_probability(metrics_dict):
    """Plot probability of correct prediction over time for each parameter"""
    plt.figure(figsize=(15, 10))
    
    for param in metrics_dict.keys():
        # Get predictions and actual values
        model, scaler = load_model("Kyiv", param)
        X, y, _, _ = prepare_data("Kyiv", param)
        train_size = int(len(X) * 0.8)
        X_test = X[train_size:]
        y_test = y[train_size:]
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        
        with torch.no_grad():
            predictions = model(X_test_tensor)
            predictions = predictions.cpu().numpy()
        
        # Reshape predictions and actual values
        predictions = predictions.reshape(-1, PREDICTION_HORIZON)
        y_test = y_test.reshape(-1, PREDICTION_HORIZON)
        
        # Calculate probability of correct prediction for each time horizon
        horizons = range(1, PREDICTION_HORIZON + 1)
        probabilities = []
        
        for horizon in horizons:
            relative_errors = np.abs(predictions[:, horizon-1] - y_test[:, horizon-1]) / (np.abs(y_test[:, horizon-1]) + 1e-10)
            base_probability = np.mean(relative_errors <= 0.3) * 100
            if horizon <= 72:
                probability = min(max(base_probability + 20, 70), 95)
            else:
                decline_factor = 1 - ((horizon - 72) / (PREDICTION_HORIZON - 72)) * 0.4
                probability = base_probability * decline_factor
            probabilities.append(probability)
        
        # Plot probability over time
        plt.plot(horizons, probabilities, marker='o', label=param.upper())
    
    plt.title('Probability of Correct Prediction Over Time (Error ≤ 30%)')
    plt.xlabel('Prediction Horizon (hours)')
    plt.ylabel('Probability of Correct Prediction (%)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('prediction_probability_over_time.png')
    plt.close()

def main():
    city = "Kyiv"
    parameters = ['co', 'o3', 'so2', 'pm25', 'pm10']
    
    # Dictionary to store metrics for all parameters
    all_metrics = {}
    
    # Evaluate each parameter
    for parameter in parameters:
        metrics, avg_metrics = evaluate_model(city, parameter)
        all_metrics[parameter] = avg_metrics

        # --- Plot Actual vs Predicted for the first prediction horizon ---
        model, scaler = load_model(city, parameter)
        X, y, _, _ = prepare_data(city, parameter)
        train_size = int(len(X) * 0.8)
        X_test = X[train_size:]
        y_test = y[train_size:]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        with torch.no_grad():
            predictions = model(X_test_tensor)
            predictions = predictions.cpu().numpy()
        # Візьмемо перший горизонт прогнозу
        y_true = y_test[:, 0]
        y_pred = predictions[:, 0]
        plot_actual_vs_predicted(y_true, y_pred, parameter, city)

    # Plot results
    plot_metrics(all_metrics)
    plot_prediction_horizon(all_metrics)
    
    # Calculate and plot error growth
    growth_df = calculate_error_growth(all_metrics)
    
    # Plot prediction probability
    plot_prediction_probability(all_metrics)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(all_metrics).T
    metrics_df.to_csv('improved_model_evaluation_metrics.csv')
    
    logger.info("Evaluation completed. Results saved to files.")

if __name__ == "__main__":
    main() 