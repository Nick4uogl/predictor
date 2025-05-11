import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
import os
import logging
from database.db_manager import get_db_session, AirQualityMeasurement

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ARIMAModel:
    def __init__(self, order=(1, 1, 1)):
        """
        Initialize ARIMA model
        :param order: tuple of (p,d,q) parameters for ARIMA model
        """
        self.order = order
        self.model = None
        self.model_fitted = None
        
    def fit(self, data):
        """
        Fit ARIMA model to the data
        :param data: pandas Series with time index
        """
        self.model = ARIMA(data, order=self.order)
        self.model_fitted = self.model.fit()
        return self
        
    def predict(self, steps):
        """
        Make predictions for future steps
        :param steps: number of steps to predict
        :return: predictions
        """
        if self.model_fitted is None:
            raise ValueError("Model needs to be fitted before making predictions")
        return self.model_fitted.forecast(steps=steps)
    
    def save_model(self, path):
        """
        Save the fitted model
        :param path: path to save the model
        """
        if self.model_fitted is None:
            raise ValueError("No fitted model to save")
        joblib.dump(self.model_fitted, path)
    
    @classmethod
    def load_model(cls, path):
        """
        Load a saved model
        :param path: path to the saved model
        :return: loaded model
        """
        model = cls()
        model.model_fitted = joblib.load(path)
        return model

def load_data_from_db(city: str = None):
    """
    Load air quality data from database.
    
    Args:
        city (str, optional): Filter data by city. If None, load all data.
        
    Returns:
        pd.DataFrame: Air quality data
    """
    session = get_db_session()
    try:
        query = session.query(
            AirQualityMeasurement.timestamp,
            AirQualityMeasurement.city,
            AirQualityMeasurement.co,
            AirQualityMeasurement.no2,
            AirQualityMeasurement.o3,
            AirQualityMeasurement.so2,
            AirQualityMeasurement.pm25,
            AirQualityMeasurement.pm10
        )
        
        if city:
            query = query.filter(AirQualityMeasurement.city == city)
            
        query = query.order_by(AirQualityMeasurement.timestamp)
        
        df = pd.read_sql(query.statement, session.bind)
        
        # Log data statistics
        logger.info(f"Total number of records loaded: {len(df)}")
        if not city:
            city_counts = df['city'].value_counts()
            logger.info("Number of records per city:")
            for city, count in city_counts.items():
                logger.info(f"  {city}: {count} records")
        
        return df
        
    finally:
        session.close()

def train_and_evaluate_arima(city: str, target_param: str = 'pm25', order=(1, 1, 1)):
    """
    Train and evaluate ARIMA model for a specific city and parameter
    
    Args:
        city (str): City name
        target_param (str): Target parameter to predict
        order (tuple): ARIMA order parameters (p,d,q)
        
    Returns:
        tuple: (trained model, evaluation metrics)
    """
    logger.info(f"Training ARIMA model for {city} - {target_param}")
    
    # Load data
    df = load_data_from_db(city)
    
    if df.empty:
        raise ValueError(f"No data found for city: {city}")
    
    # Set timestamp as index and sort
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    
    # Handle missing values
    df[target_param] = df[target_param].interpolate(method='time')
    
    # Split data into train and test sets (80/20 split)
    train_size = int(len(df) * 0.8)
    train_data = df[target_param][:train_size]
    test_data = df[target_param][train_size:]
    
    # Initialize and train model
    model = ARIMAModel(order=order)
    model.fit(train_data)
    
    # Make predictions
    predictions = model.predict(len(test_data))
    
    # Calculate metrics
    mse = mean_squared_error(test_data, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test_data, predictions)
    r2 = r2_score(test_data, predictions)
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }
    
    # Create plots directory if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(test_data.index, test_data, label='Actual')
    plt.plot(test_data.index, predictions, label='Predicted')
    plt.title(f'ARIMA Model Predictions vs Actual Values - {city} - {target_param}')
    plt.xlabel('Time')
    plt.ylabel(target_param)
    plt.legend()
    plt.savefig(f'plots/arima_predictions_{city}_{target_param}.png')
    plt.close()
    
    # Save model
    model_path = f'models/arima_model_{city}_{target_param}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.joblib'
    if not os.path.exists('models'):
        os.makedirs('models')
    model.save_model(model_path)
    
    logger.info(f"Model metrics for {city} - {target_param}:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    return model, metrics

def main():
    """Main function to train ARIMA models for Kyiv"""
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Set city to Kyiv
    city = 'Kyiv'
    logger.info(f"\nProcessing city: {city}")
    
    # Parameters to model
    parameters = ['co', 'no2', 'o3', 'so2', 'pm25', 'pm10']
    
    for param in parameters:
        try:
            model, metrics = train_and_evaluate_arima(city, param)
            logger.info(f"ARIMA model for {city} - {param} trained and saved successfully")
        except Exception as e:
            logger.error(f"Error training model for {city} - {param}: {str(e)}")
            continue

if __name__ == "__main__":
    main() 