import pandas as pd
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from air_quality_model import AirQualityPredictor
import logging
import os
from database.db_manager import get_db_session, AirQualityMeasurement, init_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize database
init_db()

def fetch_test_data_from_db(city=None, limit=500):
    """
    Fetch test data from the AirQualityMeasurement table in the database.
    Args:
        city (str, optional): City to filter by. If None, use all cities.
        limit (int): Number of most recent records to fetch.
    Returns:
        pd.DataFrame: DataFrame with air quality data
    """
    try:
        session = get_db_session()
        query = session.query(AirQualityMeasurement)
        if city:
            query = query.filter(AirQualityMeasurement.city == city)
        query = query.order_by(AirQualityMeasurement.timestamp.desc()).limit(limit)
        records = query.all()
        session.close()
        
        if not records:
            raise ValueError("No air quality data found in the database.")
            
        # Convert to DataFrame
        data = pd.DataFrame([
            {
                'datetime': r.timestamp,
                'co': r.co,
                'no2': r.no2,
                'o3': r.o3,
                'so2': r.so2,
                'pm25': r.pm25,
                'pm10': r.pm10
            }
            for r in records
        ])
        data = data.sort_values('datetime')
        data.set_index('datetime', inplace=True)
        return data
    except Exception as e:
        logger.error(f"Error fetching data from database: {str(e)}")
        raise

def evaluate_model(model_path, scaler_path, city=None, limit=500):
    """
    Evaluate the trained model on test data from the database.
    Args:
        model_path (str): Path to the trained model file
        scaler_path (str): Path to the scaler file
        city (str, optional): City to filter by
        limit (int): Number of records to use
    """
    try:
        # Load the trained model
        predictor = AirQualityPredictor.load_model(model_path)
        logger.info("Model loaded successfully")
        
        # Fetch test data from DB
        test_data = fetch_test_data_from_db(city=city, limit=limit)
        logger.info(f"Loaded test data from DB with shape: {test_data.shape}")
        
        # Make predictions
        predictions = predictor.predict(test_data)
        logger.info(f"Generated predictions with shape: {predictions.shape}")
        
        # Get the last 24 records from test data for comparison
        test_data_last_24 = test_data.iloc[-24:]
        logger.info(f"Using last 24 records from test data for comparison")
        
        # Calculate metrics for each feature
        metrics = {}
        for feature in predictor.feature_columns:
            metrics[feature] = {
                'MSE': mean_squared_error(test_data_last_24[feature], predictions[feature]),
                'RMSE': np.sqrt(mean_squared_error(test_data_last_24[feature], predictions[feature])),
                'MAE': mean_absolute_error(test_data_last_24[feature], predictions[feature]),
                'R2': r2_score(test_data_last_24[feature], predictions[feature])
            }
        
        # Print metrics
        logger.info("\nModel Evaluation Metrics:")
        for feature, feature_metrics in metrics.items():
            logger.info(f"\n{feature.upper()}:")
            for metric_name, value in feature_metrics.items():
                logger.info(f"{metric_name}: {value:.4f}")
        
        # Plot predictions vs actual for each feature
        plt.figure(figsize=(15, 10))
        for i, feature in enumerate(predictor.feature_columns, 1):
            plt.subplot(3, 2, i)
            plt.plot(test_data_last_24.index, test_data_last_24[feature], label='Actual', alpha=0.7)
            plt.plot(predictions.index, predictions[feature], label='Predicted', alpha=0.7)
            plt.title(f'{feature.upper()} - Actual vs Predicted')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('model_evaluation_plots.png')
        logger.info("Evaluation plots saved as 'model_evaluation_plots.png'")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error during model evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    # Get the directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Paths to model files
    model_path = os.path.join(current_dir, "Kyiv_lstm_model.pth")
    scaler_path = os.path.join(current_dir, "Kyiv_lstm_scaler.pkl")
    city = "Kyiv"  # Set to a city name like 'Kyiv' if you want to filter
    limit = 500  # Number of records to use
    
    # Run evaluation
    metrics = evaluate_model(model_path, scaler_path, city=city, limit=limit) 