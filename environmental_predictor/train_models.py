import logging
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.db_manager import AirQualityMeasurement, get_db_session
from models.lstm_model import LSTMModel, train_model, predict, save_model
from models.prophet_model import prepare_data, train_model as train_prophet_model, save_model as save_prophet_model
from models.xgboost_model import prepare_data as prepare_xgboost_data, train_model as train_xgboost_model, save_model as save_xgboost_model
from models.random_forest_model import train_random_forest_model
import pandas as pd
from datetime import datetime, timedelta
from config import DATABASE_URL
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define parameters to model
PARAMETERS = ['co', 'no2', 'o3', 'so2', 'pm25', 'pm10']

def load_data_from_db():
    """Load air quality data from the database"""
    session = get_db_session()
    
    try:
        # Query all measurements
        measurements = session.query(AirQualityMeasurement).all()
        
        # Convert to DataFrame
        data = pd.DataFrame([{
            'city': m.city_name,
            'timestamp': m.timestamp,
            'lat': m.lat,
            'lon': m.lon,
            'co': m.co,
            'no2': m.no2,
            'o3': m.o3,
            'so2': m.so2,
            'pm2_5': m.pm2_5,
            'pm10': m.pm10
        } for m in measurements])
        
        # Log total number of records and records per city
        logger.info(f"Total number of records loaded from database: {len(data)}")
        city_counts = data['city'].value_counts()
        logger.info("Number of records per city:")
        for city, count in city_counts.items():
            logger.info(f"  {city}: {count} records")
        
        return data
        
    finally:
        session.close()

def train_lstm_models(city_data, city):
    """Train LSTM models for a specific city"""
    logger.info(f"Training LSTM models for {city}")
    
    # Get the number of records available for training
    num_records = len(city_data)
    logger.info(f"Number of records available for LSTM training: {num_records}")
    
    try:
        # Train the model
        trained_model, scaler, original_data = train_model(city)
        
        # Get predictions for all parameters
        predictions = predict(trained_model, scaler, original_data)
        
        # Calculate and log RMSE for each parameter
        param_indices = {'co': 0, 'no2': 1, 'o3': 2, 'so2': 3, 'pm25': 4, 'pm10': 5}
        for param, index in param_indices.items():
            # Get actual values
            actual_values = original_data[param].values[-len(predictions):]
            
            # Calculate RMSE
            rmse = np.sqrt(np.mean((predictions[:, index] - actual_values) ** 2))
            logger.info(f"RMSE for {param} in {city}: {rmse:.4f}")
        
        # Save the model
        save_model(trained_model, scaler, city)
        logger.info(f"LSTM model for {city} trained and saved successfully")
        
    except Exception as e:
        logger.error(f"Error training LSTM model for {city}: {str(e)}")
        raise

def train_prophet_models(city_data, city_name):
    """Train Prophet models for each parameter"""
    logger.info(f"Training Prophet models for {city_name}")
    logger.info(f"Number of records available for Prophet training: {len(city_data)}")
    
    for param in PARAMETERS:
        logger.info(f"Training Prophet model for {param} in {city_name}")
        
        try:
            # Prepare data for current parameter
            df, scaler = prepare_data(city_name, param)
            
            if len(df) == 0:
                logger.warning(f"Not enough data for Prophet model for {param} in {city_name}")
                continue
            
            logger.info(f"Number of records prepared for Prophet training ({param}): {len(df)}")
            
            # Train model
            model, scaler = train_prophet_model(city_name, param)
            
            # Save model
            model_path = os.path.join('models', 'prophet', city_name, param)
            os.makedirs(model_path, exist_ok=True)
            save_prophet_model(model, scaler, city_name, param)
            logger.info(f"Prophet model for {param} in {city_name} saved to {model_path}")
            
        except Exception as e:
            logger.error(f"Error training Prophet model for {param} in {city_name}: {str(e)}")
            continue

def train_xgboost_models(city_data, city_name):
    """Train XGBoost models for each parameter"""
    logger.info(f"Training XGBoost models for {city_name}")
    logger.info(f"Number of records available for XGBoost training: {len(city_data)}")
    
    for param in PARAMETERS:
        logger.info(f"Training XGBoost model for {param} in {city_name}")
        
        try:
            # Prepare data for current parameter
            X, y, scaler, feature_columns = prepare_xgboost_data(city_name, param)
            
            if len(X) == 0:
                logger.warning(f"Not enough data for XGBoost model for {param} in {city_name}")
                continue
            
            logger.info(f"Number of records prepared for XGBoost training ({param}): {len(X)}")
            
            # Train model
            model, scaler, feature_columns = train_xgboost_model(city_name, param)
            
            # Save model
            model_path = os.path.join('models', 'xgboost', city_name, param)
            os.makedirs(model_path, exist_ok=True)
            save_xgboost_model(model, scaler, feature_columns, city_name, param)
            logger.info(f"XGBoost model for {param} in {city_name} saved to {model_path}")
            
        except Exception as e:
            logger.error(f"Error training XGBoost model for {param} in {city_name}: {str(e)}")
            continue

def train_random_forest_models(city_data, city):
    """Train Random Forest models for a specific city"""
    logger.info(f"Training Random Forest models for {city}")
    
    try:
        # Train the model
        model = train_random_forest_model(city)
        logger.info(f"Random Forest model for {city} trained and saved successfully")
        
    except Exception as e:
        logger.error(f"Error training Random Forest model for {city}: {str(e)}")
        raise

def main():
    """Main function to train all models"""
    # Load data
    data = load_data_from_db()
    
    # Get unique cities
    cities = data['city'].unique()
    
    for city in cities:
        logger.info(f"\nProcessing city: {city}")
        
        # Filter data for current city
        city_data = data[data['city'] == city]
        
        # Train LSTM model
        train_lstm_models(city_data, city)
        
        # Train Random Forest model
        train_random_forest_models(city_data, city)

if __name__ == "__main__":
    main() 