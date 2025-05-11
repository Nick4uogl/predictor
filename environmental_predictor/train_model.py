import logging
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.db_manager import AirQualityMeasurement, get_db_session
from models.lstm_model import LSTMModel
from models.prophet_model import prepare_data, train_model, save_model
from models.xgboost_model import prepare_data as prepare_xgboost_data, train_model as train_xgboost_model, save_model as save_xgboost_model
import pandas as pd
from datetime import datetime, timedelta
from config import DATABASE_URL

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

def main():
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Load data from database
    data = load_data_from_db()
    
    if data.empty:
        logger.error("No data found in database")
        return
    
    # Train a model for each city
    for city in data['city'].unique():
        logger.info(f"Training model for {city}")
        
        # Filter data for current city
        city_data = data[data['city'] == city].copy()
        
        # Initialize model
        model = LSTMModel(
            sequence_length=24,  # Use last 24 hours to predict
            prediction_horizon=24  # Predict next 24 hours
        )
        
        # Prepare data
        X, y = model.prepare_data(city_data)
        
        if len(X) == 0:
            logger.warning(f"Not enough data for {city}")
            continue
        
        # Train model
        model.train(X, y, epochs=100, batch_size=32)
        
        # Save model
        model_path = os.path.join('models', city)
        os.makedirs(model_path, exist_ok=True)
        model.save_model(model_path)
        
        logger.info(f"Model for {city} saved to {model_path}")

if __name__ == "__main__":
    main() 