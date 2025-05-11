from prophet import Prophet
import pandas as pd
from database.db_manager import get_db_session, AirQualityMeasurement
import logging
import joblib
import os
from sklearn.preprocessing import MinMaxScaler

# Налаштування логування
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_data(city, target_param):
    """Prepare data for Prophet training"""
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
        logger.info(f"Initial data shape for {city}: {df.shape}")
        
        # Convert timestamp to datetime and set as index
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Remove duplicate timestamps by taking the mean of values
        df = df.groupby('timestamp').mean().reset_index()
        logger.info(f"After removing duplicates: {df.shape}")
        
        # Handle missing values
        air_quality_columns = ['co', 'no2', 'o3', 'so2', 'pm25', 'pm10']
        
        # Check for missing values before filling
        missing_before = df[air_quality_columns].isna().sum()
        logger.info(f"Missing values before filling:\n{missing_before}")
        
        # First fill missing values with forward fill
        df[air_quality_columns] = df[air_quality_columns].ffill()
        
        # Then fill any remaining missing values with backward fill
        df[air_quality_columns] = df[air_quality_columns].bfill()
        
        # If there are still any NaN values, fill with column mean
        df[air_quality_columns] = df[air_quality_columns].fillna(df[air_quality_columns].mean())
        
        # Check for missing values after filling
        missing_after = df[air_quality_columns].isna().sum()
        logger.info(f"Missing values after filling:\n{missing_after}")
        
        # Prepare data for Prophet
        prophet_df = pd.DataFrame()
        prophet_df['ds'] = df['timestamp']
        prophet_df['y'] = df[target_param]
        
        # Add additional regressors
        for param in air_quality_columns:
            if param != target_param:
                prophet_df[param] = df[param]
        
        # Check for NaN values in prophet_df
        prophet_nan = prophet_df.isna().sum()
        logger.info(f"NaN values in prophet_df:\n{prophet_nan}")
        
        # Normalize regressors
        scaler = MinMaxScaler()
        regressor_columns = [col for col in air_quality_columns if col != target_param]
        prophet_df[regressor_columns] = scaler.fit_transform(prophet_df[regressor_columns])
        
        # Verify we have enough data
        if len(prophet_df) < 2:
            raise ValueError(f"Not enough data for city {city} after preprocessing. Shape: {prophet_df.shape}")
        
        logger.info(f"Final prophet_df shape: {prophet_df.shape}")
        return prophet_df, scaler
    
    finally:
        session.close()

def train_model(city, target_param='pm25'):
    """Train Prophet model for a specific city and parameter"""
    # Prepare data
    df, scaler = prepare_data(city, target_param)
    
    # Initialize Prophet model with custom seasonality
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        interval_width=0.95
    )
    
    # Add regressors
    air_quality_columns = ['co', 'no2', 'o3', 'so2', 'pm25', 'pm10']
    for param in air_quality_columns:
        if param != target_param:
            model.add_regressor(param)
    
    # Fit model
    model.fit(df)
    
    return model, scaler

def save_model(model, scaler, city, target_param):
    """Save trained model and scaler"""
    os.makedirs('models', exist_ok=True)
    
    # Save model
    joblib.dump(model, f'models/{city}_{target_param}_prophet_model.pkl')
    
    # Save scaler
    joblib.dump(scaler, f'models/{city}_{target_param}_prophet_scaler.pkl')

def predict_future(model, scaler, last_data, target_param, periods=72):
    """Make predictions for future time steps"""
    # Create future dataframe
    future = model.make_future_dataframe(periods=periods, freq='H')
    
    # Add regressors for future periods
    air_quality_columns = ['co', 'no2', 'o3', 'so2', 'pm25', 'pm10']
    for param in air_quality_columns:
        if param != target_param:
            future[param] = last_data[param].iloc[-1]
    
    # Make predictions
    forecast = model.predict(future)
    
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)

if __name__ == "__main__":
    air_quality_params = ['co', 'no2', 'o3', 'so2', 'pm25', 'pm10']
    for city in ['Kyiv', 'Lviv', 'Kharkiv', 'Odesa', 'Dnipro']:
        for param in air_quality_params:
            logger.info(f"Training Prophet model for {city} - {param}")
            model, scaler = train_model(city, param)
            save_model(model, scaler, city, param)
            logger.info(f"Prophet model for {city} - {param} saved") 