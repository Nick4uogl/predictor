import xgboost as xgb
import numpy as np
import pandas as pd
from database.db_manager import get_db_session, AirQualityMeasurement
from config import SEQUENCE_LENGTH, PREDICTION_HORIZON
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import logging
import joblib
import os

# Налаштування логування
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_features(df, target_param):
    """Create features for XGBoost model"""
    # Create lag features
    lag_features = {}
    air_quality_params = ['co', 'no2', 'o3', 'so2', 'pm25', 'pm10']
    
    for param in air_quality_params:
        for i in range(1, SEQUENCE_LENGTH + 1):
            lag_features[f'{param}_lag_{i}'] = df[param].shift(i)
    
    # Create time-based features
    time_features = {
        'hour': df.index.hour,
        'day_of_week': df.index.dayofweek,
        'month': df.index.month
    }
    
    # Create target variable (future values)
    target_features = {f'target_{i}': df[target_param].shift(-i) for i in range(1, PREDICTION_HORIZON + 1)}
    
    # Combine all features
    all_features = {**lag_features, **time_features, **target_features}
    
    # Add all features to dataframe at once
    df = pd.concat([df, pd.DataFrame(all_features)], axis=1)
    
    return df

def prepare_data(city, target_param='pm25'):
    """Prepare data for XGBoost training"""
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
        
        # Handle missing values
        air_quality_columns = ['co', 'no2', 'o3', 'so2', 'pm25', 'pm10']
        df[air_quality_columns] = df[air_quality_columns].interpolate(method='time')
        
        # Create features
        df = create_features(df, target_param)
        
        # Drop rows with NaN values (from lag features and future targets)
        df = df.dropna()
        
        # Separate features and targets
        feature_columns = [col for col in df.columns if col.startswith(tuple([f'{param}_lag_' for param in air_quality_columns] + ['hour', 'day_of_week', 'month']))]
        target_columns = [col for col in df.columns if col.startswith('target_')]
        
        X = df[feature_columns]
        y = df[target_columns]
        
        # Normalize features
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, y, scaler, feature_columns
    
    finally:
        session.close()

def train_model(city, target_param='pm25', num_boost_round=200, early_stopping_rounds=20):
    """Train XGBoost model for a specific city and parameter"""
    # Prepare data
    X, y, scaler, feature_columns = prepare_data(city, target_param)
    
    # Split into train and test sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Set parameters
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'min_child_weight': 2,
        'gamma': 0.1,
        'tree_method': 'hist',
        'random_state': 42
    }
    
    # Train model
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=10
    )
    
    return model, scaler, feature_columns

def save_model(model, scaler, feature_columns, city, target_param):
    """Save trained model and scaler"""
    os.makedirs('models', exist_ok=True)
    
    # Save model
    model.save_model(f'models/{city}_{target_param}_xgboost_model.json')
    
    # Save scaler and feature columns
    joblib.dump({
        'scaler': scaler,
        'feature_columns': feature_columns
    }, f'models/{city}_{target_param}_xgboost_scaler.pkl')

def predict_future(model, scaler, feature_columns, last_sequence):
    """Make predictions for future time steps"""
    # Create features from last sequence
    df = pd.DataFrame(last_sequence, columns=feature_columns)
    
    # Create DMatrix
    dtest = xgb.DMatrix(df)
    
    # Make predictions
    predictions = model.predict(dtest)
    
    return predictions

if __name__ == "__main__":
    air_quality_params = ['co', 'no2', 'o3', 'so2', 'pm25', 'pm10']
    for city in ['Kyiv', 'Lviv', 'Kharkiv', 'Odesa', 'Dnipro']:
        for param in air_quality_params:
            logger.info(f"Training XGBoost model for {city} - {param}")
            model, scaler, feature_columns = train_model(city, param)
            save_model(model, scaler, feature_columns, city, param)
            logger.info(f"XGBoost model for {city} - {param} saved") 