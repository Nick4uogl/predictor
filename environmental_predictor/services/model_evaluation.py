import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import torch
import xgboost as xgb
from prophet import Prophet
import logging
from datetime import datetime, timedelta
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

logger = logging.getLogger(__name__)

def load_model(city, model_type):
    """Load a trained model for a specific city"""
    try:
        if model_type == 'lstm':
            model_path = f'models/{city}_lstm_model.pth'
            scaler_path = f'models/{city}_scaler.pkl'
            model = torch.load(model_path)
            scaler = joblib.load(scaler_path)
            return model, scaler
        
        elif model_type == 'xgboost':
            model_path = f'models/{city}_xgboost_model.json'
            scaler_path = f'models/{city}_xgboost_scaler.pkl'
            model = xgb.Booster()
            model.load_model(model_path)
            scaler_data = joblib.load(scaler_path)
            return model, scaler_data
        
        elif model_type == 'prophet':
            model_path = f'models/{city}_prophet_model.pkl'
            scaler_path = f'models/{city}_prophet_scaler.pkl'
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            return model, scaler
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    except Exception as e:
        logger.error(f"Error loading {model_type} model for {city}: {str(e)}")
        return None, None

def evaluate_model(city, model_type, test_data):
    """Evaluate a model's performance on test data"""
    try:
        model, scaler = load_model(city, model_type)
        if model is None:
            return None
        
        if model_type == 'lstm':
            return evaluate_lstm(model, scaler, test_data)
        elif model_type == 'xgboost':
            return evaluate_xgboost(model, scaler, test_data)
        elif model_type == 'prophet':
            return evaluate_prophet(model, scaler, test_data)
    
    except Exception as e:
        logger.error(f"Error evaluating {model_type} model for {city}: {str(e)}")
        return None

def evaluate_lstm(model, scaler, test_data):
    """Evaluate LSTM model"""
    # Scale test data
    scaled_test = scaler.transform(test_data)
    
    # Convert to tensor
    test_tensor = torch.FloatTensor(scaled_test).unsqueeze(0)
    
    # Make predictions
    with torch.no_grad():
        predictions = model(test_tensor)
        predictions = predictions.squeeze().numpy()
    
    # Inverse transform predictions
    predictions = scaler.inverse_transform(np.column_stack([predictions, np.zeros((len(predictions), 4))]))[:, 0]
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(test_data['pm25'], predictions))
    mae = mean_absolute_error(test_data['pm25'], predictions)
    r2 = r2_score(test_data['pm25'], predictions)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': predictions.tolist()
    }

def evaluate_xgboost(model, scaler_data, test_data):
    """Evaluate XGBoost model"""
    # Scale test data
    scaled_test = scaler_data['scaler'].transform(test_data)
    
    # Create DMatrix
    dtest = xgb.DMatrix(scaled_test)
    
    # Make predictions
    predictions = model.predict(dtest)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(test_data['pm25'], predictions))
    mae = mean_absolute_error(test_data['pm25'], predictions)
    r2 = r2_score(test_data['pm25'], predictions)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': predictions.tolist()
    }

def evaluate_prophet(model, scaler, test_data):
    """Evaluate Prophet model"""
    # Create future dataframe
    future = model.make_future_dataframe(periods=len(test_data), freq='H')
    
    # Add regressors
    for col in ['pm10', 'temperature', 'humidity', 'wind_speed']:
        future[col] = test_data[col]
    
    # Make predictions
    forecast = model.predict(future)
    predictions = forecast['yhat'].tail(len(test_data))
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(test_data['pm25'], predictions))
    mae = mean_absolute_error(test_data['pm25'], predictions)
    r2 = r2_score(test_data['pm25'], predictions)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': predictions.tolist()
    }

def get_model_comparison(city, test_data):
    """Get comparison of all models for a specific city"""
    session = get_db_session()
    try:
        # Get the latest evaluation for each model type
        latest_evaluations = session.query(ModelEvaluation).filter(
            ModelEvaluation.city == city
        ).order_by(
            ModelEvaluation.evaluation_date.desc()
        ).all()
        
        if not latest_evaluations:
            logger.warning(f"No evaluation data found for {city}")
            return None
            
        comparisons = {}
        for eval in latest_evaluations:
            comparisons[eval.model_type] = {
                'mae': eval.mae,
                'rmse': eval.rmse,
                'r2': eval.r2,
                'metrics': eval.metrics
            }
            
        return comparisons
        
    except Exception as e:
        logger.error(f"Error getting model comparison for {city}: {str(e)}")
        return None
        
    finally:
        session.close() 