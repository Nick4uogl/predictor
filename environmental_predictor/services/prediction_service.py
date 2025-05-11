from database.db_manager import get_db_session
from database.models import ModelConfiguration
import joblib
import torch
import xgboost as xgb
from prophet import Prophet
import logging
import numpy as np

logger = logging.getLogger(__name__)

class PredictionService:
    def __init__(self):
        self.session = get_db_session()
    
    def get_active_model(self):
        """Get the active model configuration"""
        config = self.session.query(ModelConfiguration).first()
        if not config:
            logger.warning("No model configuration found, using LSTM as default")
            return 'lstm'
        return config.active_model
    
    def load_model(self, city, model_type):
        """Load the appropriate model based on type"""
        try:
            if model_type == 'lstm':
                model_path = f'models/{city}_lstm_model.pth'
                scaler_path = f'models/{city}_lstm_scaler.pkl'
                
                # Load LSTM model
                model = torch.load(model_path)
                scaler = joblib.load(scaler_path)
                return model, scaler
            
            elif model_type == 'xgboost':
                model_path = f'models/{city}_xgboost_model.json'
                scaler_path = f'models/{city}_xgboost_scaler.pkl'
                
                # Load XGBoost model
                model = xgb.Booster()
                model.load_model(model_path)
                scaler_data = joblib.load(scaler_path)
                return model, scaler_data
            
            elif model_type == 'prophet':
                model_path = f'models/{city}_prophet_model.pkl'
                scaler_path = f'models/{city}_prophet_scaler.pkl'
                
                # Load Prophet model
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                return model, scaler
            
            else:
                raise ValueError(f"Unknown model type: {model_type}")
                
        except Exception as e:
            logger.error(f"Error loading model for {city}: {str(e)}")
            raise
    
    def predict(self, city, input_data):
        """Make prediction using the active model"""
        try:
            # Get active model type
            model_type = self.get_active_model()
            
            # Load model and scaler
            model, scaler = self.load_model(city, model_type)
            
            # Make prediction based on model type
            if model_type == 'lstm':
                return self._predict_lstm(model, scaler, input_data)
            elif model_type == 'xgboost':
                return self._predict_xgboost(model, scaler, input_data)
            elif model_type == 'prophet':
                return self._predict_prophet(model, scaler, input_data)
            
        except Exception as e:
            logger.error(f"Error making prediction for {city}: {str(e)}")
            raise
    
    def _predict_lstm(self, model, scaler, input_data):
        """Make prediction using LSTM model"""
        # Scale input data
        scaled_input = scaler.transform(input_data)
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(scaled_input).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            predictions = output.squeeze().numpy()
        
        # Inverse transform predictions
        predictions = scaler.inverse_transform(np.column_stack([predictions, np.zeros((len(predictions), 4))]))[:, 0]
        return predictions
    
    def _predict_xgboost(self, model, scaler_data, input_data):
        """Make prediction using XGBoost model"""
        # Scale input data
        scaled_input = scaler_data['scaler'].transform(input_data)
        
        # Create DMatrix
        dtest = xgb.DMatrix(scaled_input)
        
        # Make prediction
        predictions = model.predict(dtest)
        return predictions
    
    def _predict_prophet(self, model, scaler, input_data):
        """Make prediction using Prophet model"""
        # Create future dataframe
        future = model.make_future_dataframe(periods=72, freq='H')
        
        # Add regressors
        for col in ['pm10', 'temperature', 'humidity', 'wind_speed']:
            future[col] = input_data[col].iloc[-1]
        
        # Make prediction
        forecast = model.predict(future)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(72) 