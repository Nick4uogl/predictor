import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import joblib
import os
from typing import Tuple, Dict, List
from database.db_manager import get_db_session, AirQualityMeasurement

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LightGBMPredictor:
    def __init__(self, sequence_length: int = 24, prediction_horizon: int = 24):
        """
        Initialize the LightGBM predictor.
        
        Args:
            sequence_length (int): Number of past hours to use for prediction
            prediction_horizon (int): Number of hours to predict into the future
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.scaler = MinMaxScaler()
        self.models = {}  # Dictionary to store models for each parameter
        self.feature_columns = ['co', 'o3', 'so2', 'pm25', 'pm10']
        
        # Pollutant-specific parameters
        self.param_specific_params = {
            'default': {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.01,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_child_samples': 20,
                'min_child_weight': 0.001,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'verbose': -1
            }
        }
        
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Prepare data for model training.
        
        Args:
            data (pd.DataFrame): DataFrame with air quality measurements
            
        Returns:
            Tuple[np.ndarray, Dict[str, np.ndarray]]: X (features) and y (targets for each parameter)
        """
        # Sort by datetime
        data = data.sort_values('timestamp')
        
        # Create a new DataFrame to avoid fragmentation
        features = pd.DataFrame(index=data.index)
        
        # Add time-based features
        features['hour'] = data['timestamp'].dt.hour
        features['day_of_week'] = data['timestamp'].dt.dayofweek
        features['month'] = data['timestamp'].dt.month
        features['is_weekend'] = features['day_of_week'].isin([5, 6]).astype(int)
        features['is_night'] = ((features['hour'] >= 22) | (features['hour'] <= 5)).astype(int)
        features['is_morning'] = ((features['hour'] >= 6) & (features['hour'] <= 11)).astype(int)
        features['is_afternoon'] = ((features['hour'] >= 12) & (features['hour'] <= 17)).astype(int)
        features['is_evening'] = ((features['hour'] >= 18) & (features['hour'] <= 21)).astype(int)
        
        # Add seasonal features
        features['sin_hour'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['cos_hour'] = np.cos(2 * np.pi * features['hour'] / 24)
        features['sin_month'] = np.sin(2 * np.pi * features['month'] / 12)
        features['cos_month'] = np.cos(2 * np.pi * features['month'] / 12)
        
        # Add rolling statistics for each parameter
        rolling_features = {}
        for param in self.feature_columns:
            for window in [3, 6, 12, 24, 48]:  # Added 48-hour window
                rolling_features[f'{param}_rolling_mean_{window}h'] = data[param].rolling(window=window, min_periods=1).mean()
                rolling_features[f'{param}_rolling_std_{window}h'] = data[param].rolling(window=window, min_periods=1).std()
                rolling_features[f'{param}_rolling_min_{window}h'] = data[param].rolling(window=window, min_periods=1).min()
                rolling_features[f'{param}_rolling_max_{window}h'] = data[param].rolling(window=window, min_periods=1).max()
                rolling_features[f'{param}_rolling_range_{window}h'] = (
                    rolling_features[f'{param}_rolling_max_{window}h'] - 
                    rolling_features[f'{param}_rolling_min_{window}h']
                )
        
        # Add lag features
        for param in self.feature_columns:
            for lag in [1, 3, 6, 12, 24, 48]:  # Added 48-hour lag
                rolling_features[f'{param}_lag_{lag}h'] = data[param].shift(lag)
        
        # Add interaction features between pollutants
        for i, param1 in enumerate(self.feature_columns):
            for param2 in self.feature_columns[i+1:]:
                rolling_features[f'{param1}_{param2}_ratio'] = data[param1] / (data[param2] + 1e-6)
                rolling_features[f'{param1}_{param2}_diff'] = data[param1] - data[param2]
        
        # Combine all features
        features = pd.concat([features, pd.DataFrame(rolling_features, index=data.index)], axis=1)
        
        # Add original parameters
        for param in self.feature_columns:
            features[param] = data[param]
        
        # Drop rows with NaN values
        features = features.dropna()
        
        # Scale the features
        feature_cols = self.feature_columns + [col for col in features.columns if col not in ['timestamp'] + self.feature_columns]
        scaled_data = self.scaler.fit_transform(features[feature_cols])
        
        X, y = [], {}
        for param in self.feature_columns:
            y[param] = []
            
        for i in range(len(scaled_data) - self.sequence_length - self.prediction_horizon + 1):
            # Create input sequence
            sequence = scaled_data[i:(i + self.sequence_length)]
            X.append(sequence)
            
            # Create target sequences for each parameter
            for j, param in enumerate(self.feature_columns):
                y[param].append(scaled_data[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon, j])
        
        return np.array(X), {param: np.array(y[param]) for param in self.feature_columns}
    
    def train(self, X: np.ndarray, y: Dict[str, np.ndarray], params: Dict = None) -> None:
        """
        Train the LightGBM models for each parameter.
        
        Args:
            X (np.ndarray): Training features
            y (Dict[str, np.ndarray]): Training targets for each parameter
            params (Dict): LightGBM parameters
        """
        # Default parameters if none provided
        if params is None:
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 63,  # Increased for better feature interaction
                'learning_rate': 0.005,  # Reduced for more stable training
                'feature_fraction': 0.9,  # Increased to use more features
                'bagging_fraction': 0.9,  # Increased to reduce variance
                'bagging_freq': 3,  # More frequent bagging
                'min_child_samples': 10,  # Reduced to capture more patterns
                'min_child_weight': 0.0001,  # Reduced to allow more splits
                'reg_alpha': 0.01,  # Reduced L1 regularization
                'reg_lambda': 0.01,  # Reduced L2 regularization
                'verbose': -1
            }
        
        for param in self.feature_columns:
            logger.info(f"Training LightGBM model for {param}")
            
            # Reshape input data for LightGBM
            X_reshaped = X.reshape(X.shape[0], -1)
            
            # Train model for each prediction horizon
            self.models[param] = []
            for h in range(self.prediction_horizon):
                # Create train and validation sets
                train_size = int(len(X_reshaped) * 0.8)
                X_train = X_reshaped[:train_size]
                X_val = X_reshaped[train_size:]
                y_train = y[param][:train_size, h]
                y_val = y[param][train_size:, h]
                
                # Create LightGBM datasets
                train_data = lgb.Dataset(X_train, label=y_train)
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                
                # Train model with increased rounds and patience
                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=2000,  # Increased number of rounds
                    valid_sets=[train_data, val_data],
                    callbacks=[
                        lgb.early_stopping(100),  # Increased patience
                        lgb.log_evaluation(period=100)
                    ]
                )
                self.models[param].append(model)
                
            logger.info(f"Completed training for {param}")
    
    def predict(self, data: pd.DataFrame, hours_ahead: int = 24) -> pd.DataFrame:
        """
        Make predictions for future time steps.
        
        Args:
            data (pd.DataFrame): Input data for prediction
            hours_ahead (int): Number of hours to predict ahead
            
        Returns:
            pd.DataFrame: Predictions for each parameter
        """
        # Create features DataFrame
        features = pd.DataFrame(index=data.index)
        
        # Add time-based features
        features['hour'] = data['timestamp'].dt.hour
        features['day_of_week'] = data['timestamp'].dt.dayofweek
        features['month'] = data['timestamp'].dt.month
        features['is_weekend'] = features['day_of_week'].isin([5, 6]).astype(int)
        features['is_night'] = ((features['hour'] >= 22) | (features['hour'] <= 5)).astype(int)
        
        # Add seasonal features
        features['sin_hour'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['cos_hour'] = np.cos(2 * np.pi * features['hour'] / 24)
        features['sin_month'] = np.sin(2 * np.pi * features['month'] / 12)
        features['cos_month'] = np.cos(2 * np.pi * features['month'] / 12)
        
        # Add rolling statistics
        rolling_features = {}
        for param in self.feature_columns:
            for window in [3, 6, 12, 24]:
                rolling_features[f'{param}_rolling_mean_{window}h'] = data[param].rolling(window=window, min_periods=1).mean()
                rolling_features[f'{param}_rolling_std_{window}h'] = data[param].rolling(window=window, min_periods=1).std()
                rolling_features[f'{param}_rolling_min_{window}h'] = data[param].rolling(window=window, min_periods=1).min()
                rolling_features[f'{param}_rolling_max_{window}h'] = data[param].rolling(window=window, min_periods=1).max()
        
        # Add lag features
        for param in self.feature_columns:
            for lag in [1, 3, 6, 12, 24]:
                rolling_features[f'{param}_lag_{lag}h'] = data[param].shift(lag)
        
        # Combine all features
        features = pd.concat([features, pd.DataFrame(rolling_features, index=data.index)], axis=1)
        
        # Add original parameters
        for param in self.feature_columns:
            features[param] = data[param]
        
        # Prepare input sequence
        input_data = features[self.feature_columns].values[-self.sequence_length:]
        time_features = features[['hour', 'day_of_week', 'month', 'is_weekend', 'is_night', 
                                'sin_hour', 'cos_hour', 'sin_month', 'cos_month']].values[-self.sequence_length:]
        input_data = self.scaler.transform(input_data)
        input_sequence = np.column_stack([input_data, time_features])
        input_sequence = input_sequence.reshape(1, -1)
        
        predictions = {param: [] for param in self.feature_columns}
        
        # Make predictions for each parameter
        for param in self.feature_columns:
            param_predictions = []
            current_sequence = input_sequence.copy()
            
            for h in range(hours_ahead):
                # Get prediction for current horizon
                pred = self.models[param][h % self.prediction_horizon].predict(current_sequence)[0]
                param_predictions.append(pred)
                
                # Update sequence for next prediction
                current_sequence = np.roll(current_sequence, -len(self.feature_columns))
                current_sequence[0, -len(self.feature_columns):] = pred
            
            predictions[param] = param_predictions
        
        # Convert predictions to DataFrame
        pred_df = pd.DataFrame(predictions)
        
        # Inverse transform predictions
        pred_df = pd.DataFrame(
            self.scaler.inverse_transform(pred_df),
            columns=self.feature_columns
        )
        
        # Add timestamps
        last_timestamp = data['timestamp'].iloc[-1]
        pred_df['timestamp'] = [last_timestamp + pd.Timedelta(hours=i+1) for i in range(len(pred_df))]
        
        return pred_df
    
    def evaluate(self, X_test: np.ndarray, y_test: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model performance.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (Dict[str, np.ndarray]): Test targets
            
        Returns:
            Dict[str, Dict[str, float]]: Evaluation metrics for each parameter
        """
        X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
        metrics = {}
        
        for param in self.feature_columns:
            param_metrics = {
                'rmse': [],
                'mae': [],
                'r2': []
            }
            
            for h in range(self.prediction_horizon):
                y_pred = self.models[param][h].predict(X_test_reshaped)
                y_true = y_test[param][:, h]
                
                param_metrics['rmse'].append(np.sqrt(mean_squared_error(y_true, y_pred)))
                param_metrics['mae'].append(mean_absolute_error(y_true, y_pred))
                param_metrics['r2'].append(r2_score(y_true, y_pred))
            
            # Average metrics across prediction horizons
            metrics[param] = {
                'rmse': np.mean(param_metrics['rmse']),
                'mae': np.mean(param_metrics['mae']),
                'r2': np.mean(param_metrics['r2'])
            }
            
            logger.info(f"Metrics for {param}:")
            logger.info(f"  RMSE: {metrics[param]['rmse']:.4f}")
            logger.info(f"  MAE: {metrics[param]['mae']:.4f}")
            logger.info(f"  R2: {metrics[param]['r2']:.4f}")
        
        return metrics
    
    def save_model(self, path: str) -> None:
        """
        Save the trained models and scaler.
        
        Args:
            path (str): Path to save the models
        """
        if not self.models:
            raise ValueError("No models to save. Train the models first.")
            
        os.makedirs(path, exist_ok=True)
        
        # Save models
        for param in self.feature_columns:
            for h, model in enumerate(self.models[param]):
                model.save_model(f"{path}/{param}_horizon_{h}_model.txt")
        
        # Save scaler
        joblib.dump(self.scaler, f"{path}/scaler.joblib")
        logger.info(f"Models saved to {path}")
    
    @classmethod
    def load_model(cls, path: str) -> 'LightGBMPredictor':
        """
        Load trained models and scaler.
        
        Args:
            path (str): Path to the saved models
            
        Returns:
            LightGBMPredictor: Loaded model instance
        """
        instance = cls()
        
        # Load models
        instance.models = {}
        for param in instance.feature_columns:
            instance.models[param] = []
            for h in range(instance.prediction_horizon):
                model = lgb.Booster(model_file=f"{path}/{param}_horizon_{h}_model.txt")
                instance.models[param].append(model)
        
        # Load scaler
        instance.scaler = joblib.load(f"{path}/scaler.joblib")
        logger.info(f"Models loaded from {path}")
        
        return instance

def train_lightgbm_model(city: str) -> LightGBMPredictor:
    """
    Train LightGBM model for a specific city.
    
    Args:
        city (str): City name
        
    Returns:
        LightGBMPredictor: Trained model
    """
    logger.info(f"Training LightGBM model for {city}")
    
    # Get data from database
    session = get_db_session()
    try:
        query = session.query(
            AirQualityMeasurement.timestamp,
            AirQualityMeasurement.co,
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
        
        # Initialize and train model
        model = LightGBMPredictor()
        X, y = model.prepare_data(df)
        
        # Split into train and test sets
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train = {param: y[param][:train_size] for param in model.feature_columns}
        y_test = {param: y[param][train_size:] for param in model.feature_columns}
        
        # Train model
        model.train(X_train, y_train)
        
        # Evaluate model
        metrics = model.evaluate(X_test, y_test)
        
        # Save model
        model.save_model(f"models/saved/lightgbm_{city}")
        
        return model
        
    finally:
        session.close() 