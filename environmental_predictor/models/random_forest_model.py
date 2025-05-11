import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import joblib
import os
from typing import Tuple, Dict, List
from database.db_manager import get_db_session, AirQualityMeasurement
import multiprocessing

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get number of CPU cores for parallel processing
N_JOBS = max(1, multiprocessing.cpu_count() - 1)

class RandomForestPredictor:
    def __init__(self, sequence_length: int = 24, prediction_horizon: int = 24):
        """
        Initialize the Random Forest predictor.
        
        Args:
            sequence_length (int): Number of past hours to use for prediction
            prediction_horizon (int): Number of hours to predict into the future
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.scaler = MinMaxScaler()
        self.models = {}  # Dictionary to store models for each parameter
        self.feature_columns = ['co', 'o3', 'so2', 'pm25', 'pm10']  # Removed no2
        
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
        
        # Scale the features
        scaled_data = self.scaler.fit_transform(data[self.feature_columns])
        
        X, y = [], {}
        for param in self.feature_columns:
            y[param] = []
            
        for i in range(len(scaled_data) - self.sequence_length - self.prediction_horizon + 1):
            # Create input sequence
            X.append(scaled_data[i:(i + self.sequence_length)])
            
            # Create target sequences for each parameter
            for j, param in enumerate(self.feature_columns):
                y[param].append(scaled_data[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon, j])
        
        return np.array(X), {param: np.array(y[param]) for param in self.feature_columns}
    
    def train(self, X: np.ndarray, y: Dict[str, np.ndarray], n_estimators: int = 50) -> None:
        """
        Train the Random Forest models for each parameter.
        
        Args:
            X (np.ndarray): Training features
            y (Dict[str, np.ndarray]): Training targets for each parameter
            n_estimators (int): Number of trees in the forest
        """
        for param in self.feature_columns:
            logger.info(f"Training Random Forest model for {param}")
            
            # Reshape input data for Random Forest
            X_reshaped = X.reshape(X.shape[0], -1)
            
            # Train model for each prediction horizon
            self.models[param] = []
            for h in range(self.prediction_horizon):
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=10,  # Limit tree depth
                    min_samples_split=5,  # Increase minimum samples for split
                    min_samples_leaf=3,  # Increase minimum samples per leaf
                    max_features='sqrt',  # Use sqrt of features for each split
                    n_jobs=N_JOBS,  # Use parallel processing
                    random_state=42
                )
                model.fit(X_reshaped, y[param][:, h])
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
        # Prepare input sequence
        input_data = data[self.feature_columns].values[-self.sequence_length:]
        input_data = self.scaler.transform(input_data)
        input_sequence = input_data.reshape(1, -1)
        
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
                joblib.dump(model, f"{path}/{param}_horizon_{h}_model.joblib")
        
        # Save scaler
        joblib.dump(self.scaler, f"{path}/scaler.joblib")
        logger.info(f"Models saved to {path}")
    
    @classmethod
    def load_model(cls, path: str) -> 'RandomForestPredictor':
        """
        Load trained models and scaler.
        
        Args:
            path (str): Path to the saved models
            
        Returns:
            RandomForestPredictor: Loaded model instance
        """
        instance = cls()
        
        # Load models
        instance.models = {}
        for param in instance.feature_columns:
            instance.models[param] = []
            for h in range(instance.prediction_horizon):
                model = joblib.load(f"{path}/{param}_horizon_{h}_model.joblib")
                instance.models[param].append(model)
        
        # Load scaler
        instance.scaler = joblib.load(f"{path}/scaler.joblib")
        logger.info(f"Models loaded from {path}")
        
        return instance

def train_random_forest_model(city: str) -> RandomForestPredictor:
    """
    Train Random Forest model for a specific city.
    
    Args:
        city (str): City name
        
    Returns:
        RandomForestPredictor: Trained model
    """
    logger.info(f"Training Random Forest model for {city}")
    
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
        model = RandomForestPredictor()
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
        model.save_model(f"models/saved/random_forest_{city}")
        
        return model
        
    finally:
        session.close() 