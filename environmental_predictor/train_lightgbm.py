import os
import logging
import pandas as pd
from models.lightgbm_model import train_lightgbm_model
from database.db_manager import get_db_session, AirQualityMeasurement
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data_from_db(city: str = None) -> pd.DataFrame:
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
        
        return pd.read_sql(query.statement, session.bind)
    finally:
        session.close()

def plot_feature_importance(model, city: str) -> None:
    """
    Plot feature importance for the LightGBM model.
    
    Args:
        model: Trained LightGBM model
        city (str): City name
    """
    plt.figure(figsize=(12, 6))
    
    # Get feature importance for each parameter
    for param in model.feature_columns:
        importance = model.models[param][0].feature_importance()
        feature_names = [f"{f}_{i}" for f in ['value', 'hour', 'day', 'month'] 
                        for i in range(model.sequence_length)]
        
        # Plot feature importance
        plt.subplot(2, 3, model.feature_columns.index(param) + 1)
        sns.barplot(x=importance, y=feature_names)
        plt.title(f'Feature Importance - {param.upper()}')
        plt.xlabel('Importance')
        plt.ylabel('Features')
    
    plt.tight_layout()
    plt.savefig(f'plots/lightgbm_{city}_feature_importance.png')
    plt.close()

def plot_predictions_vs_actual(model, X_test, y_test, city: str) -> None:
    """
    Plot predictions vs actual values for each parameter.
    
    Args:
        model: Trained LightGBM model
        X_test: Test features
        y_test: Test targets
        city (str): City name
    """
    X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
    
    plt.figure(figsize=(15, 10))
    
    for i, param in enumerate(model.feature_columns):
        plt.subplot(2, 3, i + 1)
        
        # Get predictions for first horizon
        y_pred = model.models[param][0].predict(X_test_reshaped)
        y_true = y_test[param][:, 0]
        
        # Plot predictions vs actual
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        plt.title(f'{param.upper()} - Predictions vs Actual')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
    
    plt.tight_layout()
    plt.savefig(f'plots/lightgbm_{city}_predictions_vs_actual.png')
    plt.close()

def main():
    """Main function to train LightGBM models"""
    # Create necessary directories
    os.makedirs('models/saved', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Set city to Kyiv
    city = 'Kyiv'
    logger.info(f"\nProcessing city: {city}")
    
    try:
        # Train model
        model = train_lightgbm_model(city)
        
        # Load test data for visualization
        test_data = load_data_from_db(city)
        X, y = model.prepare_data(test_data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create visualizations
        plot_feature_importance(model, city)
        plot_predictions_vs_actual(model, X_test, y_test, city)
        
        logger.info(f"LightGBM model for {city} trained and saved successfully")
        logger.info(f"Visualizations saved in 'plots' directory")
        
    except Exception as e:
        logger.error(f"Error processing city {city}: {str(e)}")

if __name__ == "__main__":
    main() 