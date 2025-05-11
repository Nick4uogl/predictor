import logging
import os
import pandas as pd
import numpy as np
from database.db_manager import get_db_session, AirQualityMeasurement
from models.random_forest_model import RandomForestPredictor, train_random_forest_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data_from_db(city: str = None):
    """Load air quality data from the database"""
    session = get_db_session()
    
    try:
        # Query measurements
        query = session.query(
            AirQualityMeasurement.timestamp,
            AirQualityMeasurement.city,
            AirQualityMeasurement.co,
            AirQualityMeasurement.o3,
            AirQualityMeasurement.so2,
            AirQualityMeasurement.pm25,
            AirQualityMeasurement.pm10
        )
        
        if city:
            query = query.filter(AirQualityMeasurement.city == city)
            
        query = query.order_by(AirQualityMeasurement.timestamp)
        
        # Convert to DataFrame
        df = pd.read_sql(query.statement, session.bind)
        
        # Log data statistics
        logger.info(f"Total number of records loaded: {len(df)}")
        if not city:
            city_counts = df['city'].value_counts()
            logger.info("Number of records per city:")
            for city, count in city_counts.items():
                logger.info(f"  {city}: {count} records")
        
        return df
        
    finally:
        session.close()

def plot_feature_importance(model: RandomForestPredictor, city: str):
    """Plot feature importance for each parameter"""
    plt.figure(figsize=(15, 10))
    
    for i, param in enumerate(model.feature_columns):
        plt.subplot(2, 3, i+1)
        
        # Get feature importance from the first horizon model
        importance = model.models[param][0].feature_importances_
        
        # Create feature names based on sequence length
        feature_names = []
        for t in range(model.sequence_length):
            for f in model.feature_columns:
                feature_names.append(f"{f}_t-{model.sequence_length-t}")
        
        # Sort features by importance
        indices = np.argsort(importance)[::-1]
        
        # Plot
        plt.bar(range(len(importance)), importance[indices])
        plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=90)
        plt.title(f"Feature Importance for {param.upper()}")
        plt.tight_layout()
    
    # Save plot
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/feature_importance_{city}.png')
    plt.close()

def plot_predictions_vs_actual(model: RandomForestPredictor, X_test: np.ndarray, y_test: dict, city: str):
    """Plot predictions vs actual values for each parameter"""
    plt.figure(figsize=(15, 10))
    
    X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
    
    for i, param in enumerate(model.feature_columns):
        plt.subplot(2, 3, i+1)
        
        # Get predictions for first horizon
        y_pred = model.models[param][0].predict(X_test_reshaped)
        y_true = y_test[param][:, 0]
        
        # Plot
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f"{param.upper()} - Predictions vs Actual")
    
    plt.tight_layout()
    plt.savefig(f'plots/predictions_vs_actual_{city}.png')
    plt.close()

def main():
    """Main function to train Random Forest models"""
    # Create necessary directories
    os.makedirs('models/saved', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Set city to Kyiv
    city = "Kyiv"
    logger.info(f"\nProcessing city: {city}")
    
    try:
        # Train model
        model = train_random_forest_model(city)
        
        # Load test data for visualization
        test_data = load_data_from_db(city)
        X, y = model.prepare_data(test_data)
        
        # Create visualizations
        plot_feature_importance(model, city)
        plot_predictions_vs_actual(model, X, y, city)
        
        logger.info(f"Random Forest model for {city} trained and saved successfully")
        logger.info(f"Visualizations saved in 'plots' directory")
        
    except Exception as e:
        logger.error(f"Error processing city {city}: {str(e)}")
        raise

if __name__ == "__main__":
    main() 