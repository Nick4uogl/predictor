import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, List, Optional
from pathlib import Path

from environmental_predictor.config import (
    AIR_QUALITY_PARAMETERS,
    MISSING_DATA_THRESHOLD,
    PROCESSED_DATA_DIR
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AirQualityDataCleaner:
    def __init__(self):
        self.parameters = AIR_QUALITY_PARAMETERS
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the air quality data.
        
        Args:
            df: Raw DataFrame from OpenAQ
            
        Returns:
            Cleaned and processed DataFrame
        """
        if df.empty:
            logger.warning("Empty DataFrame provided")
            return df
            
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date.utc'])
        
        # Sort by date
        df = df.sort_values('date')
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Remove outliers
        df = self._remove_outliers(df)
        
        # Resample to hourly data
        df = self._resample_to_hourly(df)
        
        # Add time-based features
        df = self._add_time_features(df)
        
        # Save processed data
        self._save_processed_data(df)
        
        return df
        
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        # Calculate percentage of missing values for each parameter
        missing_stats = df[self.parameters].isnull().mean()
        
        # Log missing value statistics
        for param, missing_rate in missing_stats.items():
            logger.info(f"Missing rate for {param}: {missing_rate:.2%}")
            
        # Remove parameters with too many missing values
        valid_params = missing_stats[missing_stats <= MISSING_DATA_THRESHOLD].index
        df = df[['date'] + list(valid_params)]
        
        # Forward fill missing values (within 6 hours)
        df = df.set_index('date')
        df = df.fillna(method='ffill', limit=6)
        
        # Backward fill remaining missing values (within 6 hours)
        df = df.fillna(method='bfill', limit=6)
        
        # Remove rows with remaining missing values
        df = df.dropna()
        
        return df.reset_index()
        
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using IQR method."""
        for param in self.parameters:
            if param in df.columns:
                Q1 = df[param].quantile(0.25)
                Q3 = df[param].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Replace outliers with bounds
                df[param] = df[param].clip(lower_bound, upper_bound)
                
        return df
        
    def _resample_to_hourly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample data to hourly frequency."""
        df = df.set_index('date')
        df = df.resample('H').mean()
        return df.reset_index()
        
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features to the dataset."""
        df['hour'] = df['date'].dt.hour
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_year'] = df['date'].dt.dayofyear
        df['month'] = df['date'].dt.month
        
        # Add cyclical features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
        
        return df
        
    def _save_processed_data(self, df: pd.DataFrame) -> None:
        """Save processed data to CSV file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"processed_air_quality_{timestamp}.csv"
        filepath = PROCESSED_DATA_DIR / filename
        
        df.to_csv(filepath, index=False)
        logger.info(f"Saved processed data to {filepath}") 