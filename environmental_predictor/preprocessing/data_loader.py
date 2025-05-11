import os
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from config import OPENWEATHER_API_BASE_URL, OPENWEATHER_API_KEY, AIR_QUALITY_PARAMETERS

logger = logging.getLogger(__name__)

class OpenWeatherDataLoader:
    def __init__(self):
        self.base_url = OPENWEATHER_API_BASE_URL
        self.api_key = OPENWEATHER_API_KEY
        logger.info(f"Initializing OpenWeatherDataLoader with base URL: {self.base_url}")

    def get_available_locations(self, country_code: str) -> List[Dict]:
        """
        Get available measurement locations for a specific country.
        For OpenWeather, we'll use a predefined list of major Ukrainian cities.
        
        Args:
            country_code (str): Two-letter country code (e.g., 'UA')
            
        Returns:
            List[Dict]: List of location dictionaries with coordinates
        """
        # Predefined list of major Ukrainian cities with their coordinates
        ua_cities = [
            {"name": "Київ", "coord": {"lat": 50.4501, "lon": 30.5234}},
            {"name": "Харків", "coord": {"lat": 49.9935, "lon": 36.2304}},
            {"name": "Одеса", "coord": {"lat": 46.4825, "lon": 30.7233}},
            {"name": "Дніпро", "coord": {"lat": 48.4647, "lon": 35.0462}},
            {"name": "Львів", "coord": {"lat": 49.8397, "lon": 24.0297}},
            {"name": "Запоріжжя", "coord": {"lat": 47.8388, "lon": 35.1396}},
            {"name": "Вінниця", "coord": {"lat": 49.2331, "lon": 28.4682}},
            {"name": "Полтава", "coord": {"lat": 49.5883, "lon": 34.5514}},
            {"name": "Чернівці", "coord": {"lat": 48.2921, "lon": 25.9358}},
            {"name": "Тернопіль", "coord": {"lat": 49.5535, "lon": 25.5948}}
        ]
        
        if country_code.upper() == 'UA':
            logger.info(f"Returning {len(ua_cities)} predefined Ukrainian cities")
            return ua_cities
        else:
            logger.warning(f"Country code {country_code} not supported. Only UA is supported.")
            return []

    def fetch_measurements(
        self,
        lat: float,
        lon: float,
        parameters: List[str],
        days_back: int = 30
    ) -> Optional[pd.DataFrame]:
        """
        Fetch measurements for a specific location.
        
        Args:
            lat (float): Latitude of the location
            lon (float): Longitude of the location
            parameters (List[str]): List of parameters to fetch
            days_back (int): Number of days of historical data to fetch
            
        Returns:
            Optional[pd.DataFrame]: DataFrame containing measurements or None if error
        """
        endpoint = f"{self.base_url}/air_pollution/history"
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        params = {
            'lat': lat,
            'lon': lon,
            'start': int(start_date.timestamp()),
            'end': int(end_date.timestamp()),
            'appid': self.api_key
        }
        
        logger.info(f"Fetching measurements from {endpoint} with params: {params}")
        
        try:
            response = requests.get(endpoint, params=params)
            logger.info(f"API Response Status: {response.status_code}")
            
            if response.status_code != 200:
                logger.error(f"API Error: {response.text}")
                return None
                
            response.raise_for_status()
            data = response.json()
            
            if not data.get('list'):
                logger.warning(f"No measurements found for location {lat}, {lon}")
                return None
                
            # Convert to DataFrame
            measurements = []
            for entry in data['list']:
                measurement = {
                    'datetime': datetime.fromtimestamp(entry['dt']),
                    'lat': lat,
                    'lon': lon
                }
                # Add air quality components
                for component, value in entry['components'].items():
                    measurement[component] = value
                measurements.append(measurement)
            
            df = pd.DataFrame(measurements)
            logger.info(f"Successfully created DataFrame with {len(df)} rows")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching measurements: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response content: {e.response.text}")
            return None 