import requests
import pandas as pd
from datetime import datetime, timedelta
import time
from database.db_manager import get_db_session, AirQualityMeasurement
from config import API_KEY, CITIES
import logging

# Налаштування логування
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_air_pollution_data(lat, lon, start, end):
    """Get historical air pollution data from OpenWeather API"""
    url = "http://api.openweathermap.org/data/2.5/air_pollution/history"
    params = {
        'lat': lat,
        'lon': lon,
        'start': int(start.timestamp()),
        'end': int(end.timestamp()),
        'appid': API_KEY
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching data: {e}")
        return None

def get_weather_data(lat, lon, dt):
    """Get weather data for a specific timestamp"""
    url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'lat': lat,
        'lon': lon,
        'dt': int(dt.timestamp()),
        'appid': API_KEY,
        'units': 'metric'
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching weather data: {e}")
        return None

def get_city_coordinates(city):
    """Get coordinates for a city"""
    url = "http://api.openweathermap.org/geo/1.0/direct"
    params = {
        'q': city,
        'limit': 1,
        'appid': API_KEY
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if data:
            return data[0]['lat'], data[0]['lon']
        return None, None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error getting coordinates: {e}")
        return None, None

def collect_historical_data():
    """Collect historical air pollution data for all cities"""
    session = get_db_session()
    
    # Define time periods (OpenWeather API has a limit of 30 days per request)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # One year ago
    
    # Split into 30-day periods
    current_date = start_date
    while current_date < end_date:
        period_end = min(current_date + timedelta(days=30), end_date)
        
        logger.info(f"Processing period from {current_date} to {period_end}")
        
        for city in CITIES:
            lat, lon = get_city_coordinates(city)
            if lat is None or lon is None:
                logger.error(f"Could not get coordinates for {city}")
                continue
            
            # Get air pollution data
            pollution_data = get_air_pollution_data(lat, lon, current_date, period_end)
            if not pollution_data or 'list' not in pollution_data:
                logger.error(f"No pollution data for {city} in period {current_date} to {period_end}")
                continue
            
            # Logging: кількість записів, перший і останній таймстемп, приклад запису
            data_list = pollution_data['list']
            logger.info(f"{city}: отримано {len(data_list)} записів за період {current_date} - {period_end}")
            if data_list:
                first_dt = datetime.fromtimestamp(data_list[0]['dt'])
                last_dt = datetime.fromtimestamp(data_list[-1]['dt'])
                logger.info(f"{city}: перший запис: {first_dt}, останній запис: {last_dt}")
                logger.info(f"{city}: приклад запису: {data_list[0]}")

            # Process each data point
            for item in data_list:
                logger.debug(f"{city}: запис: {item}")
                dt = datetime.fromtimestamp(item['dt'])
                
                # Create measurement record with only the fields that exist in the model
                measurement = AirQualityMeasurement(
                    city=city,
                    timestamp=dt,
                    pm25=item['components']['pm2_5'],
                    pm10=item['components']['pm10'],
                    o3=item['components']['o3'],
                    no2=item['components']['no2'],
                    so2=item['components']['so2'],
                    co=item['components']['co']
                )
                
                session.add(measurement)
            
            # Commit after each city to avoid losing too much data if something fails
            session.commit()
            logger.info(f"Processed data for {city} from {current_date} to {period_end}")
            
            # Sleep to avoid hitting API rate limits
            time.sleep(1)  # Reduced from 2 seconds to 1 second
        
        current_date = period_end
        # Sleep between periods
        time.sleep(2)  # Reduced from 5 seconds to 2 seconds
    
    session.close()

def clear_existing_data():
    """Clear existing air quality measurements"""
    session = get_db_session()
    try:
        session.query(AirQualityMeasurement).delete()
        session.commit()
        logger.info("Cleared existing air quality measurements")
    except Exception as e:
        session.rollback()
        logger.error(f"Error clearing existing data: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    clear_existing_data()  # Clear existing data before collecting new data
    collect_historical_data() 