import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.db_manager import AirQualityMeasurement, get_db_session
from config import DATABASE_URL
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_database():
    """Check the contents of the database"""
    session = get_db_session()
    
    try:
        # Get total number of records
        total_records = session.query(AirQualityMeasurement).count()
        logger.info(f"Total number of records in database: {total_records}")
        
        if total_records == 0:
            logger.warning("No records found in the database")
            return
        
        # Get records per city
        city_counts = session.query(
            AirQualityMeasurement.city,
            func.count(AirQualityMeasurement.id)
        ).group_by(AirQualityMeasurement.city).all()
        
        logger.info("Records per city:")
        for city, count in city_counts:
            logger.info(f"  {city}: {count} records")
        
        # Get date range of data
        first_record = session.query(AirQualityMeasurement).order_by(AirQualityMeasurement.timestamp).first()
        last_record = session.query(AirQualityMeasurement).order_by(AirQualityMeasurement.timestamp.desc()).first()
        
        if first_record and last_record:
            logger.info(f"Data date range: from {first_record.timestamp} to {last_record.timestamp}")
        
        # Get sample of data
        sample = session.query(AirQualityMeasurement).limit(5).all()
        logger.info("\nSample of data:")
        for record in sample:
            logger.info(f"City: {record.city}, Timestamp: {record.timestamp}")
            logger.info(f"  PM2.5: {record.pm25}, PM10: {record.pm10}")
            logger.info(f"  CO: {record.co}, NO2: {record.no2}")
            logger.info(f"  O3: {record.o3}, SO2: {record.so2}")
            logger.info("---")
        
    finally:
        session.close()

if __name__ == "__main__":
    check_database() 