import logging
from sqlalchemy import func
from database.db_manager import get_db_session, AirQualityMeasurement
from datetime import datetime

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
        
        logger.info("\nRecords per city:")
        for city, count in city_counts:
            logger.info(f"  {city}: {count} records")
        
        # Get date range of data
        first_record = session.query(AirQualityMeasurement).order_by(AirQualityMeasurement.timestamp).first()
        last_record = session.query(AirQualityMeasurement).order_by(AirQualityMeasurement.timestamp.desc()).first()
        
        if first_record and last_record:
            logger.info(f"\nData date range: from {first_record.timestamp} to {last_record.timestamp}")
        
        # Get sample of data
        sample = session.query(AirQualityMeasurement).limit(5).all()
        logger.info("\nSample of data (first 5 records):")
        for record in sample:
            logger.info(f"\nCity: {record.city}, Timestamp: {record.timestamp}")
            logger.info(f"  PM2.5: {record.pm25}, PM10: {record.pm10}")
            logger.info(f"  CO: {record.co}, NO2: {record.no2}")
            logger.info(f"  O3: {record.o3}, SO2: {record.so2}")
        
        # Check for null values
        logger.info("\nChecking for null values:")
        for column in ['pm25', 'pm10', 'co', 'no2', 'o3', 'so2']:
            null_count = session.query(AirQualityMeasurement).filter(
                getattr(AirQualityMeasurement, column) == None
            ).count()
            logger.info(f"  {column}: {null_count} null values")
        
    finally:
        session.close()

def check_city_data(city_name):
    """Check the contents of the database for a specific city"""
    session = get_db_session()
    
    try:
        # Get total number of records for the city
        total_records = session.query(AirQualityMeasurement).filter_by(city=city_name).count()
        logger.info(f"Total number of records for {city_name}: {total_records}")
        
        if total_records == 0:
            logger.warning(f"No records found for {city_name}")
            return
        
        # Get date range of data
        first_record = session.query(AirQualityMeasurement)\
            .filter_by(city=city_name)\
            .order_by(AirQualityMeasurement.timestamp).first()
            
        last_record = session.query(AirQualityMeasurement)\
            .filter_by(city=city_name)\
            .order_by(AirQualityMeasurement.timestamp.desc()).first()
        
        if first_record and last_record:
            logger.info(f"Data date range for {city_name}: from {first_record.timestamp} to {last_record.timestamp}")
        
        # Get sample of most recent data
        recent_records = session.query(AirQualityMeasurement)\
            .filter_by(city=city_name)\
            .order_by(AirQualityMeasurement.timestamp.desc())\
            .limit(5)\
            .all()
            
        logger.info(f"\nMost recent records for {city_name}:")
        for record in recent_records:
            logger.info(f"\nTimestamp: {record.timestamp}")
            logger.info(f"  PM2.5: {record.pm25}, PM10: {record.pm10}")
            logger.info(f"  CO: {record.co}, NO2: {record.no2}")
            logger.info(f"  O3: {record.o3}, SO2: {record.so2}")
        
    finally:
        session.close()

def check_kyiv_data():
    """Check the contents of the database for Kyiv specifically"""
    session = get_db_session()
    
    try:
        # Get total number of records for Kyiv
        total_records = session.query(AirQualityMeasurement).filter_by(city="Kyiv").count()
        logger.info(f"Total number of records for Kyiv: {total_records}")
        
        if total_records == 0:
            logger.warning("No records found for Kyiv")
            return
        
        # Get date range of data
        first_record = session.query(AirQualityMeasurement)\
            .filter_by(city="Kyiv")\
            .order_by(AirQualityMeasurement.timestamp).first()
            
        last_record = session.query(AirQualityMeasurement)\
            .filter_by(city="Kyiv")\
            .order_by(AirQualityMeasurement.timestamp.desc()).first()
        
        if first_record and last_record:
            logger.info(f"Data date range for Kyiv: from {first_record.timestamp} to {last_record.timestamp}")
            
            # Calculate how old the last record is
            time_diff = datetime.now() - last_record.timestamp
            logger.info(f"Last record is {time_diff.total_seconds() / 3600:.2f} hours old")
        
        # Get sample of most recent data
        recent_records = session.query(AirQualityMeasurement)\
            .filter_by(city="Kyiv")\
            .order_by(AirQualityMeasurement.timestamp.desc())\
            .limit(5)\
            .all()
            
        logger.info(f"\nMost recent records for Kyiv:")
        for record in recent_records:
            logger.info(f"\nTimestamp: {record.timestamp}")
            logger.info(f"  PM2.5: {record.pm25}, PM10: {record.pm10}")
            logger.info(f"  CO: {record.co}, NO2: {record.no2}")
            logger.info(f"  O3: {record.o3}, SO2: {record.so2}")
        
    finally:
        session.close()

if __name__ == "__main__":
    check_kyiv_data() 