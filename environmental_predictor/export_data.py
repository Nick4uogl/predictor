import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection parameters
DATABASE_URL = "postgresql://postgres:root@localhost:5432/air_quality"

def export_city_data(city):
    """Export air quality data for a specific city to CSV"""
    try:
        # Create database engine
        engine = create_engine(DATABASE_URL)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Query data for the city
        query = f"""
            SELECT timestamp, co, no2, o3, so2, pm25, pm10
            FROM air_quality_measurements
            WHERE city = '{city}'
            ORDER BY timestamp
        """
        
        # Read data into DataFrame
        df = pd.read_sql(query, session.bind)
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Save to CSV
        csv_path = f'data/{city.lower()}_air_quality.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"Data for {city} exported to {csv_path}")
        
        # Print data summary
        logger.info(f"Number of records: {len(df)}")
        logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        logger.info(f"Columns: {', '.join(df.columns)}")
        
    except Exception as e:
        logger.error(f"Error exporting data for {city}: {str(e)}")
    finally:
        session.close()

def main():
    # List of cities to export
    cities = ['Kyiv', 'Lviv', 'Kharkiv', 'Odesa', 'Dnipro']
    
    for city in cities:
        logger.info(f"Exporting data for {city}")
        export_city_data(city)

if __name__ == "__main__":
    main() 