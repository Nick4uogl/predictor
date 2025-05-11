import logging
from preprocessing.data_loader import OpenWeatherDataLoader
from database.db_manager import DatabaseManager
from config import AIR_QUALITY_PARAMETERS

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Initialize data loader and database manager
    data_loader = OpenWeatherDataLoader()
    db_manager = DatabaseManager()
    
    try:
        # Set parameters
        country_code = 'UA'
        days_back = 30
        
        # Get available locations
        locations = data_loader.get_available_locations(country_code)
        logger.info(f"Found {len(locations)} locations in {country_code}")
        
        # Fetch data for each location
        for location in locations:
            try:
                city_name = location['name']
                lat = location['coord']['lat']
                lon = location['coord']['lon']
                
                logger.info(f"Fetching data for {city_name}")
                
                # Fetch measurements
                measurements = data_loader.fetch_measurements(
                    lat=lat,
                    lon=lon,
                    parameters=AIR_QUALITY_PARAMETERS,
                    days_back=days_back
                )
                
                if measurements is not None and not measurements.empty:
                    # Save to database
                    db_manager.save_measurements(city_name, measurements)
                else:
                    logger.warning(f"No measurements found for {city_name}")
                    
            except Exception as e:
                logger.error(f"Error processing location {city_name}: {str(e)}")
                continue
                
    finally:
        # Close database connection
        db_manager.close()

if __name__ == "__main__":
    main() 