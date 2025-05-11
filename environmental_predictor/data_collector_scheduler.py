import logging
import time
import schedule
from threading import Thread
from datetime import datetime, timedelta
from data_collector import get_air_pollution_data, get_city_coordinates
from database.db_manager import get_db_session, AirQualityMeasurement
from config import CITIES, API_KEY

logger = logging.getLogger(__name__)

class DataCollectorScheduler:
    def __init__(self):
        """Initialize data collector scheduler"""
        self.scheduler_thread = None
        self.running = False
        
    def start(self):
        """Start data collector scheduler in background thread"""
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            logger.warning("Data collector scheduler is already running")
            return False
            
        self.running = True
        self.scheduler_thread = Thread(target=self._run_scheduler)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        logger.info("Data collector scheduler started")
        return True
        
    def stop(self):
        """Stop data collector scheduler"""
        if not self.scheduler_thread or not self.scheduler_thread.is_alive():
            logger.warning("Data collector scheduler is not running")
            return False
            
        self.running = False
        self.scheduler_thread.join(timeout=5)
        logger.info("Data collector scheduler stopped")
        return True
        
    def _run_scheduler(self):
        """Run the scheduler loop"""
        # Schedule data collection every hour
        schedule.every().hour.do(self._collect_recent_data)
        
        # Run initial collection
        self._collect_recent_data()
        
        logger.info("Scheduler started with hourly data collection")
        
        while self.running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute for pending jobs
            
    def _collect_recent_data(self):
        """Collect recent air quality data for all cities"""
        session = get_db_session()
        
        try:
            # Get data for the last 2 hours to ensure we don't miss any updates
            end_date = datetime.now()
            start_date = end_date - timedelta(hours=2)
            
            logger.info(f"Collecting data from {start_date} to {end_date}")
            
            for city in CITIES:
                try:
                    lat, lon = get_city_coordinates(city)
                    if lat is None or lon is None:
                        logger.error(f"Could not get coordinates for {city}")
                        continue
                    
                    # Get air pollution data
                    pollution_data = get_air_pollution_data(lat, lon, start_date, end_date)
                    if not pollution_data or 'list' not in pollution_data:
                        logger.error(f"No pollution data for {city} in period {start_date} to {end_date}")
                        continue
                    
                    # Process each data point
                    for item in pollution_data['list']:
                        dt = datetime.fromtimestamp(item['dt'])
                        
                        # Check if we already have this data point
                        existing = session.query(AirQualityMeasurement)\
                            .filter_by(city=city, timestamp=dt)\
                            .first()
                            
                        if existing:
                            continue
                            
                        # Create new measurement record
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
                    
                    # Commit after each city
                    session.commit()
                    logger.info(f"Processed data for {city} from {start_date} to {end_date}")
                    
                except Exception as e:
                    logger.error(f"Error processing city {city}: {str(e)}")
                    session.rollback()
                    continue
                    
        finally:
            session.close()

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Start the scheduler
    scheduler = DataCollectorScheduler()
    scheduler.start()
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        scheduler.stop() 