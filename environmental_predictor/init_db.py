import logging
from database.db_manager import init_db, Base
from sqlalchemy import inspect

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_database_schema():
    """Check if the database schema matches our models"""
    engine = init_db()
    inspector = inspect(engine)
    
    # Get all table names
    table_names = inspector.get_table_names()
    logger.info(f"Existing tables: {table_names}")
    
    # Check if our tables exist
    required_tables = [
        'air_quality_measurements',
        'air_quality_forecasts',
        'users',
        'notification_settings',
        'notification_history',
        'notifications',
        'notification_logs',
        'model_evaluations'
    ]
    
    missing_tables = [table for table in required_tables if table not in table_names]
    if missing_tables:
        logger.warning(f"Missing tables: {missing_tables}")
        return False
    
    # Check columns in air_quality_measurements table
    columns = inspector.get_columns('air_quality_measurements')
    column_names = [col['name'] for col in columns]
    logger.info(f"Columns in air_quality_measurements: {column_names}")
    
    required_columns = [
        'id', 'city', 'timestamp', 'co', 'no2', 'o3', 'so2', 'pm25', 'pm10'
    ]
    
    missing_columns = [col for col in required_columns if col not in column_names]
    if missing_columns:
        logger.warning(f"Missing columns in air_quality_measurements: {missing_columns}")
        return False
    
    return True

def initialize_database():
    """Initialize the database with our schema"""
    engine = init_db()
    
    # Drop all existing tables
    Base.metadata.drop_all(engine)
    logger.info("Dropped all existing tables")
    
    # Create all tables
    Base.metadata.create_all(engine)
    logger.info("Created all tables")
    
    # Verify schema
    if check_database_schema():
        logger.info("Database schema verified successfully")
    else:
        logger.error("Database schema verification failed")

if __name__ == "__main__":
    logger.info("Starting database initialization...")
    initialize_database()
    logger.info("Database initialization complete") 