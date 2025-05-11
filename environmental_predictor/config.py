import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://postgres:root@localhost:5432/air_quality')

# API configuration
API_KEY = os.getenv('API_KEY', '8e1458dc219c3d90a9fa267017943918')

# Model configuration
SEQUENCE_LENGTH = 72  # Number of hours to use for prediction (3 days)
PREDICTION_HORIZON = 72  # Number of hours to predict ahead (3 days)

# List of cities to monitor
CITIES = [
    'Kyiv',
    'Lviv',
    'Kharkiv',
    'Odesa',
    'Dnipro'
]

# Email configuration for notifications
EMAIL_CONFIG = {
    'SMTP_SERVER': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
    'SMTP_PORT': int(os.getenv('SMTP_PORT', 587)),
    'EMAIL_USERNAME': os.getenv('EMAIL_USERNAME', 'your_email@gmail.com'),
    'EMAIL_PASSWORD': os.getenv('EMAIL_PASSWORD', 'your_app_password'),
    'FROM_EMAIL': os.getenv('FROM_EMAIL', 'your_email@gmail.com')
}

# Email configuration variables for direct import
EMAIL_HOST = EMAIL_CONFIG['SMTP_SERVER']
EMAIL_PORT = EMAIL_CONFIG['SMTP_PORT']
EMAIL_USER = EMAIL_CONFIG['EMAIL_USERNAME']
EMAIL_PASSWORD = EMAIL_CONFIG['EMAIL_PASSWORD']
EMAIL_FROM = EMAIL_CONFIG['FROM_EMAIL']

# Scheduler configuration
SCHEDULER_CONFIG = {
    'CHECK_INTERVAL': 60,  # Check every 60 minutes
    'DAILY_SUMMARY_TIME': '07:00'  # Send daily summary at 7:00 AM
}

class Config:
    # Database configuration
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'postgresql://postgres:root@localhost:5432/air_quality')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # JWT configuration
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'your-secret-key-for-development')
    JWT_ACCESS_TOKEN_EXPIRES = 86400  # 24 hours
    JWT_REFRESH_TOKEN_EXPIRES = 2592000  # 30 days
    
    # Email configuration
    EMAIL_HOST = EMAIL_HOST
    EMAIL_PORT = EMAIL_PORT
    EMAIL_USER = EMAIL_USER
    EMAIL_PASSWORD = EMAIL_PASSWORD
    EMAIL_FROM = EMAIL_FROM
    
    # Model configuration
    MODEL_SAVE_PATH = os.getenv('MODEL_SAVE_PATH', 'models')
    ACTIVE_MODEL = os.getenv('ACTIVE_MODEL', 'lstm')
    
    # Logging configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    # List of cities to monitor
    CITIES = [
        'Kyiv',
        'Lviv',
        'Kharkiv',
        'Odesa',
        'Dnipro'
    ]
    
    # Scheduler configuration
    SCHEDULER_CONFIG = {
        'CHECK_INTERVAL': 60,  # Check every 60 minutes
        'DAILY_SUMMARY_TIME': '07:00'  # Send daily summary at 7:00 AM
    }