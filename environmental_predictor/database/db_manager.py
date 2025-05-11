import logging
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime
import os
from config import DATABASE_URL

# Налаштування логування
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ініціалізація SQLAlchemy
Base = declarative_base()

class AirQualityMeasurement(Base):
    __tablename__ = 'air_quality_measurements'
    
    id = Column(Integer, primary_key=True)
    city = Column(String(100), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    co = Column(Float)
    no2 = Column(Float)
    o3 = Column(Float)
    so2 = Column(Float)
    pm25 = Column(Float)
    pm10 = Column(Float)
    
    def __repr__(self):
        return f"AirQualityMeasurement(id={self.id}, city={self.city}, timestamp={self.timestamp})"


class AirQualityForecast(Base):
    __tablename__ = 'air_quality_forecasts'
    
    id = Column(Integer, primary_key=True)
    city = Column(String(100), nullable=False)
    forecast_timestamp = Column(DateTime, nullable=False)
    created_timestamp = Column(DateTime, default=datetime.utcnow)
    aqi = Column(Float)
    pm25 = Column(Float)
    pm10 = Column(Float)
    o3 = Column(Float)
    no2 = Column(Float)
    so2 = Column(Float)
    co = Column(Float)
    temperature = Column(Float)
    humidity = Column(Float)
    wind_speed = Column(Float)
    wind_direction = Column(String(50))
    
    def __repr__(self):
        return f"AirQualityForecast(id={self.id}, city={self.city}, forecast_timestamp={self.forecast_timestamp})"


# Нові класи для системи авторизації та сповіщень

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(256), nullable=False)
    name = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Зв'язки з іншими таблицями
    notifications = relationship("NotificationSetting", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"User(id={self.id}, username={self.username}, email={self.email})"


class NotificationSetting(Base):
    __tablename__ = 'notification_settings'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    city_name = Column(String(100), nullable=False)
    
    # Пороги для сповіщень (концентрації забруднювачів)
    pm25_threshold = Column(Float, default=25.0)  # мкг/м³
    pm10_threshold = Column(Float, default=50.0)  # мкг/м³
    o3_threshold = Column(Float, default=100.0)   # мкг/м³
    no2_threshold = Column(Float, default=40.0)   # мкг/м³
    so2_threshold = Column(Float, default=20.0)   # мкг/м³
    co_threshold = Column(Float, default=7000.0)  # мкг/м³
    
    # Налаштування
    is_enabled = Column(Boolean, default=True)
    notification_frequency = Column(String(20), default='daily')  # daily, realtime, etc.
    email_notifications = Column(Boolean, default=True)
    push_notifications = Column(Boolean, default=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Зв'язки з іншими таблицями
    user = relationship("User", back_populates="notifications")
    
    def __repr__(self):
        return f"NotificationSetting(id={self.id}, user_id={self.user_id}, city={self.city_name})"


class NotificationHistory(Base):
    __tablename__ = 'notification_history'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    city_name = Column(String(100), nullable=False)
    notification_type = Column(String(50), nullable=False)  # email, push
    message = Column(Text, nullable=False)
    is_read = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"NotificationHistory(id={self.id}, user_id={self.user_id}, city={self.city_name})"

# Додаємо класи, які потрібні для model_api.py
class Notification(Base):
    __tablename__ = 'notifications'
    
    id = Column(Integer, primary_key=True)
    email = Column(String(100), nullable=False)
    city_name = Column(String(100), nullable=False)
    
    # Threshold values
    threshold_pm25 = Column(Float, default=25.0)
    threshold_pm10 = Column(Float, default=50.0)
    threshold_o3 = Column(Float, default=100.0)
    threshold_no2 = Column(Float, default=40.0)
    threshold_so2 = Column(Float, default=20.0)
    threshold_co = Column(Float, default=7000.0)
    threshold_aqi = Column(Float, default=100.0)
    
    # Settings
    daily_summary = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"Notification(id={self.id}, email={self.email}, city={self.city_name})"

class NotificationLog(Base):
    __tablename__ = 'notification_logs'
    
    id = Column(Integer, primary_key=True)
    notification_id = Column(Integer, ForeignKey('notifications.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    message = Column(Text, nullable=False)
    sent_successfully = Column(Boolean, default=True)
    
    def __repr__(self):
        return f"NotificationLog(id={self.id}, notification_id={self.notification_id})"

class ModelEvaluation(Base):
    __tablename__ = 'model_evaluations'
    
    id = Column(Integer, primary_key=True)
    city = Column(String(100), nullable=False)
    model_type = Column(String(50), nullable=False)
    rmse = Column(Float)
    mae = Column(Float)
    r2 = Column(Float)
    evaluation_date = Column(DateTime, default=datetime.utcnow)
    metrics = Column(Text)  # JSON string with detailed metrics
    
    def __repr__(self):
        return f"ModelEvaluation(id={self.id}, city={self.city}, model_type={self.model_type})"

# Функція для ініціалізації БД
_engine = None

def init_db():
    global _engine
    if _engine is None:
        _engine = create_engine(DATABASE_URL)
        Base.metadata.create_all(_engine)
        logger.info("Database initialized successfully")
    return _engine

# Створення сесії для роботи з БД
def get_db_session():
    engine = init_db()
    Session = sessionmaker(bind=engine)
    return Session()

# Alias для get_db_session для сумісності з model_api.py
def get_session(engine=None):
    if engine is None:
        return get_db_session()
    else:
        Session = sessionmaker(bind=engine)
        return Session()