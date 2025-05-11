from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class AirQualityMeasurement(Base):
    __tablename__ = 'air_quality_measurements'
    
    id = Column(Integer, primary_key=True)
    city = Column(String(100), nullable=False)
    timestamp = Column(DateTime, nullable=False)
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

class AirQualityForecast(Base):
    __tablename__ = 'air_quality_forecasts'
    
    id = Column(Integer, primary_key=True)
    city = Column(String(100), nullable=False)
    forecast_timestamp = Column(DateTime, nullable=False)
    created_timestamp = Column(DateTime)
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

class ModelConfiguration(Base):
    __tablename__ = 'model_configurations'
    
    id = Column(Integer, primary_key=True)
    active_model = Column(String(50), default='lstm')
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<ModelConfiguration(active_model='{self.active_model}')>"

class ModelEvaluation(Base):
    __tablename__ = 'model_evaluations'
    
    id = Column(Integer, primary_key=True)
    city = Column(String(50))
    model_type = Column(String(50))
    rmse = Column(Float)
    mae = Column(Float)
    r2 = Column(Float)
    evaluation_date = Column(DateTime, default=datetime.utcnow)
    metrics = Column(JSON)  # Store additional metrics if needed
    
    def __repr__(self):
        return f"<ModelEvaluation(city='{self.city}', model_type='{self.model_type}', rmse={self.rmse})>" 