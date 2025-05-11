from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from .models import ModelConfiguration, ModelEvaluation
import os

Base = declarative_base()

def get_db_session():
    """Create and return a database session"""
    engine = create_engine(os.getenv('DATABASE_URL'))
    Session = sessionmaker(bind=engine)
    return Session()

def init_db():
    """Initialize the database"""
    engine = create_engine(os.getenv('DATABASE_URL'))
    Base.metadata.create_all(engine) 