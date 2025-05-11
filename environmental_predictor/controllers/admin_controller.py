from flask import Blueprint, render_template, request, redirect, url_for, flash
from database.db_manager import get_db_session
from database.models import ModelConfiguration, AirQualityMeasurement
from services.model_evaluation import get_model_comparison
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

@admin_bp.route('/model_management')
def model_management():
    """Show model management page"""
    session = get_db_session()
    try:
        config = session.query(ModelConfiguration).first()
        active_model = config.active_model if config else 'lstm'
        
        # Get test data for comparison
        test_data = get_test_data()
        
        # Get model comparisons for each city
        cities = ['Kyiv', 'Lviv', 'Kharkiv', 'Odesa', 'Dnipro']
        comparisons = {}
        
        for city in cities:
            comparisons[city] = get_model_comparison(city, test_data)
        
        return render_template('admin/model_management.html', 
                            active_model=active_model,
                            comparisons=comparisons)
    
    finally:
        session.close()

@admin_bp.route('/model_management', methods=['POST'])
def update_model_config():
    """Update active model configuration"""
    session = get_db_session()
    try:
        active_model = request.form.get('active_model')
        
        if not active_model:
            flash('Please select a model', 'error')
            return redirect(url_for('admin.model_management'))
        
        config = session.query(ModelConfiguration).first()
        if not config:
            config = ModelConfiguration(active_model=active_model)
            session.add(config)
        else:
            config.active_model = active_model
        
        session.commit()
        flash('Model configuration updated successfully', 'success')
        return redirect(url_for('admin.model_management'))
    
    except Exception as e:
        session.rollback()
        logger.error(f"Error updating model configuration: {str(e)}")
        flash('Error updating model configuration', 'error')
        return redirect(url_for('admin.model_management'))
    
    finally:
        session.close()

@admin_bp.route('/reset_model_config', methods=['POST'])
def reset_model_config():
    """Reset model configuration to default (LSTM)"""
    session = get_db_session()
    try:
        config = session.query(ModelConfiguration).first()
        if config:
            config.active_model = 'lstm'
            session.commit()
            flash('Model configuration reset to LSTM', 'success')
        return redirect(url_for('admin.model_management'))
    
    except Exception as e:
        session.rollback()
        logger.error(f"Error resetting model configuration: {str(e)}")
        flash('Error resetting model configuration', 'error')
        return redirect(url_for('admin.model_management'))
    
    finally:
        session.close()

def get_test_data():
    """Get test data for model evaluation"""
    # Get last 72 hours of data for each city
    session = get_db_session()
    try:
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=72)
        
        query = session.query(
            AirQualityMeasurement.timestamp,
            AirQualityMeasurement.city,
            AirQualityMeasurement.aqi,
            AirQualityMeasurement.pm25,
            AirQualityMeasurement.pm10,
            AirQualityMeasurement.o3,
            AirQualityMeasurement.no2,
            AirQualityMeasurement.so2,
            AirQualityMeasurement.co,
            AirQualityMeasurement.temperature,
            AirQualityMeasurement.humidity,
            AirQualityMeasurement.wind_speed,
            AirQualityMeasurement.wind_direction
        ).filter(
            AirQualityMeasurement.timestamp.between(start_time, end_time)
        )
        
        df = pd.read_sql(query.statement, session.bind)
        return df
    
    finally:
        session.close() 