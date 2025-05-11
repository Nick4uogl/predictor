from flask import Blueprint, render_template, request, redirect, url_for, flash
from database.db_manager import get_db_session
from database.models import ModelConfiguration

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

@admin_bp.route('/models', methods=['GET', 'POST'])
def model_management():
    session = get_db_session()
    
    if request.method == 'POST':
        active_model = request.form.get('active_model')
        
        # Validate input
        if active_model not in ['lstm', 'xgboost', 'prophet']:
            flash('Invalid model selected', 'error')
            return redirect(url_for('admin.model_management'))
        
        # Update or create configuration
        config = session.query(ModelConfiguration).first()
        if config:
            config.active_model = active_model
        else:
            config = ModelConfiguration(active_model=active_model)
            session.add(config)
        
        session.commit()
        flash(f'Model configuration updated to {active_model}', 'success')
        return redirect(url_for('admin.model_management'))
    
    # Get current configuration
    config = session.query(ModelConfiguration).first()
    active_model = config.active_model if config else 'lstm'
    
    return render_template('admin/model_management.html',
                         active_model=active_model)

@admin_bp.route('/models/reset', methods=['POST'])
def reset_model_config():
    session = get_db_session()
    config = session.query(ModelConfiguration).first()
    
    if config:
        session.delete(config)
        session.commit()
        flash('Model configuration reset to default (LSTM)', 'success')
    else:
        flash('No configuration to reset', 'info')
    
    return redirect(url_for('admin.model_management')) 