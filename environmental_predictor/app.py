from flask import Flask
import logging
from flask_jwt_extended import JWTManager
from datetime import timedelta
from api.model_api import model_api
from api.auth_api import auth_api
from api.notifications_api import notifications_api
from api.notification_scheduler import start_scheduler, stop_scheduler
import os
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager, UserMixin
from config import Config
from database import init_db
from controllers.admin_controller import admin_bp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class User(UserMixin):
    def __init__(self, id):
        self.id = id

def create_app():
    """Create and configure Flask application"""
    app = Flask(__name__)
    
    # Load configuration
    app.config.from_object(Config)
    
    # Configure CORS to allow all origins
    CORS(app, resources={r"/*": {"origins": "*"}})
    
    # Initialize extensions
    db = SQLAlchemy(app)
    migrate = Migrate(app, db)
    login_manager = LoginManager(app)
    login_manager.login_view = 'auth.login'
    
    @login_manager.user_loader
    def load_user(user_id):
        return User(user_id)
    
    # Configure JWT
    jwt = JWTManager(app)
    
    # Register blueprints
    app.register_blueprint(model_api, url_prefix='/api')
    app.register_blueprint(auth_api, url_prefix='/api/auth')
    app.register_blueprint(notifications_api, url_prefix='/api')
    app.register_blueprint(admin_bp)
    
    # Initialize database
    with app.app_context():
        db.create_all()
    
    # Start notification scheduler
    start_scheduler()
    logger.info("Notification scheduler started")
    
    # Register shutdown handler
    @app.teardown_appcontext
    def shutdown_scheduler(exception=None):
        stop_scheduler()
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)