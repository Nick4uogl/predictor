from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from database.db_manager import User, get_db_session
from flask_jwt_extended import create_access_token, create_refresh_token, jwt_required, get_jwt_identity
from email_validator import validate_email, EmailNotValidError
import logging

# Create blueprint
auth_api = Blueprint('auth', __name__)
logger = logging.getLogger(__name__)

@auth_api.route('/register', methods=['POST'])
def register():
    """Register a new user"""
    session = None
    try:
        # Get data from request
        data = request.get_json()
        name = data.get('name', '').strip()
        email = data.get('email', '').strip()
        password = data.get('password', '')
        
        # Validate data
        if not email or not password:
            return jsonify({"error": "Email та пароль обов'язкові"}), 400
            
        if len(password) < 8:
            return jsonify({"error": "Пароль повинен містити не менше 8 символів"}), 400
            
        # Validate email
        try:
            valid = validate_email(email)
            email = valid.email
        except EmailNotValidError as e:
            return jsonify({"error": f"Некоректна електронна адреса: {str(e)}"}), 400
            
        # Check if user exists
        session = get_db_session()
        existing_user = session.query(User).filter(User.email == email).first()
        
        if existing_user:
            return jsonify({"error": "Користувач з такою електронною адресою вже існує"}), 409
                
        # Create new user
        password_hash = generate_password_hash(password)
        new_user = User(username=email, email=email, password_hash=password_hash, name=name)
        
        session.add(new_user)
        session.commit()
        
        # Generate tokens
        access_token = create_access_token(identity=new_user.id)
        refresh_token = create_refresh_token(identity=new_user.id)
        
        return jsonify({
            "message": "Реєстрація успішна",
            "user": {
                "id": new_user.id,
                "email": new_user.email,
                "name": new_user.name
            },
            "access_token": access_token,
            "refresh_token": refresh_token
        }), 201
        
    except Exception as e:
        logger.error(f"Error during registration: {str(e)}", exc_info=True)
        if session:
            session.rollback()
        return jsonify({
            "error": "Помилка під час реєстрації. Спробуйте пізніше.",
            "details": str(e)
        }), 500
    finally:
        if session:
            session.close()
        
@auth_api.route('/login', methods=['POST'])
def login():
    """Login user and return JWT tokens"""
    try:
        data = request.get_json()
        email = data.get('email', '')
        password = data.get('password', '')
        
        if not email or not password:
            return jsonify({"error": "Введіть email та пароль"}), 400
            
        session = get_db_session()
        
        # Check if user exists
        user = session.query(User).filter(User.email == email).first()
        
        if not user or not check_password_hash(user.password_hash, password):
            return jsonify({"error": "Невірний email або пароль"}), 401
            
        if not user.is_active:
            return jsonify({"error": "Цей акаунт деактивовано. Зверніться до адміністратора."}), 403
            
        # Generate tokens with string user ID
        access_token = create_access_token(identity=str(user.id))
        refresh_token = create_refresh_token(identity=str(user.id))
        
        return jsonify({
            "message": "Вхід успішний",
            "user": {
                "id": user.id,
                "username": user.username,
                "email": user.email
            },
            "access_token": access_token,
            "refresh_token": refresh_token
        }), 200
        
    except Exception as e:
        logger.error(f"Error during login: {str(e)}")
        return jsonify({"error": "Помилка під час входу. Спробуйте пізніше."}), 500
        
@auth_api.route('/refresh', methods=['POST'])
@jwt_required(refresh=True)
def refresh():
    """Refresh access token"""
    try:
        current_user_id = get_jwt_identity()
        access_token = create_access_token(identity=str(current_user_id))
        
        return jsonify({
            "access_token": access_token
        }), 200
        
    except Exception as e:
        logger.error(f"Error refreshing token: {str(e)}")
        return jsonify({"error": "Помилка оновлення токена. Спробуйте увійти знову."}), 500
        
@auth_api.route('/me', methods=['GET'])
@jwt_required()
def get_profile():
    """Get current user profile"""
    try:
        current_user_id = get_jwt_identity()
        
        session = get_db_session()
        user = session.query(User).filter_by(id=int(current_user_id)).first()
        
        if not user:
            return jsonify({"error": "Користувача не знайдено"}), 404
            
        return jsonify({
            "user": {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "name": user.name,
                "created_at": user.created_at.isoformat() if user.created_at else None
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting profile: {str(e)}")
        return jsonify({"error": "Помилка отримання профілю. Спробуйте пізніше."}), 500
        
@auth_api.route('/me', methods=['PUT'])
@jwt_required()
def update_profile():
    """Update user profile"""
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()
        
        session = get_db_session()
        user = session.query(User).filter_by(id=current_user_id).first()
        
        if not user:
            return jsonify({"error": "Користувача не знайдено"}), 404
            
        # Update fields
        if 'username' in data and data['username'].strip():
            new_username = data['username'].strip()
            # Check if username is taken
            existing = session.query(User).filter_by(username=new_username).first()
            if existing and existing.id != current_user_id:
                return jsonify({"error": "Це ім'я користувача вже зайняте"}), 409
                
            user.username = new_username
            
        if 'email' in data and data['email'].strip():
            new_email = data['email'].strip()
            
            # Validate email
            try:
                valid = validate_email(new_email)
                new_email = valid.email
            except EmailNotValidError as e:
                return jsonify({"error": f"Некоректна електронна адреса: {str(e)}"}), 400
                
            # Check if email is taken
            existing = session.query(User).filter_by(email=new_email).first()
            if existing and existing.id != current_user_id:
                return jsonify({"error": "Ця електронна адреса вже зареєстрована"}), 409
                
            user.email = new_email
            
        if 'password' in data and data['password']:
            new_password = data['password']
            
            if len(new_password) < 8:
                return jsonify({"error": "Пароль повинен містити не менше 8 символів"}), 400
                
            user.password_hash = generate_password_hash(new_password)
            
        session.commit()
        
        return jsonify({
            "message": "Профіль оновлено успішно",
            "user": {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "created_at": user.created_at.isoformat()
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error updating profile: {str(e)}")
        session.rollback()
        return jsonify({"error": "Помилка оновлення профілю. Спробуйте пізніше."}), 500