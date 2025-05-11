import logging
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from database.db_manager import NotificationSetting, NotificationHistory, get_db_session

logger = logging.getLogger(__name__)

# Створюємо blueprint для сповіщень
notifications_api = Blueprint('notifications_api', __name__)

@notifications_api.route('/notifications', methods=['GET'])
@jwt_required()
def get_notifications():
    """Get user notification settings"""
    session = get_db_session()
    try:
        user_id = get_jwt_identity()
        
        # Get all notification settings for user
        notifications = session.query(NotificationSetting).filter_by(user_id=user_id).all()
        
        result = []
        for notification in notifications:
            result.append({
                "id": notification.id,
                "city_name": notification.city_name,
                "is_enabled": notification.is_enabled,
                "notification_frequency": notification.notification_frequency,
                "email_notifications": notification.email_notifications,
                "push_notifications": notification.push_notifications,
                "pm25_threshold": notification.pm25_threshold,
                "pm10_threshold": notification.pm10_threshold,
                "o3_threshold": notification.o3_threshold,
                "no2_threshold": notification.no2_threshold,
                "so2_threshold": notification.so2_threshold,
                "co_threshold": notification.co_threshold,
                "created_at": notification.created_at.isoformat(),
                "updated_at": notification.updated_at.isoformat(),
            })
            
        return jsonify({
            "notification_settings": result
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting notifications: {str(e)}")
        return jsonify({"error": "Failed to get notifications"}), 500
    finally:
        session.close()

@notifications_api.route('/notifications', methods=['POST'])
@jwt_required()
def create_notification():
    """Create new notification setting"""
    session = get_db_session()
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        
        # Validate required fields
        if 'city_name' not in data:
            return jsonify({"error": "City name is required"}), 400
            
        # Check if notification for this city already exists
        existing = session.query(NotificationSetting).filter_by(
            user_id=user_id,
            city_name=data['city_name']
        ).first()
        
        if existing:
            return jsonify({"error": "Notification setting for this city already exists"}), 409
            
        # Create new notification setting
        new_notification = NotificationSetting(
            user_id=user_id,
            city_name=data['city_name'],
            is_enabled=data.get('is_enabled', True),
            notification_frequency=data.get('notification_frequency', 'daily'),
            email_notifications=data.get('email_notifications', True),
            push_notifications=data.get('push_notifications', True),
            pm25_threshold=data.get('pm25_threshold', 25.0),
            pm10_threshold=data.get('pm10_threshold', 50.0),
            o3_threshold=data.get('o3_threshold', 100.0),
            no2_threshold=data.get('no2_threshold', 40.0),
            so2_threshold=data.get('so2_threshold', 20.0),
            co_threshold=data.get('co_threshold', 7000.0)
        )
        
        session.add(new_notification)
        session.commit()
        
        return jsonify({
            "message": "Notification setting created successfully",
            "notification": {
                "id": new_notification.id,
                "city_name": new_notification.city_name,
                "is_enabled": new_notification.is_enabled,
                "notification_frequency": new_notification.notification_frequency,
                "email_notifications": new_notification.email_notifications,
                "push_notifications": new_notification.push_notifications
            }
        }), 201
        
    except Exception as e:
        session.rollback()
        logger.error(f"Error creating notification: {str(e)}")
        return jsonify({"error": "Failed to create notification setting"}), 500
    finally:
        session.close()

@notifications_api.route('/notifications/<int:notification_id>', methods=['GET'])
@jwt_required()
def get_notification(notification_id):
    """Get single notification setting"""
    session = get_db_session()
    try:
        user_id = get_jwt_identity()
        
        # Get notification setting
        notification = session.query(NotificationSetting).filter_by(
            id=notification_id,
            user_id=user_id
        ).first()
        
        if not notification:
            return jsonify({"error": "Notification setting not found"}), 404
            
        return jsonify({
            "notification": {
                "id": notification.id,
                "city_name": notification.city_name,
                "is_enabled": notification.is_enabled,
                "notification_frequency": notification.notification_frequency,
                "email_notifications": notification.email_notifications,
                "push_notifications": notification.push_notifications,
                "pm25_threshold": notification.pm25_threshold,
                "pm10_threshold": notification.pm10_threshold,
                "o3_threshold": notification.o3_threshold,
                "no2_threshold": notification.no2_threshold,
                "so2_threshold": notification.so2_threshold,
                "co_threshold": notification.co_threshold,
                "created_at": notification.created_at.isoformat(),
                "updated_at": notification.updated_at.isoformat(),
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting notification: {str(e)}")
        return jsonify({"error": "Failed to get notification setting"}), 500
    finally:
        session.close()

@notifications_api.route('/notifications/<int:notification_id>', methods=['PUT'])
@jwt_required()
def update_notification(notification_id):
    """Update notification setting"""
    session = get_db_session()
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        
        # Get notification setting
        notification = session.query(NotificationSetting).filter_by(
            id=notification_id,
            user_id=user_id
        ).first()
        
        if not notification:
            return jsonify({"error": "Notification setting not found"}), 404
            
        # Update fields if provided
        if 'is_enabled' in data:
            notification.is_enabled = data['is_enabled']
            
        if 'notification_frequency' in data:
            notification.notification_frequency = data['notification_frequency']
            
        if 'email_notifications' in data:
            notification.email_notifications = data['email_notifications']
            
        if 'push_notifications' in data:
            notification.push_notifications = data['push_notifications']
            
        # Update thresholds
        for field in [
            'pm25_threshold', 'pm10_threshold', 'o3_threshold',
            'no2_threshold', 'so2_threshold', 'co_threshold'
        ]:
            if field in data:
                setattr(notification, field, data[field])
        
        session.commit()
        
        return jsonify({
            "message": "Notification setting updated successfully",
            "notification": {
                "id": notification.id,
                "city_name": notification.city_name,
                "is_enabled": notification.is_enabled,
                "notification_frequency": notification.notification_frequency,
                "email_notifications": notification.email_notifications,
                "push_notifications": notification.push_notifications,
                "pm25_threshold": notification.pm25_threshold,
                "pm10_threshold": notification.pm10_threshold,
                "o3_threshold": notification.o3_threshold,
                "no2_threshold": notification.no2_threshold,
                "so2_threshold": notification.so2_threshold,
                "co_threshold": notification.co_threshold
            }
        }), 200
        
    except Exception as e:
        session.rollback()
        logger.error(f"Error updating notification: {str(e)}")
        return jsonify({"error": "Failed to update notification setting"}), 500
    finally:
        session.close()

@notifications_api.route('/notifications/<int:notification_id>', methods=['DELETE'])
@jwt_required()
def delete_notification(notification_id):
    """Delete notification setting"""
    session = get_db_session()
    try:
        user_id = get_jwt_identity()
        
        # Get notification setting
        notification = session.query(NotificationSetting).filter_by(
            id=notification_id,
            user_id=user_id
        ).first()
        
        if not notification:
            return jsonify({"error": "Notification setting not found"}), 404
            
        session.delete(notification)
        session.commit()
        
        return jsonify({
            "message": "Notification setting deleted successfully"
        }), 200
        
    except Exception as e:
        session.rollback()
        logger.error(f"Error deleting notification: {str(e)}")
        return jsonify({"error": "Failed to delete notification setting"}), 500
    finally:
        session.close()

@notifications_api.route('/notifications/history', methods=['GET'])
@jwt_required()
def get_notification_history():
    """Get user notification history"""
    session = get_db_session()
    try:
        user_id = get_jwt_identity()
        
        # Get notification history with pagination
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        
        # Limit per_page to 50 to prevent abuse
        per_page = min(per_page, 50)
        
        history = session.query(NotificationHistory)\
            .filter_by(user_id=user_id)\
            .order_by(NotificationHistory.created_at.desc())\
            .limit(per_page)\
            .offset((page - 1) * per_page)\
            .all()
            
        # Count total items for pagination
        total_count = session.query(NotificationHistory)\
            .filter_by(user_id=user_id)\
            .count()
            
        # Format response
        result = []
        for item in history:
            result.append({
                "id": item.id,
                "city_name": item.city_name,
                "notification_type": item.notification_type,
                "message": item.message,
                "is_read": item.is_read,
                "created_at": item.created_at.isoformat()
            })
            
        return jsonify({
            "history": result,
            "pagination": {
                "total_items": total_count,
                "page": page,
                "per_page": per_page,
                "total_pages": (total_count + per_page - 1) // per_page
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting notification history: {str(e)}")
        return jsonify({"error": "Failed to get notification history"}), 500
    finally:
        session.close()

@notifications_api.route('/notifications/history/<int:history_id>/read', methods=['PUT'])
@jwt_required()
def mark_notification_read(history_id):
    """Mark notification as read"""
    session = get_db_session()
    try:
        user_id = get_jwt_identity()
        
        # Get notification history item
        history_item = session.query(NotificationHistory).filter_by(
            id=history_id,
            user_id=user_id
        ).first()
        
        if not history_item:
            return jsonify({"error": "Notification history item not found"}), 404
            
        history_item.is_read = True
        session.commit()
        
        return jsonify({
            "message": "Notification marked as read"
        }), 200
        
    except Exception as e:
        session.rollback()
        logger.error(f"Error marking notification as read: {str(e)}")
        return jsonify({"error": "Failed to mark notification as read"}), 500
    finally:
        session.close()