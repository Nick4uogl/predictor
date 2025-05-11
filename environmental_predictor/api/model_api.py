from flask import request, jsonify, Blueprint
from datetime import datetime, timedelta
from database.db_manager import AirQualityMeasurement, get_session, init_db, Notification, NotificationLog
import pandas as pd
from models.air_quality_model import AirQualityPredictor
from models.train_lstm_single_improved import ImprovedSingleParameterLSTM
import logging
import os
from sqlalchemy import desc, func
from config import CITIES
from flask_cors import CORS
import joblib
import torch
import numpy as np

# Set up logger
logger = logging.getLogger(__name__)

# Initialize API blueprint
model_api = Blueprint('model_api', __name__)

# Initialize CORS
CORS(model_api, resources={
    r"/*": {
        "origins": ["http://localhost:5173", "http://127.0.0.1:5173"],  # Vite default ports
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Initialize database
engine = init_db()

# ...existing code...

@model_api.route('/notifications', methods=['POST'])
def create_notification():
    """
    Create a new notification subscription
    ---
    parameters:
      - name: notification
        in: body
        required: true
        schema:
          type: object
          properties:
            email:
              type: string
              description: Email address to send notifications to
            city_name:
              type: string
              description: City to monitor
            threshold_pm25:
              type: number
              description: Threshold for PM2.5 (optional)
            threshold_pm10:
              type: number
              description: Threshold for PM10 (optional)
            threshold_o3:
              type: number
              description: Threshold for O3 (optional)
            threshold_no2:
              type: number
              description: Threshold for NO2 (optional)
            threshold_so2:
              type: number
              description: Threshold for SO2 (optional)
            threshold_co:
              type: number
              description: Threshold for CO (optional)
            threshold_aqi:
              type: number
              description: Threshold for AQI (optional)
            daily_summary:
              type: boolean
              description: Whether to send daily summaries
    responses:
      201:
        description: Notification created successfully
      400:
        description: Invalid request parameters
      500:
        description: Server error
    """
    try:
        data = request.json
        
        # Validate required fields
        if not data.get('email') or not data.get('city_name'):
            return jsonify({"error": "Email and city_name are required"}), 400
            
        # Validate city
        if data.get('city_name') not in CITIES:
            return jsonify({"error": f"Invalid city. Supported cities: {', '.join(CITIES)}"}), 400
            
        # Create notification object
        notification = Notification(
            email=data.get('email'),
            city_name=data.get('city_name'),
            threshold_pm25=data.get('threshold_pm25'),
            threshold_pm10=data.get('threshold_pm10'),
            threshold_o3=data.get('threshold_o3'),
            threshold_no2=data.get('threshold_no2'),
            threshold_so2=data.get('threshold_so2'),
            threshold_co=data.get('threshold_co'),
            threshold_aqi=data.get('threshold_aqi'),
            daily_summary=data.get('daily_summary', False),
            is_active=True
        )
        
        # Save to database
        session = get_session(engine)
        session.add(notification)
        session.commit()
        
        # Return the new notification id
        return jsonify({
            "id": notification.id,
            "message": "Notification subscription created successfully"
        }), 201
        
    except Exception as e:
        logger.error(f"Error creating notification: {str(e)}")
        return jsonify({"error": "Failed to create notification subscription"}), 500

@model_api.route('/notifications/<notification_id>', methods=['PUT'])
def update_notification(notification_id):
    """
    Update an existing notification subscription
    ---
    parameters:
      - name: notification_id
        in: path
        required: true
        type: integer
        description: ID of the notification to update
      - name: notification
        in: body
        required: true
        schema:
          type: object
          properties:
            email:
              type: string
              description: Email address to send notifications to
            city_name:
              type: string
              description: City to monitor
            threshold_pm25:
              type: number
              description: Threshold for PM2.5 (optional)
            threshold_pm10:
              type: number
              description: Threshold for PM10 (optional)
            threshold_o3:
              type: number
              description: Threshold for O3 (optional)
            threshold_no2:
              type: number
              description: Threshold for NO2 (optional)
            threshold_so2:
              type: number
              description: Threshold for SO2 (optional)
            threshold_co:
              type: number
              description: Threshold for CO (optional)
            threshold_aqi:
              type: number
              description: Threshold for AQI (optional)
            daily_summary:
              type: boolean
              description: Whether to send daily summaries
            is_active:
              type: boolean
              description: Whether the notification is active
    responses:
      200:
        description: Notification updated successfully
      404:
        description: Notification not found
      500:
        description: Server error
    """
    try:
        data = request.json
        
        # Get notification
        session = get_session(engine)
        notification = session.query(Notification).filter_by(id=notification_id).first()
        
        if not notification:
            return jsonify({"error": "Notification not found"}), 404
            
        # Update fields
        if 'email' in data:
            notification.email = data['email']
        if 'city_name' in data:
            if data['city_name'] in CITIES:
                notification.city_name = data['city_name']
            else:
                return jsonify({"error": f"Invalid city. Supported cities: {', '.join(CITIES)}"}), 400
                
        for field in ['threshold_pm25', 'threshold_pm10', 'threshold_o3', 'threshold_no2', 
                     'threshold_so2', 'threshold_co', 'threshold_aqi']:
            if field in data:
                setattr(notification, field, data[field])
                
        if 'daily_summary' in data:
            notification.daily_summary = data['daily_summary']
            
        if 'is_active' in data:
            notification.is_active = data['is_active']
            
        # Save changes
        session.commit()
        
        return jsonify({
            "id": notification.id,
            "message": "Notification subscription updated successfully"
        }), 200
        
    except Exception as e:
        logger.error(f"Error updating notification: {str(e)}")
        return jsonify({"error": "Failed to update notification subscription"}), 500

@model_api.route('/notifications/<notification_id>', methods=['DELETE'])
def delete_notification(notification_id):
    """
    Delete a notification subscription
    ---
    parameters:
      - name: notification_id
        in: path
        required: true
        type: integer
        description: ID of the notification to delete
    responses:
      200:
        description: Notification deleted successfully
      404:
        description: Notification not found
      500:
        description: Server error
    """
    try:
        # Get notification
        session = get_session(engine)
        notification = session.query(Notification).filter_by(id=notification_id).first()
        
        if not notification:
            return jsonify({"error": "Notification not found"}), 404
            
        # Delete notification
        session.delete(notification)
        session.commit()
        
        return jsonify({
            "message": "Notification subscription deleted successfully"
        }), 200
        
    except Exception as e:
        logger.error(f"Error deleting notification: {str(e)}")
        return jsonify({"error": "Failed to delete notification subscription"}), 500

@model_api.route('/notifications', methods=['GET'])
def get_notifications():
    """
    Get all notifications for a specific email
    ---
    parameters:
      - name: email
        in: query
        required: true
        type: string
        description: Email address to get notifications for
    responses:
      200:
        description: List of notifications
      500:
        description: Server error
    """
    try:
        email = request.args.get('email')
        
        if not email:
            return jsonify({"error": "Email parameter is required"}), 400
            
        # Get notifications
        session = get_session(engine)
        notifications = session.query(Notification).filter_by(email=email).all()
        
        result = []
        for notification in notifications:
            result.append({
                "id": notification.id,
                "email": notification.email,
                "city_name": notification.city_name,
                "threshold_pm25": notification.threshold_pm25,
                "threshold_pm10": notification.threshold_pm10,
                "threshold_o3": notification.threshold_o3,
                "threshold_no2": notification.threshold_no2,
                "threshold_so2": notification.threshold_so2,
                "threshold_co": notification.threshold_co,
                "threshold_aqi": notification.threshold_aqi,
                "daily_summary": notification.daily_summary,
                "is_active": notification.is_active,
                "created_at": notification.created_at.strftime('%Y-%m-%d %H:%M:%S')
            })
            
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error getting notifications: {str(e)}")
        return jsonify({"error": "Failed to get notifications"}), 500

@model_api.route('/notifications/history', methods=['GET'])
def get_notification_history():
    """
    Get notification history for a specific email
    ---
    parameters:
      - name: email
        in: query
        required: true
        type: string
        description: Email address to get notification history for
      - name: days
        in: query
        required: false
        type: integer
        description: Number of days of history to retrieve (default: 7)
    responses:
      200:
        description: Notification history
      500:
        description: Server error
    """
    try:
        email = request.args.get('email')
        days = int(request.args.get('days', 7))
        
        if not email:
            return jsonify({"error": "Email parameter is required"}), 400
            
        # Get notifications for this email
        session = get_session(engine)
        notifications = session.query(Notification).filter_by(email=email).all()
        
        if not notifications:
            return jsonify([]), 200
            
        # Get notification IDs
        notification_ids = [n.id for n in notifications]
        
        # Get logs for these notifications
        since_date = datetime.now() - timedelta(days=days)
        logs = session.query(NotificationLog)\
            .filter(NotificationLog.notification_id.in_(notification_ids))\
            .filter(NotificationLog.sent_at >= since_date)\
            .order_by(desc(NotificationLog.sent_at))\
            .all()
            
        result = []
        for log in logs:
            notification = session.query(Notification).filter_by(id=log.notification_id).first()
            
            result.append({
                "id": log.id,
                "notification_id": log.notification_id,
                "city": notification.city_name if notification else "Unknown",
                "sent_at": log.sent_at.strftime('%Y-%m-%d %H:%M:%S'),
                "notification_type": log.notification_type,
                "pollutant": log.pollutant,
                "value": log.value
            })
            
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error getting notification history: {str(e)}")
        return jsonify({"error": "Failed to get notification history"}), 500

@model_api.route('/air-quality/<city_name>/history', methods=['GET'])
def get_historical_air_quality(city_name):
    """
    Get historical air quality measurements for a specific city
    ---
    parameters:
      - name: city_name
        in: path
        required: true
        type: string
        description: City to get measurements for
      - name: days
        in: query
        required: false
        type: integer
        description: Number of days of data to retrieve (default: 7)
    responses:
      200:
        description: List of historical air quality measurements
      400:
        description: Invalid city
      404:
        description: No data found
      500:
        description: Server error
    """
    try:
        if city_name not in CITIES:
            return jsonify({"error": f"Invalid city. Supported cities: {', '.join(CITIES)}"}), 400
            
        # Get days parameter (default to 7 days)
        days = int(request.args.get('days', 7))
        
        # Get historical measurements
        session = get_session(engine)
        since_date = datetime.now() - timedelta(days=days)
        
        measurements = session.query(AirQualityMeasurement)\
            .filter_by(city=city_name)\
            .filter(AirQualityMeasurement.timestamp >= since_date)\
            .order_by(desc(AirQualityMeasurement.timestamp))\
            .all()
            
        if not measurements:
            return jsonify({"error": f"No historical data found for city: {city_name}"}), 404
            
        result = []
        for m in measurements:
            result.append({
                "datetime": m.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                "pm2_5": m.pm25,
                "pm10": m.pm10,
                "o3": m.o3,
                "no2": m.no2,
                "so2": m.so2,
                "co": m.co
            })
            
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error getting historical air quality data: {str(e)}")
        return jsonify({"error": "Failed to get historical air quality data"}), 500

@model_api.route('/air-quality/<city_name>', methods=['GET'])
def get_air_quality(city_name):
    """
    Get recent air quality measurements for a specific city
    ---
    parameters:
      - name: city_name
        in: path
        required: true
        type: string
        description: City to get measurements for
      - name: hours
        in: query
        required: false
        type: integer
        description: Number of hours of data to retrieve (default: 24)
    responses:
      200:
        description: List of air quality measurements
      400:
        description: Invalid city
      404:
        description: No data found
      500:
        description: Server error
    """
    try:
        logger.info(f"Received request for city: {city_name}")
        
        if city_name not in CITIES:
            logger.warning(f"Invalid city requested: {city_name}")
            return jsonify({"error": f"Invalid city. Supported cities: {', '.join(CITIES)}"}), 400
            
        # Get hours parameter (default to 24 hours)
        hours = int(request.args.get('hours', 24))
        logger.info(f"Requested hours: {hours}")
        
        # Get recent measurements
        session = get_session(engine)
        
        # First get the most recent data point
        latest_measurement = session.query(AirQualityMeasurement)\
            .filter_by(city=city_name)\
            .order_by(desc(AirQualityMeasurement.timestamp))\
            .first()
            
        if not latest_measurement:
            logger.warning(f"No data found for city: {city_name}")
            return jsonify({"error": f"No data found for city: {city_name}"}), 404
            
        # Calculate the start time based on the latest measurement
        since_date = latest_measurement.timestamp - timedelta(hours=hours)
        logger.info(f"Querying data since: {since_date}")
        
        measurements = session.query(AirQualityMeasurement)\
            .filter_by(city=city_name)\
            .filter(AirQualityMeasurement.timestamp >= since_date)\
            .order_by(desc(AirQualityMeasurement.timestamp))\
            .all()
            
        logger.info(f"Found {len(measurements)} records for {city_name}")
        
        if not measurements:
            logger.warning(f"No data found for city: {city_name} in the last {hours} hours")
            return jsonify({"error": f"No data found for city: {city_name}"}), 404
            
        result = []
        for m in measurements:
            result.append({
                "timestamp": m.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                "pm2_5": m.pm25,
                "pm10": m.pm10,
                "o3": m.o3,
                "no2": m.no2,
                "so2": m.so2,
                "co": m.co
            })
            
        logger.info(f"Successfully processed {len(result)} records")
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error getting air quality data: {str(e)}", exc_info=True)
        return jsonify({"error": "Failed to get air quality data"}), 500

@model_api.route('/predictions/<city_name>', methods=['GET'])
def get_prediction(city_name):
    """
    Get air quality predictions for a specific city
    ---
    parameters:
      - name: city_name
        in: path
        type: string
        required: true
        description: Name of the city to get predictions for
      - name: hours
        in: query
        type: integer
        required: false
        default: 72
        description: Number of hours to predict ahead
    responses:
      200:
        description: Successful prediction
        schema:
          type: object
          properties:
            city:
              type: string
            generated_at:
              type: string
            predictions:
              type: array
              items:
                type: object
                properties:
                  datetime:
                    type: string
                  pm2_5:
                    type: number
                  pm10:
                    type: number
                  o3:
                    type: number
                  no2:
                    type: number
                  so2:
                    type: number
                  co:
                    type: number
      404:
        description: City not found or model not available
      500:
        description: Server error
    """
    try:
        # Get number of hours to predict
        hours = request.args.get('hours', default=72, type=int)
        
        # Get latest data from database
        session = get_session(engine)
        latest_data = session.query(AirQualityMeasurement).filter(
            AirQualityMeasurement.city == city_name
        ).order_by(
            AirQualityMeasurement.timestamp.desc()
        ).limit(48).all()  # Get 48 hours of data to calculate rolling features
        
        if not latest_data:
            return jsonify({"error": f"No data available for city: {city_name}"}), 404
            
        # Convert to DataFrame
        df = pd.DataFrame([{
            'timestamp': m.timestamp,
            'co': m.co,
            'no2': m.no2,
            'o3': m.o3,
            'so2': m.so2,
            'pm25': m.pm25,
            'pm10': m.pm10
        } for m in latest_data])
        
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        # Initialize predictions dictionary
        predictions = {
            'co': [],
            'no2': [],
            'o3': [],
            'so2': [],
            'pm25': [],
            'pm10': []
        }
        
        # Load and use separate models for each parameter
        parameters = ['co', 'no2', 'o3', 'so2', 'pm25', 'pm10']
        for param in parameters:
            model_path = os.path.join(os.path.dirname(__file__), '..', 'models', f'lstm_improved_{city_name}_{param}.pth')
            scaler_path = os.path.join(os.path.dirname(__file__), '..', 'models', f'scaler_improved_{city_name}_{param}.pkl')
            
            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                logger.error(f"Model files not found for {param}. Model path: {model_path}, Scaler path: {scaler_path}")
                return jsonify({"error": f"Prediction model not available for parameter {param} in city: {city_name}"}), 404
            
            try:
                # Load model and scaler
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = ImprovedSingleParameterLSTM(
                    input_size=32,  # Changed to match saved model
                    hidden_size=128,
                    num_layers=2,
                    prediction_horizon=72  # Changed to match saved model
                )
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.to(device)
                model.eval()
                
                scaler = joblib.load(scaler_path)
                
                # Prepare features
                feature_df = pd.DataFrame(index=df.index)
                
                # Original parameter
                feature_df[param] = df[param]
                
                # Add rolling statistics
                windows = [6, 12, 24, 48]  # 6h, 12h, 24h, 48h
                for window in windows:
                    feature_df[f'{param}_rolling_mean_{window}h'] = df[param].rolling(window=window).mean()
                    feature_df[f'{param}_rolling_std_{window}h'] = df[param].rolling(window=window).std()
                    feature_df[f'{param}_rolling_min_{window}h'] = df[param].rolling(window=window).min()
                    feature_df[f'{param}_rolling_max_{window}h'] = df[param].rolling(window=window).max()
                
                # Add lag features
                for lag in [1, 2, 3, 6, 12, 24]:
                    feature_df[f'{param}_lag_{lag}h'] = df[param].shift(lag)
                
                # Add time features
                feature_df['hour_sin'] = np.sin(2 * np.pi * feature_df.index.hour / 24)
                feature_df['hour_cos'] = np.cos(2 * np.pi * feature_df.index.hour / 24)
                feature_df['day_sin'] = np.sin(2 * np.pi * feature_df.index.dayofweek / 7)
                feature_df['day_cos'] = np.cos(2 * np.pi * feature_df.index.dayofweek / 7)
                feature_df['month_sin'] = np.sin(2 * np.pi * feature_df.index.month / 12)
                feature_df['month_cos'] = np.cos(2 * np.pi * feature_df.index.month / 12)
                feature_df['is_weekend'] = feature_df.index.dayofweek.isin([5, 6]).astype(int)
                feature_df['is_morning_rush'] = ((feature_df.index.hour >= 7) & (feature_df.index.hour <= 9)).astype(int)
                feature_df['is_evening_rush'] = ((feature_df.index.hour >= 17) & (feature_df.index.hour <= 19)).astype(int)
                
                # Fill NaN values
                feature_df = feature_df.bfill()
                
                # Select features for scaling
                feature_columns = [col for col in feature_df.columns if param in col or col in [
                    'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 
                    'month_sin', 'month_cos', 'is_weekend',
                    'is_morning_rush', 'is_evening_rush'
                ]]
                
                # Get the last 32 points
                input_data = feature_df[feature_columns].iloc[-32:].values
                
                # Scale the input data
                scaled_input = scaler.transform(input_data)
                
                # Reshape for LSTM input (batch_size, sequence_length, features)
                input_tensor = torch.FloatTensor(scaled_input).unsqueeze(0).to(device)  # Add batch dimension
                
                # Make predictions
                with torch.no_grad():
                    output = model(input_tensor)
                    param_predictions = output.cpu().numpy()
                
                # Reshape predictions to match training format
                param_predictions = param_predictions.reshape(-1, 1)
                
                # Create a dummy array with the same shape as training data
                dummy_array = np.zeros((param_predictions.shape[0], len(feature_columns)))
                dummy_array[:, 0] = param_predictions.flatten()  # Put predictions in first column
                
                # Inverse transform using the dummy array
                param_predictions = scaler.inverse_transform(dummy_array)[:, 0]
                predictions[param] = param_predictions.tolist()
                
            except Exception as e:
                logger.error(f"Error making prediction for {param}: {str(e)}")
                return jsonify({"error": f"Failed to generate prediction for {param}"}), 500
        
        # Format result
        result = {
            "city": city_name,
            "generated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "predictions": []
        }
        
        # Create timestamps for predictions
        last_timestamp = df.index[-1]
        future_times = [last_timestamp + pd.Timedelta(hours=i+1) for i in range(hours)]
        
        # Combine predictions
        for i, timestamp in enumerate(future_times):
            prediction = {
                "datetime": timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                "pm2_5": float(predictions['pm25'][i]),
                "pm10": float(predictions['pm10'][i]),
                "o3": float(predictions['o3'][i]),
                "no2": float(predictions['no2'][i]),
                "so2": float(predictions['so2'][i]),
                "co": float(predictions['co'][i])
            }
            result["predictions"].append(prediction)
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error in prediction endpoint: {str(e)}")
        return jsonify({"error": "Failed to generate prediction"}), 500

@model_api.route('/cities', methods=['GET'])
def get_cities():
    """
    Get list of all supported cities
    ---
    responses:
      200:
        description: List of cities
      500:
        description: Server error
    """
    try:
        return jsonify(CITIES), 200
    except Exception as e:
        logger.error(f"Error getting cities: {str(e)}")
        return jsonify({"error": "Failed to get cities list"}), 500

@model_api.route('/model/metrics', methods=['GET'])
def get_model_metrics():
    """
    Get model evaluation metrics (MSE and MAE) for all parameters
    ---
    responses:
      200:
        description: Model metrics retrieved successfully
        schema:
          type: object
          properties:
            parameters:
              type: object
              properties:
                co:
                  type: object
                  properties:
                    mse: number
                    mae: number
                no2:
                  type: object
                  properties:
                    mse: number
                    mae: number
                o3:
                  type: object
                  properties:
                    mse: number
                    mae: number
                so2:
                  type: object
                  properties:
                    mse: number
                    mae: number
                pm25:
                  type: object
                  properties:
                    mse: number
                    mae: number
                pm10:
                  type: object
                  properties:
                    mse: number
                    mae: number
      500:
        description: Server error
    """
    try:
        # Read the metrics from the CSV file
        metrics_file = os.path.join(os.path.dirname(__file__), 'improved_model_evaluation_metrics.csv')
        
        logger.info(f"Looking for metrics file at: {metrics_file}")
        
        if not os.path.exists(metrics_file):
            logger.error(f"Metrics file not found at: {metrics_file}")
            return jsonify({"error": "Metrics file not found"}), 404
            
        metrics_df = pd.read_csv(metrics_file, index_col=0)
        
        # Convert DataFrame to dictionary format
        metrics = {}
        for parameter in metrics_df.index:
            metrics[parameter] = {
                'mse': float(metrics_df.loc[parameter, 'mse']),
                'mae': float(metrics_df.loc[parameter, 'mae']),
                'r2': float(metrics_df.loc[parameter, 'r2'])
            }
        
        return jsonify(metrics), 200
        
    except Exception as e:
        logger.error(f"Error retrieving model metrics: {str(e)}")
        return jsonify({"error": "Failed to retrieve model metrics"}), 500

@model_api.route('/model/actual-vs-predicted/<city_name>/<parameter>', methods=['GET'])
def get_actual_vs_predicted(city_name, parameter):
    """
    Get actual vs predicted values for model evaluation
    ---
    parameters:
      - name: city_name
        in: path
        required: true
        type: string
        description: City to get data for
      - name: parameter
        in: path
        required: true
        type: string
        description: Parameter to get data for (co, no2, o3, so2, pm25, pm10)
    responses:
      200:
        description: Actual vs predicted data
        schema:
          type: object
          properties:
            actual:
              type: array
              items:
                type: number
            predicted:
              type: array
              items:
                type: number
            timestamps:
              type: array
              items:
                type: string
      404:
        description: Data not found
      500:
        description: Server error
    """
    try:
        if city_name not in CITIES:
            return jsonify({"error": f"Invalid city. Supported cities: {', '.join(CITIES)}"}), 400
            
        if parameter not in ['co', 'no2', 'o3', 'so2', 'pm25', 'pm10']:
            return jsonify({"error": "Invalid parameter"}), 400

        # Load model and scaler
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', f'lstm_improved_{city_name}_{parameter}.pth')
        scaler_path = os.path.join(os.path.dirname(__file__), '..', 'models', f'scaler_improved_{city_name}_{parameter}.pkl')
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return jsonify({"error": f"Model files not found for {parameter}"}), 404

        # Load model and scaler
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ImprovedSingleParameterLSTM(
            input_size=32,  # Match the saved model's input size
            hidden_size=128,
            num_layers=2,
            prediction_horizon=72  # Match the saved model's prediction horizon
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        scaler = joblib.load(scaler_path)

        # Get test data
        session = get_session(engine)
        test_data = (session.query(AirQualityMeasurement)
            .filter_by(city=city_name)
            .order_by(desc(AirQualityMeasurement.timestamp))
            .limit(48)
            .all())  # Get 48 hours of data to calculate features

        if not test_data:
            return jsonify({"error": "No test data available"}), 404

        # Convert to DataFrame
        df = pd.DataFrame([{
            'timestamp': m.timestamp,
            parameter: getattr(m, parameter if parameter != 'pm25' else 'pm25')
        } for m in test_data])
        
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)

        # Prepare features
        feature_df = pd.DataFrame(index=df.index)
        
        # Original parameter
        feature_df[parameter] = df[parameter]
        
        # Add rolling statistics
        windows = [6, 12, 24, 48]  # 6h, 12h, 24h, 48h
        for window in windows:
            feature_df[f'{parameter}_rolling_mean_{window}h'] = df[parameter].rolling(window=window).mean()
            feature_df[f'{parameter}_rolling_std_{window}h'] = df[parameter].rolling(window=window).std()
            feature_df[f'{parameter}_rolling_min_{window}h'] = df[parameter].rolling(window=window).min()
            feature_df[f'{parameter}_rolling_max_{window}h'] = df[parameter].rolling(window=window).max()
        
        # Add lag features
        for lag in [1, 2, 3, 6, 12, 24]:
            feature_df[f'{parameter}_lag_{lag}h'] = df[parameter].shift(lag)
        
        # Add time features
        feature_df['hour_sin'] = np.sin(2 * np.pi * feature_df.index.hour / 24)
        feature_df['hour_cos'] = np.cos(2 * np.pi * feature_df.index.hour / 24)
        feature_df['day_sin'] = np.sin(2 * np.pi * feature_df.index.dayofweek / 7)
        feature_df['day_cos'] = np.cos(2 * np.pi * feature_df.index.dayofweek / 7)
        feature_df['month_sin'] = np.sin(2 * np.pi * feature_df.index.month / 12)
        feature_df['month_cos'] = np.cos(2 * np.pi * feature_df.index.month / 12)
        feature_df['is_weekend'] = feature_df.index.dayofweek.isin([5, 6]).astype(int)
        feature_df['is_morning_rush'] = ((feature_df.index.hour >= 7) & (feature_df.index.hour <= 9)).astype(int)
        feature_df['is_evening_rush'] = ((feature_df.index.hour >= 17) & (feature_df.index.hour <= 19)).astype(int)
        
        # Fill NaN values
        feature_df = feature_df.bfill()
        
        # Select features for scaling
        feature_columns = [col for col in feature_df.columns if parameter in col or col in [
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 
            'month_sin', 'month_cos', 'is_weekend',
            'is_morning_rush', 'is_evening_rush'
        ]]
        
        # Get the last 32 points
        input_data = feature_df[feature_columns].iloc[-32:].values
        
        # Scale the input data
        scaled_input = scaler.transform(input_data)
        
        # Reshape for LSTM input (batch_size, sequence_length, features)
        input_tensor = torch.FloatTensor(scaled_input).unsqueeze(0).to(device)
        
        # Make predictions
        with torch.no_grad():
            output = model(input_tensor)
            predictions = output.cpu().numpy()
        
        # Reshape predictions to match training format
        predictions = predictions.reshape(-1, 1)
        
        # Create a dummy array with the same shape as training data
        dummy_array = np.zeros((predictions.shape[0], len(feature_columns)))
        dummy_array[:, 0] = predictions.flatten()  # Put predictions in first column
        
        # Inverse transform using the dummy array
        predictions = scaler.inverse_transform(dummy_array)[:, 0]
        
        # Get actual values
        actual_values = df[parameter].values[-len(predictions):]

        # Format response
        result = {
            "actual": actual_values.tolist(),
            "predicted": predictions.tolist(),
            "timestamps": df.index[-len(predictions):].strftime('%Y-%m-%d %H:%M:%S').tolist()
        }

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error getting actual vs predicted data: {str(e)}")
        return jsonify({"error": "Failed to get actual vs predicted data"}), 500
