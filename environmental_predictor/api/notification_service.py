import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.db_manager import NotificationSetting, NotificationHistory, User, AirQualityMeasurement
from datetime import datetime, timedelta
import os
from config import EMAIL_HOST, EMAIL_PORT, EMAIL_USER, EMAIL_PASSWORD, EMAIL_FROM
from models.air_quality_model import AirQualityPredictor
import pandas as pd

logger = logging.getLogger(__name__)

class NotificationService:
    def __init__(self, db_session=None):
        """Initialize notification service"""
        if db_session:
            self.session = db_session
        else:
            from database.db_manager import get_db_session
            self.session = get_db_session()
        
    def calculate_aqi(self, data):
        """Calculate Air Quality Index from pollutant values"""
        # Simplified AQI calculation - should be more complex in a real app
        if not data:
            return 0
            
        pm25_index = round((data.pm2_5 / 12) * 50)
        pm10_index = round((data.pm10 / 50) * 50)
        o3_index = round((data.o3 / 100) * 50)
        no2_index = round((data.no2 / 100) * 50)
        
        # Return the max of all pollutant indices
        return max(pm25_index, pm10_index, o3_index, no2_index)
        
    def get_air_quality_category(self, aqi):
        """Get air quality category based on AQI value"""
        if aqi <= 50:
            return "Добра"
        if aqi <= 100:
            return "Помірна"
        if aqi <= 150:
            return "Шкідлива для чутливих груп"
        if aqi <= 200:
            return "Шкідлива"
        if aqi <= 300:
            return "Дуже шкідлива"
        return "Небезпечна"
        
    def check_threshold_notifications(self):
        """Check all threshold-based notifications and send if needed"""
        try:
            # Get all active notifications
            notifications = self.session.query(NotificationSetting).filter_by(is_enabled=True).all()
            
            for notification in notifications:
                # Get latest measurement for the city
                latest = self.session.query(AirQualityMeasurement)\
                    .filter_by(city=notification.city_name)\
                    .order_by(AirQualityMeasurement.timestamp.desc())\
                    .first()
                    
                if not latest:
                    continue
                    
                # Check each threshold
                thresholds_exceeded = []
                
                if latest.pm25 > notification.pm25_threshold:
                    thresholds_exceeded.append(('PM2.5', latest.pm25, notification.pm25_threshold))
                    
                if latest.pm10 > notification.pm10_threshold:
                    thresholds_exceeded.append(('PM10', latest.pm10, notification.pm10_threshold))
                    
                if latest.o3 > notification.o3_threshold:
                    thresholds_exceeded.append(('O3', latest.o3, notification.o3_threshold))
                    
                if latest.no2 > notification.no2_threshold:
                    thresholds_exceeded.append(('NO2', latest.no2, notification.no2_threshold))
                    
                if latest.so2 > notification.so2_threshold:
                    thresholds_exceeded.append(('SO2', latest.so2, notification.so2_threshold))
                    
                if latest.co > notification.co_threshold:
                    thresholds_exceeded.append(('CO', latest.co, notification.co_threshold))
                    
                # Check AQI threshold
                aqi = self.calculate_aqi(latest)
                
                # Send notification if any thresholds exceeded
                if thresholds_exceeded:
                    user = self.session.query(User).filter_by(id=notification.user_id).first()
                    if user and user.is_active:
                        self._send_threshold_notification(user, notification, thresholds_exceeded, latest.timestamp)
                    
                        # Log notification
                        for pollutant, value, threshold in thresholds_exceeded:
                            history_entry = NotificationHistory(
                                user_id=notification.user_id,
                                city_name=notification.city_name,
                                notification_type="threshold",
                                message=f"Перевищення рівня {pollutant}: {value:.1f} (поріг: {threshold:.1f})"
                            )
                            self.session.add(history_entry)
                        
            self.session.commit()
            
        except Exception as e:
            logger.error(f"Error checking threshold notifications: {str(e)}")
            self.session.rollback()
            
    def send_daily_summaries(self):
        """Send daily summary notifications"""
        try:
            # Get all active notifications with daily frequency
            notifications = self.session.query(NotificationSetting)\
                .filter_by(is_enabled=True, notification_frequency='daily')\
                .all()
                
            for notification in notifications:
                user = self.session.query(User).filter_by(id=notification.user_id).first()
                
                if not user or not user.is_active:
                    continue
                    
                # Generate and send daily summary
                self._send_daily_summary(user, notification)
                
                # Log the notification
                history_entry = NotificationHistory(
                    user_id=notification.user_id,
                    city_name=notification.city_name,
                    notification_type="daily",
                    message="Щоденний звіт про якість повітря"
                )
                self.session.add(history_entry)
                
            self.session.commit()
            
        except Exception as e:
            logger.error(f"Error sending daily summaries: {str(e)}")
            self.session.rollback()
            
    def _send_threshold_notification(self, user, notification, thresholds_exceeded, measurement_time):
        """Send email notification for exceeded thresholds"""
        recipient = user.email
        subject = f"Увага! Перевищення показників якості повітря в {notification.city_name}"
        
        # Create message body
        body = f"""
        <html>
        <body>
            <h2>Перевищення показників якості повітря</h2>
            <p>Місто: <b>{notification.city_name}</b></p>
            <p>Час вимірювання: {measurement_time.strftime('%Y-%m-%d %H:%M')}</p>
            <h3>Перевищені показники:</h3>
            <ul>
        """
        
        for pollutant, value, threshold in thresholds_exceeded:
            body += f"<li><b>{pollutant}</b>: Поточне значення {value:.1f} перевищує встановлений поріг {threshold:.1f}</li>"
            
        body += """
            </ul>
            <p>Рекомендуємо обмежити перебування на відкритому повітрі та закрити вікна.</p>
            <p>--<br>Сервіс прогнозування якості повітря</p>
        </body>
        </html>
        """
        
        if notification.email_notifications:
            self._send_email(recipient, subject, body)
            
        if notification.push_notifications:
            self._send_push_notification(user.id, subject, body)
        
    def _send_daily_summary(self, user, notification):
        """Send daily summary of air quality and forecast"""
        recipient = user.email
        subject = f"Щоденний звіт про якість повітря: {notification.city_name}"
        
        try:
            # Get latest measurement
            latest = self.session.query(AirQualityMeasurement)\
                .filter_by(city=notification.city_name)\
                .order_by(AirQualityMeasurement.timestamp.desc())\
                .first()
                
            if not latest:
                return
                
            # Calculate AQI
            aqi = self.calculate_aqi(latest)
            quality_category = self.get_air_quality_category(aqi)
            
            # Try to get forecast
            forecast_available = False
            forecast_html = ""
            
            try:
                # Try to load model and make prediction
                model_path = os.path.join('models', notification.city_name)
                if os.path.exists(model_path):
                    # Get data for prediction
                    measurements = self.session.query(AirQualityMeasurement)\
                        .filter_by(city=notification.city_name)\
                        .order_by(AirQualityMeasurement.timestamp.desc())\
                        .limit(24)\
                        .all()
                        
                    if len(measurements) > 0:
                        # Convert to dataframe
                        df = pd.DataFrame([{
                            'datetime': m.timestamp,
                            'co': m.co,
                            'no2': m.no2,
                            'o3': m.o3,
                            'so2': m.so2,
                            'pm2_5': m.pm25,
                            'pm10': m.pm10
                        } for m in measurements])
                        
                        model = AirQualityPredictor.load_model(model_path)
                        predictions = model.predict(df)
                        
                        forecast_available = True
                        forecast_html = "<h3>Прогноз на наступну добу:</h3><ul>"
                        
                        # Just show 4 time points (now + 6, 12, 18, 24 hours)
                        hours = [6, 12, 18, 24]
                        for hour in hours:
                            if hour < len(predictions):
                                pred = predictions.iloc[hour]
                                pred_aqi = self.calculate_aqi(pred)
                                pred_category = self.get_air_quality_category(pred_aqi)
                                
                                forecast_html += f"""
                                <li>Через {hour} годин: AQI = {pred_aqi}, 
                                    якість повітря: <b>{pred_category}</b>
                                    (PM2.5: {pred.pm2_5:.1f}, PM10: {pred.pm10:.1f})
                                </li>
                                """
                        
                        forecast_html += "</ul>"
            except Exception as e:
                logger.error(f"Error generating forecast for daily summary: {str(e)}")
                
            # Create message body
            body = f"""
            <html>
            <body>
                <h2>Щоденний звіт про якість повітря</h2>
                <p>Місто: <b>{notification.city_name}</b></p>
                <p>Дата: {datetime.now().strftime('%Y-%m-%d')}</p>
                
                <h3>Поточний стан:</h3>
                <p>Час вимірювання: {latest.timestamp.strftime('%Y-%m-%d %H:%M')}</p>
                <p>Індекс якості повітря (AQI): <b>{aqi}</b></p>
                <p>Категорія: <b>{quality_category}</b></p>
                
                <h3>Детальні показники:</h3>
                <ul>
                    <li>PM2.5: {latest.pm25:.1f} мкг/м³</li>
                    <li>PM10: {latest.pm10:.1f} мкг/м³</li>
                    <li>O3 (озон): {latest.o3:.1f} мкг/м³</li>
                    <li>NO2 (діоксид азоту): {latest.no2:.1f} мкг/м³</li>
                    <li>SO2 (діоксид сірки): {latest.so2:.1f} мкг/м³</li>
                    <li>CO (монооксид вуглецю): {latest.co:.1f} мкг/м³</li>
                </ul>
            """
            
            if forecast_available:
                body += forecast_html
                
            body += f"""
                <p>Щоб змінити налаштування сповіщень, відвідайте наш <a href="https://your-app-url.com/settings">сайт</a>.</p>
                <p>--<br>Сервіс прогнозування якості повітря</p>
            </body>
            </html>
            """
            
            if notification.email_notifications:
                self._send_email(recipient, subject, body)
                
            if notification.push_notifications:
                self._send_push_notification(user.id, subject, f"Щоденний звіт про якість повітря для міста {notification.city_name}. AQI: {aqi}, якість: {quality_category}")
            
        except Exception as e:
            logger.error(f"Error creating daily summary: {str(e)}")
            
    def _send_email(self, recipient, subject, html_body):
        """Send email using SMTP"""
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = EMAIL_FROM
            msg['To'] = recipient
            
            # Attach HTML content
            html_part = MIMEText(html_body, 'html')
            msg.attach(html_part)
            
            # Connect to SMTP server and send
            with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as server:
                server.starttls()
                server.login(EMAIL_USER, EMAIL_PASSWORD)
                server.send_message(msg)
                
            logger.info(f"Email notification sent to {recipient}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email: {str(e)}")
            return False
            
    def _send_push_notification(self, user_id, title, message):
        """Send push notification (placeholder for actual implementation)"""
        try:
            logger.info(f"Push notification sent to user {user_id}: {title}")
            # Here you would implement actual push notification sending
            # This could use Firebase Cloud Messaging, OneSignal, or other service
            return True
        except Exception as e:
            logger.error(f"Error sending push notification: {str(e)}")
            return False