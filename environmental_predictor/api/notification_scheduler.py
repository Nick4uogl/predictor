import logging
import time
import schedule
from threading import Thread
from api.notification_service import NotificationService

logger = logging.getLogger(__name__)

class NotificationScheduler:
    def __init__(self):
        """Initialize notification scheduler"""
        self.notification_service = NotificationService()
        self.scheduler_thread = None
        self.running = False
        
    def start(self):
        """Start notification scheduler in background thread"""
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            logger.warning("Notification scheduler is already running")
            return False
            
        self.running = True
        self.scheduler_thread = Thread(target=self._run_scheduler)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        logger.info("Notification scheduler started")
        return True
        
    def stop(self):
        """Stop notification scheduler"""
        if not self.scheduler_thread or not self.scheduler_thread.is_alive():
            logger.warning("Notification scheduler is not running")
            return False
            
        self.running = False
        self.scheduler_thread.join(timeout=5)
        logger.info("Notification scheduler stopped")
        return True
        
    def _run_scheduler(self):
        """Run the scheduler loop"""
        # Schedule threshold checks every 15 minutes
        schedule.every(15).minutes.do(self._check_thresholds)
        
        # Schedule daily summaries at 8:00 AM
        schedule.every().day.at("08:00").do(self._send_daily_summaries)
        
        logger.info("Scheduler started with jobs: check thresholds every 15 minutes, daily summaries at 8:00 AM")
        
        while self.running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute for pending jobs
            
    def _check_thresholds(self):
        """Run threshold notifications check"""
        try:
            logger.info("Running threshold notifications check")
            self.notification_service.check_threshold_notifications()
            return True
        except Exception as e:
            logger.error(f"Error in threshold notifications check: {str(e)}")
            return False
            
    def _send_daily_summaries(self):
        """Send daily summary notifications"""
        try:
            logger.info("Sending daily summary notifications")
            self.notification_service.send_daily_summaries()
            return True
        except Exception as e:
            logger.error(f"Error sending daily summaries: {str(e)}")
            return False

# Singleton instance
scheduler = NotificationScheduler()

def start_scheduler():
    """Start the notification scheduler"""
    return scheduler.start()
    
def stop_scheduler():
    """Stop the notification scheduler"""
    return scheduler.stop()