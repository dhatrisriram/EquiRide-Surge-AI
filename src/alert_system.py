"""
Alert System with Twilio Integration
Sends SMS alerts for surge predictions
"""
import os
import sys
from datetime import datetime, timedelta
from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.logging_config import setup_logging, log_stage

# Load environment variables
load_dotenv()

class AlertSystem:
    def __init__(self, config, logger=None):
        """Initialize Alert System"""
        self.config = config
        self.logger = logger or setup_logging()
        self.twilio_enabled = config.get('twilio', {}).get('enabled', False)
        self.alert_cooldown = config.get('alerts', {}).get('alert_cooldown_minutes', 30)
        self.last_alert_time = {}
        
        # Initialize Twilio if enabled
        if self.twilio_enabled:
            self._initialize_twilio()
        else:
            self.logger.info("Twilio alerts disabled in config")
    
    def _initialize_twilio(self):
        """Initialize Twilio client"""
        try:
            from twilio.rest import Client
            
            account_sid = os.getenv('TWILIO_ACCOUNT_SID')
            auth_token = os.getenv('TWILIO_AUTH_TOKEN')
            self.twilio_phone = os.getenv('TWILIO_PHONE_NUMBER')
            self.alert_phone = os.getenv('ALERT_PHONE_NUMBER')
            
            if not all([account_sid, auth_token, self.twilio_phone, self.alert_phone]):
                self.logger.warning("Twilio credentials not configured. Alerts will be logged only.")
                self.twilio_enabled = False
                return
            
            self.twilio_client = Client(account_sid, auth_token)
            self.logger.info("SUCCESS Twilio client initialized successfully")
            
        except ImportError:
            self.logger.warning("Twilio package not installed. Run: pip install twilio")
            self.twilio_enabled = False
        except Exception as e:
            self.logger.error(f"FAILED Twilio initialization failed: {str(e)}")
            self.twilio_enabled = False
    
    def check_alert_cooldown(self, area_name):
        """Check if enough time has passed since last alert for this area"""
        if area_name not in self.last_alert_time:
            return True
        
        time_since_last = datetime.now() - self.last_alert_time[area_name]
        return time_since_last.total_seconds() > (self.alert_cooldown * 60)
    
    def send_sms_alert(self, message, area_name):
        """Send SMS alert via Twilio"""
        if not self.twilio_enabled:
            self.logger.info(f"[ALERT - SMS DISABLED] {message}")
            return False
        
        # Check cooldown
        if not self.check_alert_cooldown(area_name):
            self.logger.info(f"Alert cooldown active for {area_name}. Skipping.")
            return False
        
        try:
            msg = self.twilio_client.messages.create(
                body=message,
                from_=self.twilio_phone,
                to=self.alert_phone
            )
            
            self.last_alert_time[area_name] = datetime.now()
            self.logger.info(f"SUCCESS SMS sent successfully. SID: {msg.sid}")
            return True
            
        except Exception as e:
            self.logger.error(f"FAILED Failed to send SMS: {str(e)}")
            return False
    
    def check_surge_conditions(self, df):
        """
        Check for surge conditions that warrant alerts
        Returns list of alert dictionaries
        """
        try:
            log_stage(self.logger, 'CHECK_SURGE', 'START')
            
            thresholds = self.config.get('alerts', {})
            demand_threshold = thresholds.get('demand_surge_threshold', 2.0)
            congestion_threshold = thresholds.get('congestion_threshold', 80.0)
            
            alerts = []
            
            # Check for demand surges
            surge_rows = df[
                (df['demand_supply_ratio'] > demand_threshold) &
                (df['Congestion Level'] > congestion_threshold)
            ]
            
            for _, row in surge_rows.iterrows():
                alert = {
                    'h3_index': row['h3_index'],
                    'area_name': row.get('Area Name', 'Unknown'),
                    'alert_type': 'DEMAND_SURGE',
                    'severity': self._calculate_severity(row),
                    'demand_supply_ratio': row['demand_supply_ratio'],
                    'congestion_level': row['Congestion Level'],
                    'timestamp': datetime.now().isoformat()
                }
                alerts.append(alert)
            
            # Check for anomalies
            anomaly_rows = df[df.get('any_anomaly', 0) == 1]
            
            for _, row in anomaly_rows.iterrows():
                if row['demand_anomaly'] == 1:
                    alert = {
                        'h3_index': row['h3_index'],
                        'area_name': row.get('Area Name', 'Unknown'),
                        'alert_type': 'DEMAND_ANOMALY',
                        'severity': 'HIGH',
                        'demand_supply_ratio': row.get('demand_supply_ratio', 0),
                        'congestion_level': row.get('Congestion Level', 0),
                        'timestamp': datetime.now().isoformat()
                    }
                    alerts.append(alert)
            
            log_stage(self.logger, 'CHECK_SURGE', 'SUCCESS', alerts_found=len(alerts))
            return alerts
            
        except Exception as e:
            log_stage(self.logger, 'CHECK_SURGE', 'FAILURE', error=str(e))
            return []
    
    def _calculate_severity(self, row):
        """Calculate alert severity based on metrics"""
        demand_ratio = row.get('demand_supply_ratio', 1.0)
        congestion = row.get('Congestion Level', 0)
        
        if demand_ratio > 3.0 or congestion > 95:
            return 'CRITICAL'
        elif demand_ratio > 2.5 or congestion > 85:
            return 'HIGH'
        elif demand_ratio > 2.0 or congestion > 75:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def format_alert_message(self, alert):
        """Format alert message for SMS"""
        msg = f""" ALERT!! NAMMA YATRI SURGE ALERT!!

Area: {alert['area_name']}
Type: {alert['alert_type']}
Severity: {alert['severity']}

Demand/Supply Ratio: {alert['demand_supply_ratio']:.2f}
Congestion Level: {alert['congestion_level']:.1f}%

Time: {datetime.now().strftime('%H:%M')}

Action Required: Reposition drivers to this zone."""
        return msg.strip()
    
    def process_alerts(self, alerts, feature_store=None):
        """Process and send alerts"""
        try:
            log_stage(self.logger, 'PROCESS_ALERTS', 'START', count=len(alerts))
            
            sent_count = 0
            for alert in alerts:
                # Format message
                message = self.format_alert_message(alert)
                
                # Send SMS
                success = self.send_sms_alert(message, alert['area_name'])
                
                if success:
                    sent_count += 1
                
                # Log to feature store if available
                if feature_store:
                    feature_store.log_surge_alert(
                        h3_index=alert['h3_index'],
                        area_name=alert['area_name'],
                        alert_type=alert['alert_type'],
                        severity=alert['severity'],
                        demand_supply_ratio=alert['demand_supply_ratio'],
                        congestion_level=alert['congestion_level']
                    )
            
            log_stage(self.logger, 'PROCESS_ALERTS', 'SUCCESS', sent=sent_count)
            return sent_count
            
        except Exception as e:
            log_stage(self.logger, 'PROCESS_ALERTS', 'FAILURE', error=str(e))
            return 0

# Standalone testing
if __name__ == "__main__":
    import yaml
    import pandas as pd
    
    config_path = 'config/config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize alert system
    alert_sys = AlertSystem(config)
    
    # Create test alert
    test_alert = {
        'h3_index': '8961892e55bffff',
        'area_name': 'Whitefield',
        'alert_type': 'DEMAND_SURGE',
        'severity': 'HIGH',
        'demand_supply_ratio': 2.5,
        'congestion_level': 85.0,
        'timestamp': datetime.now().isoformat()
    }
    
    # Test sending alert
    message = alert_sys.format_alert_message(test_alert)
    print("\nTest Alert Message:")
    print(message)
    
    if alert_sys.twilio_enabled:
        print("\nSending test SMS...")
        alert_sys.send_sms_alert(message, 'Whitefield')
    else:
        print("\nTwilio not configured. Set up .env file to enable SMS alerts.")