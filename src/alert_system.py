"""
EquiRide Surge AI - Alert System (Compatible with Existing FeatureStore)
Processes all alerts, logs them properly, and sends SMS only for top 5.
"""
import os
from datetime import datetime
from dotenv import load_dotenv
from config.logging_config import setup_logging, log_stage

# Load environment variables for Twilio
load_dotenv()

class AlertSystem:
    def __init__(self, config, logger=None):
        """Initialize Alert System"""
        self.config = config
        self.logger = logger or setup_logging()
        self.twilio_enabled = config.get('twilio', {}).get('enabled', False)
        self.alert_cooldown = config.get('alerts', {}).get('alert_cooldown_minutes', 30)
        self.last_alert_time = {}

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

        except Exception as e:
            self.logger.error(f"FAILED Twilio initialization failed: {str(e)}")
            self.twilio_enabled = False

    def check_alert_cooldown(self, area_name):
        """Ensure we don’t spam the same area repeatedly"""
        if area_name not in self.last_alert_time:
            return True
        time_since_last = datetime.now() - self.last_alert_time[area_name]
        return time_since_last.total_seconds() > (self.alert_cooldown * 60)

    def send_sms_alert(self, message, area_name):
        """Send SMS alert via Twilio"""
        if not self.twilio_enabled:
            self.logger.info(f"[ALERT - SMS DISABLED] {message}")
            return False

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

    # ===========================================================
    # SURGE DETECTION LOGIC (based on searches/bookings)
    # ===========================================================
    def check_surge_conditions(self, df):
        """Detect surge zones using statistical thresholds."""
        try:
            log_stage(self.logger, 'CHECK_SURGE', 'START')

            search_mean = df['Searches'].mean()
            search_std = df['Searches'].std()
            booking_mean = df['Bookings'].mean()
            booking_std = df['Bookings'].std()

            # Derived metric to capture abnormal search/booking patterns
            df['demand_index'] = df['Searches'] / (df['Bookings'] + 1)

            surge_rows = df[
                (df['Searches'] > search_mean + 2 * search_std) |
                (df['Bookings'] > booking_mean + 2 * booking_std) |
                (df['demand_index'] > df['demand_index'].mean() + 2 * df['demand_index'].std())
            ]

            alerts = []
            for _, row in surge_rows.iterrows():
                alerts.append({
                    'h3_index': row['h3_index'],
                    'area_name': row.get('Area Name', 'Unknown'),
                    'alert_type': 'DEMAND_SURGE',
                    'severity': self._calculate_custom_severity(row),
                    'searches': row['Searches'],
                    'bookings': row['Bookings'],
                    'demand_index': row['demand_index'],
                    'congestion_level': row.get('Congestion Level', 0),
                    # compatibility fields for FeatureStore
                    'demand_supply_ratio': row['demand_index'],  # mapped to demand_index
                    'timestamp': datetime.now().isoformat()
                })

            log_stage(self.logger, 'CHECK_SURGE', 'SUCCESS', alerts_found=len(alerts))
            return alerts

        except Exception as e:
            log_stage(self.logger, 'CHECK_SURGE', 'FAILURE', error=str(e))
            return []

    def _calculate_custom_severity(self, row):
        """Calculate severity based on demand index and searches."""
        idx = row.get('demand_index', 1.0)
        searches = row.get('Searches', 0)
        if idx > 4 or searches > 1200:
            return 'CRITICAL'
        elif idx > 3 or searches > 900:
            return 'HIGH'
        elif idx > 2 or searches > 700:
            return 'MEDIUM'
        else:
            return 'LOW'

    # ===========================================================
    # MESSAGE FORMATTING
    # ===========================================================
    def format_alert_message(self, alert):
        """Create a Twilio-friendly alert text"""
        msg = f"""🚨 NAMMA YATRI DEMAND ALERT 🚨

Area: {alert['area_name']}
Severity: {alert['severity']}
Searches: {int(alert['searches'])}
Bookings: {int(alert['bookings'])}
Demand Index: {alert['demand_index']:.2f}

Time: {datetime.now().strftime('%H:%M')}
Action: Increase driver availability in this zone immediately.
"""
        return msg.strip()

    # ===========================================================
    # PROCESS ALERTS (LOG ALL, SEND SMS TO TOP 5)
    # ===========================================================
    def process_alerts(self, alerts, feature_store=None):
        """Process all alerts, log them to FeatureStore, and send SMS only for top 5."""
        try:
            log_stage(self.logger, 'PROCESS_ALERTS', 'START', count=len(alerts))
            if len(alerts) == 0:
                self.logger.info("No alerts to process.")
                return 0

            sent_count = 0

            # ✅ Log all alerts to the feature store (existing structure)
            if feature_store:
                for a in alerts:
                    feature_store.log_surge_alert(
                        h3_index=a['h3_index'],
                        area_name=a['area_name'],
                        alert_type=a['alert_type'],
                        severity=a['severity'],
                        demand_supply_ratio=a['demand_supply_ratio'],
                        congestion_level=a['congestion_level']
                    )

            # ✅ Sort and pick top 5 alerts for Twilio SMS
            priority_order = {"CRITICAL": 3, "HIGH": 2, "MEDIUM": 1, "LOW": 0}
            alerts.sort(key=lambda a: (priority_order[a['severity']], a['demand_index']), reverse=True)
            top_alerts = alerts[:5]

            self.logger.info(f"Sending SMS for top {len(top_alerts)} critical alerts (out of {len(alerts)} total).")

            for alert in top_alerts:
                message = self.format_alert_message(alert)
                success = self.send_sms_alert(message, alert['area_name'])
                if success:
                    sent_count += 1

            log_stage(self.logger, 'PROCESS_ALERTS', 'SUCCESS', sent=sent_count)
            return sent_count

        except Exception as e:
            log_stage(self.logger, 'PROCESS_ALERTS', 'FAILURE', error=str(e))
            return 0
