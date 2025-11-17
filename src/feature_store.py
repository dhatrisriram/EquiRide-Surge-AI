"""
Feature Store Management Module
Manages centralized feature registry using SQLite
"""
import pandas as pd
import sqlite3
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.logging_config import setup_logging, log_stage
from src.utils import ensure_dir

class FeatureStore:
    def __init__(self, config, logger=None):
        """Initialize Feature Store"""
        self.config = config
        self.logger = logger or setup_logging()
        self.db_path = config['data']['feature_store_path']
        ensure_dir(os.path.dirname(self.db_path))
        self.conn = None
        
    def connect(self):
        """Connect to SQLite database"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.logger.info(f"SUCCESS Connected to feature store: {self.db_path}")
        except Exception as e:
            self.logger.error(f"FAILED Failed to connect: {str(e)}")
            raise
    
    def create_tables(self):
        """Create feature store tables"""
        try:
            log_stage(self.logger, 'CREATE_TABLES', 'START')
            
            cursor = self.conn.cursor()
            
            # Main features table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS features (
                    feature_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    h3_index TEXT NOT NULL,
                    date TEXT NOT NULL,
                    time_bucket TEXT NOT NULL,
                    area_name TEXT,
                    ward TEXT,
                    traffic_volume REAL,
                    avg_speed REAL,
                    congestion_level REAL,
                    searches REAL,
                    completed_trips REAL,
                    bookings REAL,
                    demand_supply_ratio REAL,
                    booking_success_rate REAL,
                    congestion_change REAL,
                    unfulfilled_demand REAL,
                    demand_anomaly INTEGER,
                    traffic_anomaly INTEGER,
                    congestion_anomaly INTEGER,
                    any_anomaly INTEGER,
                    weather_encoded INTEGER,
                    is_rush_hour INTEGER,
                    is_weekend INTEGER,
                    hour INTEGER,
                    day_of_week INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(h3_index, date, time_bucket)
                )
            ''')
            
            # Metadata table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS feature_metadata (
                    metadata_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feature_name TEXT UNIQUE,
                    feature_type TEXT,
                    description TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Ingestion log
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ingestion_log (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_id TEXT,
                    records_inserted INTEGER,
                    records_failed INTEGER,
                    ingestion_time TEXT DEFAULT CURRENT_TIMESTAMP,
                    status TEXT
                )
            ''')
            
            # Surge alerts log
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS surge_alerts (
                    alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    h3_index TEXT,
                    area_name TEXT,
                    alert_type TEXT,
                    severity TEXT,
                    demand_supply_ratio REAL,
                    congestion_level REAL,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    notification_sent INTEGER DEFAULT 0
                )
            ''')
            
            self.conn.commit()
            log_stage(self.logger, 'CREATE_TABLES', 'SUCCESS')
            
        except Exception as e:
            log_stage(self.logger, 'CREATE_TABLES', 'FAILURE', error=str(e))
            raise
    
    def insert_features(self, df, batch_id=None):
        """Insert features into feature store"""
        try:
            log_stage(self.logger, 'INSERT_FEATURES', 'START', batch=batch_id)
            
            # Select columns to insert
            insert_cols = [
                'h3_index', 'timestamp', 'time_bucket', 'Area Name', 'Ward',
                'Traffic Volume', 'Average Speed', 'Congestion Level',
                'Searches', 'Completed Trips', 'Bookings',
                'demand_supply_ratio', 'booking_success_rate', 'congestion_change',
                'unfulfilled_demand', 'demand_anomaly', 'traffic_anomaly',
                'congestion_anomaly', 'any_anomaly', 'weather_encoded',
                'is_rush_hour', 'is_weekend', 'hour', 'day_of_week'
            ]
            
            # Filter existing columns
            available_cols = [col for col in insert_cols if col in df.columns]
            insert_df = df[available_cols].copy()
            
            # Rename columns to match table schema
            column_mapping = {
                'timestamp': 'date',
                'Area Name': 'area_name',
                'Ward': 'ward',
                'Traffic Volume': 'traffic_volume',
                'Average Speed': 'avg_speed',
                'Congestion Level': 'congestion_level',
                'Searches': 'searches',
                'Completed Trips': 'completed_trips',
                'Bookings': 'bookings'
            }
            insert_df = insert_df.rename(columns=column_mapping)
            
            # Convert date to string
            if 'date' in insert_df.columns:
                insert_df['date'] = insert_df['date'].astype(str)
            if 'time_bucket' in insert_df.columns:
                insert_df['time_bucket'] = insert_df['time_bucket'].astype(str)
            
            # Insert (replace on conflict)
            records_inserted = 0
            records_failed = 0
            
            for _, row in insert_df.iterrows():
                try:
                    cursor = self.conn.cursor()
                    placeholders = ', '.join(['?'] * len(row))
                    columns = ', '.join(row.index)
                    
                    query = f'''
                        INSERT OR REPLACE INTO features ({columns})
                        VALUES ({placeholders})
                    '''
                    cursor.execute(query, tuple(row))
                    records_inserted += 1
                except Exception as row_error:
                    records_failed += 1
                    if records_failed <= 5:  # Log first 5 failures
                        self.logger.warning(f"Failed to insert row: {str(row_error)}")
            
            self.conn.commit()
            
            # Log ingestion
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO ingestion_log (batch_id, records_inserted, records_failed, status)
                VALUES (?, ?, ?, ?)
            ''', (batch_id, records_inserted, records_failed, 'SUCCESS'))
            self.conn.commit()
            
            log_stage(self.logger, 'INSERT_FEATURES', 'SUCCESS',
                     inserted=records_inserted, failed=records_failed)
            
            return records_inserted
            
        except Exception as e:
            log_stage(self.logger, 'INSERT_FEATURES', 'FAILURE', error=str(e))
            raise
    
    def query_features(self, h3_index=None, start_date=None, end_date=None, limit=1000):
        """Query features from feature store"""
        try:
            query = "SELECT * FROM features WHERE 1=1"
            params = []
            
            if h3_index:
                query += " AND h3_index = ?"
                params.append(h3_index)
            
            if start_date:
                query += " AND date >= ?"
                params.append(str(start_date))
            
            if end_date:
                query += " AND date <= ?"
                params.append(str(end_date))
            
            query += f" ORDER BY date DESC LIMIT {limit}"
            
            df = pd.read_sql_query(query, self.conn, params=params)
            self.logger.info(f"SUCCESS Retrieved {len(df)} records from feature store")
            return df
            
        except Exception as e:
            self.logger.error(f"FAILED Query failed: {str(e)}")
            raise
    
    def log_surge_alert(self, h3_index, area_name, alert_type, severity, 
                       demand_supply_ratio, congestion_level):
        """Log a surge alert to database"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO surge_alerts 
                (h3_index, area_name, alert_type, severity, demand_supply_ratio, 
                 congestion_level, notification_sent)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (h3_index, area_name, alert_type, severity, 
                  demand_supply_ratio, congestion_level, 0))
            self.conn.commit()
            self.logger.info(f"SUCCESS Surge alert logged: {area_name} - {alert_type}")
        except Exception as e:
            self.logger.error(f"FAILED Failed to log alert: {str(e)}")
    
    def get_recent_alerts(self, hours=24, limit=100):
        """Get recent surge alerts"""
        try:
            query = f'''
                SELECT * FROM surge_alerts 
                WHERE timestamp >= datetime('now', '-{hours} hours')
                ORDER BY timestamp DESC
                LIMIT {limit}
            '''
            df = pd.read_sql_query(query, self.conn)
            return df
        except Exception as e:
            self.logger.error(f"✗ Failed to get alerts: {str(e)}")
            return pd.DataFrame()
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.logger.info("SUCCESS Feature store connection closed")

# Standalone execution
if __name__ == "__main__":
    import yaml
    
    config_path = 'config/config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    store = FeatureStore(config)
    store.connect()
    store.create_tables()
    
    # Load engineered features
    features_file = os.path.join(config['data']['features_path'], 'engineered_features.csv')
    if os.path.exists(features_file):
        df = pd.read_csv(features_file)
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        store.insert_features(df, batch_id=batch_id)
        print(f"\n✓ Features inserted into store with batch_id: {batch_id}")
    else:
        print(f"\n✗ Features file not found: {features_file}")
    
    store.close()