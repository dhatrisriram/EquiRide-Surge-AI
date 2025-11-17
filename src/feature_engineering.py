
"""
Feature Engineering Module
Generates advanced features for surge forecasting
"""
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.logging_config import setup_logging, log_stage
from src.utils import detect_anomalies_zscore, ensure_dir, safe_divide

class FeatureEngineer:
    def __init__(self, config, logger=None):
        """Initialize Feature Engineer"""
        self.config = config
        self.logger = logger or setup_logging()
        
    def load_processed_data(self, file_path):
        """Load processed data"""
        try:
            log_stage(self.logger, 'LOAD_PROCESSED', 'START')
            df = pd.read_csv(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            log_stage(self.logger, 'LOAD_PROCESSED', 'SUCCESS', rows=len(df))
            return df
        except Exception as e:
            log_stage(self.logger, 'LOAD_PROCESSED', 'FAILURE', error=str(e))
            raise
    
    def generate_demand_features(self, df):
        """Generate demand-related features"""
        try:
            log_stage(self.logger, 'DEMAND_FEATURES', 'START')
            
            # Demand/Supply ratio
            df['demand_supply_ratio'] = safe_divide(
                df['Searches'], df['Completed Trips'], default=1.0
            )
            
            # Booking success rate
            df['booking_success_rate'] = safe_divide(
                df['Completed Trips'], df['Bookings'], default=0.0
            )
            
            # Search conversion rate
            df['search_conversion_rate'] = safe_divide(
                df['Bookings'], df['Searches'], default=0.0
            )
            
            # Cancellation rate (numeric)
            df['cancellation_rate_numeric'] = safe_divide(
                df['Cancelled Bookings'], df['Bookings'], default=0.0
            )
            
            # Earnings per trip
            df['earnings_per_trip'] = safe_divide(
                df["Drivers' Earnings"], df['Completed Trips'], default=0.0
            )
            
            # Unfulfilled demand
            df['unfulfilled_demand'] = df['Searches'] - df['Completed Trips']
            df['unfulfilled_demand'] = df['unfulfilled_demand'].clip(lower=0)
            
            log_stage(self.logger, 'DEMAND_FEATURES', 'SUCCESS', features=6)
            return df
        except Exception as e:
            log_stage(self.logger, 'DEMAND_FEATURES', 'FAILURE', error=str(e))
            raise
    
    def generate_traffic_features(self, df):
        """Generate traffic-related features"""
        try:
            log_stage(self.logger, 'TRAFFIC_FEATURES', 'START')
            
            # Congestion change rate
            df['congestion_change'] = df.groupby('h3_index')['Congestion Level'].diff().fillna(0)
            
            # Traffic volume change percentage
            df['traffic_volume_pct_change'] = df.groupby('h3_index')['Traffic Volume'].pct_change().fillna(0)
            
            # Speed variance (rolling)
            df['speed_variance'] = df.groupby('h3_index')['Average Speed'].transform(
                lambda x: x.rolling(3, min_periods=1).std()
            ).fillna(0)
            
            # Capacity utilization category
            df['capacity_category'] = pd.cut(
                df['Road Capacity Utilization'],
                bins=[0, 50, 75, 90, 100],
                labels=['low', 'medium', 'high', 'critical']
            ).astype(str)
            
            log_stage(self.logger, 'TRAFFIC_FEATURES', 'SUCCESS', features=4)
            return df
        except Exception as e:
            log_stage(self.logger, 'TRAFFIC_FEATURES', 'FAILURE', error=str(e))
            raise
    
    def generate_rolling_features(self, df):
        """Generate rolling window features"""
        try:
            log_stage(self.logger, 'ROLLING_FEATURES', 'START')
            
            windows = self.config['features']['rolling_window_sizes']
            target_cols = ['Traffic Volume', 'Searches', 'Completed Trips', 'Congestion Level']
            
            features_count = 0
            for col in target_cols:
                if col not in df.columns:
                    continue
                    
                for window in windows:
                    # Rolling mean
                    df[f'{col}_rolling_mean_{window}'] = df.groupby('h3_index')[col].transform(
                        lambda x: x.rolling(window, min_periods=1).mean()
                    )
                    
                    # Rolling max
                    df[f'{col}_rolling_max_{window}'] = df.groupby('h3_index')[col].transform(
                        lambda x: x.rolling(window, min_periods=1).max()
                    )
                    
                    features_count += 2
            
            log_stage(self.logger, 'ROLLING_FEATURES', 'SUCCESS', features=features_count)
            return df
        except Exception as e:
            log_stage(self.logger, 'ROLLING_FEATURES', 'FAILURE', error=str(e))
            raise
    
    def generate_lag_features(self, df):
        """Generate lag features"""
        try:
            log_stage(self.logger, 'LAG_FEATURES', 'START')
            
            lags = self.config['features']['lag_periods']
            target_cols = ['Searches', 'Completed Trips', 'Congestion Level']
            
            features_count = 0
            for col in target_cols:
                if col not in df.columns:
                    continue
                    
                for lag in lags:
                    df[f'{col}_lag_{lag}'] = df.groupby('h3_index')[col].shift(lag)
                    features_count += 1
            
            log_stage(self.logger, 'LAG_FEATURES', 'SUCCESS', features=features_count)
            return df
        except Exception as e:
            log_stage(self.logger, 'LAG_FEATURES', 'FAILURE', error=str(e))
            raise
    
    def generate_temporal_features(self, df):
        """Generate time-based features"""
        try:
            log_stage(self.logger, 'TEMPORAL_FEATURES', 'START')
            
            # Hour category
            df['hour_category'] = pd.cut(
                df['hour'],
                bins=[-1, 6, 12, 18, 24],
                labels=['night', 'morning', 'afternoon', 'evening']
            ).astype(str)
            
            # Rush hour flag
            df['is_rush_hour'] = df['hour'].isin([8, 9, 18, 19, 20]).astype(int)
            
            # Weekend flag (already exists, ensure it's int)
            if 'is_weekend' not in df.columns:
                df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            # Day part encoding (cyclical)
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            
            log_stage(self.logger, 'TEMPORAL_FEATURES', 'SUCCESS', features=5)
            return df
        except Exception as e:
            log_stage(self.logger, 'TEMPORAL_FEATURES', 'FAILURE', error=str(e))
            raise
    
    def generate_anomaly_features(self, df):
        """Detect anomalies in key metrics"""
        try:
            log_stage(self.logger, 'ANOMALY_FEATURES', 'START')
            
            threshold = self.config['features']['anomaly_threshold']
            
            # Anomaly in demand (searches)
            df['demand_anomaly'] = detect_anomalies_zscore(df['Searches'], threshold).astype(int)
            
            # Anomaly in traffic volume
            df['traffic_anomaly'] = detect_anomalies_zscore(df['Traffic Volume'], threshold).astype(int)
            
            # Anomaly in congestion
            df['congestion_anomaly'] = detect_anomalies_zscore(df['Congestion Level'], threshold).astype(int)
            
            # Combined anomaly flag
            df['any_anomaly'] = ((df['demand_anomaly'] == 1) | 
                                 (df['traffic_anomaly'] == 1) | 
                                 (df['congestion_anomaly'] == 1)).astype(int)
            
            anomaly_count = df['any_anomaly'].sum()
            log_stage(self.logger, 'ANOMALY_FEATURES', 'SUCCESS', 
                     features=4, anomalies_detected=anomaly_count)
            return df
        except Exception as e:
            log_stage(self.logger, 'ANOMALY_FEATURES', 'FAILURE', error=str(e))
            raise
    
    def generate_weather_event_features(self, df):
        """Generate weather and event features"""
        try:
            log_stage(self.logger, 'WEATHER_EVENT_FEATURES', 'START')
            
            # Weather encoding
            weather_map = {
                'Clear': 0, 'Overcast': 1, 'Cloudy': 1,
                'Rain': 2, 'Fog': 3, 'Windy': 1, 'Unknown': 0
            }
            df['weather_encoded'] = df['Weather Conditions'].map(weather_map).fillna(0).astype(int)
            
            # Weather severity (high impact: Rain, Fog)
            df['weather_severity'] = (df['weather_encoded'] >= 2).astype(int)
            
            # Roadwork flag
            df['has_roadwork'] = (df['Roadwork and Construction Activity'] == 'Yes').astype(int)
            
            # Incident impact
            df['has_incidents'] = (df['Incident Reports'] > 0).astype(int)
            
            log_stage(self.logger, 'WEATHER_EVENT_FEATURES', 'SUCCESS', features=4)
            return df
        except Exception as e:
            log_stage(self.logger, 'WEATHER_EVENT_FEATURES', 'FAILURE', error=str(e))
            raise
    
    def run_feature_engineering_pipeline(self, input_file, output_file):
        """Execute complete feature engineering pipeline"""
        try:
            self.logger.info("="*70)
            self.logger.info("FEATURE ENGINEERING PIPELINE - STARTED")
            self.logger.info("="*70)
            
            # Load processed data
            df = self.load_processed_data(input_file)
            
            initial_cols = len(df.columns)
            
            # Generate all feature categories
            df = self.generate_demand_features(df)
            df = self.generate_traffic_features(df)
            df = self.generate_rolling_features(df)
            df = self.generate_lag_features(df)
            df = self.generate_temporal_features(df)
            df = self.generate_anomaly_features(df)
            df = self.generate_weather_event_features(df)
            
            # Fill any remaining NaNs from lag/rolling operations
            df = df.ffill().bfill().fillna(0)
            
            final_cols = len(df.columns)
            new_features = final_cols - initial_cols
            
            # Save engineered features
            ensure_dir(os.path.dirname(output_file))
            df.to_csv(output_file, index=False)
            
            self.logger.info("="*70)
            self.logger.info(f"SUCCESS Engineered features saved: {output_file}")
            self.logger.info(f"SUCCESS Shape: {df.shape}")
            self.logger.info(f"SUCCESS New features created: {new_features}")
            self.logger.info("FEATURE ENGINEERING PIPELINE - COMPLETED")
            self.logger.info("="*70)
            
            return df
            
        except Exception as e:
            self.logger.error(f"FAILED Feature engineering failed: {str(e)}")
            raise

# Standalone execution
if __name__ == "__main__":
    import yaml
    
    config_path = 'config/config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    engineer = FeatureEngineer(config)
    
    input_file = os.path.join(config['data']['processed_data_path'], 'processed_data.csv')
    output_file = os.path.join(config['data']['features_path'], 'engineered_features.csv')
    
    features_df = engineer.run_feature_engineering_pipeline(input_file, output_file)
    print(f"\nSUCCESS Feature engineering complete! Output: {output_file}")
