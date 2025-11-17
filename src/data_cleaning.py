"""
Data Cleaning and Alignment Module
Handles preprocessing of cleaned_bangalore_traffic_wards.csv
"""
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.logging_config import setup_logging, log_stage
from src.utils import parse_indian_number, clean_percentage, ensure_dir

class DataCleaner:
    def __init__(self, config, logger=None):
        """Initialize Data Cleaner"""
        self.config = config
        self.logger = logger or setup_logging()
        
    def load_data(self, file_path):
        """Load raw CSV data"""
        try:
            log_stage(self.logger, 'LOAD_DATA', 'START', file=file_path)
            df = pd.read_csv(file_path)
            log_stage(self.logger, 'LOAD_DATA', 'SUCCESS', rows=len(df), cols=len(df.columns))
            return df
        except Exception as e:
            log_stage(self.logger, 'LOAD_DATA', 'FAILURE', error=str(e))
            raise
    
    def clean_numeric_columns(self, df):
        """Clean columns with Indian number formatting"""
        try:
            log_stage(self.logger, 'CLEAN_NUMERIC', 'START')
            
            # Indian format columns (with commas)
            indian_cols = [
                'Searches', 'Searches which got estimate', 'Searches for Quotes',
                'Searches which got Quotes', 'Bookings', 'Completed Trips',
                'Cancelled Bookings', "Drivers' Earnings", 'Distance Travelled (km)'
            ]
            
            for col in indian_cols:
                if col in df.columns:
                    df[col] = df[col].apply(parse_indian_number)
            
            # Percentage columns
            pct_cols = [
                'Search-to-estimate Rate', 'Estimate-to-search for quotes Rate',
                'Quote Acceptance Rate', 'Quote-to-booking Rate',
                'Booking Cancellation Rate', 'Conversion Rate'
            ]
            
            for col in pct_cols:
                if col in df.columns:
                    df[col] = df[col].apply(clean_percentage)
            
            # Average Fare per Trip
            if 'Average Fare per Trip' in df.columns:
                df['Average Fare per Trip'] = df['Average Fare per Trip'].apply(parse_indian_number)
            
            log_stage(self.logger, 'CLEAN_NUMERIC', 'SUCCESS')
            return df
        except Exception as e:
            log_stage(self.logger, 'CLEAN_NUMERIC', 'FAILURE', error=str(e))
            raise
    
    def handle_missing_values(self, df):
        """Handle missing values with appropriate strategies"""
        try:
            log_stage(self.logger, 'HANDLE_MISSING', 'START')
            
            initial_missing = df.isnull().sum().sum()
            
            # Forward/backward fill for time-series columns
            ts_cols = ['Traffic Volume', 'Average Speed', 'Congestion Level']
            for col in ts_cols:
                if col in df.columns:
                    df[col] = df[col].ffill().bfill()
            
            # Median for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().sum() > 0:
                    df[col] = df[col].fillna(df[col].median())
            
            # Mode for categorical columns
            cat_cols = df.select_dtypes(include=['object']).columns
            for col in cat_cols:
                if df[col].isnull().sum() > 0:
                    mode_val = df[col].mode()
                    if len(mode_val) > 0:
                        df[col] = df[col].fillna(mode_val[0])
                    else:
                        df[col] = df[col].fillna('Unknown')
            
            final_missing = df.isnull().sum().sum()
            log_stage(self.logger, 'HANDLE_MISSING', 'SUCCESS', 
                     filled=initial_missing-final_missing)
            return df
        except Exception as e:
            log_stage(self.logger, 'HANDLE_MISSING', 'FAILURE', error=str(e))
            raise
    
    def parse_datetime(self, df, date_col='Date'):
        """Parse and extract datetime features"""
        try:
            log_stage(self.logger, 'PARSE_DATETIME', 'START')
            
            df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')

            # Drop or handle unparsable rows if any
            invalid_dates = df[date_col].isna().sum()
            if invalid_dates > 0:
                self.logger.warning(f"[PARSE_DATETIME] {invalid_dates} invalid dates found; dropping them.")
                df = df.dropna(subset=[date_col])

            df['year'] = df[date_col].dt.year
            df['month'] = df[date_col].dt.month
            df['day'] = df[date_col].dt.day
            df['day_of_week'] = df[date_col].dt.dayofweek  # 0=Monday, 6=Sunday
            df['hour'] = df[date_col].dt.hour
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            # Create time bucket
            bucket_min = self.config['features']['time_buckets_minutes']
            df['time_bucket'] = df[date_col].dt.floor(f'{bucket_min}T')
            
            df = df.rename(columns={'Date': 'timestamp'})

            log_stage(self.logger, 'PARSE_DATETIME', 'SUCCESS')
            return df
        except Exception as e:
            log_stage(self.logger, 'PARSE_DATETIME', 'FAILURE', error=str(e))
            raise
    
    def remove_duplicates(self, df):
        """Remove duplicate rows"""
        try:
            log_stage(self.logger, 'REMOVE_DUPLICATES', 'START')
            
            initial = len(df)
            df = df.drop_duplicates()
            removed = initial - len(df)
            
            log_stage(self.logger, 'REMOVE_DUPLICATES', 'SUCCESS', removed=removed)
            return df
        except Exception as e:
            log_stage(self.logger, 'REMOVE_DUPLICATES', 'FAILURE', error=str(e))
            raise
    
    def align_spatial_temporal(self, df):
        """Ensure spatial (H3) and temporal consistency"""
        try:
            log_stage(self.logger, 'ALIGN_SPATIAL_TEMPORAL', 'START')
            
            # Ensure h3_index is string
            df['h3_index'] = df['h3_index'].astype(str)
            
            # Sort by h3_index and time
            df = df.sort_values(['h3_index', 'timestamp']).reset_index(drop=True)
            
            log_stage(self.logger, 'ALIGN_SPATIAL_TEMPORAL', 'SUCCESS')
            return df
        except Exception as e:
            log_stage(self.logger, 'ALIGN_SPATIAL_TEMPORAL', 'FAILURE', error=str(e))
            raise
    
    def run_cleaning_pipeline(self, input_file, output_file):
        """Execute complete cleaning pipeline"""
        try:
            self.logger.info("="*70)
            self.logger.info("DATA CLEANING PIPELINE - STARTED")
            self.logger.info("="*70)
            
            # Load data
            df = self.load_data(input_file)
            
            # Clean numeric columns
            df = self.clean_numeric_columns(df)
            
            # Handle missing values
            df = self.handle_missing_values(df)
            
            # Parse datetime
            df = self.parse_datetime(df)
            
            # Remove duplicates
            df = self.remove_duplicates(df)
            
            # Align spatial-temporal
            df = self.align_spatial_temporal(df)
            
            # Save processed data
            ensure_dir(os.path.dirname(output_file))
            df.to_csv(output_file, index=False)
            
            self.logger.info("="*70)
            self.logger.info(f"SUCCESS Cleaned data saved: {output_file}")
            self.logger.info(f"SUCCESS Shape: {df.shape}")
            self.logger.info("DATA CLEANING PIPELINE - COMPLETED")
            self.logger.info("="*70)
            
            return df
            
        except Exception as e:
            self.logger.error(f"FAILED Cleaning pipeline failed: {str(e)}")
            raise

# Standalone execution
if __name__ == "__main__":
    import yaml
    
    config_path = 'config/config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    cleaner = DataCleaner(config)
    
    input_file = config['data']['input_csv']
    output_file = os.path.join(config['data']['processed_data_path'], 'processed_data.csv')
    
    cleaned_df = cleaner.run_cleaning_pipeline(input_file, output_file)
    print(f"\nCleaning complete! Output: {output_file}")
