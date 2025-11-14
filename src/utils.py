"""
Utility Functions for Data Pipeline
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yaml
import os
import re

import yaml
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_config(config_path="config/config.yaml"):
    """Loads the YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found at {config_path}")
        return {}
    except yaml.YAMLError as exc:
        logger.error(f"Error parsing YAML config: {exc}")
        return {}

def load_csv_data(filepath, index_col=None, parse_dates=False):
    """
    Loads data from a CSV file, handling common errors.
    
    Args:
        filepath (str): The path to the CSV file.
        index_col (str, optional): Column to set as index. Defaults to None.
        parse_dates (list/bool, optional): Columns to parse as dates. Defaults to False.
        
    Returns:
        pd.DataFrame: The loaded DataFrame, or an empty DataFrame on error.
    """
    try:
        # Check if file exists before trying to read
        if not pd.io.common.file_exists(filepath):
            logger.error(f"Data file not found at: {filepath}. Returning empty DataFrame.")
            return pd.DataFrame()
            
        df = pd.read_csv(filepath, index_col=index_col, parse_dates=parse_dates)
        logger.info(f"Successfully loaded data from: {filepath} with {len(df)} rows.")
        return df
    except Exception as e:
        logger.error(f"An error occurred while loading {filepath}: {e}")
        return pd.DataFrame()

def parse_indian_number(value):
    """
    Parse Indian number format
    Examples: '1,10,293' -> 110293, '₹2,92,80,129' -> 29280129
    """
    if pd.isna(value) or value == '':
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    
    # Remove currency symbols and commas
    cleaned = str(value).replace(',', '').replace('₹', '').replace('%', '').strip()
    try:
        return float(cleaned)
    except:
        return np.nan

def clean_percentage(value):
    """Clean percentage values: '39.2%' -> 39.2"""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace('%', '').strip())
    except:
        return np.nan

def create_time_buckets(df, time_col='Date', bucket_minutes=15):
    """Create time buckets for temporal aggregation"""
    df[time_col] = pd.to_datetime(df[time_col])
    df['time_bucket'] = df[time_col].dt.floor(f'{bucket_minutes}T')
    return df

def detect_anomalies_zscore(series, threshold=2.5):
    """
    Detect anomalies using Z-score method
    Returns boolean array where True indicates anomaly
    """
    if len(series) < 3:
        return pd.Series([False] * len(series), index=series.index)
    
    mean = series.mean()
    std = series.std()
    
    if std == 0:
        return pd.Series([False] * len(series), index=series.index)
    
    z_scores = np.abs((series - mean) / std)
    return z_scores > threshold

def detect_anomalies_iqr(series, multiplier=1.5):
    """
    Detect anomalies using IQR method
    Returns boolean array where True indicates anomaly
    """
    if len(series) < 4:
        return pd.Series([False] * len(series), index=series.index)
    
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return (series < lower_bound) | (series > upper_bound)

def ensure_dir(directory):
    """Ensure directory exists, create if not"""
    os.makedirs(directory, exist_ok=True)

def get_timestamp(format='%Y%m%d_%H%M%S'):
    """Get current timestamp as formatted string"""
    return datetime.now().strftime(format)

def calculate_demand_supply_ratio(searches, completed_trips):
    """Calculate demand/supply ratio safely"""
    return searches / (completed_trips + 1)

def safe_divide(numerator, denominator, default=0.0):
    import numpy as np
    import pandas as pd
    
    numerator = pd.Series(numerator) if not isinstance(numerator, pd.Series) else numerator
    denominator = pd.Series(denominator) if not isinstance(denominator, pd.Series) else denominator
    
    result = numerator / denominator.replace(0, np.nan)
    result = result.fillna(default)
    return result


def get_peak_hours():
    """Return list of peak hours (8-10 AM, 6-8 PM)"""
    return [8, 9, 18, 19]

def get_time_category(hour):
    """Categorize hour into time periods"""
    if 0 <= hour < 6:
        return 'night'
    elif 6 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 18:
        return 'afternoon'
    elif 18 <= hour < 24:
        return 'evening'
    else:
        return 'unknown'
