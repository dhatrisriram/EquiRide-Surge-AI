"""
Utility Functions for Data Pipeline
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yaml
import os
import re

def load_config(config_path='config/config.yaml'):
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

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
