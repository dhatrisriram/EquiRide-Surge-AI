"""
Unified Utility Module for EquiRide Surge AI
Includes both:
 - Member 2 (Repositioning, Fairness, Sustainability)
 - Member 3 (Data Cleaning, Feature Engineering, Anomaly Detection)
"""

import os
import re
import math
import yaml
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# Set up logging for module
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
        df = pd.read_csv(filepath, index_col=index_col, parse_dates=parse_dates)
        logger.info(f"Successfully loaded data from: {filepath} with {len(df)} rows.")
        return df
    except FileNotFoundError:
        logger.error(f"Data file not found at: {filepath}. Returning empty DataFrame.")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"An error occurred while loading {filepath}: {e}")
        return pd.DataFrame()
# =====================================================================
# 🧩 CONFIG & FILE UTILITIES
# =====================================================================

def load_config(config_path='config/config.yaml'):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def ensure_dir(directory):
    """Ensure directory exists, create if not."""
    os.makedirs(directory, exist_ok=True)

def get_timestamp(format='%Y%m%d_%H%M%S'):
    """Return current timestamp string."""
    return datetime.now().strftime(format)

# =====================================================================
# 🧮 DATA CLEANING HELPERS (Member 3)
# =====================================================================

def parse_indian_number(value):
    """Parse Indian number format. Example: '₹2,92,80,129' → 29280129"""
    if pd.isna(value) or value == '':
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = str(value).replace(',', '').replace('₹', '').replace('%', '').strip()
    try:
        return float(cleaned)
    except ValueError:
        return np.nan

def clean_percentage(value):
    """Convert strings like '39.2%' → 39.2."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace('%', '').strip())
    except ValueError:
        return np.nan

# =====================================================================
# ⏱️ TIME UTILITIES (Member 3)
# =====================================================================

def create_time_buckets(df, time_col='Date', bucket_minutes=15):
    """Create time buckets for temporal aggregation."""
    df[time_col] = pd.to_datetime(df[time_col])
    df['time_bucket'] = df[time_col].dt.floor(f'{bucket_minutes}T')
    return df

def get_peak_hours():
    """Return list of typical peak hours (8–10 AM, 6–8 PM)."""
    return [8, 9, 18, 19]

def get_time_category(hour):
    """Categorize hour into time periods."""
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

# =====================================================================
# ⚠️ ANOMALY DETECTION (Member 3)
# =====================================================================

def detect_anomalies_zscore(series, threshold=2.5):
    """Detect anomalies using Z-score method."""
    if len(series) < 3:
        return pd.Series([False] * len(series), index=series.index)
    mean = series.mean()
    std = series.std()
    if std == 0:
        return pd.Series([False] * len(series), index=series.index)
    z_scores = np.abs((series - mean) / std)
    return z_scores > threshold

def detect_anomalies_iqr(series, multiplier=1.5):
    """Detect anomalies using IQR method."""
    if len(series) < 4:
        return pd.Series([False] * len(series), index=series.index)
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - multiplier * IQR, Q3 + multiplier * IQR
    return (series < lower) | (series > upper)

# =====================================================================
# 📊 MATHEMATICAL HELPERS (Member 3)
# =====================================================================

def calculate_demand_supply_ratio(searches, completed_trips):
    """Safely compute demand/supply ratio."""
    return searches / (completed_trips + 1)

def safe_divide(numerator, denominator, default=0.0):
    """Safely divide two arrays/series."""
    numerator = pd.Series(numerator) if not isinstance(numerator, pd.Series) else numerator
    denominator = pd.Series(denominator) if not isinstance(denominator, pd.Series) else denominator
    result = numerator / denominator.replace(0, np.nan)
    return result.fillna(default)

# =====================================================================
# 🚕 REPOSITIONING / FAIRNESS / ECO HELPERS (Member 2)
# =====================================================================

EARTH_RADIUS_KM = 6371.0

def haversine(lat1, lon1, lat2, lon2):
    """Compute distance (km) between two geo-coordinates."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dlambda = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def build_distance_matrix(drivers, zones_df):
    """Compute D×Z distance matrix from driver to zone centroid."""
    D, Z = len(drivers), len(zones_df)
    dist = np.zeros((D, Z))
    for i, d in enumerate(drivers):
        for j, z in enumerate(zones_df.itertuples(index=False)):
            dist[i, j] = haversine(d["lat"], d["lon"], z.latitude, z.longitude)
    return dist

def compute_driver_history(history_df):
    """Aggregate earnings per driver (simulate last 24 h window)."""
    df = history_df.groupby("driver_id", as_index=False)["earnings"].sum()
    return dict(zip(df["driver_id"], df["earnings"]))

def compute_fairness_scores(drivers, zones, driver_histories, forecast):
    """Compute fairness scores (low earnings → high fairness)."""
    D, Z = len(drivers), len(zones)
    earnings = np.array([driver_histories.get(d["id"], 0.0) for d in drivers], dtype=float)
    inv = 1.0 / (1.0 + earnings)
    inv = (inv - inv.min()) / (inv.max() - inv.min() + 1e-9)
    zone_vals = np.array([forecast[z] for z in zones], dtype=float)
    zone_vals = (zone_vals - zone_vals.min()) / (zone_vals.max() - zone_vals.min() + 1e-9)
    return np.outer(inv, zone_vals)

def estimate_eco_scores(distance_matrix, emission_factor=0.21):
    """Convert distances → eco scores (higher = better)."""
    emissions = distance_matrix * emission_factor
    return 1 / (1 + emissions)

def get_current_available_drivers(n=5):
    """Dummy generator for available drivers (simulate GPS positions)."""
    np.random.seed(42)
    drivers = []
    for i in range(n):
        drivers.append({
            "id": f"driver_{i+1}",
            "lat": 12.90 + np.random.uniform(0, 0.05),
            "lon": 77.60 + np.random.uniform(0, 0.05),
        })
    return drivers

def get_driver_history():
    """Simulate recent driver earnings (use engineered_features.csv in real pipeline)."""
    data = {
        "driver_id": [f"driver_{i+1}" for i in range(5)],
        "earnings": np.random.randint(200, 800, 5),
    }
    return pd.DataFrame(data)

def get_zone_eco_metrics():
    """Load processed_data.csv → return distance/eco matrix."""
    df = pd.read_csv("datasets/processed_data.csv")
    zones = df[["Area Name", "latitude", "longitude"]].drop_duplicates().rename(
        columns={"Area Name": "zone", "latitude": "latitude", "longitude": "longitude"}
    )
    drivers = get_current_available_drivers(len(zones))
    dist = build_distance_matrix(drivers, zones)
    eco_scores = estimate_eco_scores(dist)
    return eco_scores


