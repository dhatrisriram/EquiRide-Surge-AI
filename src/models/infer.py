# src/models/infer.py
import pandas as pd
import numpy as np
from src.data.utils import get_target_zones

DATA_PATH = "datasets\processed_data.csv"

# ------------------------
# Forecast & Anomaly Functions
# ------------------------
def get_forecast_outputs():
    """
    Compute forecasted surge demand per zone using historical traffic.
    Here we use simple aggregation of Traffic Volume or Completed Trips per zone.
    Returns:
        dict: {zone_name: forecasted_value}
    """
    df = pd.read_csv(DATA_PATH)
    zones = get_target_zones()

    forecast = {}
    for z in zones:
        zone_df = df[df["Ward"].str.lower() == z.lower()]
        # Use Completed Trips as proxy for demand
        if not zone_df.empty:
            forecast[z] = zone_df["Completed Trips"].mean()
        else:
            forecast[z] = 0
    return forecast


def get_zone_anomaly_flags(threshold=1.5):
    """
    Detect anomaly zones based on congestion or traffic spikes.
    Here we compare each zone's Traffic Volume to mean and flag if > threshold * mean.
    Returns:
        dict: {zone_name: 0 or 1}  # 1 = anomaly
    """
    df = pd.read_csv(DATA_PATH)
    zones = get_target_zones()

    anomaly_flags = {}
    for z in zones:
        zone_df = df[df["Ward"].str.lower() == z.lower()]
        if zone_df.empty:
            anomaly_flags[z] = 0
            continue
        avg_volume = zone_df["Traffic Volume"].mean()
        std_volume = zone_df["Traffic Volume"].std() if not np.isnan(zone_df["Traffic Volume"].std()) else 0
        # latest value
        latest_volume = zone_df.iloc[-1]["Traffic Volume"]
        # flag as anomaly if significantly higher than historical mean
        anomaly_flags[z] = 1 if latest_volume > avg_volume + threshold * std_volume else 0
    return anomaly_flags
