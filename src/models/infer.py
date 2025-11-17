# src/models/infer.py
import pandas as pd
import numpy as np
from src.data.utils import get_target_zones
"""
DATA_PATH = "datasets\processed_data.csv"

# ------------------------
# Forecast & Anomaly Functions
# ------------------------
def get_forecast_outputs():
    
    Compute forecasted surge demand per zone using historical traffic.
    Here we use simple aggregation of Traffic Volume or Completed Trips per zone.
    Returns:
        dict: {zone_name: forecasted_value}
    
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
   
    Detect anomaly zones based on congestion or traffic spikes.
    Here we compare each zone's Traffic Volume to mean and flag if > threshold * mean.
    Returns:
        dict: {zone_name: 0 or 1}  # 1 = anomaly
  
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
"""
import pandas as pd
import logging
import numpy as np
import os
from src.utils import load_csv_data 
# Import the data utility functions to get zones and driver data if needed
from src.data.utils import get_target_zones 

logger = logging.getLogger(__name__)

# Define paths
FEATURES_PATH = 'datasets/engineered_features.csv'
MODEL_PREDICTIONS_PATH = 'forecast_15min_predictions.csv' 
MODEL_PATH = 'models/best_forecast_model.pt' 

def infer_predictions(features_path=FEATURES_PATH, output_path=MODEL_PREDICTIONS_PATH):
    """
    [MEMBER 1's ROLE]
    Loads the trained model (GNN/LSTM/TFT), infers future demand, and saves the output 
    in the standard format (forecast_15min_predictions.csv) needed by Member 2.
    """
    logger.info(f"Starting inference using features from {features_path}...")
    
    # 1. Load Features (Assuming Member 3 has generated features)
    feature_df = load_csv_data(features_path, parse_dates=['timestamp'])
    if feature_df.empty:
        logger.error("Feature data is empty. Cannot run inference.")
        # Ensure an empty file is created so the pipeline doesn't crash
        pd.DataFrame({'zone': [], 'next_time': [], 'pred_bookings_15min': []}).to_csv(output_path, index=False)
        return pd.DataFrame()
        
    # --- TEMPORARY SIMULATION LOGIC: MUST BE REPLACED BY MEMBER 1 (HARSHINI) ---
    # This simulation mimics the GNN/LSTM/TFT predicting demand ('pred_bookings_15min').
    
    last_time = feature_df['timestamp'].max()
    next_time = last_time + pd.Timedelta(minutes=15)
    
    # Get the zones that need a prediction
    zones = feature_df['h3_index'].unique()
    
    # Generate mock predictions (based on recent completed bookings)
    recent_bookings = feature_df.groupby('h3_index')['Bookings'].last()
    
    mock_predictions = pd.DataFrame({
        'h3_index': zones,
        'next_time': next_time,
        # Simulate predictions: 10% variation on the last recorded booking value
        'pred_bookings_15min': recent_bookings.loc[zones].values * np.random.uniform(0.9, 1.1, size=len(zones))
    })
    
    # --- END OF TEMPORARY SIMULATION LOGIC ---
    
    # 2. Save Output in the standard format for Member 2
    mock_predictions = mock_predictions[['h3_index', 'next_time', 'pred_bookings_15min']].fillna(0)
    mock_predictions.to_csv(output_path, index=False)
    logger.info(f"Inference complete. Predictions saved for {len(mock_predictions)} zones to {output_path}")
    
    return mock_predictions