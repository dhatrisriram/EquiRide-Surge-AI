import os
import math
import numpy as np
import pandas as pd
import logging
import random
from src.utils import load_csv_data # Rely on the robust loader

logger = logging.getLogger(__name__)

# --- Data Source Paths (Centralized) ---
PROCESSED_DATA_PATH = 'datasets/processed_data.csv'
GRAPH_EDGES_PATH = 'datasets/graph_edges.csv'
FORECAST_FILE = 'datasets/forecast_15min_predictions.csv' 


EARTH_RADIUS_KM = 6371.0


def haversine(lat1, lon1, lat2, lon2):
    """Compute great-circle distance (km) between two lat/lon points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dlambda = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def build_distance_matrix(drivers, zones_df):
    """
    Compute D×Z distance matrix from driver positions to zone centroids.
    NOTE: Currently unused, as Eco-Metrics are simulated.
    """
    D, Z = len(drivers), len(zones_df)
    dist = np.zeros((D, Z), dtype=float)
    for i, d in enumerate(drivers):
        for j, z in enumerate(zones_df.itertuples(index=False)):
            lat_z = getattr(z, "latitude", None) or getattr(z, "lat", None) or getattr(z, "y", None)
            lon_z = getattr(z, "longitude", None) or getattr(z, "lon", None) or getattr(z, "x", None)
            if lat_z is None or lon_z is None:
                dist[i, j] = 1.0  # fallback
            else:
                dist[i, j] = haversine(d["lat"], d["lon"], lat_z, lon_z)
    return dist


# ---------------------------
# CORE INTERFACE FUNCTIONS (Used by assignments.py)
# ---------------------------

def get_current_available_drivers() -> list[dict]:
    """
    [MEMBER 3's RESPONSIBILITY - SIMULATED]
    Fetches the current list of available drivers.
    Simulated using the latest zone data from the processed file.
    """
    processed_df = load_csv_data(PROCESSED_DATA_PATH, parse_dates=['Datetime'])
    if processed_df.empty:
        logger.error("Processed data empty. Cannot simulate drivers.")
        return []

    # Use the unique zones from the latest time step
    latest_time_df = processed_df[processed_df['Datetime'] == processed_df['Datetime'].max()]
    available_zones = latest_time_df['zone'].unique()
    
    num_drivers = 100
    drivers = []
    
    for i in range(num_drivers):
        # Simulate driver location and type
        zone = random.choice(available_zones) if available_zones.size > 0 else "unknown"
        drivers.append({
            "id": f"D{i:03d}",
            "current_zone": zone,
            "vehicle_type": random.choice(["auto", "car"]),
            # Add dummy lat/lon for distance calculation (if ever used)
            "lat": np.random.uniform(12.9, 13.1), 
            "lon": np.random.uniform(77.5, 77.7),
        })
    
    logger.info(f"Simulated {num_drivers} current available drivers.")
    return drivers


def get_target_zones() -> list[str]:
    """
    [MEMBER 3's RESPONSIBILITY - IMPLEMENTED]
    Fetches the list of zones that require prediction/repositioning.
    We get this from the zones present in the latest forecast file.
    """
    forecast_df = load_csv_data(FORECAST_FILE)
    if forecast_df.empty:
        logger.warning(f"Forecast file {FORECAST_FILE} empty. No target zones available.")
        return []
    
    zones = forecast_df['zone'].unique().tolist()
    logger.info(f"Identified {len(zones)} target zones from forecast.")
    return zones


def get_driver_history_final() -> dict:
    """
    [MEMBER 3's RESPONSIBILITY - SIMULATED]
    Fetches historical data per driver for the Fairness calculation.
    """
    drivers = get_current_available_drivers()
    history = {}
    
    # We use the raw zone data to get all possible zones for simulation consistency
    all_zones = get_target_zones()

    for driver in drivers:
        history[driver["id"]] = {
            # Simulate earnings for fairness score calculation
            "earnings": np.random.normal(500.0, 200.0).clip(50.0), 
            # Simulate recent surges handled
            "recent_zone_surges": {
                z: random.randint(0, 5) 
                for z in random.sample(all_zones, k=min(3, len(all_zones)))
            }
        }
    logger.info("Simulated driver history for fairness calculation.")
    return history


def get_zone_eco_metrics_final() -> np.ndarray:
    """
    [MEMBER 3's RESPONSIBILITY - SIMULATED]
    Fetches the driver-to-zone distance matrix (km) for Eco/Distance calculation.
    
    NOTE: This is SIMULATED because 'graph_edges.csv' only contains topology (src_zone, dst_zone),
    not the actual distances/times needed for the cost matrix.
    """
    drivers = get_current_available_drivers()
    zones = get_target_zones()
    n, m = len(drivers), len(zones)
    
    if n == 0 or m == 0:
        return np.array([])
    
    # --- SIMULATION: REPLACE WITH LOOKUP FROM ENRICHED graph_edges.csv ---
    distance_matrix = np.random.uniform(1.0, 15.0, size=(n, m))
    
    # Ensure drivers already in the target zone have low distance (e.g., 0.1km)
    driver_zone_map = {d["id"]: d["current_zone"] for d in drivers}
    
    for i in range(n):
        for j in range(m):
            if driver_zone_map.get(drivers[i]["id"]) == zones[j]:
                distance_matrix[i, j] = 0.1 
    
    logger.info(f"Simulated Eco/Distance Matrix shape: {distance_matrix.shape}")
    return distance_matrix


def get_forecast_outputs() -> dict:
    """
    [MEMBER 1's HAND-OFF - IMPLEMENTED]
    Fetches the predicted bookings/demand from the CSV generated by infer.py.
    """
    forecast_df = load_csv_data(FORECAST_FILE)
    if forecast_df.empty:
        logger.error("Forecast output is missing or empty. Returning empty dict.")
        return {}
    
    # The expected column is 'pred_bookings_15min' (output of GNN/LSTM/TFT)
    forecast_map = forecast_df.set_index('zone')['pred_bookings_15min'].to_dict()
    logger.info(f"Successfully loaded {len(forecast_map)} forecast values.")
    return forecast_map

def get_zone_anomaly_flags() -> dict:
    """
    [MEMBER 1's HAND-OFF - SIMULATED]
    Fetches anomaly/event flags (e.g., from Member 3's event integration).
    """
    zones = get_target_zones()
    # Simulate a few zones having an event/anomaly
    anomaly_flags = {
        z: random.choice([0, 0, 0, 1]) # 25% chance of flag
        for z in zones
    }
    logger.info(f"Simulated anomaly flags for {len(zones)} zones.")
    return anomaly_flags