# src/data/utils.py
import os
import math
import numpy as np
import pandas as pd

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
    """Compute D×Z distance matrix from driver positions to zone centroids.
    drivers: list of dicts with keys 'id','lat','lon'
    zones_df: DataFrame with columns ['zone','latitude','longitude'] (or similar)
    """
    D, Z = len(drivers), len(zones_df)
    dist = np.zeros((D, Z), dtype=float)
    for i, d in enumerate(drivers):
        for j, z in enumerate(zones_df.itertuples(index=False)):
            # zones_df expected to have attributes matching latitude/longitude names
            # try common names
            lat_z = getattr(z, "latitude", None) or getattr(z, "lat", None) or getattr(z, "y", None)
            lon_z = getattr(z, "longitude", None) or getattr(z, "lon", None) or getattr(z, "x", None)
            if lat_z is None or lon_z is None:
                dist[i, j] = 1.0  # fallback
            else:
                dist[i, j] = haversine(d["lat"], d["lon"], lat_z, lon_z)
    return dist


# ---------------------------
# ECO and FAIRNESS UTILITIES
# ---------------------------
def estimate_eco_scores(distance_matrix, emission_factor=0.21):
    """
    Convert distances → eco scores (higher = better sustainability).
    Smaller distances → higher eco score.
    """
    emissions = distance_matrix * emission_factor
    eco = 1 / (1 + emissions)
    return eco


# ---------------------------
# MOCK / STATIC DATA
# ---------------------------
def get_current_available_drivers_mock():
    """Mock: list of driver dicts"""
    return [
        {"id": "D1", "lat": 12.92, "lon": 77.64},
        {"id": "D2", "lat": 12.93, "lon": 77.63},
    ]


def get_target_zones_mock():
    return ["MG Road", "Indiranagar"]


def get_driver_history_mock():
    """Mock nested structure used by older fairness logic."""
    return {
        "D1": {"recent_zone_surges": {"MG Road": 2, "Indiranagar": 0}},
        "D2": {"recent_zone_surges": {"MG Road": 1, "Indiranagar": 3}},
    }


def get_zone_eco_metrics_mock():
    """Mock dict (legacy fallback)."""
    return {
        ("D1", "MG Road"): {"distance_km": 2.1},
        ("D1", "Indiranagar"): {"distance_km": 2.4},
        ("D2", "MG Road"): {"distance_km": 3.5},
        ("D2", "Indiranagar"): {"distance_km": 1.9},
    }


# ---------------------------
# REAL (DATA-DRIVEN) FUNCTIONS
# ---------------------------
def get_current_available_drivers(n=5):
    """Simulate current drivers with random GPS coords (for testing)."""
    np.random.seed(42)
    drivers = []
    for i in range(n):
        drivers.append(
            {
                "id": f"driver_{i+1}",
                "lat": 12.90 + np.random.uniform(0, 0.05),
                "lon": 77.60 + np.random.uniform(0, 0.05),
            }
        )
    return drivers


def get_zone_eco_metrics():
    """
    Build and return a driver×zone distance matrix (numpy array).
    - Loads zones from datasets/processed_data.csv (Area Name, latitude, longitude).
    - Builds a drivers list using get_current_available_drivers() (same default n as pipeline).
    Returns: np.ndarray shape (len(drivers), len(zones))
    """
    dataset_path = "datasets/processed_data.csv"
    # If dataset missing, return a small default matrix to avoid crashes
    if not os.path.exists(dataset_path):
        # fallback: create small matrix using mock drivers and mock zones
        drivers = get_current_available_drivers_mock()
        zones = get_target_zones_mock()
        return np.ones((len(drivers), len(zones)), dtype=float)

    df = pd.read_csv(dataset_path)
    if "Area Name" not in df.columns or "latitude" not in df.columns or "longitude" not in df.columns:
        # fallback
        drivers = get_current_available_drivers_mock()
        zones = get_target_zones_mock()
        return np.ones((len(drivers), len(zones)), dtype=float)

    # zones DataFrame with lat/lon
    zones_df = df[["Area Name", "latitude", "longitude"]].drop_duplicates()
    zones_df = zones_df.rename(columns={"Area Name": "zone", "latitude": "latitude", "longitude": "longitude"})
    # Build drivers using same source as pipeline (no n passed so defaults match)
    drivers = get_current_available_drivers()  # important: uses same default n
    # Compute distances
    dist_matrix = build_distance_matrix(drivers, zones_df)
    # Return raw distances (in km). Caller can convert to emissions/eco if needed.
    return dist_matrix


# ---------------------------
# COMPATIBILITY WRAPPERS (for assignments.py)
# ---------------------------
def get_driver_history_final():
    """
    Return driver history in the form expected by downstream code.
    If a DataFrame is available, convert to nested dict with 'recent_zone_surges' placeholder.
    Otherwise return mock.
    """
    # try a real DF source if present on your pipeline (this function can be adapted)
    try:
        # sample attempt to read a real source; if not available, fallback
        # keep simple: simulate the DF and convert to nested structure
        df = pd.DataFrame({"driver_id": [f"driver_{i+1}" for i in range(5)], "earnings": np.random.randint(200, 800, 5)})
        history = {}
        for _, row in df.iterrows():
            # preserve the nested recent_zone_surges structure for compatibility
            history[row["driver_id"]] = {"recent_zone_surges": {}}
        return history
    except Exception:
        return get_driver_history_mock()


def get_zone_eco_metrics_final():
    """
    Return eco_data as a numpy array (drivers x zones) when possible.
    Fallbacks:
      - If get_zone_eco_metrics() returns array -> pass through.
      - If exception occurs -> return a ones matrix sized to (len(drivers), len(zones)).
    """
    try:
        eco = get_zone_eco_metrics()
        if isinstance(eco, np.ndarray):
            return eco
    except Exception:
        pass

    # Fallback: attempt to build a ones matrix sized to the drivers & zones used by pipeline
    drivers = get_current_available_drivers()
    # Try to read zones from CSV; if not available, use mock zones
    dataset_path = "datasets/processed_data.csv"
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
        if "Area Name" in df.columns:
            zones = df["Area Name"].dropna().unique().tolist()
        else:
            zones = get_target_zones_mock()
    else:
        zones = get_target_zones_mock()

    return np.ones((len(drivers), len(zones)), dtype=float)


def get_target_zones():
    """Return list of zones; prefer CSV, fallback to mock."""
    dataset_path = "datasets/processed_data.csv"
    if not os.path.exists(dataset_path):
        return get_target_zones_mock()
    df = pd.read_csv(dataset_path)
    if "Area Name" not in df.columns:
        return get_target_zones_mock()
    return df["Area Name"].dropna().unique().tolist()
