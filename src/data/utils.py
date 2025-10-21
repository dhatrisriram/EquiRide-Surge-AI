# src/data/utils.py
import math
import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0


# ==========================================================
# 🔹 DISTANCE & GEO HELPERS
# ==========================================================
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
    """Compute D×Z distance matrix from driver positions to zone centroids."""
    D, Z = len(drivers), len(zones_df)
    dist = np.zeros((D, Z))
    for i, d in enumerate(drivers):
        for j, z in enumerate(zones_df.itertuples(index=False)):
            dist[i, j] = haversine(d["lat"], d["lon"], z.latitude, z.longitude)
    return dist


# ==========================================================
# 🔹 FAIRNESS & HISTORY UTILITIES
# ==========================================================
def compute_driver_history(history_df):
    """Aggregate earnings per driver (simulate last 24 h window)."""
    df = history_df.groupby("driver_id", as_index=False)["earnings"].sum()
    return dict(zip(df["driver_id"], df["earnings"]))


def compute_fairness_scores(drivers, zones, driver_histories, forecast):
    """
    Compute fairness scores based on inverse earnings × surge forecast.
    Returns D×Z fairness score matrix.
    """
    D, Z = len(drivers), len(zones)
    earnings = np.array(
        [driver_histories.get(d["id"], 0.0) for d in drivers], dtype=float
    )

    # Lower earnings → higher fairness priority
    inv = 1.0 / (1.0 + earnings)
    inv = (inv - inv.min()) / (inv.max() - inv.min() + 1e-9)

    zone_vals = np.array([forecast[z] for z in zones], dtype=float)
    zone_vals = (zone_vals - zone_vals.min()) / (zone_vals.max() - zone_vals.min() + 1e-9)

    return np.outer(inv, zone_vals)


# ==========================================================
# 🔹 ECOLOGICAL / SUSTAINABILITY METRICS
# ==========================================================
def estimate_eco_scores(distance_matrix, emission_factor=0.21):
    """
    Convert distances → eco scores (higher = better sustainability).
    Smaller distances → higher eco score.
    """
    emissions = distance_matrix * emission_factor
    eco = 1 / (1 + emissions)
    return eco


# ==========================================================
# 🔹 STATIC MOCK DATA (for demo/testing)
# ==========================================================
def get_driver_history_mock():
    """
    Return driver surge allocation history.
    Format: {driver_id: {"recent_zone_surges": {zone_id: int}}}
    """
    return {
        "D1": {"recent_zone_surges": {"MG Road": 2, "Indiranagar": 0}},
        "D2": {"recent_zone_surges": {"MG Road": 1, "Indiranagar": 3}},
    }


def get_zone_eco_metrics_mock():
    """
    Return emission-related distance info between driver and zones.
    Format: {(driver_id, zone_id): {"distance_km": float}}
    """
    return {
        ("D1", "MG Road"): {"distance_km": 2.1},
        ("D1", "Indiranagar"): {"distance_km": 2.4},
        ("D2", "MG Road"): {"distance_km": 3.5},
        ("D2", "Indiranagar"): {"distance_km": 1.9},
    }


def get_current_available_drivers_mock():
    """Return list of driver IDs currently available."""
    return ["D1", "D2"]


def get_target_zones_mock():
    """Return list of target zones for repositioning."""
    return ["MG Road", "Indiranagar"]


# ==========================================================
# 🔹 DATA-DRIVEN VARIANTS (using real CSVs)
# ==========================================================
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


def get_driver_history():
    """Simulate recent driver earnings (extend to use engineered_features.csv)."""
    data = {
        "driver_id": [f"driver_{i+1}" for i in range(5)],
        "earnings": np.random.randint(200, 800, 5),
    }
    return pd.DataFrame(data)


def get_zone_eco_metrics():
    """Load processed_data.csv → return distance/eco matrix."""
    df = pd.read_csv("datasets/processed_data.csv")
    zones = (
        df[["Area Name", "latitude", "longitude"]]
        .drop_duplicates()
        .rename(columns={"Area Name": "zone"})
    )
    drivers = get_current_available_drivers(len(zones))
    dist = build_distance_matrix(drivers, zones)
    eco_scores = estimate_eco_scores(dist)
    return eco_scores


# ==========================================================
# 🔹 COMPATIBILITY WRAPPERS (for assignments.py)
# ==========================================================
def get_driver_history_final():
    """Choose between static or data-driven history depending on mode."""
    try:
        df = get_driver_history()
        if isinstance(df, pd.DataFrame):
            return compute_driver_history(df)
    except Exception:
        pass
    return get_driver_history_mock()


def get_zone_eco_metrics_final():
    """Return combined eco metrics (mock fallback)."""
    try:
        eco = get_zone_eco_metrics()
        if isinstance(eco, np.ndarray):
            # Convert distance matrix into pair dictionary if needed
            drivers = get_current_available_drivers_mock()
            zones = get_target_zones_mock()
            eco_dict = {}
            for i, d in enumerate(drivers):
                for j, z in enumerate(zones):
                    eco_dict[(d, z)] = {"distance_km": float(1 / eco[i, j])}
            return eco_dict
    except Exception:
        pass
    return get_zone_eco_metrics_mock()
