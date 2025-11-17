# src/optimization/assignments.py
import logging
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from src.data.utils import (
    get_driver_history_final as get_driver_history,
    get_zone_eco_metrics_final as get_zone_eco_metrics,
    get_current_available_drivers,
    get_target_zones,
    get_forecast_outputs,
    get_zone_anomaly_flags
    )

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ------------------------
# Configurable Weights
# ------------------------
DEFAULT_WEIGHTS = {"w_profit": 1.5, "w_fair": 1.0, "w_eco": 0.8}


# ------------------------
# Utility Functions
# ------------------------
def calculate_emission(distance_km, vehicle_type="auto"):
    """Calculate estimated CO2 emissions (kg) for a given distance and vehicle type."""
    emission_factors = {"auto": 0.13, "car": 0.16}  # kg CO2/km
    return distance_km * emission_factors.get(vehicle_type, 0.13)


def calculate_trip_fare(distance_km: float, vehicle_type="auto") -> float:
    """
    Calculate realistic fare per trip based on vehicle type and distance.
    """
    # Base auto fare
    base_fare = 30.0
    base_distance = 2.0
    per_km_rate = 11.0  # average of 10-12

    if distance_km <= base_distance:
        fare = base_fare
    else:
        extra_distance = distance_km - base_distance
        fare = base_fare + extra_distance * per_km_rate

    # Adjust fare based on vehicle type
    if vehicle_type == "auto":
        return fare
    elif vehicle_type == "car":
        return fare * 1.5  # 1.5x auto fare
    elif vehicle_type == "taxi":
        return fare * 2.0  # double auto fare
    else:
        return fare  # default to auto fare



def fairness_score(driver_id, zone_id, driver_history, zone_demand):
    """
    Compute fairness score.
    """
    val = driver_history.get(driver_id, None) if isinstance(driver_history, dict) else None
    if isinstance(val, dict):
        surge_count = val.get("recent_zone_surges", {}).get(zone_id, 0)
        score = 1.0 / (1.0 + surge_count)
    else:
        earnings = val if isinstance(val, (int, float)) else 0.0
        score = 1.0 / (1.0 + earnings)
    return score * zone_demand


def normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.array(arr, dtype=float)
    if arr.size == 0:
        return arr
    mn, mx = arr.min(), arr.max()
    if np.isclose(mx, mn):
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


# ------------------------
# Core Cost Function
# ------------------------
def build_cost_matrix(drivers, zones, forecast, anomaly_flags, driver_history, eco_data, weights=None):
    if weights is None:
        weights = DEFAULT_WEIGHTS

    n, m = len(drivers), len(zones)

    # Ensure eco_data is correct shape
    if isinstance(eco_data, np.ndarray):
        eco_matrix = eco_data
        if eco_matrix.shape != (n, m):
            logger.warning("eco_data shape %s doesn't match (drivers,zones)=%d,%d. Using ones.", eco_matrix.shape, n, m)
            eco_matrix = np.ones((n, m))
    else:
        eco_matrix = np.ones((n, m))

    # Priority based on forecast + anomaly
    base_priority = np.array([forecast.get(z, 0) for z in zones], dtype=float)
    event_boost = np.array([anomaly_flags.get(z, 0) * base_priority[i] * 0.3 for i, z in enumerate(zones)], dtype=float)
    priority = base_priority + event_boost

    cost_matrix = np.zeros((n, m))
    max_priority = np.max(priority)
    ptp_priority = np.ptp(priority) + 1e-5

    for i, driver in enumerate(drivers):
        driver_id = driver["id"]
        for j, zone_id in enumerate(zones):
            # Emission / distance
            dist = float(eco_matrix[i, j])
            emissions = calculate_emission(dist)

            # Fairness
            fair = fairness_score(driver_id, zone_id, driver_history, priority[j])

            # Profit term (higher forecast -> lower cost)
            profit_term = (max_priority - priority[j]) / ptp_priority

            # Eco term normalized
            all_emissions = [calculate_emission(float(eco_matrix[i, z_idx])) for z_idx in range(m)]
            max_emission = max(all_emissions) if all_emissions else 1.0
            eco_term = emissions / max_emission if max_emission != 0 else 0.0

            # Fairness term normalized
            all_fairs = [fairness_score(d["id"], zone_id, driver_history, priority[j]) for d in drivers]
            max_fair = max(all_fairs) if all_fairs else 1.0
            fair_term = (max_fair - fair) / (max_fair + 1e-5)

            # Total cost
            cost_matrix[i, j] = weights["w_profit"]*profit_term + weights["w_eco"]*eco_term + weights["w_fair"]*fair_term

    return cost_matrix


# ------------------------
# Main Assignment Function
# ------------------------
def assign_drivers(drivers, zones, forecast, anomaly_flags, driver_history, eco_data, weights=None):
    cost_matrix = build_cost_matrix(drivers, zones, forecast, anomaly_flags, driver_history, eco_data, weights)

    # Hungarian assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    results = []
    for i, j in zip(row_ind, col_ind):
        driver = drivers[i]
        zone = zones[j]
        dist_km = float(eco_data[i, j]) if isinstance(eco_data, np.ndarray) else 1.0
        emission_kg = calculate_emission(dist_km)

        # Realistic profit using average trip fare
        trips_est = forecast.get(zone, 0)
        vehicle_type = driver.get("vehicle_type", "auto")
        fare_per_trip = calculate_trip_fare(dist_km, vehicle_type=vehicle_type)
        profit_est = trips_est * fare_per_trip  # realistic profit estimate

        results.append({
            "driver_id": driver["id"],
            "assigned_zone": zone,
            "profit_est": profit_est,
            "anomaly_flag": anomaly_flags.get(zone, 0),
            "distance_km": dist_km,
            "emission_kg": emission_kg
        })

    df = pd.DataFrame(results)
    logger.info("Assignments completed: %d drivers assigned.", len(df))
    return df


# ------------------------
# Pipeline Integration
# ------------------------
def get_repositioning_plan():
    drivers = get_current_available_drivers()
    zones = get_target_zones()
    forecast = get_forecast_outputs()
    anomaly_flags = get_zone_anomaly_flags()
    driver_history = get_driver_history()
    eco_data = get_zone_eco_metrics()  # shape (len(drivers), len(zones))

    assignments = assign_drivers(drivers, zones, forecast, anomaly_flags, driver_history, eco_data)
    return assignments
