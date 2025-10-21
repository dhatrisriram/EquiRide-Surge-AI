# src/optimization/assignments.py
import logging
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from src.data.utils import (
    get_driver_history_final as get_driver_history,
    get_zone_eco_metrics_final as get_zone_eco_metrics,
    get_current_available_drivers_mock as get_current_available_drivers,
    get_target_zones_mock as get_target_zones,
)

from src.models.infer import get_forecast_outputs, get_zone_anomaly_flags

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


def fairness_score(driver_id, zone_id, driver_history, zone_demand):
    """
    Compute fairness score:
    Drivers with fewer recent surge allocations get higher scores.
    """
    surge_count = (
        driver_history.get(driver_id, {})
        .get("recent_zone_surges", {})
        .get(zone_id, 0)
    )
    score = 1 / (1 + surge_count)
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
def build_cost_matrix(
    drivers,
    zones,
    forecast,
    anomaly_flags,
    driver_history,
    eco_data,
    weights=None,
):
    """
    Create a cost matrix balancing:
    - forecasted surge demand (profit)
    - emissions (eco)
    - fairness (equitable distribution)
    Lower cost = higher priority in Hungarian optimization.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    n, m = len(drivers), len(zones)
    base_priority = np.array([forecast.get(z, 0) for z in zones])
    event_boost = np.array(
        [anomaly_flags.get(z, 0) * base_priority[i] * 0.3 for i, z in enumerate(zones)]
    )
    priority = base_priority + event_boost

    cost_matrix = np.zeros((n, m))
    max_priority = np.max(priority)
    ptp_priority = np.ptp(priority) + 1e-5

    for i, driver_id in enumerate(drivers):
        for j, zone_id in enumerate(zones):
            # Emission cost
            dist = eco_data.get((driver_id, zone_id), {}).get("distance_km", 1)
            emissions = calculate_emission(dist)

            # Fairness
            fair = fairness_score(driver_id, zone_id, driver_history, priority[j])

            # Normalize sub-components
            profit_term = (max_priority - priority[j]) / ptp_priority

            all_emissions = [
                calculate_emission(eco_data.get((driver_id, z), {}).get("distance_km", 1))
                for z in zones
            ]
            max_emission = max(all_emissions) if all_emissions else 1
            eco_term = emissions / max_emission

            all_fairs = [fairness_score(d, zone_id, driver_history, priority[j]) for d in drivers]
            max_fair = max(all_fairs) if all_fairs else 1
            fair_term = (max_fair - fair) / (max_fair + 1e-5)

            cost_matrix[i, j] = (
                weights["w_profit"] * profit_term
                + weights["w_eco"] * eco_term
                + weights["w_fair"] * fair_term
            )

    return cost_matrix


# ------------------------
# Main Assignment Function
# ------------------------
def assign_drivers(
    drivers, zones, forecast, anomaly_flags, driver_history, eco_data, weights=None
):
    """Compute optimal driver-zone pairs minimizing total cost."""
    cost_matrix = build_cost_matrix(
        drivers, zones, forecast, anomaly_flags, driver_history, eco_data, weights
    )
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assignments = [(drivers[i], zones[j]) for i, j in zip(row_ind, col_ind)]

    results = []
    for d, z in assignments:
        results.append(
            {
                "driver_id": d,
                "assigned_zone": z,
                "profit_est": forecast.get(z, 0),
                "anomaly_flag": anomaly_flags.get(z, 0),
                "distance_km": eco_data.get((d, z), {}).get("distance_km", 0),
                "emission_kg": calculate_emission(
                    eco_data.get((d, z), {}).get("distance_km", 0)
                ),
            }
        )

    df = pd.DataFrame(results)
    logger.info("Assignments completed: %d drivers assigned.", len(df))
    return df


# ------------------------
# Integration Hook (Pipeline Entry)
# ------------------------
def get_repositioning_plan():
    """
    Entry point for repositioning — called by run_pipeline.py
    Fetches current drivers, zones, forecast, and metrics.
    Returns assignment DataFrame.
    """
    logger.info("Fetching data for repositioning plan...")

    drivers = get_current_available_drivers()
    zones = get_target_zones()
    forecast = get_forecast_outputs()
    anomaly_flags = get_zone_anomaly_flags()
    driver_history = get_driver_history()
    eco_data = get_zone_eco_metrics()

    assignments = assign_drivers(
        drivers, zones, forecast, anomaly_flags, driver_history, eco_data
    )

    logger.info("Repositioning plan generated successfully ✅")
    return assignments
