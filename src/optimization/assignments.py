import numpy as np
from scipy.optimize import linear_sum_assignment
from src.data.utils import get_driver_history, get_zone_eco_metrics, get_current_available_drivers, get_target_zones
from src.models.infer import get_forecast_outputs, get_zone_anomaly_flags

def calculate_emission(distance_km, vehicle_type='auto'):
    """
    Calculate estimated CO2 emissions (kg) for given distance and vehicle type.
    """
    emission_factors = {'auto': 0.13, 'car': 0.16}  # kg CO2/km
    return distance_km * emission_factors.get(vehicle_type, 0.13)

def fairness_score(driver_id, zone_id, driver_history, zone_demand):
    """
    Compute fairness score: drivers with fewer recent surge allocations get higher scores.
    """
    surge_count = driver_history.get(driver_id, {}).get('recent_zone_surges', {}).get(zone_id, 0)
    score = 1 / (1 + surge_count)
    return score * zone_demand

def build_cost_matrix(drivers, zones, forecast, anomaly_flags, driver_history, eco_data, w_profit=1.5, w_eco=0.8, w_fair=1.0):
    """
    Create a cost matrix for assignment, balancing forecast demand (with anomaly boost),
    estimated emission, and fairness score. Lower cost preferred.
    """
    n = len(drivers)
    m = len(zones)

    base_priority = np.array([forecast.get(z, 0) for z in zones])
    event_boost = np.array([anomaly_flags.get(z, 0) * base_priority[i] * 0.3 for i, z in enumerate(zones)])
    priority = base_priority + event_boost

    cost_matrix = np.zeros((n, m))
    max_priority = np.max(priority)
    ptp_priority = np.ptp(priority) + 1e-5

    for i, driver_id in enumerate(drivers):
        for j, zone_id in enumerate(zones):
            # Emission estimate
            dist = eco_data.get((driver_id, zone_id), {}).get("distance_km", 1)
            emissions = calculate_emission(dist)

            # Fairness score
            fair = fairness_score(driver_id, zone_id, driver_history, priority[j])

            # Normalize sub-costs
            profit_term = (max_priority - priority[j]) / ptp_priority

            # Emission normalization: normalize across emissions per driver-zone pair for stability
            all_emissions = [calculate_emission(eco_data.get((driver_id, z), {}).get("distance_km", 1)) for z in zones]
            max_emission = max(all_emissions) if all_emissions else 1
            eco_term = emissions / max_emission

            # Fairness normalization: assume fairness score varies similarly to priority
            # To avoid errors, clamp values when no variation
            all_fairs = []
            for d in drivers:
                s = fairness_score(d, zone_id, driver_history, priority[j])
                all_fairs.append(s)
            max_fair = max(all_fairs) if all_fairs else 1
            fair_term = (max_fair - fair) / (max_fair + 1e-5)

            cost_matrix[i, j] = w_profit * profit_term + w_eco * eco_term + w_fair * fair_term

    return cost_matrix

def assign_drivers(drivers, zones, forecast, anomaly_flags, driver_history, eco_data):
    """
    Compute optimal assignment pairs for drivers to zones minimizing total cost.
    """
    cost_matrix = build_cost_matrix(drivers, zones, forecast, anomaly_flags, driver_history, eco_data)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    assignments = [(drivers[i], zones[j]) for i, j in zip(row_ind, col_ind)]
    return assignments

def get_repositioning_plan():
    """
    Entry point for generating repositioning plan.
    Integrate with your data pipeline to provide actual data in these functions.
    """
    drivers = get_current_available_drivers()
    zones = get_target_zones()
    forecast = get_forecast_outputs()
    anomaly_flags = get_zone_anomaly_flags()
    driver_history = get_driver_history()
    eco_data = get_zone_eco_metrics()

    return assign_drivers(drivers, zones, forecast, anomaly_flags, driver_history, eco_data)
