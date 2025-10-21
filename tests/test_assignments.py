# tests/test_assignments.py
import sys
import os
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.optimization.assignments import assign_drivers
from src.data.utils import get_current_available_drivers, get_target_zones, get_driver_history_final, get_zone_eco_metrics
from src.models.infer import get_forecast_outputs, get_zone_anomaly_flags

def test_assign_drivers_pipeline():
    drivers = get_current_available_drivers()
    zones = get_target_zones()
    forecast = get_forecast_outputs()
    anomaly_flags = get_zone_anomaly_flags()
    driver_history = get_driver_history_final()
    eco_data = get_zone_eco_metrics()

    assignments = assign_drivers(drivers, zones, forecast, anomaly_flags, driver_history, eco_data)

    assert not assignments.empty, "Assignment output should not be empty"
    assert {"driver_id", "assigned_zone", "profit_est", "distance_km", "emission_kg"}.issubset(assignments.columns), \
        "Output must include driver_id, assigned_zone, profit_est, distance_km, emission_kg"
    print("✅ Test passed: assign_drivers pipeline works as expected.")

if __name__ == "__main__":
    test_assign_drivers_pipeline()
