import os
import sys

# --- Ensure project root ('EquiRide-Surge-AI') is on sys.path ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- Now safe to import your modules ---
from src.optimization.assignments import assign_drivers
from src.models.infer import get_forecast_outputs, get_zone_anomaly_flags
from src.data.utils import (
    get_current_available_drivers_mock as get_current_available_drivers,
    get_driver_history_final as get_driver_history,
    get_zone_eco_metrics_final as get_zone_eco_metrics,
)

def test_assign_drivers_pipeline():
    forecast = get_forecast_outputs()
    anomaly_flags = get_zone_anomaly_flags()
    drivers = get_current_available_drivers()
    hist = get_driver_history()
    eco = get_zone_eco_metrics()

    assignments = assign_drivers(
        drivers,
        list(forecast.keys()),
        forecast,
        anomaly_flags,
        hist,
        eco,
    )

    assert not assignments.empty, "Assignment output should not be empty"
    assert {"driver_id", "assigned_zone"}.issubset(assignments.columns), \
        "Output must include driver_id and assigned_zone"
