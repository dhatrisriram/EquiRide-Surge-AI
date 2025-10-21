# src/pipeline/run_pipeline.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.optimization.assignments import assign_drivers
from src.data.utils import get_current_available_drivers, get_target_zones, get_driver_history_final, get_zone_eco_metrics
from src.models.infer import get_forecast_outputs, get_zone_anomaly_flags

def run_pipeline():
    drivers = get_current_available_drivers()
    zones = get_target_zones()          # list of strings
    eco_data = get_zone_eco_metrics()   # np.ndarray of shape (len(drivers), len(zones))
    forecast = get_forecast_outputs()
    anomaly_flags = get_zone_anomaly_flags()
    driver_history = get_driver_history_final()

    assignments = assign_drivers(drivers, zones, forecast, anomaly_flags, driver_history, eco_data)

    print("\n=== DRIVER REPOSITIONING PLAN ===")
    print(assignments)
    print("\n✅ Pipeline executed successfully!")

if __name__=="__main__":
    run_pipeline()
