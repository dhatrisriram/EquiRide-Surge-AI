from src.models.infer import get_forecast_outputs, get_zone_anomaly_flags
from src.optimization.assignments import assign_drivers
from src.data.utils import get_current_available_drivers, get_driver_history, get_zone_eco_metrics

def run_pipeline():
    forecast = get_forecast_outputs()
    anomaly_flags = get_zone_anomaly_flags()

    drivers = get_current_available_drivers()  
    zones = list(forecast.keys())
    driver_histories = get_driver_history()  
    eco_data = get_zone_eco_metrics()  

    assignments = assign_drivers(drivers, zones, forecast, anomaly_flags, driver_histories, eco_data)
    # Proceed with alerts and visualization integration here
    return assignments

if __name__ == "__main__":
    run_pipeline()
