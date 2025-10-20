def get_forecast_outputs():
    # Output forecast dict after running TFT/GNN: {zone_id: surge_score}
    return {
        "MG Road": 1350,
        "Indiranagar": 2200,
        "Koramangala": 1650,
        # ...
    }

def get_zone_anomaly_flags():
    # Output anomaly/event flags: {zone_id: 1 if flagged, else 0}
    return {
        "MG Road": 1,
        "Indiranagar": 0,
        "Koramangala": 1,
    }
