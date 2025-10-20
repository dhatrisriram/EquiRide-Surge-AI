from src.optimization.assignments import build_cost_matrix, assign_drivers

def test_assignments_shape():
    # Minimal mock test for coverage
    drivers = ['D1', 'D2']
    zones = ['MG Road', 'Indiranagar']
    forecast = {'MG Road': 100, 'Indiranagar': 200}
    anomaly_flags = {'MG Road': 1, 'Indiranagar': 0}
    driver_history = {'D1': {}, 'D2': {}}
    eco_data = {('D1','MG Road'):{'distance_km':3}, ('D2','Indiranagar'):{'distance_km':5}}
    assignments = assign_drivers(drivers, zones, forecast, anomaly_flags, driver_history, eco_data)
    assert len(assignments) == 2
