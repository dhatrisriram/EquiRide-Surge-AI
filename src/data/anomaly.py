from sklearn.ensemble import IsolationForest

def detect_demand_anomalies(demand_series):
    clf = IsolationForest(contamination=0.05, random_state=42)
    values = demand_series.values.reshape(-1, 1)
    preds = clf.fit_predict(values)
    # Convert: -1 means anomaly, 1 means normal
    anomaly_flags = (preds == -1).astype(int)
    return anomaly_flags
