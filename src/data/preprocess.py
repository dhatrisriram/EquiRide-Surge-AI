import pandas as pd

def preprocess_events(events_json):
    df = pd.DataFrame(events_json['results'])
    df['bucket'] = pd.to_datetime(df['start']).dt.floor('15T')
    # Feature: event intensity (scale by expected attendees, type, proximity)
    df['weight'] = df['attendance'].fillna(1) * (df['type'].map({'concert':1.2, 'sports':1.1, 'other':1}))
    return df

def merge_traffic_weather_events(traffic, weather, events):
    # Merge/join all objects by location and time bucket, filling missing with zeros
    traffic_df = pd.DataFrame(traffic)
    weather_df = pd.DataFrame(weather)
    events_df = preprocess_events(events)
    merged = pd.merge(traffic_df, weather_df, on=['location','bucket'], how='outer')
    merged = pd.merge(merged, events_df, on=['location','bucket'], how='outer')
    merged = merged.fillna(0)
    return merged
