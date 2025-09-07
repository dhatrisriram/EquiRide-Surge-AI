import pandas as pd
from sklearn.cluster import DBSCAN
import h3

def load_and_validate(csv_path: str):
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print("Missing values:")
    print(df.isnull().sum())
    df = df.dropna(subset=['latitude', 'longitude'])
    return df

def add_spatial_features(df):
    coords = df[['latitude', 'longitude']].values
    clustering = DBSCAN(eps=0.005, min_samples=3).fit(coords)
    df['cluster'] = clustering.labels_

    if 'h3_index' not in df.columns:
        df['h3_index'] = df.apply(lambda row: h3.latlng_to_cell(row['latitude'], row['longitude'], 9), axis=1)

    return df

def add_temporal_features(df, date_col='Date'):
    df[date_col] = pd.to_datetime(df[date_col])
    df['hour'] = df[date_col].dt.hour
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['month'] = df[date_col].dt.month
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df = df.sort_values(by=date_col)
    df['traffic_volume_rolling_3'] = df['Traffic Volume'].rolling(window=3, min_periods=1).mean()
    return df
