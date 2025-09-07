import pandas as pd
import os

FEATURE_STORE_PATH = 'data/processed/feature_store.parquet'

def save_feature_store(df: pd.DataFrame):
    os.makedirs(os.path.dirname(FEATURE_STORE_PATH), exist_ok=True)
    df.to_parquet(FEATURE_STORE_PATH, index=False)
    print(f"Feature Store saved at {FEATURE_STORE_PATH}")

def load_feature_store():
    return pd.read_parquet(FEATURE_STORE_PATH)
