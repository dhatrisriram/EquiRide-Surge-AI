import pandas as pd

# Load the feature store
df = pd.read_parquet('data/processed/feature_store.parquet')

# Show first few rows
print(df.head())

# Check columns
print(df.columns)
