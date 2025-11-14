import sqlite3
import pandas as pd

db_path = 'data/feature_store/surge_features.db'
output_csv = 'data/features/surge_alerts.csv'

conn = sqlite3.connect(db_path)
df = pd.read_sql_query("SELECT * FROM surge_alerts", conn)
df.to_csv(output_csv, index=False)
conn.close()

print(f'Saved surge_alerts to {output_csv}')