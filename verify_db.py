from data_manager import DataManager
import config
import pandas as pd

dm = DataManager()
print(f"Connecting to DB: {config.DB_URL}")

# Check 1m data
df = dm.get_data_from_db(timeframe='1m', limit=500000)
print(f"Total rows in DB (1m): {len(df)}")
if not df.empty:
    print(f"Start: {df['timestamp'].min()}")
    print(f"End: {df['timestamp'].max()}")
    print("Duplicates:", df.duplicated(subset=['timestamp']).sum())
