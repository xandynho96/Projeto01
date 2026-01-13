from data_manager import DataManager
import config
import pandas as pd
from contextlib import redirect_stdout

with open('binance_backfill_debug.txt', 'w', encoding='utf-8') as f:
    with redirect_stdout(f):
        print(" Testing fetch_deep_history (Binance)...")
        dm = DataManager()
        
        # Test 1 month
        print("\n--- Testing 1 Month Backfill ---")
        try:
            dm.fetch_deep_history(months=1)
            print("Done backfill call.")
        except Exception as e:
            print(f"Error: {e}")

        # Check count in DB
        df = dm.get_data_from_db(limit=100000)
        print(f"Count after backfill: {len(df)}")
        if not df.empty:
            print(f"First: {df['timestamp'].min()}")
            print(f"Last:  {df['timestamp'].max()}")
