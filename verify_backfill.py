from data_manager import DataManager
import config
import pandas as pd
from contextlib import redirect_stdout
import time

with open('backfill_debug.txt', 'w', encoding='utf-8') as f:
    with redirect_stdout(f):
        print(" Testing fetch_ohlcv 5m...")
        dm = DataManager()
        if not dm.exchange:
            print("No exchange.")
            exit()
            
        # Target: 30 days ago
        start_date = pd.Timestamp.now() - pd.Timedelta(days=30)
        ts_ms = int(start_date.timestamp() * 1000)
        
        print(f"Target Date: {start_date}")
        print(f"TS (ms): {ts_ms}")
        
        # Test 7: Since 2020 (MS)
        print("\n--- Test 7: Since 2020 (MS) ---")
        ts_2020 = 1577836800000 # 2020-01-01
        try:
            ohlcv = dm.exchange.fetch_ohlcv('BTC/USD', '1m', since=ts_2020)
            if ohlcv:
                print(f"First Candle: {pd.to_datetime(ohlcv[0][0], unit='ms')}")
                print(f"Last Candle:  {pd.to_datetime(ohlcv[-1][0], unit='ms')}")
            else:
                print("Empty.")
        except Exception as e:
            print(f"Error: {e}")
