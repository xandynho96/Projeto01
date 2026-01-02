from data_manager import DataManager
import logging

# Configure logging to see output clearly
logging.basicConfig(level=logging.INFO)

print("🚀 Starting Data Manager Test...")
dm = DataManager()

print("\n--- Testing fetch_full_history (Bulk YFinance/Fallback) ---")
# This should now handle the 1m limit gracefully and not crash
df = dm.fetch_full_history(symbol='BTC/USD', timeframe='1m')

if df is not None and not df.empty:
    print(f"\n✅ SUCCESS: Fetched {len(df)} rows.")
    print(df.head())
    print(df.tail())
else:
    print("\n❌ FAILURE: No data returned.")

print("\n--- Checking DB Content ---")
df_db = dm.get_data_from_db(limit=5)
print(f"DB Content Sample: {len(df_db)} rows")
print(df_db)
