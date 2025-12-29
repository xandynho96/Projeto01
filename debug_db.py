from data_manager import DataManager
from sqlalchemy import text
import pandas as pd

dm = DataManager()
print("Connecting to DB...")
with dm.engine.connect() as conn:
    print("Tables:", conn.execute(text("SELECT name FROM sqlite_master WHERE type='table';")).fetchall())
    
    try:
        # Count total rows
        count = conn.execute(text("SELECT COUNT(*) FROM market_data")).scalar()
        print(f"Total Rows in market_data: {count}")
        
        # Group by timeframe
        print("Rows by Timeframe:")
        rows = conn.execute(text("SELECT timeframe, COUNT(*) FROM market_data GROUP BY timeframe")).fetchall()
        for r in rows:
            print(r)
            
        # Sample row
        print("Sample Row:")
        sample = conn.execute(text("SELECT * FROM market_data LIMIT 1")).fetchone()
        print(sample)
        
    except Exception as e:
        print(f"Error querying: {e}")
