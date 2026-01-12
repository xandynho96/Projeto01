import sqlite3
import pandas as pd
import os

db_path = "dist/crypto_data.db"

if not os.path.exists(db_path):
    print(f"Database {db_path} not found.")
else:
    try:
        conn = sqlite3.connect(db_path)
        
        # Strategies
        try:
            df = pd.read_sql_query("SELECT * FROM strategies ORDER BY rowid DESC LIMIT 5", conn)
            print("--- Last 5 strategies in DIST DB ---")
            print(df[['id', 'origin', 'regime', 'winrate', 'trades', 'fitness']])
        except Exception as e:
            print(f"Could not read strategies: {e}")
            
        conn.close()
    except Exception as e:
        print(f"Error inspecting DB: {e}")
