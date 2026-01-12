import sqlite3
import pandas as pd
import os

db_path = "crypto_data.db"

if not os.path.exists(db_path):
    print(f"Database {db_path} not found.")
else:
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # List tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print("Tables found:", [t[0] for t in tables])
        
        for table in tables:
            table_name = table[0]
            count = pd.read_sql_query(f"SELECT COUNT(*) FROM {table_name}", conn).iloc[0,0]
            print(f"Table '{table_name}' has {count} rows.")
            
            # Show last 3 rows if it looks like timeseries or log
            try:
                df = pd.read_sql_query(f"SELECT * FROM {table_name} ORDER BY rowid DESC LIMIT 3", conn)
                print(f"--- Last 3 rows of {table_name} ---")
                print(df)
            except Exception as e:
                print(f"Could not read content for {table_name}: {e}")
                
        conn.close()
    except Exception as e:
        print(f"Error inspecting DB: {e}")
