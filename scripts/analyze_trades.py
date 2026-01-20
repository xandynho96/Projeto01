import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import os
import json
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.data_manager import DataManager
from app.utils import config

def load_keys():
    # Try user_config_roe.json first (as seen in trader.py)
    paths = ["config/user_config_roe.json", "config/user_config.json"]
    for p in paths:
        if os.path.exists(p):
            try:
                with open(p, "r") as f:
                    data = json.load(f)
                    if data.get("api_key"):
                        print(f"Loaded keys from {p}")
                        return data.get("api_key"), data.get("secret"), data
            except Exception as e:
                print(f"Error reading {p}: {e}")
    return None, None, {}

def main():
    print("--- STARTING TRADE ANALYSIS ---")
    
    # 1. DB Analysis
    print("\n[LOCAL DB Check]")
    db_path = "data/crypto_data.db"
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            df = pd.read_sql_query("SELECT * FROM trades", conn)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                # Filter last 48h
                recent = df[df['timestamp'] > (datetime.now() - timedelta(hours=48))]
                print(f"Trades in DB (Total): {len(df)}")
                print(f"Trades in DB (Last 48h): {len(recent)}")
                if not recent.empty:
                    print(recent[['timestamp', 'side', 'amount', 'price', 'pnl', 'status']].to_string())
            else:
                print("Trades table is empty.")
            conn.close()
        except Exception as e:
            print(f"DB Error: {e}")
    else:
        print("DB file not found.")

    # 2. API Analysis
    print("\n[KRAKEN API Check]")
    api_key, secret, settings = load_keys()
    
    if api_key and secret:
        dm = DataManager()
        # Connect
        trading_mode = settings.get('trading_mode', 'Spot')
        demo_mode = settings.get('demo_mode', False)
        print(f"Connecting with mode: {trading_mode} (Demo: {demo_mode})")
        
        if dm.connect_exchange(api_key, secret, demo_mode=demo_mode, trading_mode=trading_mode):
            try:
                # Fetch recent trades from Exchange
                print("Fetching recent trades from Exchange (last 48h)...")
                since = int((datetime.now() - timedelta(hours=48)).timestamp() * 1000)
                
                my_trades = dm.exchange.fetch_my_trades(symbol=None, since=since, limit=50) # symbol=None for all
                
                if my_trades:
                    print(f"Found {len(my_trades)} trades on Exchange:")
                    # Convert to DF for display
                    data = []
                    for t in my_trades:
                        data.append({
                            'time': t['datetime'],
                            'symbol': t['symbol'],
                            'side': t['side'],
                            'price': t['price'],
                            'amount': t['amount'],
                            'cost': t['cost'],
                            'fee': t['fee']['cost'] if t.get('fee') else 0,
                            'pnl': t['info'].get('realizedPnl', 'N/A') if 'info' in t else 'N/A'
                        })
                    df_api = pd.DataFrame(data)
                    print(df_api.to_string())
                    
                    # Analyze PnL
                    try:
                        total_pnl = 0
                        valid_pnl_count = 0
                        for d in data:
                            if d['pnl'] != 'N/A':
                                total_pnl += float(d['pnl'])
                                valid_pnl_count += 1
                        
                        if valid_pnl_count > 0:
                            print(f"\nTotal Realized PnL (API): ${total_pnl:.2f}")
                        else:
                            print("\nNo PnL data available in API response (Spot?).")
                            
                    except Exception as e:
                        print(f"Error calculating API PnL: {e}")

                else:
                    print("No trades found on Exchange in the last 48 hours.")
                    
            except Exception as e:
                print(f"API Fetch Error: {e}")
        else:
            print("Failed to connect to Exchange.")
    else:
        print("No API keys found in config.")

if __name__ == "__main__":
    main()
