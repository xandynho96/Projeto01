
import ccxt
import os
import json
import time

def debug_kraken_futures():
    print("--- STARTING KRAKEN FUTURES DEBUG ---")
    
    api_key = ""
    secret = ""
    
    # Checking config paths
    config_pool = ["user_config_roe.json", os.path.join("config", "user_config_roe.json")]
    data = None
    
    try:
        found = False
        for cfg in config_pool:
            if os.path.exists(cfg):
                try:
                    with open(cfg, "r") as f:
                        data = json.load(f)
                        print(f"Loaded config from: {cfg}")
                        found = True
                        break
                except: pass
        
        if found and data:
            api_key = data.get("api_key")
            secret = data.get("secret")
            if api_key: print(f"Key loaded: {api_key[:4]}...")
            
    except Exception as e:
        print(f"Could not load JSON config: {e}")
        
    if not api_key:
        api_key = input("Enter Kraken API Key: ").strip()
        secret = input("Enter Kraken Secret: ").strip()
        
    if not api_key or not secret:
        print("No keys provided. Exiting.")
        return

    print("\n1. Connecting to Kraken Futures (ccxt.krakenfutures)...")
    try:
        exchange = ccxt.krakenfutures({
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True,
        })
        
        print("2. Fetching Balance...")
        balance = exchange.fetch_balance()
        
        print("\n--- RAW BALANCE RESPONSE ---")
        print(json.dumps(balance, indent=2, default=str))
        print("----------------------------")
        
        print("\n3. Analyzing Keys...")
        total = balance.get('total', {})
        free = balance.get('free', {})
        print(f"Total: {total}")
        print(f"Free: {free}")
        
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_kraken_futures()
