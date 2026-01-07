import ccxt
import json
import logging

def debug_demo():
    print("--- Kraken Futures DEMO Debugger ---")
    
    # Load Keys
    try:
        with open('dist/user_config.json', 'r') as f:
            config = json.load(f)
            api_key = config.get('api_key', '').strip()
            secret = config.get('secret', '').strip()
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    # Inspect Default URLs
    temp = ccxt.krakenfutures()
    # print("Default URLs:", json.dumps(temp.urls, indent=2))
    
    # Init exchange
    exchange = ccxt.krakenfutures({
        'apiKey': api_key,
        'secret': secret,
        'enableRateLimit': True,
    })
    
    # Correct way to use 'test' urls in CCXT
    print("Enabling Sandbox Mode...")
    exchange.set_sandbox_mode(True)
    
    print("Loading Markets...")
    try:
        markets = exchange.load_markets()
        print(f"✅ Loaded {len(markets)} markets.")
        
        # Filter for BTC/XBT
        print("\n--- BTC/XBT Symbols ---")
        btc_symbols = [s for s in markets.keys() if 'BTC' in s or 'XBT' in s]
        for s in btc_symbols:
            print(f"Symbol: {s} | ID: {markets[s]['id']} | Type: {markets[s]['type']}")
            
    except Exception as e:
        print(f"❌ Failed to load markets: {e}")

    # Test Balance
    print("\n--- Testing Balance ---")
    try:
        # Try default
        bal = exchange.fetch_balance()
        print("✅ Balance Fetched (Default):")
        print(bal.get('total', 'No Total'))
    except Exception as e:
        print(f"❌ Default fetch_balance failed: {e}")
        
    try:
        # Try params for flex/margin
        # Some versions of CCXT require specific types for Kraken Futures
        pass
    except Exception as e:
        pass

if __name__ == "__main__":
    debug_demo()
