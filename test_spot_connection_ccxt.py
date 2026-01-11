import ccxt
import json
import os
import sys

def load_config():
    try:
        if os.path.exists('user_config.json'):
            with open('user_config.json', 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
    return {}

def test_connection():
    config = load_config()
    api_key = config.get('api_key')
    secret = config.get('secret')
    
    if not api_key or not secret:
        print("❌ No API Keys found in user_config.json")
        return

    print(f"Testing Kraken Spot connection with Key: {api_key[:6]}...")
    
    try:
        exchange = ccxt.kraken({
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True,
        })
        
        # Load Markets (Crucial for validation)
        print("Loading markets...")
        markets = exchange.load_markets()
        print(f"✅ Markets Loaded. Found {len(markets)} symbols.")
        
        if 'BTC/USD' in markets:
            print("✅ 'BTC/USD' symbol found.")
        else:
            print("❌ 'BTC/USD' symbol NOT found. Check symbol naming.")
            
        # Fetch Balance
        print("Fetching Balance...")
        balance = exchange.fetch_balance()
        print("✅ Balance Fetched:")
        # Show Total USD
        total_usd = balance.get('total', {}).get('USD', 0)
        total_usdt = balance.get('total', {}).get('USDT', 0)
        print(f"   USD: {total_usd}")
        print(f"   USDT: {total_usdt}")
        
    except Exception as e:
        print(f"❌ Connection Failed: {e}")

if __name__ == "__main__":
    test_connection()
