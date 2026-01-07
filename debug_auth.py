import json
import ccxt
import sys

def debug_auth():
    print("--- Kraken Futures Auth Debugger ---")
    
    # Load Keys
    try:
        with open('dist/user_config.json', 'r') as f:
            config = json.load(f)
            api_key = config.get('api_key', '').strip()
            secret = config.get('secret', '').strip()
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    print(f"API Key Length: {len(api_key)}")
    print(f"Secret Length: {len(secret)}")
    
    if len(api_key) < 10 or len(secret) < 10:
        print("❌ Keys look too short/empty. Please check user_config.json")
        return

    print(f"API Key (first 4): {api_key[:4]}...")
    
    # Initialize CCXT
    try:
        exchange = ccxt.krakenfutures({
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True,
        })
        
        # Test Public (should work)
        # print("\nTesting Public API (Server Time)...")
        # time = exchange.fetch_time()
        # print(f"✅ Server Time: {time}")
        
        # Test Private (Balance)
        print("\nTesting Private API (Balance)...")
        balance = exchange.fetch_balance()
        print("✅ Balance Fetch Successful!")
        print("Keys are VALID and have 'General API' permission.")
        
        # Check specific wallet
        print("\nChecking Wallets:")
        if 'total' in balance:
            print(f"Total: {balance['total']}")
            
    except ccxt.AuthenticationError as e:
        print(f"\n❌ AUTHENTICATION ERROR: {e}")
        print("Reason: Most likely this is a SPOT key being used on FUTURES, or incorrect characters.")
        print("Tip: Go to Kraken Futures -> Settings -> API Keys.")
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")

if __name__ == "__main__":
    debug_auth()
