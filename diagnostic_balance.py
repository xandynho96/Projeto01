import logging
import json
import os
import ccxt
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BalanceDiag")

def load_config():
    if os.path.exists("user_config.json"):
        with open("user_config.json", "r") as f:
            return json.load(f)
    return {}

def run_diagnostic():
    config = load_config()
    api_key = config.get("api_key")
    secret = config.get("secret")
    mode = config.get("trading_mode", "Spot Margin")
    
    if not api_key or not secret:
        logger.error("❌ API Key/Secret not found in user_config.json")
        return

    logger.info(f"🔵 Connecting to Kraken [{mode}]...")
    
    try:
        if "Futures" in mode:
            exchange = ccxt.krakenfutures({
                'apiKey': api_key,
                'secret': secret,
                'options': {'defaultType': 'future'}
            })
        else: # Spot
            exchange = ccxt.kraken({
                'apiKey': api_key,
                'secret': secret,
                'options': {'defaultType': 'spot'}
            })
            
        exchange.load_markets()
        logger.info("✅ Connected!")
        
        logger.info("💰 Fetching Balance...")
        balance = exchange.fetch_balance()
        
        logger.info("--- BALANCE KEYS ---")
        if 'total' in balance:
            logger.info(f"Total keys: {list(balance['total'].keys())}")
            for currency, amount in balance['total'].items():
                if amount > 0:
                    logger.info(f"  {currency}: {amount}")
        else:
            logger.warning("'total' key not found in balance response.")

        logger.info("--- FREE KEYS ---")
        if 'free' in balance:
             for currency, amount in balance['free'].items():
                if amount > 0:
                    logger.info(f"  {currency}: {amount}")
                    
        # Check specific USD variants
        logger.info("--- USD CHECK ---")
        usd_keys = ['USD', 'USDT', 'ZUSD', 'ZUSDT', 'USDC', 'ZEUR']
        for k in usd_keys:
            val = balance['total'].get(k, 0)
            logger.info(f"  {k}: {val}")
            
        logger.info("--- RAW INFO (First 500 chars) ---")
        # logger.info(str(balance.get('info', 'No Info'))[:500])
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_diagnostic()
