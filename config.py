import os
from dotenv import load_dotenv

load_dotenv()

# Trading settings
# Trading settings
SYMBOL = 'PF_XBTUSD' # Kraken Futures Perpetual (Linear)
TIMEFRAME = '1m' # Changed to 1m for High-Frequency Scalping
LIMIT = 1000  # Number of candles to fetch
LEVERAGE = 5 # Default Leverage

# API Keys (Kraken) - Leave empty for public data
# API Keys (Kraken) - Leave empty for public data
KRAKEN_API_KEY = os.getenv('KRAKEN_API_KEY', '')
KRAKEN_SECRET = os.getenv('KRAKEN_SECRET', '')

# Deepseek API
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', 'sk-5df7fcf6533b4ff19dcdcaf706f4b030')

# Helper to load from JSON if ENV is missing
import json
if not KRAKEN_API_KEY or not DEEPSEEK_API_KEY:
    try:
        if os.path.exists("user_config.json"):
            with open("user_config.json", "r") as f:
                data = json.load(f)
                if not KRAKEN_API_KEY:
                    KRAKEN_API_KEY = data.get('api_key', '')
                    KRAKEN_SECRET = data.get('secret', '')
                if not DEEPSEEK_API_KEY:
                    DEEPSEEK_API_KEY = data.get('deepseek_api_key', '')
    except Exception as e:
        print(f"Warning: Could not load user_config.json: {e}")

# Database
DB_URL = 'sqlite:///crypto_data.db'
