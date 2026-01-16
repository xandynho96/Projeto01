import os
from dotenv import load_dotenv

load_dotenv()

# Trading settings
# Trading settings
SYMBOL = 'BTC/USD' # Kraken Spot
TIMEFRAME = '1m' # Changed to 1m for High-Frequency Scalping
LIMIT = 1000  # Number of candles to fetch
LEVERAGE = 10 # Spot Margin Leverage

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
        if os.path.exists(os.path.join("config", "user_config.json")):
            with open(os.path.join("config", "user_config.json"), "r") as f:
                data = json.load(f)
                if not KRAKEN_API_KEY:
                    KRAKEN_API_KEY = data.get('api_key', '').strip()
                    KRAKEN_SECRET = data.get('secret', '').strip()
                if not DEEPSEEK_API_KEY:
                    DEEPSEEK_API_KEY = data.get('deepseek_api_key', '').strip()
    except Exception as e:
        print(f"Warning: Could not load user_config.json: {e}")


# Database
import sys
if getattr(sys, 'frozen', False):
    # If Frozen (Exe), look in the same folder as the Executable
    BASE_DIR = os.path.dirname(sys.executable)
else:
    # If Script, look 2 levels up from app/utils/config.py -> Root
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

DB_PATH = os.path.join(BASE_DIR, "data", "crypto_data.db")
# Ensure the data directory exists
data_dir = os.path.dirname(DB_PATH)
if not os.path.exists(data_dir):
    try:
        os.makedirs(data_dir)
    except Exception as e:
        print(f"Warning: Could not create data directory at {data_dir}: {e}")

DB_URL = f'sqlite:///{DB_PATH}'
