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
KRAKEN_API_KEY = os.getenv('KRAKEN_API_KEY', '')
KRAKEN_SECRET = os.getenv('KRAKEN_SECRET', '')

# Deepseek API
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', '')

# Database
DB_URL = 'sqlite:///crypto_data.db'
