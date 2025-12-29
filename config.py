import os
from dotenv import load_dotenv

load_dotenv()

# Trading settings
SYMBOL = 'BTC/USD'
TIMEFRAME = '1m' # Changed to 1m for High-Frequency Scalping
LIMIT = 1000  # Number of candles to fetch

# API Keys (Kraken) - Leave empty for public data
KRAKEN_API_KEY = os.getenv('KRAKEN_API_KEY', '')
KRAKEN_SECRET = os.getenv('KRAKEN_SECRET', '')

# Deepseek API
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', '')

# Database
DB_URL = 'sqlite:///crypto_data.db'
