import ccxt
import pandas as pd
from datetime import datetime

def debug_fetch():
    exchange = ccxt.kraken()
    symbol = 'BTC/USD'
    timeframe = '1h'
    since = exchange.parse8601('2023-01-01T00:00:00Z')
    
    print(f"Testing fetch for {symbol} since {datetime.fromtimestamp(since/1000)}")
    
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=10)
        print(f"Returned {len(ohlcv)} candles")
        if len(ohlcv) > 0:
            print(f"First candle: {ohlcv[0]}")
            print(f"Last candle: {ohlcv[-1]}")
    except Exception as e:
        print(f"Error: {e}")

    # Test without since
    print("\nTesting fetch without 'since'...")
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=10)
    print(f"Returned {len(ohlcv)} candles")
    print(f"First candle: {ohlcv[0]}")

if __name__ == "__main__":
    debug_fetch()
