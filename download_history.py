import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta
from data_manager import DataManager
import config

def download_full_history(symbol_proxy='BTC/USDT', start_date='2017-01-01', timeframe='1m'):
    """
    Downloads full historical data from Binance (as a proxy for generic BTC action) 
    and saves to the local database using the project's Symbol name.
    
    Args:
        symbol_proxy (str): The symbol to fetch from Binance (e.g., 'BTC/USDT').
        start_date (str): 'YYYY-MM-DD' string.
        timeframe (str): Timeframe to download.
    """
    dm = DataManager()
    
    # We use Binance for history as it has deep liquidity and API allows easy history fetch
    exchange = ccxt.binance({
        'enableRateLimit': True,
    })
    
    print(f"🌍 Connecting to Binance to fetch history for {symbol_proxy}...")
    
    # Convert start date to timestamp (ms)
    since = exchange.parse8601(f"{start_date}T00:00:00Z")
    end_time = exchange.milliseconds()
    
    total_candles = 0
    
    print(f"⬇️ Starting download from {start_date}...")
    
    try:
        while since < end_time:
            # Fetch batch
            ohlcv = exchange.fetch_ohlcv(symbol_proxy, timeframe, since, limit=1000)
            
            if len(ohlcv) == 0:
                print("⚠️ No more data received.")
                break
            
            # Prepare DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Save to DB (Using the project's configured symbol, e.g., PF_XBTUSD, so backtest finds it)
            # We map the downloaded data to our target symbol
            target_symbol = config.SYMBOL
            
            # Save
            dm.save_data(df, symbol=target_symbol, timeframe=timeframe)
            
            count = len(df)
            total_candles += count
            
            # Update 'since' for next batch
            # Last timestamp inside ohlcv is ohlcv[-1][0]
            # Next batch starts at last timestamp + 1 timeframe (approx)
            # But safer to just take last timestamp + 1ms to avoid overlap/gaps 
            # (fetch_ohlcv usually includes the start time, so we need to move forward)
            last_ts = ohlcv[-1][0]
            since = last_ts + 1 
            
            curr_date = pd.to_datetime(last_ts, unit='ms')
            print(f"   ✅ Saved {count} candles. Last: {curr_date} | Total: {total_candles}")
            
            # Rate limit sleep (ccxt handles this mostly, but good to be safe for big loops)
            # time.sleep(0.1) 
            
        print(f"\n🎉 Download Complete! Total Candles: {total_candles}")
        
    except KeyboardInterrupt:
        print("\n🛑 Download stopped by user.")
    except Exception as e:
        print(f"\n❌ Error downloading history: {e}")

if __name__ == "__main__":
    # You can change start_date if you need even more history
    # 2020 is usually a good balance for relevance
    download_full_history(start_date='2020-01-01', timeframe='1m')
