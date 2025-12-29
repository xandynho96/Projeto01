import ccxt
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import time
import config

Base = declarative_base()

class MarketData(Base):
    __tablename__ = 'market_data'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    symbol = Column(String, nullable=False)
    timeframe = Column(String, nullable=False, default='1h')
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)

class DataManager:
    def __init__(self, db_url=config.DB_URL):
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.exchange = ccxt.kraken()
        
    def fetch_historical_data(self, symbol=config.SYMBOL, timeframe=config.TIMEFRAME, limit=config.LIMIT, since=None):
        """Fetches historical OHLCV data from Kraken."""
        print(f"Fetching {limit if limit else 'all'} candles for {symbol} ({timeframe}) since {since}...")
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit, since=since)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()

    def fetch_full_history(self, start_year=2020, symbol=config.SYMBOL, timeframe=config.TIMEFRAME):
        """Fetches full history using yfinance for bulk data (bypassing Kraken limits)."""
        import yfinance as yf
        
        # Map symbol for YF (BTC/USD -> BTC-USD)
        yf_symbol = symbol.replace('/', '-')
        
        print(f"Downloading full history for {yf_symbol} from {start_year} to now via YFinance...")
        
        try:
            # YFinance interval mapping
            interval_map = {'1h': '1h', '1d': '1d', '1m': '1m'}
            interval = interval_map.get(timeframe, '1h')
            
            start_date_str = f"{start_year}-01-01"
            
            # YFinance 1h data limit check (730 days)
            if interval == '1h':
                limit_date = datetime.now() - pd.Timedelta(days=729)
                start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
                if start_dt < limit_date:
                    print(f"Warning: 1h data limited to last 730 days. Adjusting start date.")
                    start_date_str = limit_date.strftime("%Y-%m-%d")

            # YFinance 1m data limit check (7 days usually, safe 5 days)
            if interval == '1m':
                # Yahoo Finance limits 1m data to the last 7 days (sometimes 30 depending on provider quirks, staying safe with 7)
                limit_date = datetime.now() - pd.Timedelta(days=7)
                try:
                    start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
                    if start_dt < limit_date:
                        print(f"Warning: 1m data limited to last 7 days. Adjusting start date.")
                        start_date_str = limit_date.strftime("%Y-%m-%d")
                except ValueError:
                    # If start_date_str is already just a date object or formatted differently
                    pass
            
            # Download
            df_yf = yf.download(yf_symbol, start=start_date_str, interval=interval, progress=False)
            
            if df_yf.empty:
                print("No data found on YFinance.")
                return

            # Flatten MultiIndex columns if present (common in new yfinance)
            if isinstance(df_yf.columns, pd.MultiIndex):
                df_yf.columns = df_yf.columns.get_level_values(0)

            # Reset index to get Date/Datetime as column
            df_yf.reset_index(inplace=True)
            
            # Normalize columns
            # YF columns: Date/Datetime, Open, High, Low, Close, Adj Close, Volume
            # We need: timestamp, open, high, low, close, volume
            
            # Rename columns (case insensitive usually, but let's be precise)
            df_yf.rename(columns={
                'Date': 'timestamp', 
                'Datetime': 'timestamp',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }, inplace=True)
            
            # Ensure timestamp type
            df_yf['timestamp'] = pd.to_datetime(df_yf['timestamp'])
            
            # Select only needed columns
            df_final = df_yf[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
            
            # Save to DB
            print(f"Downloaded {len(df_final)} candles. Saving to database...")
            # Note: We use the requested 'timeframe' (e.g. 1m or 1h) as label, even if we map it to YF interval
            self.save_data(df_final, symbol, timeframe=timeframe)
            print("Bulk import complete.")
            
        except Exception as e:
            print(f"Error in bulk download: {e}")

    def save_data(self, df, symbol=config.SYMBOL, timeframe=config.TIMEFRAME):
        """Saves DataFrame to SQLite database, avoiding duplicates."""
        if df.empty:
            return
            
        session = self.Session()
        count = 0
        try:
            # Check existing for this symbol AND timeframe
            existing_timestamps = set(
                dt[0] for dt in session.query(MarketData.timestamp)
                .filter(MarketData.symbol == symbol, MarketData.timeframe == timeframe)
                .all()
            )
            
            for _, row in df.iterrows():
                if row['timestamp'] in existing_timestamps:
                    continue
                    
                market_data = MarketData(
                    timestamp=row['timestamp'],
                    symbol=symbol,
                    timeframe=timeframe,
                    open=row['open'],
                    high=row['high'],
                    low=row['low'],
                    close=row['close'],
                    volume=row['volume']
                )
                session.add(market_data)
                count += 1
            
            session.commit()
            print(f"Saved {count} new records to database ({timeframe}).")
            
        except Exception as e:
            session.rollback()
            print(f"Error saving to DB: {e}")
        finally:
            session.close()

    def get_data_from_db(self, symbol=config.SYMBOL, timeframe=config.TIMEFRAME, limit=1000):
        """Retrieves data from local DB for analysis."""
        try:
            query = f"SELECT * FROM market_data WHERE symbol = '{symbol}' AND timeframe = '{timeframe}' ORDER BY timestamp ASC"
            df = pd.read_sql(query, self.engine)
            if not df.empty and limit:
                df = df.tail(limit)
            return df
        except Exception as e:
            print(f"Error reading DB: {e}")
            return pd.DataFrame()

if __name__ == "__main__":
    # Test the module
    dm = DataManager()
    print("Fetching data from Kraken...")
    df = dm.fetch_historical_data()
    print(df.head())
    print("Saving to DB...")
    dm.save_data(df)
    print("Done.")
