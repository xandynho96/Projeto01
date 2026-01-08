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

class Trade(Base):
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    symbol = Column(String)
    side = Column(String) # buy/sell
    amount = Column(Float)
    price = Column(Float)
    pnl = Column(Float, nullable=True)
    status = Column(String) # open/closed

class StrategyModel(Base):
    __tablename__ = 'strategies'
    
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    origin = Column(String) # evolution, ai, user
    regime = Column(String) # UPTREND, DOWNTREND, SIDEWAYS
    genes = Column(String) # JSON or String representation
    winrate = Column(Float)
    trades = Column(Integer)
    fitness = Column(Float)

class DataManager:
    def __init__(self, db_url=config.DB_URL):
        self.engine = create_engine(db_url)
        print(f"🔌 Database Connection: {self.engine.url}")
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.exchange = None
        
        # Initial connection if keys are present (optional, can be overridden)
        if config.KRAKEN_API_KEY:
            self.connect_exchange(config.KRAKEN_API_KEY, config.KRAKEN_SECRET)

    def connect_exchange(self, api_key, secret, demo_mode=False):
        """Connects to Kraken Futures with provided keys."""
        try:
            exchange_config = {
                'apiKey': api_key,
                'secret': secret,
                'enableRateLimit': True,
            }
            
            if demo_mode:
                self.exchange = ccxt.krakenfutures(exchange_config)
                # Correct way to enable Sandbox/Demo environment in CCXT
                print("⚠️  USING DEMO/SANDBOX ENVIRONMENT ⚠️")
                self.exchange.set_sandbox_mode(True)
                # Explicitly override URLs just in case CCXT defaults are old
                # self.exchange.urls['api'] = {
                #     'public': 'https://demo-futures.kraken.com/derivatives/api/v3',
                #     'private': 'https://demo-futures.kraken.com/derivatives/api/v3',
                # }
            else:
                self.exchange = ccxt.krakenfutures(exchange_config)
            self.exchange.load_markets()
            print("Connected to Kraken Futures.")
            return True
        except Exception as e:
            print(f"Failed to connect to Kraken: {e}")
            self.exchange = None
            return False
        
    def fetch_historical_data(self, symbol=config.SYMBOL, timeframe=config.TIMEFRAME, limit=config.LIMIT, since=None):
        """Fetches historical OHLCV data from Kraken."""
        print(f"Fetching {limit if limit else 'all'} candles for {symbol} ({timeframe}) since {since}...")
        try:
            if not self.exchange:
                print("Exchange not connected.")
                return pd.DataFrame()

            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit, since=since)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            if not df.empty:
                 self.save_data(df, symbol, timeframe)
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            if self.exchange and "does not have market symbol" in str(e):
                print("DEBUG: Available Symbols:", [m for m in self.exchange.markets.keys() if 'BTC' in m or 'XBT' in m])
            return pd.DataFrame() # Empty DF

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

            # YFinance 1m data limit check (7 days strict)
            if interval == '1m':
                print("ℹ️ Note: 1m data on YFinance is strictly limited to the last 7 days.")
                limit_date = datetime.now() - pd.Timedelta(days=7)
                # Force start date to be within limit
                start_date_str = limit_date.strftime("%Y-%m-%d")
                print(f"🔄 Adjusting start date to {start_date_str} for 1m interval.")
            
            # Download
            print(f"⬇️ Downloading {interval} data for {yf_symbol} starting {start_date_str}...")
            df_yf = yf.download(yf_symbol, start=start_date_str, interval=interval, progress=False, auto_adjust=True)
            
            if df_yf.empty:
                print(f"❌ No data found on YFinance for {yf_symbol} (Interval: {interval}).")
                # Fallback: Try fetching recent data from Kraken directly via CCXT as backup
                print("🔄 Trying direct Kraken fetch for recent data...")
                return self.fetch_historical_data(symbol, timeframe, limit=1440*7) # Try to get last ~7 days from Kraken directly via API


            # Flatten MultiIndex columns if present (common in new yfinance)
            # Flatten MultiIndex columns if present (common in new yfinance)
            if isinstance(df_yf.columns, pd.MultiIndex):
                # Check if it's the new format with Ticker as level 1
                if df_yf.columns.nlevels >= 2:
                     df_yf.columns = df_yf.columns.get_level_values(0)
                else:
                    df_yf.reset_index(inplace=True) # Sometimes index is involved

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
            print(f"✅ Downloaded {len(df_final)} candles from YFinance. Saving to database...")
            self.save_data(df_final, symbol, timeframe=timeframe)
            print("Bulk import complete.")
            return df_final

        except Exception as e:
            print(f"Error in bulk download: {e}")
            return pd.DataFrame()

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
                
                # Handle potential string timestamp (if not converted properly)
                ts = row['timestamp']
                if isinstance(ts, str):
                    ts = pd.to_datetime(ts)
                
                # Convert to pydatetime if it's a pandas Timestamp
                if hasattr(ts, 'to_pydatetime'):
                    ts = ts.to_pydatetime()
                    
                market_data = MarketData(
                    timestamp=ts,
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
            # SQLite returns strings for dates, ensure datetime
            if not df.empty and 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
            if not df.empty and limit:
                df = df.tail(limit)
            return df
        except Exception as e:
            print(f"Error reading DB: {e}")
            return pd.DataFrame()

    def set_leverage(self, leverage, symbol=config.SYMBOL):
        """Sets leverage for the given symbol on Kraken Futures."""
        if not self.exchange or not self.exchange.apiKey:
            return # Public mode only
            
        try:
            # Kraken Futures often sets leverage per order or account config. 
            # CCXT unify this setLeverage if supported.
            # For Kraken Futures, we might need to be specific.
            # Checking implicit support:
            self.exchange.set_leverage(leverage, symbol)
            print(f"Leverage set to {leverage}x for {symbol}")
        except Exception as e:
            print(f"Error setting leverage: {e}")

    def execute_order(self, symbol, side, amount, type='market', price=None, params={}):
        """Executes an order on Kraken Futures."""
        if not self.exchange or not self.exchange.apiKey:
            print("[SIMULATION] execute_order called without API keys.")
            return None
            
        try:
            # Kraken Futures Specific Logic
            # CCXT Unified is good, but Futures order types can be tricky.
            # 'stop' -> type='stop-loss', needs 'stopPrice' in params (or triggerPrice)
            # 'take_profit' -> type='take-profit', needs 'stopPrice' in params
            
            # Map simplified types to CCXT/Kraken types
            if type == 'stop':
                type = 'stop-loss' # Kraken Futures specific
                if price and 'stopPrice' not in params:
                    params['stopPrice'] = price
                # Ensure price arg is None for market stop (or use limit if intended)
                # Usually we want Market Stop for guaranteed exit
                price = None 
                
            elif type == 'take_profit':
                type = 'take-profit'
                if price and 'stopPrice' not in params:
                    params['stopPrice'] = price
                price = None

            print(f"Executing {side} {type} order for {amount} {symbol}...")
            # For Kraken Futures, check if they support stopLoss/takeProfit in params of create_order
            # Otherwise might need separate orders.
            # CCXT unified params usually work for simple SL/TP if exchange supports 'stopLossPrice'/'takeProfitPrice'
            
            order = self.exchange.create_order(symbol, type, side, amount, price, params)
            print(f"Order Executed: {order['id']}")
            return order
        except Exception as e:
            print(f"Error executing order: {e}")
            return None

    def get_balance(self):
        """Fetches total account balance (USD/USDT usually)"""
        if not self.exchange or not self.exchange.apiKey:
            return 0.0
        
        try:
            balance = self.exchange.fetch_balance()
            # Kraken Futures usually has 'total' in 'info' or specific keys like 'PF_USD' 
            # CCXT usually maps 'total' -> {'USDT': ..., 'BTC': ...}
            # Let's try to get total approximate USD value or free margin
            
            # Common structure: balance['total']['USD'] or balance['free']['USD']
            # For simplicity, returning total USD collateral
            if 'USD' in balance['total']:
                return balance['total']['USD']
            elif 'USDT' in balance['total']:
                return balance['total']['USDT']
            else:
                # Fallback to total equity if available
                return balance.get('info', {}).get('totalWalletBalance', 0.0) # Example key
        except Exception as e:
            print(f"Error fetching balance: {e}")
            return 0.0

    def save_trade(self, symbol, side, amount, price, status='open'):
        """Saves a trade to the database."""
        session = self.Session()
        try:
            trade = Trade(symbol=symbol, side=side, amount=amount, price=price, status=status)
            session.add(trade)
            session.commit()
            print(f"Trade saved: {side} {amount} {symbol} @ {price}")
        except Exception as e:
            session.rollback()
            print(f"Error saving trade: {e}")
        finally:
            session.close()

    def get_recent_trades(self, limit=50):
        """Fetches recent trades."""
        session = self.Session()
        try:
            trades = session.query(Trade).order_by(Trade.timestamp.desc()).limit(limit).all()
            # Convert to list of dicts for easy consumption
            return [{
                'id': t.id,
                'time': t.timestamp,
                'symbol': t.symbol,
                'side': t.side,
                'amt': t.amount,
                'price': t.price,
                'status': t.status
            } for t in trades]
        except Exception as e:
            print(f"Error fetching trades: {e}")
            return []
        finally:
            session.close()

    def save_strategy(self, genome, origin='evolution', regime='ANY'):
        """Saves a strategy genome to the database."""
        session = self.Session()
        try:
            # Avoid saving duplicates (optional, based on genes string)
            # For massive evolution, maybe only save if fitness > X?
            # User wants to catalog created strategies.
            
            strat = StrategyModel(
                origin=origin,
                regime=regime,
                genes=str(genome),
                winrate=genome.winrate,
                trades=genome.trades,
                fitness=genome.fitness
            )
            session.add(strat)
            session.commit()
            # Assign ID back to genome if needed
            if hasattr(genome, 'id'):
                genome.id = strat.id
            return strat.id
        except Exception as e:
            session.rollback()
            print(f"Error saving strategy: {e}")
            return None
        finally:
            session.close()
            
    def get_top_strategies(self, limit=50):
        """Fetches top strategies by fitness."""
        session = self.Session()
        try:
            # Order by fitness desc
            strats = session.query(StrategyModel).order_by(StrategyModel.fitness.desc()).limit(limit).all()
            return [{
                'id': s.id,
                'regime': s.regime,
                'winrate': s.winrate,
                'trades': s.trades,
                'genes': s.genes,
                'origin': s.origin
            } for s in strats]
        except Exception as e:
            print(f"Error fetching strategies: {e}")
            return []
        finally:
            session.close()

if __name__ == "__main__":
    # Test the module
    dm = DataManager()
    print("Fetching data from Kraken...")
    df = dm.fetch_historical_data()
    print(df.head())
    print("Saving to DB...")
    dm.save_data(df)
    print("Done.")
