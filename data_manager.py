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
        self.demo_mode = False
        if config.KRAKEN_API_KEY:
            self.connect_exchange(config.KRAKEN_API_KEY, config.KRAKEN_SECRET)
            
        self.leverage = config.LEVERAGE # Default leverage

    def connect_exchange(self, api_key, secret, demo_mode=False, trading_mode="Spot Margin"):
        """
        Connects to Kraken Spot or Futures based on trading_mode.
        trading_mode: "Spot Margin (10x)" or "Futures (50x)"
        """
        self.demo_mode = demo_mode
        self.trading_mode = trading_mode
        is_futures = "Futures" in trading_mode
        
        print(f"🔌 Connecting to Kraken [{trading_mode}] (Demo: {demo_mode})...")
        
        try:
            exchange_config = {
                'apiKey': api_key,
                'secret': secret,
                'enableRateLimit': True,
            }
            
            if is_futures:
                # FUTURES CONNECTION
                if demo_mode:
                    print("⚠️  USING FUTURES SANDBOX ⚠️")
                    self.exchange = ccxt.krakenfutures(exchange_config)
                    self.exchange.set_sandbox_mode(True)
                else:
                    self.exchange = ccxt.krakenfutures(exchange_config)
            else:
                # SPOT CONNECTION
                exchange_config['options'] = {'defaultType': 'spot'}
                if demo_mode:
                     print("⚠️  DEMO MODE REQUESTED: Kraken Spot (Dry Run).")
                     # Spot doesn't have Sandbox usually active for public
                     self.exchange = ccxt.kraken(exchange_config)
                else:
                    self.exchange = ccxt.kraken(exchange_config)
            
            self.exchange.load_markets()
            print(f"Connected to Kraken {'Futures' if is_futures else 'Spot'}.")
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

    def fetch_multi_timeframe_data(self, symbol=config.SYMBOL, limit=config.LIMIT):
        """
        Fetches 1m data and enriches it with 5m and 15m indicators.
        Returns a single DataFrame with 1m granularity but higher timeframe context.
        """
        # 1. Fetch Data
        df_1m = self.fetch_historical_data(symbol, '1m', limit)
        if df_1m.empty: return df_1m
        
        # Calculate 1m indicators (Trader usually does this, but we can do it here or let Trader does it)
        # Let's let Trader/AI Brain handle 1m indicators to avoid double calcs, 
        # BUT we need 5m/15m indicators pre-calculated.
        
        # Fetch Higher Timeframes
        # We need enough 5m/15m candles to cover the 1m range
        limit_htf = limit // 5 + 100 
        df_5m = self.fetch_historical_data(symbol, '5m', limit_htf)
        df_15m = self.fetch_historical_data(symbol, '15m', limit_htf)
        
        from technical_analysis import TechnicalAnalysis # Local import to avoid circular dep if any
        
        # 2. Process 5m
        if not df_5m.empty:
            ta_5m = TechnicalAnalysis(df_5m)
            df_5m = ta_5m.add_all_indicators()
            # Select relevant columns and rename
            cols_to_keep = ['timestamp', 'rsi', 'macd', 'bb_width', 'ema_200', 'supertrend']
            df_5m = df_5m[cols_to_keep].copy()
            df_5m.columns = ['timestamp'] + [f"{c}_5m" for c in cols_to_keep if c != 'timestamp']
        
        # 3. Process 15m
        if not df_15m.empty:
            ta_15m = TechnicalAnalysis(df_15m)
            df_15m = ta_15m.add_all_indicators()
            cols_to_keep = ['timestamp', 'rsi', 'macd', 'bb_width', 'ema_200', 'supertrend']
            df_15m = df_15m[cols_to_keep].copy()
            df_15m.columns = ['timestamp'] + [f"{c}_15m" for c in cols_to_keep if c != 'timestamp']

        # 4. Merge (Forward Fill)
        # Sort by timestamp
        df_1m = df_1m.sort_values('timestamp')
        
        if not df_5m.empty:
            df_5m = df_5m.sort_values('timestamp')
            df_1m = pd.merge_asof(df_1m, df_5m, on='timestamp', direction='backward')
            
        if not df_15m.empty:
            df_15m = df_15m.sort_values('timestamp')
            df_1m = pd.merge_asof(df_1m, df_15m, on='timestamp', direction='backward')
            
        # Fill NaNs (for early candles where 5m/15m might not align perfectly or start later)
        df_1m.fillna(method='ffill', inplace=True)
        df_1m.fillna(0, inplace=True)
        
        print(f"✅ Multi-Timeframe Merge Complete. Shape: {df_1m.shape}")
        return df_1m

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
        """Sets leverage preference for Kraken Spot Margin."""
        self.leverage = leverage
        print(f"Leverage set to {leverage}x (Applied per order on Kraken Spot)")

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
                type = 'stop-loss'
                # For CCXT Kraken: `price` argument is usually the stop trigger price for market stops
                # But params['stopPrice'] is safer for unification.
                if price:
                     params['stopPrice'] = price
                price = None 
                
            elif type == 'take_profit':
                type = 'take-profit'
                if price:
                     params['stopPrice'] = price
                price = None
                
            # Spot Margin DOES support 'reduceOnly' for Limit orders (verified).
            # Keeping it if passed.
            
            print(f"Executing {side} {type} order for {amount} {symbol}...")
            
            # Add Leverage to params
            if hasattr(self, 'leverage') and self.leverage > 1:
                params['leverage'] = self.leverage
                
            # Kraken Spot specific handling for Stop/TP
            # if type == 'stop-loss' or type == 'take-profit':
                 # params['price'] might be needed if it's a Limit Stop, but for Market Stop check CCXT docs.
                 # Usually 'price' arg in create_order is the Limit Price, 
                 # and 'stopPrice' in params is the trigger.
                 # If price is None, it's a Market order triggered at stopPrice.
                 
            # DRY RUN CHECK
            if self.demo_mode:
                print(f"[DRY-RUN] Simulating Order: {side} {type} {amount} {symbol} (Lev: {params.get('leverage', 'None')})")
                # Return fake order
                import random
                fake_id = f"demo_{int(time.time())}_{random.randint(1000,9999)}"
                return {
                    'id': fake_id,
                    'symbol': symbol,
                    'side': side,
                    'amount': amount,
                    'price': price if price else 0.0, # Approximate
                    'status': 'closed',
                    'timestamp': int(time.time()*1000)
                }

            order = self.exchange.create_order(symbol, type, side, amount, price, params)
            print(f"Order Executed: {order['id']}")
            return order
        except Exception as e:
            print(f"Error executing order: {e}")
            return {'error': str(e)}

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
            # Prioritize currency with actual balance
            total = balance.get('total', {})
            usd_bal = total.get('USD', 0.0)
            usdt_bal = total.get('USDT', 0.0)
            
            if usd_bal > 0:
                return usd_bal
            elif usdt_bal > 0:
                return usdt_bal
            else:
                return usd_bal # Return 0.0 if both are empty
            
            # Legacy fallback
            return balance.get('info', {}).get('totalWalletBalance', 0.0)
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

    def get_open_trades(self):
        """Fetches all open trades."""
        session = self.Session()
        try:
            trades = session.query(Trade).filter(Trade.status == 'open').all()
            return [{
                'id': t.id,
                'symbol': t.symbol,
                'side': t.side,
                'amount': t.amount,
                'price': t.price,
                'timestamp': t.timestamp
            } for t in trades]
        except Exception as e:
            print(f"Error fetching open trades: {e}")
            return []
        finally:
            session.close()

    def update_trade_status(self, trade_id, status, pnl=None):
        """Updates trade status and PNL."""
        session = self.Session()
        try:
            trade = session.query(Trade).filter(Trade.id == trade_id).first()
            if trade:
                trade.status = status
                if pnl is not None:
                    trade.pnl = pnl
                session.commit()
                print(f"Trade {trade_id} updated: {status} (PNL: {pnl})")
        except Exception as e:
            session.rollback()
            print(f"Error updating trade: {e}")
        finally:
            session.close()

    def get_weekly_pnl(self):
        """Calculates total PNL for the last 7 days."""
        session = self.Session()
        try:
            seven_days_ago = datetime.utcnow() - pd.Timedelta(days=7)
            # Sum PNL of closed trades in last 7 days
            result = session.query(Trade).filter(
                Trade.status == 'closed', 
                Trade.timestamp >= seven_days_ago
            ).all()
            
            total_pnl = sum(t.pnl for t in result if t.pnl is not None)
            return total_pnl
        except Exception as e:
            print(f"Error calculating weekly PNL: {e}")
            return 0.0
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
