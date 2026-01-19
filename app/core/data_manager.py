import ccxt
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import time
from app.utils import config
import json

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
    status = Column(String, default='lab') # lab (validation), active (promoted), archived

class DataManager:
    def __init__(self, db_url=config.DB_URL):
        # Initial connection if keys are present (optional, can be overridden)
        self.demo_mode = False
        
        # SQLITE OPTIMIZATIONS FOR CONCURRENCY (WAL MODE)
        if 'sqlite' in db_url:
            connect_args = {'check_same_thread': False}
            self.engine = create_engine(db_url, connect_args=connect_args)
            
            # Enable Write-Ahead Logging (WAL) to allow concurrent reads/writes
            with self.engine.connect() as conn:
                try:
                    conn.exec_driver_sql("PRAGMA journal_mode=WAL;")
                    conn.exec_driver_sql("PRAGMA busy_timeout=5000;") # Wait up to 5s if locked
                    conn.exec_driver_sql("PRAGMA synchronous=NORMAL;") # Faster writes, safe enough for WAL
                    print("✅ SQLite WAL Mode Enabled (Non-blocking I/O).")
                except Exception as e:
                    print(f"Warning: Could not enable WAL mode: {e}")
        else:
             self.engine = create_engine(db_url)
             
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
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
            if "Spot" in trading_mode:
                print(f"Connected to Kraken {'Futures' if is_futures else 'Spot'}.")
            elif is_futures:
                 # Only print explicitly for Futures if user wants confirmation, 
                 # but based on "esse log de connect kraken spot n precisa mais exibir", 
                 # we can keep Futures confirmation but ensure no Spot noise leaks.
                 print("Connected to Kraken Futures.")
            return True
        except Exception as e:
            print(f"Failed to connect to Kraken: {e}")
            self.exchange = None
            return False
        
    def fetch_historical_data(self, symbol=None, timeframe=None, limit=None, since=None):
        """Fetches historical OHLCV data from Kraken."""
        symbol = symbol or config.SYMBOL
        timeframe = timeframe or config.TIMEFRAME
        limit = limit or config.LIMIT
        
        # print(f"Fetching {limit if limit else 'all'} candles for {symbol} ({timeframe}) since {since}...")

        try:
            if not self.exchange:
                print("Exchange not connected.")
                return pd.DataFrame()

            # Retry logic for data fetching
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit, since=since)
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    if not df.empty:
                        self.save_data(df, symbol, timeframe)
                    return df
                except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
                     if attempt < max_retries - 1:
                         time.sleep(2)
                         continue
                     print(f"⚠️ Network error fetching data ({timeframe}): {e}")
                     return pd.DataFrame()
                except Exception as e:
                     if "socket" in str(e) or "NameResolutionError" in str(e):
                         if attempt < max_retries - 1:
                             time.sleep(2)
                             continue
                         print(f"⚠️ Connection lost fetching data ({timeframe}).")
                         return pd.DataFrame()
                     raise e

        except Exception as e:
            print(f"Error fetching data: {e}")
            if self.exchange and "does not have market symbol" in str(e):
                print("DEBUG: Available Symbols:", [m for m in self.exchange.markets.keys() if 'BTC' in m or 'XBT' in m])
            return pd.DataFrame() # Empty DF

    def fetch_multi_timeframe_data(self, symbol=None, limit=None):
        """
        Fetches 1m data and enriches it with 5m and 15m indicators.
        Returns a single DataFrame with 1m granularity but higher timeframe context.
        """
        symbol = symbol or config.SYMBOL
        limit = limit or config.LIMIT
        
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
        
        from app.core.technical_analysis import TechnicalAnalysis # Local import to avoid circular dep if any
        
        # 2. Process 5m
        if not df_5m.empty:
            ta_5m = TechnicalAnalysis(df_5m)
            df_5m = ta_5m.add_all_indicators()
            # Select relevant columns and rename
            cols_to_keep = ['timestamp', 'rsi', 'macd', 'bb_width', 'ema_200', 'supertrend', 'stoch_k', 'stoch_rsi_k']
            df_5m = df_5m[cols_to_keep].copy()
            df_5m.columns = ['timestamp'] + [f"{c}_5m" for c in cols_to_keep if c != 'timestamp']
        
        # 3. Process 15m
        if not df_15m.empty:
            ta_15m = TechnicalAnalysis(df_15m)
            df_15m = ta_15m.add_all_indicators()
            cols_to_keep = ['timestamp', 'rsi', 'macd', 'bb_width', 'ema_200', 'supertrend', 'stoch_k', 'stoch_rsi_k']
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
        df_1m.ffill(inplace=True)
        df_1m.fillna(0, inplace=True)
        
        # df_1m.fillna(0, inplace=True)
        # print(f"✅ Multi-Timeframe Merge Complete. Shape: {df_1m.shape}")
        return df_1m

    def fetch_full_history(self, start_year=2020, symbol=None, timeframe=None):
        """Fetches full history using yfinance for bulk data (bypassing Kraken limits)."""
        import yfinance as yf
        symbol = symbol or config.SYMBOL
        timeframe = timeframe or config.TIMEFRAME
        
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
            if count > 0:
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
                params['leverage'] = int(self.leverage) # Force int for Kraken API
                
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

    def get_open_positions(self):
        """Fetches open positions from the exchange (Futures only)."""
        if not self.exchange: return []
        try:
            # Only Futures has 'positions'. Spot has 'open orders' but not 'positions' in the same sense.
            if hasattr(self.exchange, 'fetch_positions'):
                positions = self.exchange.fetch_positions()
                # Filter for active positions
                active = [p for p in positions if float(p['contracts']) > 0]
                return active
            return []
        except Exception as e:
            print(f"Error fetching positions: {e}")
            return []

    def get_balance(self, type='free'):
        """
        Fetches account balance (USD/USDT).
        type: 'free' (Available for trade) or 'total' (Equity).
        """
        if not self.exchange or not self.exchange.apiKey:
            return 0.0
        
        # Retry logic for network flakiness
        max_retries = 3
        for attempt in range(max_retries):
            try:
                balance = self.exchange.fetch_balance()
                
                # Select Free or Total
                bal_data = balance.get(type, {}) if type in balance else balance.get('total', {})
                
                # 1. Standard CCXT (Spot/Margin)
                usd_bal = bal_data.get('USD', 0.0)
                zusd_bal = bal_data.get('ZUSD', 0.0)
                usdt_bal = bal_data.get('USDT', 0.0)
                usdt_b_bal = bal_data.get('USDT.B', 0.0) # Bridged Tether
                usdt_m_bal = bal_data.get('USDT.M', 0.0) # Margin Tether?
                
                # 2. CCXT Unified (Futures often here)
                # Check balance['USD']['free'] directly if not found above
                if usd_bal == 0:
                    usd_bal = float(balance.get('USD', {}).get(type, 0.0))

                # 3. Kraken Futures Specific Keys
                pf_usd = bal_data.get('PF_USD', 0.0)
                pf_xbt = bal_data.get('PF_XBTUSD', 0.0) # Sometimes XBT balance shows as equity
                
                # 4. Deep Inspection (Flex Wallet)
                flex_usd = 0.0
                if 'info' in balance:
                    try:
                        if 'accounts' in balance['info']:
                             accounts = balance['info']['accounts']
                             if 'flex' in accounts:
                                 currencies = accounts['flex'].get('currencies', {})
                                 if 'USD' in currencies:
                                     flex_usd = float(currencies['USD'].get('quantity', 0.0))
                    except: pass

                # Sum all sources
                # FIX: Flex USD often mirrors USD balance. Only add if distinct or if main USD is 0.
                if usd_bal > 0 and flex_usd > 0 and abs(usd_bal - flex_usd) < 0.01:
                    # Likely same balance reported twice
                    final_bal = usd_bal + zusd_bal + usdt_bal + usdt_b_bal + usdt_m_bal + pf_usd
                else:
                    final_bal = usd_bal + zusd_bal + usdt_bal + usdt_b_bal + usdt_m_bal + pf_usd + flex_usd
                
                return final_bal
                
            except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1)) # Backoff: 2s, 4s, ...
                    continue
                print(f"⚠️ Network error fetching balance: {e}")
                return 0.0
            except Exception as e:
                # Filter out verbose socket/SSL errors from user view, just log strict warning
                if "socket" in str(e) or "NameResolutionError" in str(e):
                     if attempt < max_retries - 1:
                         time.sleep(2)
                         continue
                     print("⚠️ Connection lost while checking balance (DNS/Network).")
                     return 0.0
                
                print(f"Error fetching balance: {e}")
                # import traceback
                # traceback.print_exc()
                return 0.0
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

    def get_24h_pnl(self):
        """Calculates total PNL for the last 24 hours (DB Based)."""
        session = self.Session()
        try:
            one_day_ago = datetime.utcnow() - pd.Timedelta(hours=24)
            result = session.query(Trade).filter(
                Trade.status == 'closed', 
                Trade.timestamp >= one_day_ago
            ).all()
            
            total_pnl = sum(t.pnl for t in result if t.pnl is not None)
            return total_pnl
        except Exception as e:
            print(f"Error calculating 24h PNL: {e}")
            return 0.0
        finally:
            session.close()

    def get_api_pnl_24h(self):
        """Calculates total PNL for the last 24 hours directly from Kraken API."""
        if not self.exchange:
            # Fallback to DB if no exchange connection
            return self.get_24h_pnl()
            
        try:
            # Calculate timestamp 24h ago in ms
            since_ts = int((time.time() - 24 * 3600) * 1000)
            
            # Fetch closed orders seems most reliable for cost/net
            # Note: Detailed PnL per trade might be in 'trades' or 'ledger'
            # Kraken Futures uses 'fetchMyTrades' or 'fetchClosedOrders'
            # Let's try fetch_closed_orders first
            
            pnl = 0.0
            
            # Use fetch_my_trades which usually contains realized pnl info
            my_trades = self.exchange.fetch_my_trades(symbol=None, since=since_ts, limit=100)
            
            for t in my_trades:
                # Check if it has 'realizedPnl' (Futures) or calculate from cost/fee (Spot)
                # Kraken Futures usually provides 'info' -> 'realizedPnl'
                
                # Check widely accessible logic first
                # If Futures
                if 'info' in t and isinstance(t['info'], dict) and 'realizedPnl' in t['info']:
                     pnl += float(t['info'].get('realizedPnl', 0))
                # Fallback for generic structure?
                elif 'realizedPnl' in t: # Some CCXT drivers normalize it
                     pnl += float(t.get('realizedPnl', 0))
                
                # If Spot, it's harder. Realized PnL is diff between buy/sell. 
                # This is complex for Spot. 
                # Assuming Futures/Margin primarily as requested.
                
            # If 0 and we have trades, maybe we are in Spot mode? 
            # In Spot, PnL is conceptual until measured against entry. 
            # Let's trust logic for now.
            
            return pnl
            
        except Exception as e:
            error_msg = str(e)
            if "Invalid key" in error_msg or "permission" in error_msg.lower():
                # Suppress spam for key permission errors
                pass
            else:
                print(f"Error fetching API PnL: {e}")
            return self.get_24h_pnl() # Fallback

    def save_strategy(self, genome, origin='evolution', regime='ANY', status='lab'):
        """Saves a strategy genome to the database."""
        session = self.Session()
        try:
            # Avoid saving duplicates (optional, based on genes string)
            # For massive evolution, maybe only save if fitness > X?
            # User wants to catalog created strategies.
            
            # Handle Genome Object vs Dictionary
            if isinstance(genome, dict):
                genes_str = json.dumps(genome.get('params', genome))
                win = genome.get('winrate', 0)
                trd = genome.get('trades', 0)
                fit = genome.get('fitness', 0)
            else:
                # Assume Object (Genome class)
                if hasattr(genome, 'params'):
                    genes_str = json.dumps(genome.params)
                else:
                    genes_str = str(genome)
                win = genome.winrate
                trd = genome.trades
                fit = genome.fitness

            strat = StrategyModel(
                origin=origin,
                regime=regime,
                genes=genes_str,
                winrate=win,
                trades=trd,
                fitness=fit,
                status=status
            )
            session.add(strat)
            session.commit()
            
            # Assign ID back if object
            if not isinstance(genome, dict) and hasattr(genome, 'id'):
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

    def demote_strategy(self, strat_id, reason="performance"):
        """Demotes a strategy to archived status."""
        session = self.Session()
        try:
            strat = session.query(StrategyModel).filter(StrategyModel.id == strat_id).first()
            if strat:
                old_status = strat.status
                strat.status = 'archived'
                # Optionally log reason in a new column or just console
                print(f"📉 Strategy {strat_id} DEMOTED ({old_status} -> archived). Reason: {reason}")
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            print(f"Error demoting strategy: {e}")
            return False
        finally:
            session.close()

    def get_active_strategies(self):
        """Fetches all strategies currently marked as active."""
        session = self.Session()
        try:
            strats = session.query(StrategyModel).filter(StrategyModel.status == 'active').all()
            return [{
                'id': s.id,
                'regime': s.regime,
                'winrate': s.winrate,
                'trades': s.trades,
                'genes': s.genes,
                'origin': s.origin
            } for s in strats]
        except Exception as e:
            print(f"Error fetching active strategies: {e}")
            return []
        finally:
            session.close()

    def get_best_active_strategy(self):
        """Fetches the single best strategy marked as 'active'."""
        session = self.Session()
        try:
            strat = session.query(StrategyModel).filter_by(status='active').order_by(StrategyModel.fitness.desc(), StrategyModel.id.desc()).first()

            if strat:
                 return {
                    'id': strat.id,
                    'regime': strat.regime,
                    'winrate': strat.winrate,
                    'trades': strat.trades,
                    'genes': strat.genes,
                    'origin': strat.origin
                }
            return None
        except Exception as e:
            print(f"Error fetching best active strategy: {e}")
            return None
        finally:
            session.close()

    def get_best_lab_strategy(self):
        """Fetches the best candidate from the lab to promote."""
        session = self.Session()
        try:
            # Get best 'lab' strategy with decent metrics
            strat = session.query(StrategyModel).filter(
                StrategyModel.status == 'lab'
            ).order_by(StrategyModel.fitness.desc()).first()
            
            if strat:
                 return {
                    'id': strat.id,
                    'regime': strat.regime,
                    'winrate': strat.winrate,
                    'trades': strat.trades,
                    'genes': strat.genes
                }
            return None
        except Exception as e:
            print(f"Error fetching best lab strategy: {e}")
            return None
        finally:
            session.close()

    def promote_strategy(self, strat_id):
        """Promotes a lab strategy to active."""
        session = self.Session()
        try:
            strat = session.query(StrategyModel).filter(StrategyModel.id == strat_id).first()
            if strat:
                strat.status = 'active'
                session.commit()
                print(f"🏆 Strategy {strat_id} PROMOTED to ACTIVE!")
                return True
            return False
        except Exception as e:
            session.rollback()
            return False
        finally:
            session.close()

    def fetch_binance_history(self, symbol=config.SYMBOL, timeframe='1m', months=6):
        """
        Fetches deep historical data from Binance (BTC/USDT) which usually has much longer 1m retention.
        Maps it to the requested symbol (e.g. BTC/USD) and saves to DB.
        """
        print(f"🌍 Conectando à Binance para baixar histórico profundo de {months} meses...")
        try:
            binance = ccxt.binance()
            # Map symbol: Internal 'BTC/USD' -> Binance 'BTC/USDT'
            # (Assuming high correlation, adequate for pattern recognition training)
            remote_symbol = 'BTC/USDT'
            
            end_date = datetime.now()
            start_date = end_date - pd.Timedelta(days=30 * months)
            since_ts = int(start_date.timestamp() * 1000)
            
            all_candles = []
            
            # Binance limits: 1000 candles per call
            limit = 1000
            current_since = since_ts
            
            while True:
                current_date_str = pd.to_datetime(current_since, unit='ms').strftime('%Y-%m-%d %H:%M')
                print(f"   -> Baixando lote Binance a partir de {current_date_str}...")
                
                ohlcv = binance.fetch_ohlcv(remote_symbol, timeframe, limit=limit, since=current_since)
                if not ohlcv:
                    print("   -> Lote vazio. Fim do histórico.")
                    break
                
                all_candles.extend(ohlcv)
                
                # Check timestamps
                last_candle_ts = ohlcv[-1][0]
                
                # Stop if we reached close to now
                if last_candle_ts >= int(end_date.timestamp() * 1000) - 60000:
                    break
                    
                # Setup next call
                current_since = last_candle_ts + 60000 # +1 min
                
                # Safety break to avoid infinite loop
                if len(ohlcv) < limit: 
                    # If we got less than limit, we likely hit head
                    break
                    
                time.sleep(0.1) # Be nice to public API
                
            print(f"✅ Download Binance Concluído: {len(all_candles)} velas.")
            
            if not all_candles:
                return

            # Convert to DataFrame
            df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Save to DB under the LOCAL symbol (BTC/USD)
            self.save_data(df, symbol=symbol, timeframe=timeframe)
            
        except Exception as e:
            print(f"❌ Erro ao baixar da Binance: {e}")

    def fetch_deep_history(self, symbol=config.SYMBOL, timeframe='1m', months=6):
        """
        Wrapper to decide best source for deep history.
        """
        # Strategy: Try Binance first for deep 1m data as it is more reliable for 6+ months history.
        self.fetch_binance_history(symbol, timeframe, months)

if __name__ == "__main__":
    # Test the module
    dm = DataManager()
    print("Fetching data from Kraken...")
    df = dm.fetch_historical_data()
    print(df.head())
    print("Saving to DB...")
    dm.save_data(df)
    print("Done.")
