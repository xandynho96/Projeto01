import pandas as pd
import numpy as np
from data_manager import DataManager
import config
from technical_analysis import TechnicalAnalysis
import json
import os
import time

class Backtester:
    def __init__(self, initial_capital=1000.0, fee_pct=0.05, symbol='PF_XBTUSD', timeframe='1m'):
        self.dm = DataManager()
        
        # Load Config for Keys/Demo Mode
        try:
            if os.path.exists("dist/user_config.json"):
                path = "dist/user_config.json"
            elif os.path.exists("user_config.json"):
                path = "user_config.json"
            else:
                path = None
                
            if path:
                with open(path, 'r') as f:
                    data = json.load(f)
                    api_key = data.get('api_key')
                    secret = data.get('secret')
                    demo_mode = data.get('demo_mode', False)
                    
                    if api_key and secret:
                        print(f"🔑 Loading keys from {path} (Demo: {demo_mode})")
                        self.dm.connect_exchange(api_key, secret, demo_mode=demo_mode)
                    else:
                        print("ℹ️ Config found but no keys. Using Public/Demo connection.")
                        self.dm.connect_exchange('', '', demo_mode=True)
            else:
                 print("⚠️ No config file found. Using Public/Demo connection.")
                 self.dm.connect_exchange('', '', demo_mode=True)

            if self.dm.exchange is None:
                print("❌ CRITICAL: DataManager exchange is None after init.")
        except Exception as e:
            print(f"⚠️ Error loading config: {e}")
        
        self.initial_capital = initial_capital
        self.fee_pct = fee_pct / 100
        self.symbol = symbol
        self.timeframe = timeframe
        self.df = pd.DataFrame()

    def load_data(self, limit=None, fetch_if_missing=False):
        """
        Loads data from DB.
        limit: Number of candles to load. If None, loads ALL available data (Big Data Mode).
        """
        if limit:
             print(f"📉 Loading last {limit} candles for {self.symbol} ({self.timeframe})...")
        else:
             print(f"📉 Loading FULL HISTORY for {self.symbol} ({self.timeframe})...")
        
        # Try DB first
        self.df = self.dm.get_data_from_db(symbol=self.symbol, timeframe=self.timeframe, limit=limit)
        
        # If not enough data and we explicitly want to fetch from Kraken (Live/Demo mode)
        if (self.df.empty or len(self.df) < 100) and fetch_if_missing:
            print("⚠️ Not enough local data. Fetching from Kraken (can take a moment)...")
            self.dm.fetch_historical_data(symbol=self.symbol, timeframe=self.timeframe, limit=1440) # Fetch ~1 day
            self.df = self.dm.get_data_from_db(symbol=self.symbol, timeframe=self.timeframe, limit=limit)

        if not self.df.empty:
            print(f"✅ Loaded {len(self.df)} candles.")
            # Add Indicators
            print("   -> Calculating indicators (this might take a while for large datasets)...")
            ta = TechnicalAnalysis(self.df)
            self.df = ta.add_all_indicators()
            self.df.dropna(inplace=True)
            return True
        else:
            print("❌ Failed to load data. Please run 'download_history.py' to fetch initial dataset.")
            return False

    def run_vectorized_backtest(self, signals):
        """
        Runs a fully vectorized backtest. Faster by orders of magnitude.
        signals: pd.Series of -1 (Short), 0 (Neutral), 1 (Long)
        """
        if self.df.empty: return {}
        
        # 1. Align signals
        # We assume signals are aligned with self.df.index
        # Shift signal by 1 because signal calculated at T affects position at T+1 (conceptually)
        # However, usually we trade AT close or Open of next. 
        # Standard approach: Return = PctChange * Position.shift(1)
        
        # Calculate Percentage Change of Price
        market_rets = self.df['close'].pct_change()
        market_rets.fillna(0, inplace=True)
        
        # Position: We need to turn signals into continuous positions
        # If signal is 1, position becomes 1. If 0, position 0.
        # If signal is 'buy', position stays 1 until 'close'.
        # For simplicity in vectorization, let's assume 'signals' IS the target position.
        
        position = signals.shift(1).fillna(0)
        
        # Strategy Returns
        strat_rets = position * market_rets
        
        # Minus Fees
        # Fee is paid when position CHANGES
        # Change mask
        pos_change = position.diff().abs() # 0 to 1 = 1, 1 to -1 = 2, etc.
        # But fee is % of total transaction value. 
        # 0->1: Pay fee on 1 unit. 1->-1: Close 1, Open 1 (Pay on 2 units? Backtester logic usually simplifies).
        # Let's assume simple fee per turnover.
        costs = pos_change * self.fee_pct 
        
        net_rets = strat_rets - costs
        
        # Equity Curve
        # (1 + r) * (1 + r2)...
        equity_curve = (1 + net_rets).cumprod() * self.initial_capital
        
        # Final Metrics
        final_equity = equity_curve.iloc[-1]
        total_pnl = final_equity - self.initial_capital
        roi = (total_pnl / self.initial_capital) * 100
        
        # Trades Count (Approximate by changes in position)
        trades_count = pos_change[pos_change > 0].count()
        
        # Winrate (Approximate by positive return periods? No, that's candle winrate.)
        # Trade Winrate is hard in full vectorization without loop.
        # Hybrid approach: Vectorized equity, Loop for trade stats (iterating only changes is fast).
        # OR just aggregate non-zero blocks.
        
        # Fast Winrate Calc:
        # Group by trade: create a group ID that increments every time pos_change != 0
        trade_ids = pos_change.cumsum()
        # Sum returns per trade_id (ignoring id 0 if flat)
        trade_pnl = net_rets.groupby(trade_ids).sum()
        # Filter out flat periods (where position was 0) - tricky if no return but pos was 0. 
        # Better: Filter where position was != 0
        
        real_trades_mask = position != 0
        active_trade_ids = trade_ids[real_trades_mask]
        
        if not active_trade_ids.empty:
            trade_returns = net_rets[real_trades_mask].groupby(active_trade_ids).sum()
            wins = (trade_returns > 0).sum()
            total = len(trade_returns)
            win_rate = (wins / total * 100) if total > 0 else 0
        else:
            win_rate = 0
            
        return {
            'equity': final_equity,
            'roi': roi,
            'winrate': win_rate,
            'trades': trades_count
        }

    # --- Vectorized Strategy Generators ---
    def generate_rsi_signals(self, low=30, high=70):
        signals = pd.Series(0, index=self.df.index)
        rsi = self.df['rsi']
        signals[rsi < low] = 1
        signals[rsi > high] = -1
        # Fill 'hold' logic: typically we want to STAY in position until exit condition.
        # Simple Vector Signal: 1 (Long), -1 (Short), 0 (Flat).
        # If we want detailed logic (close only if > 70), we need a stateful apply or 'ffill' approach.
        # Logic: Buy < 30, Hold until > 70 -> Sell/Flat.
        
        # Vectorized State Machine using cumsum/masking is complex. 
        # For 'Acceleration', pure vector is best if simple.
        # Let's start with raw signal: Long if <30, Short if >70. Else 0? No, that's mean reversion.
        return signals

    def generate_ai_signals(self):
        # AI Pred already in DF
        if 'ai_pred' not in self.df.columns: return pd.Series(0, index=self.df.index)
        
        pred = self.df['ai_pred']
        close = self.df['close']
        
        signals = pd.Series(0, index=self.df.index)
        signals[pred > close * 1.001] = 1 # Long
        signals[pred < close * 0.999] = -1 # Short
        return signals

    def run_strategy(self, strategy_name, strategy_func):
        """Runs a backtest for a specific strategy function."""
        if self.df.empty:
            print("No data to backtest.")
            return

        # print(f"\n🧪 Testing Strategy: {strategy_name.upper()}...")
        
        capital = self.initial_capital
        position = 0 # 0, 1 (Long), -1 (Short)
        entry_price = 0.0
        trades = []
        equity_curve = [capital]

        for i in range(len(self.df)):
            row = self.df.iloc[i]
            price = row['close']
            
            # Get Signal from Strategy: 'buy', 'sell', 'hold', 'close'
            signal = strategy_func(row, position)
            
            # Execution Logic (Simplified)
            if signal == 'buy' and position == 0:
                amount = capital / price
                cost = (amount * price) * self.fee_pct
                capital -= cost
                position = 1
                entry_price = price
                trades.append({'type': 'long', 'price': price, 'time': row['timestamp']})
                
            elif signal == 'sell' and position == 0:
                amount = capital / price
                cost = (amount * price) * self.fee_pct
                capital -= cost
                position = -1
                entry_price = price
                trades.append({'type': 'short', 'price': price, 'time': row['timestamp']})

            elif (signal == 'sell' or signal == 'close') and position == 1:
                # Close Long
                revenue = (capital / entry_price) * price
                cost = revenue * self.fee_pct
                pnl = revenue - capital - cost
                capital += pnl
                position = 0
                trades[-1]['pnl'] = pnl
                trades[-1]['exit'] = price
                
            elif (signal == 'buy' or signal == 'close') and position == -1:
                # Close Short
                entry_val = capital
                pct_change = (entry_price - price) / entry_price
                revenue = entry_val * (1 + pct_change)
                cost = revenue * self.fee_pct 
                pnl = revenue - entry_val - cost
                capital += pnl
                position = 0
                trades[-1]['pnl'] = pnl
                trades[-1]['exit'] = price

            equity_curve.append(capital)

        # Metrics
        df_trades = pd.DataFrame(trades)
        total_pnl = capital - self.initial_capital
        roi = (total_pnl / self.initial_capital) * 100
        
        win_rate = 0.0
        if not df_trades.empty and 'pnl' in df_trades.columns:
            wins = len(df_trades[df_trades['pnl'] > 0])
            win_rate = (wins / len(df_trades)) * 100 if len(df_trades) > 0 else 0

        return {
            'equity': capital,
            'roi': roi,
            'winrate': win_rate,
            'trades': len(trades)
        }

# --- Strategy Definitions ---

def strat_rsi_scalp(row, position):
    """Simple RSI Mean Reversion"""
    rsi = row['rsi']
    if rsi < 30 and position == 0:
        return 'buy'
    elif rsi > 70 and position == 0:
        return 'sell'
    elif rsi > 70 and position == 1: # Close long
        return 'close'
    elif rsi < 30 and position == -1: # Close short
        return 'close'
    return 'hold'

def strat_macd_cross(row, position):
    """MACD Signal Line Crossover"""
    macd = row['macd']
    signal = row['macd_signal']
    
    if macd > signal and position == 0:
        return 'buy'
    elif macd < signal and position == 0:
        return 'sell'
    elif macd < signal and position == 1:
        return 'close'
    elif macd > signal and position == -1:
        return 'close'
    return 'hold'

def strat_bb_squeeze(row, position):
    """Bollinger Band Breakout (Simplified)"""
    # Falling back to basic trend follow using available EMA
    close = row['close']
    ema = row['ema_21'] # Using EMA 21 as Trend Baseline
    
    if close > ema * 1.001 and position == 0:
        return 'buy'
    elif close < ema * 0.999 and position == 0:
        return 'sell'
    
    # Trailing stop logic simulation (or simple reversal)
    elif close < ema and position == 1:
        return 'close'
    elif close > ema and position == -1:
        return 'close'
        
    return 'hold'

# --- Continuous Validation Loop ---

def run_continuous_training_loop(bt):
    """
    Runs a continuous loop:
    1. Loads ALL available Data.
    2. Trains AI on this data.
    3. Runs Backtest Validation.
    4. Sleeps and repeats (simulating 'always learning').
    """
    from ai_brain import AIBrain
    brain = AIBrain()
    
    print("🚀 Starting Continuous AI Training & Backtest Loop...")
    print("Press Ctrl+C to stop.")
    
    cycle = 0
    try:
        while True:
            cycle += 1
            print(f"\n\n🔁 CYCLE {cycle} START")
            
            # 1. Update Data (Optional: Run downloader incrementally? For now assume downloader runs separately or data is static for backtest)
            # In a real live loop, we would fetch latest candles here.
            # bt.dm.fetch_historical_data(limit=60) 
            
            # 2. Load ALL Data
            if not bt.load_data(limit=None, fetch_if_missing=False):
                print("Total failure to load data. Retrying in 60s...")
                time.sleep(60)
                continue
                
            # 3. Train AI
            print("🧠 Training AI on updated dataset...")
            brain.train(bt.df)
            
            # 3b. Pre-calculate AI Predictions for Backtest Speed
            print("🔮 Generating AI Predictions for backtest...")
            predictions = brain.predict_batch(bt.df)
            
            if predictions is None:
                print("⚠️ AI Predictions unavailable (Model not ready or no TF). Using Close Price as fallback.")
                bt.df['ai_pred'] = bt.df['close']
            else:
                # Align predictions
                # Sequence length = brain.sequence_length (default 60)
                seq_len = getattr(brain, 'sequence_length', 60)
                padding = [np.nan] * seq_len
                
                # Check dimensions
                if isinstance(predictions, np.ndarray) and predictions.ndim == 0:
                     print("⚠️ Prediction was scalar. Converting to array.")
                     predictions = np.array([predictions])
                
                # If still empty
                if len(predictions) == 0:
                     print("⚠️ Predictions array is empty. Fallback.")
                     bt.df['ai_pred'] = bt.df['close']
                else:
                    try:
                        pred_series = np.concatenate([padding, predictions])
                        if len(pred_series) == len(bt.df):
                            bt.df['ai_pred'] = pred_series
                        else:
                            # Try to force fit or fallback
                            print(f"⚠️ Prediction mismatch ({len(pred_series)} vs {len(bt.df)}). Fallback.")
                            bt.df['ai_pred'] = bt.df['close']
                    except Exception as e:
                        print(f"⚠️ Error aligning predictions: {e}. Fallback.")
                        bt.df['ai_pred'] = bt.df['close'] 
            bt.df['ai_pred'] = 0.0 # Reset
            
            # ... (Existing alignment logic) ...
            try:
                 aligned_preds = bt.align_predictions(brain.predict_batch(bt.df), brain.sequence_length)
                 if len(aligned_preds) == len(bt.df):
                     bt.df['ai_pred'] = aligned_preds
                 else:
                     print("⚠️ Prediction alignment mismatch.")
            except Exception as e:
                 print(f"⚠️ Prediction error: {e}")
            
            # Define AI Strategy Wrapper
            def strat_ai_wrapper(row, position):
                # Simple Logic: If AI predicts price > current * 1.001 -> Buy
                if pd.isna(row['ai_pred']): return 'hold'
                
                pred = row['ai_pred']
                curr = row['close']
                
                if pred > curr * 1.0005 and position == 0:
                    return 'buy'
                elif pred < curr * 0.9995 and position == 0:
                    return 'sell'
                elif pred < curr and position == 1:
                    return 'close'
                elif pred > curr and position == -1:
                    return 'close'
                return 'hold'

            # 4. Run Validation (Vectorized)
            print("🧪 Validating Strategies (Vectorized Mode)...")
            
            results = {}
            # Standard Strategies (Vectorized)
            signals_rsi = bt.generate_rsi_signals()
            results['RSI Scalp'] = bt.run_vectorized_backtest(signals_rsi)
            
            # AI Strategy (Vectorized)
            signals_ai = bt.generate_ai_signals()
            results['AI Enhanced'] = bt.run_vectorized_backtest(signals_ai)
            
            # 5. Dashboard
            print("-" * 80)
            print(f"{'STRATEGY':<20} | {'WINRATE':<10} | {'EQUITY':<12} | {'TRADES':<8}")
            print("-" * 80)
            
            best_strat_name = None
            best_strat_equity = -1
            best_strat_wr = 0
            
            for name, res in results.items():
                if not res: continue
                print(f"{name:<20} | {res['winrate']:>6.2f}%    | ${res['equity']:>10.2f} | {res['trades']:>6}")
                if res['equity'] > best_strat_equity:
                    best_strat_equity = res['equity']
                    best_strat_name = name
                    best_strat_wr = res['winrate']

            print("-" * 80)
            if best_strat_name:
                print(f"🏆 CYCLE {cycle} WINNER: {best_strat_name} (WR: {best_strat_wr:.2f}%)")
            else:
                 print(f"🏆 CYCLE {cycle}: No trades executed.")
            
            target_wr = 70.0
            if 'AI Enhanced' in results and results['AI Enhanced'] and results['AI Enhanced']['winrate'] >= target_wr:
                print(f"✅ AI GOAL MET! Winrate {results['AI Enhanced']['winrate']:.2f}% >= {target_wr}%")
            elif 'AI Enhanced' in results and results['AI Enhanced']:
                print(f"⚠️ AI Winrate ({results['AI Enhanced']['winrate']:.2f}%) is below goal ({target_wr}%). Training continues...")
            
            print("-" * 80)
            
            print("💤 Cooling down (fast mode)...")
            time.sleep(10) # 10s is safe with reduced training load
            
    except KeyboardInterrupt:
        print("\n🛑 Loop Stopped.")

if __name__ == "__main__":
    bt = Backtester(symbol=config.SYMBOL, timeframe='1m')
    
    # Check if we should run the simple dashboard or the full training loop
    # For this task, we default to the training loop as requested
    run_continuous_training_loop(bt)


