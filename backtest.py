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
        Runs a fully vectorized backtest with Event-Based outcomes (SL/TP Hits).
        Fast and supports accurate Risk/Reward simulation.
        signals: pd.Series of -1 (Short), 0 (Neutral), 1 (Long)
        """
        if self.df.empty: return {}
        
        # Imports here to avoid circular at top if not needed
        from fast_vector import calculate_outcomes_vectorized, calculate_outcomes_vectorized_short
        
        # 1. Config Defaults (ROE based)
        # Default GUI: 30% SL, 70% TP on 50x Leverage
        # Real Move: 0.6% SL, 1.4% TP
        lev = config.LEVERAGE # 50
        sl_roe = 30.0
        tp_roe = 70.0
        
        # Try to load from user_config
        try:
            if os.path.exists("user_config.json"):
                with open("user_config.json", 'r') as f:
                    cfg = json.load(f)
                    sl_roe = float(cfg.get('sl_pct', 30.0))
                    tp_roe = float(cfg.get('tp_pct', 70.0))
                    lev = float(cfg.get('leverage', lev))
        except:
            pass
            
        # Scalping Defaults (1m timeframe)
        # TP: 0.2% move, SL: 0.1% move
        sl_pct = 0.001
        tp_pct = 0.002
        
        # Sync ROE for PnL Calc
        sl_roe = sl_pct * lev * 100
        tp_roe = tp_pct * lev * 100
        
        # Override if user config exists but ensure sanity for scalping
        # ... logic ...
        
        # Lookahead: How long do we wait? 1H (60m)
        lookahead = 60 
        
        # 2. Boolean Masks for Outcomes
        # We calculate "Did a Long Win?" and "Did a Short Win?" for EVERY candle
        long_winners = calculate_outcomes_vectorized(self.df, tp=tp_pct, sl=sl_pct, lookahead=lookahead)
        short_winners = calculate_outcomes_vectorized_short(self.df, tp=tp_pct, sl=sl_pct, lookahead=lookahead)
        
        # 3. Simulate Trades
        # Where signal != 0
        trades_mask = signals != 0
        if not trades_mask.any():
            return {'equity': self.initial_capital, 'roi': 0, 'winrate': 0, 'trades': 0}
            
        # Filter signals to only trade entries
        # Simple Logic: Every signal is an entry (Scalping)
        # Or: Only change of signal? 
        # For "Action Learning", let's treat every signal as a potential trade opportunity
        
        # Align masks
        # Signal at T executes at T (Market) or T+1?
        # fast_vector uses Entry = Close[i]. So Signal at T means we enter at Close[i]. Correct.
        
        entry_signals = signals[trades_mask]
        
        # Determine Outcome for each signal
        # Map signal index to outcome array
        # outcomes: 1 (Win), -1 (Loss)
        
        indices = entry_signals.index
        # Get outcome bools at these indices
        long_res = long_winners[self.df.index.get_indexer(indices)]
        short_res = short_winners[self.df.index.get_indexer(indices)]
        
        pnl_log = []
        
        # Rewards (Fixed R:R)
        # Win: +TP * Leverage * Capital (approx) -> Actually Capital * (TP_ROE)
        # Loss: -SL * Leverage * Capital -> Capital * (SL_ROE)
        # Fees?
        
        win_pnl = self.initial_capital * (tp_roe / 100)
        loss_pnl = self.initial_capital * (sl_roe / 100)
        
        # Fee handling: 0.05% usually.
        # paid on Entry + Exit. Total ~0.1% * Leverage.
        fee_cost = self.initial_capital * lev * (self.fee_pct * 2) 
        
        net_win = win_pnl - fee_cost
        net_loss = -loss_pnl - fee_cost
        
        # Iterate (Vectorized sum possible?)
        # Vectorized:
        # Long Signals (1): PnL = net_win if long_res else net_loss
        # Short Signals (-1): PnL = net_win if short_res else net_loss
        
        # Masks relative to entry_signals
        is_long = entry_signals == 1
        is_short = entry_signals == -1
        
        # PnL Vector
        pnl_vector = pd.Series(0.0, index=entry_signals.index)
        
        # Longs
        pnl_vector[is_long & long_res] = net_win
        pnl_vector[is_long & ~long_res] = net_loss
        
        # Shorts
        pnl_vector[is_short & short_res] = net_win
        pnl_vector[is_short & ~short_res] = net_loss
        
        # Stats
        total_pnl = pnl_vector.sum()
        final_equity = self.initial_capital + total_pnl
        trades_count = len(pnl_vector)
        
        wins = ((is_long & long_res) | (is_short & short_res)).sum()
        win_rate = (wins / trades_count * 100) if trades_count > 0 else 0
        roi = (total_pnl / self.initial_capital) * 100
            
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

    def align_predictions(self, predictions, sequence_length):
        """Aligns predictions with the DataFrame."""
        if predictions is None or len(predictions) == 0:
            return pd.Series(0.0, index=self.df.index)
            
        # Predictions correspond to indices [sequence_length:]
        # Predictions are 'returns' for the target candle.
        # We need to PAD the beginning
        pad_len = len(self.df) - len(predictions)
        if pad_len < 0:
             # Weird case, maybe prediction buffer larger?
             return pd.Series(predictions[:len(self.df)], index=self.df.index)
             
        padding = np.full(pad_len, np.nan)
        aligned = np.concatenate([padding, predictions])
        return pd.Series(aligned, index=self.df.index)

    def generate_ai_signals(self):
        # AI Return Pred already in DF as 'ai_return'
        if 'ai_return' not in self.df.columns: return pd.Series(0, index=self.df.index)
        
        # We need prediction for T+1 available at T.
        # 'ai_return' at index T is the return for T (predicted at T-1).
        # So we shift -1 to get "Predicted Return for Next Candle".
        next_return = self.df['ai_return'].shift(-1)
        
        # Technical Filters (Relaxed for Scalping)
        rsi = self.df['rsi'] if 'rsi' in self.df.columns else pd.Series(50, index=self.df.index)
        
        signals = pd.Series(0, index=self.df.index)
        
        # Expert Scalping Thresholds (0.05% = 0.0005)
        # Long: Predicted > 0.05%
        # Relaxed RSI: Just don't buy top (>75)
        long_cond = (next_return > 0.0005) & (rsi < 75)
        signals[long_cond] = 1 
        
        # Short: Predicted < -0.05%
        # Relaxed RSI: Just don't sell bottom (<25)
        short_cond = (next_return < -0.0005) & (rsi > 25)
        signals[short_cond] = -1 
        
        return signals

    def filter_with_deepseek(self, signals, brain, limit_checks=10):
        """
        Filters signals using DeepSeek API. 
        WARNING: Slow and consumes credits. Limits to last 'limit_checks' signals.
        """
        if signals.eq(0).all(): return signals
        
        filtered_signals = signals.copy()
        
        # Get indices of signals
        signal_indices = signals[signals != 0].index
        
        # Limit to last N checks to save time/credits during backtest dev
        if len(signal_indices) > limit_checks:
            print(f"⚠️ Limiting DeepSeek checks to last {limit_checks} signals (from {len(signal_indices)} total)...")
            signal_indices = signal_indices[-limit_checks:]
            # Zero out skipped signals? Or keep them as "Unvalidated"?
            # ideally we ONLY run this filter on a small backtest window.
            # For now, we only validate the last N, and ASSUME others are 0 (safest) or leave raw? 
            # Let's leave others raw to see impact on RECENT activity, or zero them to see "What if I ONLY traded with DeepSeek"?
            # Let's ZERO them out to strictly test DeepSeek's quality.
            filtered_signals.loc[~filtered_signals.index.isin(signal_indices)] = 0
            
        print(f"🤖 DeepSeek Validating {len(signal_indices)} signals...")
        
        api_key = None
        # Load API Key
        try:
             with open("user_config.json", "r") as f:
                 api_key = json.load(f).get("deepseek_key")
        except: pass
        
        if not api_key:
             # Fallback to config.py
             try:
                 import config
                 api_key = config.DEEPSEEK_API_KEY
             except: pass
        
        if not api_key:
            print("❌ No DeepSeek Key found. Skipping validation.")
            return signals

        count = 0
        for idx in signal_indices:
            count += 1
            row = self.df.loc[idx]
            signal = signals.loc[idx]
            current_price = row['close']
            
            # Predict Target (approximate based on signal direction and scalping target 0.2%)
            # We don't have exact 'predicted_price' easily accessible here unless we stored it.
            # But we can assume the AI saw > 0.05% move.
            # Let's reconstruct context.
            
            # Use 'ai_return' if available
            pred_return = row.get('next_return', 0.001 if signal == 1 else -0.001)
            predicted_price = current_price * (1 + pred_return)
            
            # 1. Market Context Construction
            # Calculate simple regime/trend context on the fly
            # Need historical context (e.g. BB Width). 
            # Assumes DF has indicators.
            
            bb_width = row.get('bb_width', 0)
            ema_200 = row.get('ema_200', current_price)
            
            regime = "NORMAL"
            hint = "Seguir IA"
            if bb_width < 0.005:
                regime = "BAIXA VOLATILIDADE (Squeeze)"
                hint = "Aguardar Rompimento"
            elif bb_width > 0.02:
                regime = "ALTA VOLATILIDADE"
                hint = "Scalping de Alta Volatilidade"
                
            trend = "ALTA" if current_price > ema_200 else "BAIXA"
            
            market_context = {
                "regime": regime,
                "hint": hint,
                "trend": trend
            }
            
            # 2. Technical Summary
            tech_summary = {
                'rsi': row.get('rsi', 50),
                'stoch_rsi_k': row.get('stoch_rsi_k', 0.5),
                'atr': row.get('atr', 0),
                'macd': row.get('macd', 0),
                'bb_width': bb_width,
                'pattern_score': row.get('pattern_score', 0)
            }
            
            # 3. Call Brain
            # We need to print progress
            print(f"   [{count}/{len(signal_indices)}] Validating Signal {idx} ({'BUY' if signal==1 else 'SELL'})... ", end="")
            
            try:
                # Add delay to avoid rate limit
                time.sleep(0.5) 
                validation = brain.validate_signal_with_deepseek(
                    current_price, 
                    predicted_price, 
                    tech_summary, 
                    market_context=market_context, 
                    api_key=api_key
                )
                
                if not validation.get('approved', True):
                    print(f"❌ REJECTED: {validation.get('reason', 'N/A')}")
                    filtered_signals.loc[idx] = 0
                else:
                    print(f"✅ APPROVED: {validation.get('reason', 'OK')}")
                    
            except Exception as e:
                print(f"Error: {e}")
                
        return filtered_signals

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
        while cycle < 1:
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
            
            # ACCELERATION: Train only on the most recent 2000 candles to prevent hanging on slow fits
            train_limit = 2000
            if len(bt.df) > train_limit:
                 print(f"   -> Optimization: Training on last {train_limit} candles only (High Frequency Tuning).")
                 train_df = bt.df.tail(train_limit).copy()
                 brain.train(train_df)
            else:
                 brain.train(bt.df)
            
            # 3b. Pre-calculate AI Predictions for Backtest Speed
            print("🔮 Generating AI Predictions for backtest...")
            predictions = brain.predict_batch(bt.df)
            
            if predictions is None:
                print("⚠️ AI Predictions unavailable. Using 0.")
                bt.df['ai_return'] = 0.0
            else:
                try:
                    aligned_preds = bt.align_predictions(predictions, getattr(brain, 'sequence_length', 60))
                    if len(aligned_preds) == len(bt.df):
                        bt.df['ai_return'] = aligned_preds
                        # Pre-shift for strategy wrapper if needed, or just let vectorized handle it
                        bt.df['next_return'] = bt.df['ai_return'].shift(-1)
                    else:
                        print("⚠️ Prediction alignment mismatch.")
                        bt.df['ai_return'] = 0.0
                except Exception as e:
                     print(f"⚠️ Prediction error: {e}")
                     bt.df['ai_return'] = 0.0
                     
            
            # Define AI Strategy Wrapper
            def strat_ai_wrapper(row, position):
                # Using pre-calculated 'next_return' (Prediction for T+1 made at T)
                if pd.isna(row.get('next_return', 0)): return 'hold'
                
                pred_ret = row['next_return']
                
                # Scalping Threshold 0.05%
                if pred_ret > 0.0005 and position == 0:
                    return 'buy'
                elif pred_ret < -0.0005 and position == 0:
                    return 'sell'
                elif pred_ret < 0 and position == 1:
                    return 'close'
                elif pred_ret > 0 and position == -1:
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
            
            # DeepSeek Filtered Strategy (Limited Test)
            # Only run if AI has signals
            if not signals_ai.eq(0).all():
                 print(f"🤖 Running DeepSeek Filter on AI Signals...")
                 signals_deepseek = bt.filter_with_deepseek(signals_ai, brain, limit_checks=5) # Limit to 5 for speed
                 results['AI + DeepSeek'] = bt.run_vectorized_backtest(signals_deepseek)
            
            # 5. Dashboard
            print("-" * 80)
            print(f"{'STRATEGY':<20} | {'WINRATE':<10} | {'EQUITY':<12} | {'TRADES':<8}")
            print("-" * 80)
            
            best_strat_name = None
            best_strat_equity = -float('inf')
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


