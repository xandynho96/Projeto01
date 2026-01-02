import pandas as pd
import numpy as np
from technical_analysis import TechnicalAnalysis
from ai_brain import AIBrain
from data_manager import DataManager

class StrategyOptimizer:
    def __init__(self):
        self.brain = None # Load lazy
        self.results = {
            "UPTREND": {},
            "DOWNTREND": {},
            "SIDEWAYS": {}
        }
        self.strategies = [
            "scalp_trend_ema",
            "scalp_trend_aggressive",
            "scalp_reversion_stoch",
            "scalp_breakout_bb",
            "scalp_golden_cross",
            "ai_pure"
        ]
        
    def determine_regime(self, candle):
        # Define market regime
        price = candle['close']
        ema_50 = candle['ema_50']
        ema_200 = candle['ema_200']
        adx = candle['adx']
        
        if adx < 20:
             return "SIDEWAYS"
        elif price > ema_50 and ema_50 > ema_200:
             return "UPTREND"
        elif price < ema_50 and ema_50 < ema_200:
             return "DOWNTREND"
        else:
             return "SIDEWAYS" # Default to messy

    def run_grid_search(self, candles=10000): 
        print("Loading Data for Grid Search...")
        dm = DataManager()
        df = dm.get_data_from_db(limit=20000)
        df = df[df['timeframe'] == '1m'].copy()
        
        print(f"Data Loaded: {len(df)}. Calculating Indicators...")
        ta = TechnicalAnalysis(df)
        df = ta.add_all_indicators()
        df.dropna(inplace=True)
        
        # Grid Search Parameters (User Specific: 0.14% TP, 0.04% SL)
        # Scalping with 50x leverage.
        # We test exact user values and slight variations.
        
        tp_options = [0.0014] # 0.14%
        sl_options = [0.0004] # 0.04%
        stoch_thresholds = [10, 20, 30] 
        
        strategies = ["scalp_reversion_stoch", "scalp_trend_aggressive", "scalp_breakout_bb"] # Widen search for this hard target
        
        best_config = None
        
        print("Starting Refined Grid Search...")
        
        import itertools
        combinations = list(itertools.product(strategies, tp_options, sl_options, stoch_thresholds))
        
        rec = []
        
        # Optimization: Pre-calculate signals to avoid re-evaluating logic
        # For 'scalp_reversion_stoch', signal is stoch_k < 20.
        # For 'scalp_trend_aggressive', signal is rsi > 55 & close > ema_9.
        
        # Let's run the simulation for each combo
        # To be fast, we can just vectorise the exit logic check?
        # Or just loop. 20k candles * 20 combos is 400k iterations. Fast in C, slow in Python.
        # Let's pick a smaller test set for grid search: 5000 candles.
        
        test_df = df.iloc[-8000:] # Increase range for better stat significance
        
        for strat, tp, sl, thresh in combinations:
            # Skip invalid R:R (SL > TP is usually bad for high freq unless WR is huge)
            if sl > tp: continue
            
            stats = self._simulate_fast(test_df, strat, tp, sl, thresh)
            
            wr = stats['winrate']
            trades = stats['trades']
            pnl = stats['pnl']
            
            if trades > 20: # Lower min sample slightly for strict thresholds
                print(f"[{strat}|Thresh:{thresh}] TP: {tp*100:.1f}% | SL: {sl*100:.1f}% -> {trades} Trades | WR: {wr:.1f}%")
                
                rec.append({
                    'strat': strat,
                    'tp': tp,
                    'sl': sl,
                    'thresh': thresh,
                    'wr': wr,
                    'trades': trades,
                    'pnl': pnl
                })
            
        # Find Best
        rec.sort(key=lambda x: x['wr'], reverse=True)
        
        print("\n=== TOP 5 CONFIGURATIONS ===")
        for r in rec[:5]:
             print(f"Strat: {r['strat']} (Thresh {r['thresh']}) | TP: {r['tp']} | SL: {r['sl']} | WR: {r['wr']:.1f}% ({r['trades']} tr)")
             
        # Save best to file for Imitation Learning to use
        # if rec:
        #     best = rec[0]
        #     if best['wr'] > 60: # Threshold to accept as teacher
        #          print(f"\nAccepted Best Config: {best}")
        #     else:
        #          print("\nNo config met 60% with >30 trades. Needs broader search.")

    def _simulate_fast(self, df, strat, tp, sl, thresh):
        # Fast vectorised simulation if possible, or tight loop
        trades = 0
        wins = 0
        total_pnl = 0.0
        
        # Vectorized Signal
        if strat == "scalp_reversion_stoch":
            entries = df['stoch_k'] < thresh
        # elif strat == "scalp_trend_aggressive":
        #     entries = (df['rsi'] > 55) & (df['close'] > df['ema_9'])
        
        # Get indices of entries
        entry_indices = np.where(entries)[0]
        
        # We need to skip processing if we are already in a trade
        # Sequential processing is required for accurate trade count (can't take new trade while in one)
        
        last_exit_idx = -1
        
        # df values to numpy for speed
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        
        for idx in entry_indices:
            if idx <= last_exit_idx: continue
            if idx + 61 >= len(closes): break # End of data + lookahead
            
            entry_price = closes[idx]
            
            # Look forward
            won = False
            lost = False
            
            # Check next 60 minutes
            # Using slice for speed instead of loop
            # slice_highs = highs[idx+1:idx+61]
            # slice_lows = lows[idx+1:idx+61]
            
            # Manual loop is acceptable for logic clarity as per User Request replacement
            # But we must use the numpy arrays defined outside
            
            for future_i in range(idx+1, idx+61):
                # 1. Checa Stop Loss primeiro (Conservador/Seguro)
                current_low = lows[future_i]
                loss_pct = (current_low - entry_price) / entry_price
                
                if loss_pct <= -sl:
                    lost = True
                    last_exit_idx = future_i
                    break
                
                # 2. Checa Take Profit
                current_high = highs[future_i]
                gain_pct = (current_high - entry_price) / entry_price
                
                if gain_pct >= tp:
                    won = True
                    last_exit_idx = future_i
                    break
            
            if won:
                trades += 1
                wins += 1
                total_pnl += tp
            elif lost:
                trades += 1
                total_pnl -= sl
            # else: Timed out - Ignore or Flat close? User logic ignored timeouts previously. Keep consistent.
            
        winrate = (wins / trades * 100) if trades > 0 else 0
        return {"trades": trades, "winrate": winrate, "pnl": total_pnl}

if __name__ == "__main__":
    opt = StrategyOptimizer()

    def _print_report(self):
        print("\n=== STRATEGY OPTIMIZATION REPORT ===")
        for regime in self.results:
            print(f"\n--- Market Regime: {regime} ---")
            best_strat = None
            best_wr = 0
            
            for strat, stats in self.results[regime].items():
                total = stats['trades']
                if total == 0: continue
                winrate = (stats['wins'] / total) * 100
                print(f"  [{strat}]: {total} Trades | WR: {winrate:.1f}% | PnL: {stats['pnl']*100:.2f}%")
                
                if winrate > best_wr and total > 5: # Filter low sample
                    best_wr = winrate
                    best_strat = strat
            
            if best_strat:
                print(f"  >> WINNER: {best_strat} ({best_wr:.1f}%)")
            else:
                print("  >> No clear winner (insufficient trades).")

if __name__ == "__main__":
    opt = StrategyOptimizer()
    opt.run_grid_search()
