import numpy as np
import pandas as pd

def calculate_outcomes_vectorized(df, tp=0.0014, sl=0.0004, lookahead=60):
    """
    Calculates trade outcomes (Win/Loss) for every candle in the dataframe
    employing a vectorized sliding window approach.
    
    Returns:
        numpy.array of booleans (True = Winner, False = Loser)
    """
    # Prepare Numpy Arrays
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    n = len(closes)
    
    # 1. Create a Strided View (Rolling Window)
    # Shape: (N-lookahead, lookahead)
    # We want to compare High[i+j] with Close[i]
    
    # Pad arrays to handle boundary conditions easily or just cut off the end
    # We'll cut off the end for now (simpler), last candles can't be evaluated
    
    valid_n = n - lookahead
    if valid_n <= 0:
        return np.zeros(n, dtype=bool)
        
    # Create windows for Highs and Lows
    # stride_tricks requires knowing the bytes per step
    # Easier to use simple manual indexing for clarity or library
    # actually optimized broadcast:
    
    # Construct indices: rows + columns
    # Rows: 0..valid_n
    # Cols: 1..lookahead
    
    # matrix of indices (valid_n, lookahead)
    # i range
    idx_row = np.arange(valid_n).reshape(-1, 1)
    # j range
    idx_col = np.arange(1, lookahead + 1).reshape(1, -1)
    
    indices = idx_row + idx_col
    
    # Get Windows
    window_highs = highs[indices] # Shape (valid_n, lookahead)
    window_lows = lows[indices]   # Shape (valid_n, lookahead)
    
    # Entry prices (broadcast to match window columns)
    entries = closes[:valid_n].reshape(-1, 1)
    
    # 2. Calculate Returns Matrix
    # We don't need exact % for speed, just check thresholds
    # Gain > TP  =>  (High - Entry) / Entry > TP  => High > Entry * (1 + TP)
    # Loss < -SL =>  (Low - Entry) / Entry < -SL  => Low < Entry * (1 - SL)
    
    tp_prices = entries * (1 + tp)
    sl_prices = entries * (1 - sl)
    
    # Boolean Matrices
    tp_hits = window_highs > tp_prices
    sl_hits = window_lows < sl_prices
    
    # 3. Find First Occurrence
    # argmax returns index of first True. If all False, returns 0.
    # We need to handle "all False" case.
    
    tp_idx = np.argmax(tp_hits, axis=1)
    sl_idx = np.argmax(sl_hits, axis=1)
    
    # Check if they actually happened
    # max() over boolean axis checks if ANY is True
    any_tp = tp_hits.max(axis=1)
    any_sl = sl_hits.max(axis=1)
    
    # Logic:
    # Win if TP hit AND (SL not hit OR TP_index < SL_index)
    # Tie-breaker (TP_index == SL_index): Assume Loss (Conservative) or Win?
    # Conservative: SL happens first in the candle usually. Loss.
    
    # Initialize Result Array (default False)
    results = np.zeros(valid_n, dtype=bool)
    
    # Set Winners
    # CASE 1: TP Hit, No SL Hit
    mask_clean_win = any_tp & (~any_sl)
    
    # CASE 2: TP Hit AND SL Hit, but TP was earlier
    # If indices equal, it's a tie. tp_idx < sl_idx means TP strict earlier.
    mask_race_win = any_tp & any_sl & (tp_idx < sl_idx)
    
    final_win_mask = mask_clean_win | mask_race_win
    results[:] = final_win_mask
    
    # Pad result to match original length (fill with False)
    padding = np.zeros(lookahead, dtype=bool)
    full_results = np.concatenate([results, padding])
    
    return full_results
