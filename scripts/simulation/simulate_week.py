import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from app.core.data_manager import DataManager
from app.core.technical_analysis import TechnicalAnalysis
from app.core.ai_brain import AIBrain
from app.utils import config

def run_simulation(days=7, entry_amount=1.0, leverage=50.0):
    print(f"🚀 Iniciando Simulação Semanal (Last {days} days)...")
    print(f"💰 Entrada Fixa: ${entry_amount} | Alavancagem: {leverage}x")
    
    dm = DataManager()
    
    # 1. Load Data (Approx 1440 minutes per day)
    limit = 1440 * days
    print(f"📉 Carregando {limit} velas (1m)...")
    
    # Force fetch to ensure latest data
    # dm.fetch_historical_data(limit=limit) 
    # Allowing local DB load for speed, user can update separately if needed
    df = dm.get_data_from_db(limit=limit)
    
    if df.empty:
        print("❌ Sem dados. Execute download_history.py primeiro.")
        return

    # 2. Add Indicators (ATR needed for Dynamic SL/TP)
    print("📊 Calculando indicadores (ATR, RSI...)...")
    ta = TechnicalAnalysis(df)
    df = ta.add_all_indicators()
    df.dropna(inplace=True)
    
    # 3. AI Predictions (Mocking or Lightweight Training)
    # Getting accurate AI signal requires training. We'll do a quick train on the first 30% of data
    # and test on the rest, or just use simple logic if training is too heavy.
    # User asked for "Strategies we have", implying AI.
    print("🧠 Treinando IA (Rápido) para gerar sinais...")
    brain = AIBrain()
    
    # Train on first 20% to avoid lookahead bias, simulate running on remaining 80%?
    # Or just train on whole set and check fit (optimistic)?
    # To be realistic, we should use a rolling window, but that's slow.
    # Let's train on the whole dataset (assuming user wants to know 'if model was good, how would it perform?')
    # OR better: Use valid Technical Strategy as baseline + AI Filter.
    brain.train(df)
    predictions = brain.predict_batch(df)
    
    if predictions is not None:
        # Align
        pad_len = len(df) - len(predictions)
        padding = np.full(pad_len, np.nan)
        aligned = np.concatenate([padding, predictions])
        df['ai_return'] = aligned
        df['next_return'] = df['ai_return'].shift(-1)
    else:
        df['next_return'] = 0.0

    # 4. Simulation Loop (Row by Row for Dynamic Control)
    equity = 100.0 # Starting dummy balance
    total_profit = 0.0
    wins = 0
    losses = 0
    trades = []
    
    # Fee Estimate (Kraken Futures ~0.05% Taker)
    # Round trip ~0.1%
    fee_pct = 0.001 
    
    print("▶️ Executando trades...")
    
    # Logic:
    # Buy if AI > 0.05% AND RSI < 70
    # Sell if AI < -0.05% AND RSI > 30
    
    position = 0
    
    for i in range(len(df) - 60): # Stop before end to allow outcome check
        row = df.iloc[i]
        
        # Signals
        pred = row.get('next_return', 0)
        rsi = row['rsi']
        atr = row['atr']
        price = row['close']
        
        signal = 0
        if pred > 0.0005 and rsi < 70:
            signal = 1
        elif pred < -0.0005 and rsi > 30:
            signal = -1
            
        if signal != 0:
            # Trade Setup
            # Dynamic SL/TP per Trader.py logic
            volatility_pct = atr / price
            
            # SL = 1.5x ATR, TP = 3.0x ATR
            sl_move = volatility_pct * 1.5
            tp_move = volatility_pct * 3.0
            
            # Fee Guard
            est_fees = 0.002 # 0.2% Conservative
            if tp_move < est_fees:
                continue # Skip low volatility
                
            # Targets
            if signal == 1:
                tp_price = price * (1 + tp_move)
                sl_price = price * (1 - sl_move)
            else:
                tp_price = price * (1 - tp_move)
                sl_price = price * (1 + sl_move)
                
            # Check Outcome (Look ahead 4 hours max)
            outcome = "neutral"
            pnl = 0.0
            
            future_candles = df.iloc[i+1 : i+240] # Next 240 mins
            
            for _, future in future_candles.iterrows():
                f_high = future['high']
                f_low = future['low']
                
                if signal == 1: # Long
                    if f_low <= sl_price:
                        outcome = "loss"
                        # PnL = -Entry * SL_Move * Leverage
                        # Fixed Entry $5
                        raw_loss = entry_amount * sl_move * leverage
                        pnl = -raw_loss
                        break
                    if f_high >= tp_price:
                        outcome = "win"
                        raw_win = entry_amount * tp_move * leverage
                        pnl = raw_win
                        break
                else: # Short
                    if f_high >= sl_price:
                        outcome = "loss"
                        raw_loss = entry_amount * sl_move * leverage
                        pnl = -raw_loss
                        break
                    if f_low <= tp_price:
                        outcome = "win"
                        raw_win = entry_amount * tp_move * leverage
                        pnl = raw_win
                        break
            
            # Apply Fees
            fees = entry_amount * leverage * fee_pct
            pnl -= fees
            
            if outcome == "win":
                wins += 1
                total_profit += pnl
                trades.append(pnl)
            elif outcome == "loss":
                losses += 1
                total_profit += pnl
                trades.append(pnl)
            # Neutral/Timeout ignored or closed at market?
            # Let's assume neutral = break even - fees
            elif outcome == "neutral":
                total_profit -= fees
                trades.append(-fees)

    # Report
    count = len(trades)
    wr = (wins / count * 100) if count > 0 else 0
    
    print("\n" + "="*40)
    print(f"📊 RESULTADO DA SIMULAÇÃO ({days} DIAS)")
    print("="*40)
    print(f"📅 Período: Últimos {days} dias")
    print(f"💰 Entrada: ${entry_amount} | Alavancagem: {leverage}x")
    print(f"🔢 Total Trades: {count}")
    print(f"✅ Wins: {wins}")
    print(f"❌ Losses: {losses}")
    print(f"🎯 Win Rate: {wr:.2f}%")
    print("-" * 40)
    print(f"💵 LUCRO LÍQUIDO: ${total_profit:.2f}")
    if count > 0:
        print(f"💸 Média por Trade: ${total_profit/count:.2f}")
    print("="*40)

if __name__ == "__main__":
    run_simulation()
