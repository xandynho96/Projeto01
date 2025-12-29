import pandas as pd
import numpy as np
from data_manager import DataManager
from technical_analysis import TechnicalAnalysis
from ai_brain import AIBrain
import config

def run_backtest():
    print("Carregando dados para Backtest (Scalping 1m)...")
    dm = DataManager()
    # Fetch data from DB (ensure we use the correct timeframe)
    df = dm.get_data_from_db(timeframe='1m', limit=10000)
    
    if df.empty:
        print("Nenhum dado de 1m encontrado no BD.")
        return

    print(f"Iniciando Backtest em {len(df)} candles...")
    if len(df) < 100:
        print("Dados insuficientes!")
    
    # Add indicators
    ta = TechnicalAnalysis(df)
    df = ta.add_all_indicators()
    print(f"Tamanho após indicadores: {len(df)}")
    df.dropna(inplace=True)
    print(f"Tamanho após limpeza: {len(df)}")
    
    # Simulation Settings
    TEST_LEN = 5000 # Test last 5000 candles (~3.5 days)
    
    # Load Brain
    brain = AIBrain()
    if brain.model is None:
        print("Modelo IA não encontrado.")
        return

    # Simulation Variables
    balance = 1000.0 # USD
    position = 0 # 0: None, 1: Long
    entry_price = 0
    trades = []
    
    # Prepare data for batch prediction (much faster than loop)
    # Let's iterate through the last N candles
    test_range = df.tail(TEST_LEN).copy()
    
    print("Simulando operações...")
    
    wins = 0
    losses = 0
    
    print(f"Test Range Length: {len(test_range)}", flush=True)
    
    for i in range(60, len(test_range)):
        # Taking a slice for prediction (needs to be sequence_length=60)
        current_slice = test_range.iloc[i-60:i+1] # Need at least sequence_length
        current_candle = current_slice.iloc[-1]
        timestamp = current_candle['timestamp']
        
        # Predict Probability of Win
        try:
             prob = brain.predict_proba(current_slice)
        except Exception as e:
             # print(f"Prediction Error: {e}", flush=True)
             continue
        
        should_enter = False
        
        # Logic: 
        # Buy if Probability > 60% (Lowered for 50x leverage testing)
        if prob > 0.60:
            should_enter = True
            
        # Debug
        if i < 80: # Print first 20 checks
             # print(f"Prob: {prob:.4f} -> Enter: {should_enter}", flush=True)
             pass
        
        if position == 0:
            if should_enter:
                # Buy
                position = 1
                entry_price = current_price
                print(f"COMPRA (IA: {prob*100:.1f}%) em {entry_price:.2f} ({timestamp})")
                
        elif position == 1:
            # Check exit conditions
            pnl_pct = (current_price - entry_price) / entry_price
            
            # User Targets: TP 0.14%, SL 0.04%
            if pnl_pct > 0.0014: # TP 0.14%
                reason = "TP"
            elif pnl_pct < -0.0004: # SL 0.04%
                 reason = "SL"
            else:
                reason = None
                
            if reason:
                # Sell / Close
                # Leverage logic: 50x
                leverage = 50
                realized_pnl = pnl_pct * leverage
                
                balance = balance * (1 + realized_pnl)
                trades.append(realized_pnl)
                position = 0
                
                if pnl_pct > 0:
                    wins += 1
                else:
                    losses += 1
                
                print(f"VENDA ({reason}) em {current_price:.2f} | PnL: {pnl_pct*100:.4f}% (Lev: {realized_pnl*100:.2f}%) | Saldo: {balance:.2f}")

    total_trades = wins + losses
    winrate = (wins / total_trades * 100) if total_trades > 0 else 0
    
    # Print Probability Stats
    if len(probs) > 0:
        import numpy as np
        print(f"Estatísticas de Confiança (IA) -> Média: {np.mean(probs):.4f} | Máx: {np.max(probs):.4f} | Mín: {np.min(probs):.4f}")

    print("\n--- Resultados do Backtest (Últimos 500m) ---")
    print(f"Total de Operações: {total_trades}")
    print(f"Vitórias (Wins): {wins}")
    print(f"Derrotas (Losses): {losses}")
    print(f"Taxa de Acerto (Winrate): {winrate:.2f}%")
    print(f"Saldo Final: ${balance:.2f} (Início: $1000.00)")
    print(f"Retorno: {((balance-1000)/1000)*100:.2f}%")

if __name__ == "__main__":
    run_backtest()
