import pandas as pd
import numpy as np
from technical_analysis import TechnicalAnalysis
from ai_brain import AIBrain
from data_manager import DataManager

class WinningSignalExtractor:
    def __init__(self):
        self.brain = None
        
    def determine_regime(self, candle):
        price = candle['close']
        ema_50 = candle['ema_50']
        ema_200 = candle['ema_200']
        adx = candle['adx']
        
        if adx < 20: return "SIDEWAYS"
        elif price > ema_50 and ema_50 > ema_200: return "UPTREND"
        elif price < ema_50 and ema_50 < ema_200: return "DOWNTREND"
        else: return "SIDEWAYS"

    def extract(self, strategy_genome=None):
        print("Carregando Dados para Extração...")
        dm = DataManager()
        df = dm.get_data_from_db(limit=20000)
        df = df[df['timeframe'] == '1m'].copy()
        
        print("Calculando Indicadores...")
        ta = TechnicalAnalysis(df)
        df = ta.add_all_indicators()
        df.dropna(inplace=True)
        
        print("Simulando Estratégias para Encontrar Vencedores (Candidatos)...")
        winners = []
        
        # Test on as much data as possible
        test_data = df.copy()
        
        for i in range(60, len(test_data)):
            # Future looking for outcome
            if i + 10 >= len(test_data): break
            
            entry_price = test_data.iloc[i]['close']
            
            # Check next 60 candles
            future = test_data.iloc[i+1:i+61]
            
            hit_tp = False
            hit_sl = False
            
            tp_target = 0.0014 # 0.14% (User Request)
            sl_target = 0.0004 # 0.04% (User Request)
            
            for idx, row in future.iterrows():
                high = row['high']
                low = row['low']
                
                gain = (high - entry_price) / entry_price
                loss = (low - entry_price) / entry_price
                
                if loss < -sl_target: # SL Hit
                    hit_sl = True
                    break 
                
                if gain > tp_target: # TP Hit
                    hit_tp = True
                    break
            
            if hit_tp and not hit_sl:
                # Winner! Now check if it was a valid setup 
                current_candle = test_data.iloc[i]
                
                is_valid_setup = False
                
                if strategy_genome:
                    # Use the Evolved Strategy
                    if strategy_genome.check_signal(current_candle):
                        is_valid_setup = True
                else:
                    # Fallback to defaults (or hardcoded)
                    # --- STRATEGY DEFINITIONS ---
                    # 1. Stoch Reversion (Reversão de Estocástico)
                    strat_stoch = current_candle['stoch_k'] < 25
                    # 2. Aggressive Trend (Tendência Agressiva)
                    strat_trend = (current_candle['rsi'] > 55) and (current_candle['close'] > current_candle['ema_9'])
                    
                    if strat_stoch or strat_trend:
                        is_valid_setup = True
                
                if is_valid_setup:
                    winners.append(test_data.index[i]) 
                    
        print(f"Encontrados {len(winners)} setups vencedores compatíveis.")
        
        # Save to CSV
        df['target'] = 0
        df.loc[winners, 'target'] = 1
        
        # Save dataset
        df.to_csv("training_data_filtered.csv")
        print("Dataset salvo em training_data_filtered.csv")

if __name__ == "__main__":
    extractor = WinningSignalExtractor()
    extractor.extract()
