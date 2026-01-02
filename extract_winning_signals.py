import pandas as pd
import numpy as np
from technical_analysis import TechnicalAnalysis
from ai_brain import AIBrain
from data_manager import DataManager

class WinningSignalExtractor:
    def __init__(self):
        self.brain = None
        
    # determine_regime removed in favor of MarketRegime class


    def extract(self, strategy_genome=None):
        print("Carregando Dados para Extração...")
        dm = DataManager()
        df = dm.get_data_from_db(limit=50000) # Increased limit to match Evolution
        df = df[df['timeframe'] == '1m'].copy()
        
        print("Calculando Indicadores...")
        ta = TechnicalAnalysis(df)
        df = ta.add_all_indicators()
        
        # Add Regimes
        from market_regime import MarketRegime
        mr = MarketRegime()
        df = mr.add_regime_column(df)
        
        df.dropna(inplace=True)
        
        print("Simulando Estratégias para Encontrar Vencedores (Otimizado via Numpy)...")
        from fast_vector import calculate_outcomes_vectorized
        
        # 1. Vectorized Outcome Calculation (The heavy lifting)
        # Returns boolean array where True = This candle WOULD hit TP before SL
        winners_mask = calculate_outcomes_vectorized(
            df, 
            tp=0.0014, 
            sl=0.0004, 
            lookahead=60
        )
        
        # 2. Filter Candidates (Only process those that are potential winners)
        # We need the indices of True values
        potential_winner_indices = np.where(winners_mask)[0]
        
        # We need to intersect this with the Strategy Logic
        # Strategy Logic: Does the candle meet the entry criteria?
        
        final_winners = []
        
        # Convert df to records for fast iteration (or use itertuples)
        # But we need random access by integer index? No, we have indices.
        
        # Optimization: Use DataFrame Query for Strategy if possible
        # Iterate only potential winners?
        # Or better: Apply Strategy Filter to ALL data, then intersect with Winners.
        
        # --- CANDIDATE SELECTION ---
        # We want to find candles that match EITHER the Evolved Strategy OR the User Strategies
        
        candidates_indices = []
        
        # 1. EVOLVED STRATEGIES (Context-Aware)
        if hasattr(strategy_genome, 'keys'): # Check if it's a dict (Strategies per Regime)
            strategies_dict = strategy_genome
            print("Aplicando Estratégias Contextuais (Uptrend/Downtrend/Sideways)...")
            
            for regime_name, genome in strategies_dict.items():
                if not genome: continue
                
                # Build query for this regime + genome logic
                query_parts = [f"(regime == '{regime_name}')"]
                
                for gene in genome.genes:
                    query_parts.append(f"({gene.indicator} {gene.operator} {gene.threshold})")
                
                query_str = " & ".join(query_parts)
                try:
                    evolved_candidates = df.query(query_str).index
                    candidates_indices.append(evolved_candidates)
                    print(f"   -> {regime_name}: {len(evolved_candidates)} sinais.")
                except Exception as e:
                    print(f"Erro ao filtrar regime {regime_name}: {e}")

        elif strategy_genome: # Legacy single genome support
            # Fallback for old calls or single genome
            pass 
        
        # 2. USER STRATEGIES (Always applied to enrich dataset)
        print("Aplicando Estratégias do Usuário (Trend Sniper, Breakout, Fib Zone)...")
        
        # Trend Sniper
        c_sniper = (
            (df['adx'] > 15) &
            (df['close'] > df['ema_50']) &
            (df['close'] <= df['ema_9']) &
            (df['rsi'] > 40) & (df['rsi'] < 65)
        )
        
        # Momentum Breakout
        c_breakout = (
            (df['adx'] > 30) &
            (df['close'] > df['bb_high']) &
            (df['rsi'] > 60) & (df['rsi'] < 85)
        )
        
        # Fibonacci Golden Zone
        c_fib = (
            (df['adx'] > 20) &
            (df['stoch_rsi_k'] < 0.2) &
            (df['close'] >= df['fib_500']) & 
            (df['close'] <= df['fib_618'])
        )
        
        user_strategy_mask = c_sniper | c_breakout | c_fib
        user_candidates = df[user_strategy_mask].index
        candidates_indices.append(user_candidates)
        
        # UNION of all candidates
        if candidates_indices:
            all_candidates = candidates_indices[0]
            for idx in candidates_indices[1:]:
                all_candidates = all_candidates.union(idx)
        else:
            all_candidates = []
            
        # INTERSECT with Real Winners (The Oracle)
        # map potential_winner_indices (integer) to datetime index
        winner_dt_map = df.index[potential_winner_indices]
        
        if len(all_candidates) > 0:
            final_winners = winner_dt_map.intersection(all_candidates)
        else:
            final_winners = []
            
        print(f"Encontrados {len(final_winners)} setups vencedores compatíveis.")
        
        # Save to CSV
        df['target'] = 0
        df.loc[final_winners, 'target'] = 1
        
        # Save dataset
        df.to_csv("training_data_filtered.csv")
        print("Dataset salvo em training_data_filtered.csv")

if __name__ == "__main__":
    extractor = WinningSignalExtractor()
    extractor.extract()
