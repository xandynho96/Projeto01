
import pandas as pd
from data_manager import DataManager
from technical_analysis import TechnicalAnalysis
from market_regime import MarketRegime

class DebugEvol:
    def __init__(self):
        self.dm = DataManager()
        self.df = self.dm.get_data_from_db(limit=5000)
        self.df = self.df[self.df['timeframe'] == '1m'].copy()
        
        ta = TechnicalAnalysis(self.df)
        self.df = ta.add_all_indicators()
        
        mr = MarketRegime()
        self.df = mr.add_regime_column(self.df)
        self.df.dropna(inplace=True)
        
        print(f"Loaded DF: {len(self.df)}")
        print(self.df['regime'].value_counts())
        
    def test_query(self):
        regime = "UPTREND"
        gene_str = "rsi < 90"
        
        query_str = f"(regime == '{regime}') & ({gene_str})"
        print(f"Testing Query: {query_str}")
        
        try:
            subset = self.df.query(query_str)
            print(f"Subset Len: {len(subset)}")
        except Exception as e:
            print(f"Query Error: {e}")
            
        # Test 2: Double condition
        q2 = f"(regime == '{regime}') & (rsi < 90) & (adx > 10)"
        print(f"Testing Q2: {q2}")
        print(f"Subset Q2: {len(self.df.query(q2))}")

if __name__ == "__main__":
    dbg = DebugEvol()
    dbg.test_query()
