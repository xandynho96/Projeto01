
import pandas as pd
from data_manager import DataManager
from technical_analysis import TechnicalAnalysis
from market_regime import MarketRegime

def debug_queries():
    print("Loading data...")
    dm = DataManager()
    df = dm.get_data_from_db(limit=5000)
    
    ta = TechnicalAnalysis(df)
    df = ta.add_all_indicators()
    
    mr = MarketRegime()
    df = mr.add_regime_column(df)
    df.dropna(inplace=True)
    
    print(f"Total Rows: {len(df)}")
    
    # Test 1: Simple Regime
    uptrend = df.query("regime == 'UPTREND'")
    print(f"Uptrend Rows: {len(uptrend)}")
    
    # Test 2: Single Indicator in Regime
    q2 = "regime == 'UPTREND' & rsi < 40"
    res2 = df.query(q2)
    print(f"Query '{q2}': {len(res2)} rows")
    
    # Test 3: DeepSeek logic example
    # [Genes] rsi < 40.0 AND dist_ema_200 > 0.005 AND adx > 35.0
    q3 = "regime == 'UPTREND' & rsi < 40 & dist_ema_200 > 0.005 & adx > 35"
    res3 = df.query(q3)
    print(f"Query '{q3}': {len(res3)} rows")
    
    # Test 4: Check Value Ranges
    print("\nStats for UPTREND:")
    print(uptrend[['rsi', 'dist_ema_200', 'adx']].describe())

if __name__ == "__main__":
    debug_queries()
