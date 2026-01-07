
import pandas as pd
from data_manager import DataManager
from technical_analysis import TechnicalAnalysis
from market_regime import MarketRegime

def check():
    dm = DataManager()
    df = dm.get_data_from_db(limit=2000)
    
    if df.empty:
        print("Empty DF.")
        return
        
    ta = TechnicalAnalysis(df)
    df = ta.add_all_indicators()
    
    mr = MarketRegime()
    df = mr.add_regime_column(df)
    df.dropna(inplace=True)
    
    print(f"Rows: {len(df)}")
    print("Regimes:")
    print(df['regime'].value_counts())
    
    print("\nStats:")
    print(df[['close', 'rsi', 'adx', 'bb_width']].describe())

if __name__ == "__main__":
    check()
