import pandas as pd
import numpy as np

class MarketRegime:
    """
    Classifies the market environment into distinct regimes:
    - UPTREND: Price above EMA200 and sufficient trend strength (ADX).
    - DOWNTREND: Price below EMA200 and sufficient trend strength (ADX).
    - SIDEWAYS: Weak trend strength (Low ADX).
    - HIGH_VOL: (Optional/Advanced) Extreme volatility conditions.
    """
    
    REGIMES = ['UPTREND', 'DOWNTREND', 'SIDEWAYS']
    
    def __init__(self, adx_threshold=25):
        self.adx_threshold = adx_threshold

    def classify(self, row):
        """
        Determines the regime for a single row of data.
        Expects 'close', 'ema_200', 'adx' in row.
        """
        # 1. Check for Sideways (Range) first
        # If ADX is low, the market is likely ranging/choppy, regardless of EMA position
        if row['adx'] < self.adx_threshold:
            return 'SIDEWAYS'
            
        # 2. Check Trend Direction
        # If ADX is high, we look at the long-term trend filter (EMA 200)
        if row['close'] > row['ema_200']:
            return 'UPTREND'
        else:
            return 'DOWNTREND'

    def add_regime_column(self, df):
        """
        Adds a 'regime' column to the DataFrame (Vectorized for speed).
        """
        # Initialize with Trend Direction
        conditions = [
            (df['close'] > df['ema_200']),
            (df['close'] <= df['ema_200'])
        ]
        choices = ['UPTREND', 'DOWNTREND']
        df['regime'] = np.select(conditions, choices, default='SIDEWAYS')
        
        # Overwrite with SIDEWAYS if ADX is low
        df.loc[df['adx'] < self.adx_threshold, 'regime'] = 'SIDEWAYS'
        
        return df

if __name__ == "__main__":
    # Test Stub
    print("Testing MarketRegime...")
    data = {
        'close': [100, 110, 90, 100, 105],
        'ema_200': [95, 95, 95, 105, 95], # 1:Up, 2:Up, 3:Down, 4:Down, 5:Up
        'adx': [30, 20, 30, 30, 15]       # 1:Str, 2:Wk, 3:Str, 4:Str, 5:Wk
    }
    df = pd.DataFrame(data)
    mr = MarketRegime(adx_threshold=25)
    df = mr.add_regime_column(df)
    
    # Exp 1: Up, Str -> UPTREND
    # Exp 2: Up, Wk -> SIDEWAYS
    # Exp 3: Down, Str -> DOWNTREND
    # Exp 4: Down, Str -> DOWNTREND (Close 100 < EMA 105)
    # Exp 5: Up, Wk -> SIDEWAYS
    
    print(df)
