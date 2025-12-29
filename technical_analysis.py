import pandas as pd
import ta
import numpy as np

class TechnicalAnalysis:
    def __init__(self, df):
        """
        Initialize with a DataFrame containing OHLCV data.
        df must have columns: 'open', 'high', 'low', 'close', 'volume'
        """
        self.df = df.copy()

    def add_all_indicators(self):
        """Adds all technical indicators to the DataFrame."""
        self._add_momentum_indicators()
        self._add_trend_indicators()
        self._add_volatility_indicators()
        self._add_candlestick_patterns()
        self._add_fibonacci_levels()
        return self.df

    def _add_momentum_indicators(self):
        # RSI
        self.df['rsi'] = ta.momentum.RSIIndicator(close=self.df['close'], window=14).rsi()
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(
            high=self.df['high'], low=self.df['low'], close=self.df['close'], window=14, smooth_window=3
        )
        self.df['stoch_k'] = stoch.stoch()
        self.df['stoch_d'] = stoch.stoch_signal()
        
        # Stochastic RSI (Good for Scalping)
        stoch_rsi = ta.momentum.StochRSIIndicator(close=self.df['close'], window=14)
        self.df['stoch_rsi_k'] = stoch_rsi.stochrsi_k()
        self.df['stoch_rsi_d'] = stoch_rsi.stochrsi_d()

    def _add_trend_indicators(self):
        # MACD
        macd = ta.trend.MACD(close=self.df['close'])
        self.df['macd'] = macd.macd()
        self.df['macd_signal'] = macd.macd_signal()
        self.df['macd_diff'] = macd.macd_diff()
        
        # SuperTrend (Trend following)
        # Calculates ATR and Upper/Lower bands
        # We need to implement it manually or check if 'ta' has it (newer versions do, but to be safe manual)
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']
        
        # Calculate ATR
        atr = ta.volatility.AverageTrueRange(high, low, close, window=10).average_true_range()
        multiplier = 3.0
        
        # Basic SuperTrend Calculation
        hl2 = (high + low) / 2
        final_upper = hl2 + (multiplier * atr)
        final_lower = hl2 - (multiplier * atr)
        
        # Initialize columns
        supertrend = [True] * len(self.df) # True = Green/Bullish
        
        for i in range(1, len(self.df)):
            curr_close = close.iloc[i]
            prev_close = close.iloc[i-1]
            
            # Trend Logic
            if curr_close > final_upper.iloc[i-1]:
                supertrend[i] = True
            elif curr_close < final_lower.iloc[i-1]:
                supertrend[i] = False
            else:
                supertrend[i] = supertrend[i-1]
                
                # Adjust bands for trend continuation
                if supertrend[i] == True and final_lower.iloc[i] < final_lower.iloc[i-1]:
                    final_lower.iloc[i] = final_lower.iloc[i-1]
                
                if supertrend[i] == False and final_upper.iloc[i] > final_upper.iloc[i-1]:
                    final_upper.iloc[i] = final_upper.iloc[i-1]

        self.df['supertrend'] = supertrend # Boolean: True (Bullish), False (Bearish)
        
        # On-Balance Volume (OBV)
        self.df['obv'] = ta.volume.OnBalanceVolumeIndicator(close=self.df['close'], volume=self.df['volume']).on_balance_volume()
        
        # OBV Slope (To detect accumulation)
        self.df['obv_slope'] = self.df['obv'].diff(5) # Change over 5 candles

        # EMAs
        self.df['ema_9'] = ta.trend.EMAIndicator(close=self.df['close'], window=9).ema_indicator()
        self.df['ema_21'] = ta.trend.EMAIndicator(close=self.df['close'], window=21).ema_indicator()
        self.df['ema_50'] = ta.trend.EMAIndicator(close=self.df['close'], window=50).ema_indicator()
        self.df['ema_200'] = ta.trend.EMAIndicator(close=self.df['close'], window=200).ema_indicator()
        
        # ADX
        self.df['adx'] = ta.trend.ADXIndicator(
            high=self.df['high'], low=self.df['low'], close=self.df['close'], window=14
        ).adx()

    def _add_volatility_indicators(self):
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(close=self.df['close'], window=20, window_dev=2)
        self.df['bb_high'] = bb.bollinger_hband()
        self.df['bb_low'] = bb.bollinger_lband()
        self.df['bb_width'] = bb.bollinger_wband()
        
        # ATR
        self.df['atr'] = ta.volatility.AverageTrueRange(
            high=self.df['high'], low=self.df['low'], close=self.df['close'], window=14
        ).average_true_range()

    def _add_candlestick_patterns(self):
        # Simple manual implementations for common patterns
        # Note: ta library doesn't have a comprehensive pattern recognition set like talib
        
        open_price = self.df['open']
        close_price = self.df['close']
        high = self.df['high']
        low = self.df['low']
        
        # Body size
        body = np.abs(close_price - open_price)
        
        # Doji: Very small body
        self.df['is_doji'] = body <= (high - low) * 0.1
        
        # Bullish Engulfing
        # Previous candle red, current candle green and engulfs previous
        prev_open = open_price.shift(1)
        prev_close = close_price.shift(1)
        
        is_bullish_engulfing = (
            (prev_close < prev_open) & # Prev Red
            (close_price > open_price) & # Curr Green
            (close_price > prev_open) & 
            (open_price < prev_close)
        )
        self.df['pattern_bullish_engulfing'] = is_bullish_engulfing
        
        # Bearish Engulfing
        is_bearish_engulfing = (
            (prev_close > prev_open) & # Prev Green
            (close_price < open_price) & # Curr Red
            (close_price < prev_open) & 
            (open_price > prev_close)
        )
        self.df['pattern_bearish_engulfing'] = is_bearish_engulfing
        
        # Hammer (Bullish Pinbar)
        # Small body near top, long lower shadow
        lower_shadow = np.where(close_price < open_price, close_price - low, open_price - low)
        upper_shadow = np.where(close_price < open_price, high - open_price, high - close_price)
        
        is_hammer = (
            (lower_shadow > 2 * body) & 
            (upper_shadow < body * 0.5) &
            (body > 0) # Avoid full doji
        )
        self.df['pattern_hammer'] = is_hammer

        # Shooting Star (Bearish Pinbar)
        # Small body near bottom, long upper shadow
        is_shooting_star = (
            (upper_shadow > 2 * body) &
            (lower_shadow < body * 0.5) &
            (body > 0)
        )
        self.df['pattern_shooting_star'] = is_shooting_star

    def _add_fibonacci_levels(self):
        # Calculates Fibonacci retracements based on the last N periods high/low
        # This is a dynamic feature, essentially telling 'where represent current price locally'
        window = 50
        rolling_high = self.df['high'].rolling(window=window).max()
        rolling_low = self.df['low'].rolling(window=window).min()
        
        diff = rolling_high - rolling_low
        
        self.df['fib_0'] = rolling_low
        self.df['fib_236'] = rolling_low + diff * 0.236
        self.df['fib_382'] = rolling_low + diff * 0.382
        self.df['fib_500'] = rolling_low + diff * 0.5
        self.df['fib_618'] = rolling_low + diff * 0.618
        self.df['fib_618'] = rolling_low + diff * 0.618
        self.df['fib_100'] = rolling_high
        
        # Distance to EMA 200 (Dynamic Support/Resist)
        # Normalized by price (Percentage distance)
        self.df['dist_ema_200'] = (self.df['close'] - self.df['ema_200']) / self.df['close']
        
        # Distance to BB Lower (Oversold Support)
        self.df['dist_bb_lower'] = (self.df['close'] - self.df['bb_low']) / self.df['close']

if __name__ == "__main__":
    # Test script
    from data_manager import DataManager
    dm = DataManager()
    print("Fetching data...")
    df = dm.fetch_historical_data(limit=200)
    
    if not df.empty:
        print("Calculating indicators...")
        ta_engine = TechnicalAnalysis(df)
        df_with_ta = ta_engine.add_all_indicators()
        print(df_with_ta[['timestamp', 'close', 'rsi', 'macd', 'fib_500']].tail())
    else:
        print("No data fetched.")
