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
        self._add_candlestick_patterns()
        self._add_chart_patterns() # New Advanced Patterns
        self._add_trend_hierarchy() # New Trend Hierarchy
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

        # Money Flow Index (MFI)
        self.df['mfi'] = ta.volume.MFIIndicator(
            high=self.df['high'], low=self.df['low'], close=self.df['close'], volume=self.df['volume'], window=14
        ).money_flow_index()
        
        # Commodity Channel Index (CCI)
        self.df['cci'] = ta.trend.CCIIndicator(
            high=self.df['high'], low=self.df['low'], close=self.df['close'], window=20
        ).cci()
        
        # Williams %R
        self.df['williams_r'] = ta.momentum.WilliamsRIndicator(
            high=self.df['high'], low=self.df['low'], close=self.df['close'], lbp=14
        ).williams_r()

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

        # Marubozu (Strong Candle)
        # Large body, very small shadows
        # Body > 2x average body (approx) or just large relative to shadows
        is_marubozu = (
            (body > (high - low) * 0.8) & # Body is 80% of total range
            (body > 0)
        )
        self.df['pattern_marubozu'] = is_marubozu

        # ADX Slope (Trend Strength Change)
        # Using shift(1) to see immediate change
        self.df['adx_slope'] = self.df['adx'].diff()

        # Distance to Donchian Channel (Support/Resistance)
        # Support = Lowest Low of last 20
        # Resistance = Highest High of last 20
        window_sr = 20
        support = self.df['low'].rolling(window=window_sr).min()
        resistance = self.df['high'].rolling(window=window_sr).max()
        
        # Distance (normalized by close)
        self.df['dist_support'] = (self.df['close'] - support) / self.df['close']
        self.df['dist_resistance'] = (resistance - self.df['close']) / self.df['close']

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

    def _add_trend_hierarchy(self):
        """
        Determines Trend Alignment (Micro vs Macro).
        Macro: EMA 200 direction.
        Micro: EMA 21 vs EMA 50.
        """
        ema_200 = self.df['ema_200']
        ema_50 = self.df['ema_50']
        ema_21 = self.df['ema_21']
        close = self.df['close']
        
        # Macro Trend (1 = Bullish, -1 = Bearish)
        macro_trend = np.where(close > ema_200, 1, -1)
        
        # Micro Trend (1 = Bullish, -1 = Bearish)
        micro_trend = np.where(ema_21 > ema_50, 1, -1)
        
        # Alignment Score (2 = Strong Bull, -2 = Strong Bear, 0 = Mixed/Choppy)
        self.df['trend_score'] = macro_trend + micro_trend
        
        # Interaction
        self.df['trend_aligned'] = (macro_trend == micro_trend).astype(int)

    def _add_chart_patterns(self):
        """
        Aims to detect complex chart patterns like OCO (Head & Shoulders) and Triangles.
        Note: Precise detection is hard in vectorized code. We use simplified logic suitable for ML features.
        """
        high = self.df['high']
        low = self.df['low']
        close = self.df['close'] # Fixed: Was referencing undefined 'close'
        
        # 1. TRIANGLE / SQUEEZE DETECTION
        # Logic: Lower Highs AND Higher Lows over a rolling window.
        window = 10
        rolling_max = high.rolling(window=window).max()
        rolling_min = low.rolling(window=window).min()
        
        # Slope check (simplified)
        # Highs are falling?
        high_slope = high.diff(3).rolling(window=10).mean()
        # Lows are rising?
        low_slope = low.diff(3).rolling(window=10).mean()
        
        # Potential Triangle: Highs falling (<0) AND Lows rising (>0) AND Volatility Dropping
        is_triangle = (high_slope < 0) & (low_slope > 0) & (self.df['bb_width'] < 0.01)
        self.df['pattern_triangle'] = is_triangle.astype(int)
        
        # 2. OCO (Head and Shoulders - Bearish)
        # Peak (Left Shoulder) -> Higher Peak (Head) -> Lower Peak (Right Shoulder)
        # This is extremely hard to vectorise perfectly. 
        # We will use "Pivot Highs" feature for the AI to learn instead of hardcoding.
        
        # Identify Local Highs (Fractals)
        # Standard Fractal: High[i] > High[i-2]...High[i+2]
        # We can pass "Is Fractal High" as a feature.
        
        # Vectorized Fractal (5 candle)
        # We need future data for perfect fractals, but for trading signal at Time T, we look at lag.
        # So we detect if T-2 was a fractal high.
        
        # Shifted comparison for backtest/realtime safety (can't know T+2 at T)
        # At Time T, we know if T-2 was a High relative to T-4, T-3, T-1, T.
        
        is_fractal_high = (
            (high.shift(2) > high.shift(3)) &
            (high.shift(2) > high.shift(4)) &
            (high.shift(2) > high.shift(1)) &
            (high.shift(2) > high)
        )
        self.df['is_pivot_high'] = is_fractal_high.astype(int)
        
        is_fractal_low = (
            (low.shift(2) < low.shift(3)) &
            (low.shift(2) < low.shift(4)) &
            (low.shift(2) < low.shift(1)) &
            (low.shift(2) < low)
        )
        self.df['is_pivot_low'] = is_fractal_low.astype(int)
        
        # The AI (RandomForest) can learn OCO patterns if we feed it the SEQUENCE of Pivots.
        # We will feed 'time_since_last_pivot' and 'price_of_last_pivot' to help it map geometry.
        
        self.df['last_pivot_high_price'] = self.df['high'].where(is_fractal_high).ffill()
        self.df['last_pivot_low_price'] = self.df['low'].where(is_fractal_low).ffill() 

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
