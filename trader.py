import time
import schedule
import pandas as pd
from datetime import datetime
from data_manager import DataManager
from technical_analysis import TechnicalAnalysis
from data_manager import DataManager
from technical_analysis import TechnicalAnalysis
from ai_brain import AIBrain
import config
from logger import setup_logger

class BitcoinTrader:
    def __init__(self, api_key=None, secret=None, user_settings={}):
        self.logger = setup_logger()
        self.logger.info("Initializing Bitcoin Trader...")
        self.dm = DataManager()
        
        # Connect if keys provided
        if api_key and secret:
            demo_mode = user_settings.get('demo_mode', False)
            self.dm.connect_exchange(api_key, secret, demo_mode=demo_mode)
            
        self.user_settings = user_settings
        self.brain = AIBrain()
        # Ensure we have a trained model or train on startup
        self._initial_training()
        
        # Set Leverage if provided
        leverage = float(user_settings.get('leverage', config.LEVERAGE))
        if leverage > 1:
            self.dm.set_leverage(leverage)

    def _initial_training(self):
        self.logger.info("Performing initial data fetch and training...")
        # Prefer loading from DB if we have history
        df = self.dm.get_data_from_db(limit=2000) # Quick check
        
        if df.empty or len(df) < 500:
             # Fallback to fetch if DB empty
            df = self.dm.fetch_historical_data(limit=1000)
            
        if not df.empty:
            self.dm.save_data(df)
            ta = TechnicalAnalysis(df)
            df = ta.add_all_indicators()
            df.dropna(inplace=True)
            # Only train if model doesn't exist? Or re-train?
            # For now, let's skip training here if it's already running in background manually
            # or just load the model. 
            pass 
        else:
            self.logger.error("Failed to fetch initial data.")

    def job(self):
        self.logger.info(f"--- Analysis Job Started ---")
        
        # 1. Fetch latest data
        # Fetch slightly more than needed to calculate indicators correctly
        df = self.dm.fetch_historical_data(limit=1000)
        
        if df.empty:
            self.logger.warning("No data received.")
            return

        # 2. Save to DB
        self.dm.save_data(df)

        # 3. Analyze
        ta = TechnicalAnalysis(df)
        df = ta.add_all_indicators()
        df.dropna(inplace=True) # Important for LSTM

        # 4. Get AI Prediction
        current_price = df['close'].iloc[-1]
        
        # New API returns 'predicted_return' (e.g. 0.005 for 0.5% up)
        predicted_return = self.brain.get_prediction(df)
        
        # Calculate Target Price from Return
        predicted_price = current_price * (1 + predicted_return)
        
        if (predicted_return == 0.0 and predicted_price == current_price):
             # Try Fallback if return is exactly 0 (might mean no prediction made)
             predicted_price = None

        if predicted_price is None:
            self.logger.warning("AI could not make a prediction. Switching to Technical Analysis Fallback.")
            # Fallback Logic: Use current price as base and let Signals decide
            predicted_price = current_price 
            
            # Simple TA Strategy Fallback
            rsi = df['rsi'].iloc[-1]
            if rsi < 30:
                signal = "buy"
                self.logger.info("Fallback Strategy: RSI Oversold (<30) -> BUY Signal")
            elif rsi > 70:
                signal = "sell"
                self.logger.info("Fallback Strategy: RSI Overbought (>70) -> SELL Signal")
            else:
                 signal = "HOLD"
                 self.logger.info(f"Fallback Strategy: RSI Neutral ({rsi:.2f}) -> HOLD")
                 return # Exit if HOLD
        else:
             # Normal AI Logic
             self.logger.info(f"Current Price: {current_price:.2f}")
             self.logger.info(f"Predicted Price (Next Candle): {predicted_price:.2f}")
             self.logger.info(f"Predicted Return: {predicted_return*100:.4f}%")
             
             # Threshold: if predicted return is > 0.1% (High Frequency Scalping)
             # User requested HFT Scalping so thresholds should be small.
             # 0.5% was for 1h Timeframe. For 1m, 0.5% is huge. Lowering to 0.05% or 0.1%?
             # Let's use 0.05% (0.0005) filter
             
             change_percent = predicted_return * 100
             self.logger.info(f"Expected Change: {change_percent:.4f}%")
             
             signal = "HOLD"
             if change_percent > 0.02: # Very sensitive for 1m scalping
                 signal = "buy"
             elif change_percent < -0.02:
                 signal = "sell"
            
        self.logger.info(f"DECISION: {signal}")
        
        if signal != "HOLD":
             # Use User Settings
             # 'amount' is now treated as USD value (e.g., 50.0)
             amount_usd = float(self.user_settings.get('amount', 50.0))
             
             # Calculate BTC amount based on current price
             amount_btc = amount_usd / current_price
             # Ensure a minimum size (Kraken Futures min is roughly 0.0001 or $1 depending on contract, safe margin)
             if amount_btc < 0.0001:
                 self.logger.warning(f"Calculated amount {amount_btc:.6f} BTC ($ {amount_usd}) is too small. Adjusting to 0.0001")
                 amount_btc = 0.0001
                 
             self.logger.info(f"Target Entry: ${amount_usd} (~{amount_btc:.5f} BTC)")

             sl_pct = float(self.user_settings.get('sl_pct', 2.0))
             tp_pct = float(self.user_settings.get('tp_pct', 4.0))
             
             # Calculate SL/TP Prices
             # BUY: SL below, TP above
             # SELL: SL above, TP below
             
             params = {}
             # CCXT Unified approach for simple SL/TP (might vary by exchange implementation)
             # Ideally we check exchange capabilities, but adding params is a good first step.
             # Kraken Futures: 'stopLossPrice', 'takeProfitPrice' might work or need 'stopLoss' dictionary
             
             if signal == "buy":
                 sl_price = current_price * (1 - sl_pct/100)
                 tp_price = current_price * (1 + tp_pct/100)
             else: # sell
                 sl_price = current_price * (1 + sl_pct/100)
                 tp_price = current_price * (1 - tp_pct/100)
                 
             # params = {'stopLossPrice': sl_price, 'takeProfitPrice': tp_price} 
             
             # Execute Entry (Market Order)
             self.logger.info(f"🚀 Executing {signal.upper()} Entry for {amount_btc:.5f} BTC...")
             entry_order = self.dm.execute_order(config.SYMBOL, signal, amount_btc, type='market')
             
             if entry_order:
                 self.logger.info(f"✅ Entry Filled. ID: {entry_order['id']}")
                 
                 # Record trade to history
                 self.dm.save_trade(
                     symbol=config.SYMBOL, 
                     side=signal, 
                     amount=amount_btc, 
                     price=current_price, 
                     status='open'
                 )
                 
                 # --- Place Exit Orders (Reduce Only) ---
                 # Stop Loss
                 sl_side = "sell" if signal == "buy" else "buy"
                 sl_params = {'reduceOnly': True}
                 self.logger.info(f"🛡️ Placing Stop Loss at {sl_price:.2f}...")
                 self.dm.execute_order(config.SYMBOL, sl_side, amount_btc, type='stop', price=sl_price, params=sl_params)
                 
                 # Take Profit
                 tp_side = "sell" if signal == "buy" else "buy"
                 tp_params = {'reduceOnly': True}
                 self.logger.info(f"💰 Placing Take Profit at {tp_price:.2f}...")
                 self.dm.execute_order(config.SYMBOL, tp_side, amount_btc, type='take_profit', price=tp_price, params=tp_params)
                 
             else:
                 self.logger.error("❌ Entry Order Failed.")
        
        # 6. Deepseek Validation (Optional)
        if signal != "HOLD":
            technical_summary = df.tail(1)[['rsi', 'macd', 'stoch_k']].to_dict('records')[0]
            validation = self.brain.validate_signal_with_deepseek(current_price, predicted_price, technical_summary)
            self.logger.info(f"Deepseek Validation: {validation}")

    def run(self):
        self.logger.info("Bot is running. Press Ctrl+C to stop.")
        
        # Run once immediately
        self.job()
        
        # Schedule every hour (since timeframe is 1h)
        schedule.every(1).hours.do(self.job)
        
        while True:
            schedule.run_pending()
            time.sleep(1)

if __name__ == "__main__":
    bot = BitcoinTrader()
    bot.run()
