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
    def __init__(self):
        self.logger = setup_logger()
        self.logger.info("Initializing Bitcoin Trader...")
        self.dm = DataManager()
        self.brain = AIBrain()
        # Ensure we have a trained model or train on startup
        self._initial_training()

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
        predicted_price = self.brain.predict(df)
        
        if predicted_price is None:
            self.logger.warning("AI could not make a prediction.")
            return

        self.logger.info(f"Current Price: {current_price:.2f}")
        self.logger.info(f"Predicted (1h): {predicted_price:.2f}")
        
        # 5. Trading Logic (Simplified)
        # Threshold: if predicted price is > 0.5% higher
        change_percent = ((predicted_price - current_price) / current_price) * 100
        self.logger.info(f"Expected Change: {change_percent:.2f}%")
        
        signal = "HOLD"
        if change_percent > 0.5:
            signal = "BUY"
        elif change_percent < -0.5:
            signal = "SELL"
            
        self.logger.info(f"DECISION: {signal}")
        
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
