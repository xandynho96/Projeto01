from data_manager import DataManager
from technical_analysis import TechnicalAnalysis
from ai_brain import AIBrain
import pandas as pd
import time

def train_deep():
    print("🚀 Starting Deep Training Session...")
    
    dm = DataManager()
    
    # 1. Load Data (Big Load)
    print("📥 Loading Data (LIMIT=500) for Instant Start...")
    # Using fetch_if_missing=False because user should have run download_history.py
    df = dm.get_data_from_db(limit=500) 
    
    if len(df) < 10000:
        print(f"⚠️ Warning: Loaded only {len(df)} candles. Did 'download_history.py' finish?")
        # Force fetch small batch if completely empty just to not crash
        if len(df) == 0:
            print("Fetching small batch for fallback...")
            dm.fetch_historical_data(limit=1000)
            df = dm.get_data_from_db(limit=1000)
    
    print(f"✅ Loaded {len(df)} candles.")
    
    # 2. Feature Engineering
    print("🛠️ Calculating Patterns & Supports (feature rich)...")
    ta = TechnicalAnalysis(df)
    
    # We need to ensure we call the relevant methods
    df = ta.add_all_indicators() # This calls standard ones
    
    # Ensure Patterns and Support/Resistance are calculated
    # add_all_indicators might verify this, but let's double check by calling them if not present?
    # Actually TechnicalAnalysis.add_all_indicators calls add_candle_patterns and add_support_resistance
    
    df.dropna(inplace=True)
    print(f"✅ Clean Data Size: {len(df)}")
    
    # 3. Train
    print("🧠 Training AI Brain (RandomForest - 100 Trees)...")
    brain = AIBrain()
    
    start_time = time.time()
    brain.train(df)
    end_time = time.time()
    
    print(f"🎉 Training Complete! Time taken: {end_time - start_time:.2f}s")
    print("Model saved to 'bitcoin_ai_model.pkl'.")

if __name__ == "__main__":
    train_deep()
