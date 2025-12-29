from data_manager import DataManager
from technical_analysis import TechnicalAnalysis
from ai_brain import AIBrain
import pandas as pd

def train_full_model():
    print("Loading full history from Database...")
    dm = DataManager()
    # Get all data from DB
    df = dm.get_data_from_db(limit=50000) 
    
    if df.empty:
        print("No data in database to train on.")
        return
        
    print(f"Loaded {len(df)} candles from database.")
    
    # Calculate Indicators
    print("Calculating Technical Indicators (this may take a moment)...")
    ta = TechnicalAnalysis(df)
    df = ta.add_all_indicators()
    df.dropna(inplace=True)
    
    print(f"Data ready for training: {len(df)} samples.")
    
    # Train AI
    # Train AI
    brain = AIBrain()
    # Use the class method to ensure scaler is saved
    brain.train(df)
    
    # brain.model.fit(...) # Removed manual fit logic
    # brain.model.save(...) # Removed manual save logic

if __name__ == "__main__":
    train_full_model()
