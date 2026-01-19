import sys
import os
import pandas as pd

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.data_manager import DataManager

def verify_m5():
    print("🧪 Verificando integração de dados M5...")
    dm = DataManager()
    
    # Fetch Data (Mock or Real)
    # We need to make sure we have data in DB or can look it up
    # fetch_multi_timeframe_data calls fetch_historical_data which saves to DB
    
    print("Fetching Multiframe Data...")
    df = dm.fetch_multi_timeframe_data(limit=100)
    
    if df.empty:
        print("❌ DataFrame vazio. Verifique conexão ou banco de dados.")
        return

    print(f"✅ DataFrame Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    required_cols = ['stoch_k_5m', 'stoch_rsi_k_5m']
    missing = [c for c in required_cols if c not in df.columns]
    
    if missing:
        print(f"❌ Falta colunas M5: {missing}")
    else:
        print(f"✅ Sucesso! Colunas M5 encontradas: {required_cols}")
        sample = df[['timestamp', 'close', 'stoch_k_5m', 'stoch_rsi_k_5m']].tail(3)
        print("Amostra:")
        print(sample)

if __name__ == "__main__":
    verify_m5()
