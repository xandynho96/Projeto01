import pandas as pd
from data_manager import DataManager, Trade
from datetime import datetime
import os

# Use a test DB
TEST_DB = "sqlite:///test_pl.db"

def test_weekly_pl():
    print("Setting up test DB...")
    dm = DataManager(db_url=TEST_DB)
    session = dm.Session()
    
    # Clear trades
    session.query(Trade).delete()
    session.commit()
    
    # 1. Insert Closed Trade TODAY (Should count) - PNL 50
    t1 = Trade(
        symbol="BTC/USD", 
        side="buy", 
        amount=0.1, 
        price=50000, 
        status="closed", 
        pnl=50.0, 
        timestamp=datetime.utcnow()
    )
    session.add(t1)
    
    # 2. Insert Closed Trade 3 Days Ago (Should count) - PNL -10
    t2 = Trade(
        symbol="BTC/USD", 
        side="buy", 
        amount=0.1, 
        price=50000, 
        status="closed", 
        pnl=-10.0, 
        timestamp=datetime.utcnow() - pd.Timedelta(days=3)
    )
    session.add(t2)
    
    # 3. Insert Closed Trade 8 Days Ago (Should NOT count) - PNL 100
    t3 = Trade(
        symbol="BTC/USD", 
        side="buy", 
        amount=0.1, 
        price=50000, 
        status="closed", 
        pnl=100.0, 
        timestamp=datetime.utcnow() - pd.Timedelta(days=8)
    )
    session.add(t3)
    
    # 4. Insert OPEN Trade (Should NOT count)
    t4 = Trade(
        symbol="BTC/USD", 
        side="buy", 
        amount=0.1, 
        price=50000, 
        status="open", 
        pnl=None, # Or 0
        timestamp=datetime.utcnow()
    )
    session.add(t4)
    
    session.commit()
    session.close()
    
    print("Trades inserted. Calculating Weekly PL...")
    pl = dm.get_weekly_pnl()
    
    expected_pl = 50.0 - 10.0 # = 40.0
    print(f"Calculated PL: {pl}")
    print(f"Expected PL: {expected_pl}")
    
    if abs(pl - expected_pl) < 0.01:
        print("✅ SUCCESS: Weekly PL calculation is correct.")
    else:
        print("❌ FAILURE: Weekly PL calculation is incorrect.")
        
    # Cleanup
    if os.path.exists("test_pl.db"):
        os.remove("test_pl.db")

if __name__ == "__main__":
    test_weekly_pl()
