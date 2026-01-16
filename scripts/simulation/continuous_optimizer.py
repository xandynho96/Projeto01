import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import traceback
from app.core.backtest import Backtester
from app.core.ai_brain import AIBrain
from app.core.evolutionary_strategy import EvolutionaryOptimizer
from app.core.data_manager import DataManager
from app.core.technical_analysis import TechnicalAnalysis

class ContinuousOptimizer:
    def __init__(self):
        self.target_winrate = 70.0
        self.iteration = 0
        self.dm = DataManager()
        
    def start(self):
        print("🚀 Starting Continuous Optimization Loop...")
        # Initialize Brain once (or reload per cycle?) - Reload per cycle to ensure fresh scaler/model if changed
        
        while True:
            self.iteration += 1
            print(f"\n\n=== OPTIMIZATION CYCLE {self.iteration} ===")
            
            try:
                # 1. Genetic Evolution (Discover new patterns)
                print("\n🧬 [Step 1] Evolving Strategies...")
                ga = EvolutionaryOptimizer()
                # Run evolution (updates internal state and can save strategies)
                best_strategies = ga.evolve()
                
                # 2. Train AI (Learn latest price action)
                print("\n🧠 [Step 2] Retraining AI Brain...")
                # Fetch fresh data for training
                df = self.dm.get_data_from_db(limit=3000) # Train on recent history
                if df.empty:
                    print("⚠️ No data for training. Fetching...")
                    df = self.dm.fetch_historical_data(limit=1000)
                    df = self.dm.get_data_from_db(limit=2000)
                
                if not df.empty:
                    # Ensure Indicators are present for AI Training features (bb_width, etc.)
                    print("   Calculating Indicators for Training...")
                    ta = TechnicalAnalysis(df)
                    df = ta.add_all_indicators()
                    df.dropna(inplace=True)
                    
                    brain = AIBrain()
                    # Train on 1m data
                    brain.train(df)
                else:
                    print("❌ Critical: Still no data.")
                
                # 3. Validation (Backtest)
                print("\n🧪 [Step 3] Validating Performance...")
                bt = Backtester(symbol='PF_XBTUSD', timeframe='1m')
                bt.load_data(limit=2000) # Load data for backtest
                
                # Generate Signals
                print("   Generating AI Signals...")
                # We need to manually inject predictions into BT df or let BT do it?
                # BT has 'generate_ai_signals' which checks 'ai_pred' column.
                # Let's interact with BT's internal DF
                
                preds = brain.predict_batch(bt.df)
                aligned_preds = bt.align_predictions(preds, brain.sequence_length)
                bt.df['ai_pred'] = aligned_preds
                
                signals = bt.generate_ai_signals()
                result = bt.run_vectorized_backtest(signals)
                
                if result:
                    winrate = result['winrate']
                    trades = result['trades']
                    print(f"\n📊 RESULT: Winrate {winrate:.2f}% | Trades: {trades} | ROI: {result['roi']:.2f}%")
                    
                    if winrate >= self.target_winrate and trades > 5:
                        print("✅ TARGET HIT! System is optimized.")
                        # Could break here or continue to maintain
                    else:
                        print(f"⚠️ Target not met (<{self.target_winrate}%). Loop continues...")
                
            except Exception as e:
                print(f"❌ Cycle Error: {e}")
                traceback.print_exc()
            
            print("\nSuggesting 10s cooldown...")
            time.sleep(10)

if __name__ == "__main__":
    opt = ContinuousOptimizer()
    opt.start()
