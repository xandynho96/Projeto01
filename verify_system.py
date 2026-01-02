from market_regime import MarketRegime
import pandas as pd
from evolutionary_strategy import EvolutionaryOptimizer
from extract_winning_signals import WinningSignalExtractor
import evolutionary_strategy

def verify():
    print("=== VERIFYING CONTEXT-AWARE SYSTEM ===")

    # 1. Test Regime
    print("\n[1/3] Testing Regime Detection...")
    mr = MarketRegime()
    # Mock DF
    df = pd.DataFrame({
        'close': [100, 80, 100], 
        'ema_200': [90, 90, 90], 
        'adx': [30, 25, 15]
    })
    # Row 0: 100 > 90, ADX 30 -> UPTREND
    # Row 1: 80 < 90, ADX 25 -> DOWNTREND
    # Row 2: 100 > 90, ADX 15 -> SIDEWAYS
    
    r0 = mr.classify(df.iloc[0])
    r1 = mr.classify(df.iloc[1])
    r2 = mr.classify(df.iloc[2])
    print(f"Row 0: {r0} (Exp: UPTREND)")
    print(f"Row 1: {r1} (Exp: DOWNTREND)")
    print(f"Row 2: {r2} (Exp: SIDEWAYS)")
    
    # 2. Test Evolution (Stub)
    print("\n[2/3] Testing Evolution (Fast Mode)...")
    # Monkey patch for speed
    evolutionary_strategy.GENERATIONS = 1
    evolutionary_strategy.POPULATION_SIZE = 5
    
    opt = EvolutionaryOptimizer()
    strategies = opt.evolve()
    print(f"Strategies Generated: {list(strategies.keys())}")
    
    if 'UPTREND' in strategies and 'DOWNTREND' in strategies:
        print("✅ Strategy Dictionary Structure Valid.")
    else:
        print("❌ Strategy Dictionary Invalid.")
    
    # 3. Test Extraction
    print("\n[3/3] Testing Signal Extraction...")
    ext = WinningSignalExtractor()
    ext.extract(strategy_genome=strategies)
    
    print("\n=== VERIFICATION COMPLETE ===")

if __name__ == "__main__":
    verify()
