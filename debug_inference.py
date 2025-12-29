from ai_brain import AIBrain
import pandas as pd
import numpy as np

def debug():
    print("Initializing Brain...")
    brain = AIBrain()
    
    print("Creating dummy input (1, 60, 5)...")
    # 5 features: close, rsi, macd, bb_width, adx
    dummy = np.random.random((1, 60, 5))
    
    # We need to bypass scaler or match it. 
    # predict_proba expects DATAFRAME to scale.
    # Let's mock a dataframe.
    cols = ['close', 'rsi', 'macd', 'bb_width', 'adx']
    df = pd.DataFrame(np.random.random((60, 5)), columns=cols)
    
    print("Testing predict_proba...")
    try:
        prob = brain.predict_proba(df)
        print(f"Success! Probability: {prob}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug()
