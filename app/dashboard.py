import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from data_manager import DataManager
from ai_brain import AIBrain
from technical_analysis import TechnicalAnalysis
import time

st.set_page_config(page_title="Bitcoin AI Trader", layout="wide")

st.title("Bitcoin AI Trader Dashboard")

# Initialize components
dm = DataManager()
# We don't load AI brain here to avoid tensorflow memory conflict if running same env, 
# but for visualization we might want to run predictions on the fly or read from a 'signals' table.
# For now, let's just show Data + Technicals.

# Auto-refresh
if st.checkbox('Auto-Refresh (30s)'):
    time.sleep(30)
    st.rerun()

# Load Data
st.subheader("Market Data & Technicals")
limit = st.slider("Lookback Candles", min_value=100, max_value=2000, value=500)
df = dm.get_data_from_db(limit=limit)

if not df.empty:
    # Calculate Indicators on the fly for visualization
    ta = TechnicalAnalysis(df)
    df = ta.add_all_indicators()
    
    # Plot Candle Chart with Bollinger Bands
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['timestamp'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='OHLC'))
    
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['bb_high'], line=dict(color='gray', width=1), name='BB High'))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['bb_low'], line=dict(color='gray', width=1), name='BB Low'))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # KPIs
    current_price = df['close'].iloc[-1]
    rsi = df['rsi'].iloc[-1]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Price", f"${current_price:.2f}")
    col2.metric("RSI (14)", f"{rsi:.2f}")
    col3.metric("Volume", f"{df['volume'].iloc[-1]:.2f}")
    
    # Mock AI Prediction (or load if we want)
    # If we want real prediction, we need to load the model. 
    # Let's add a button to Predict
    if st.button("Generate AI Prediction"):
        try:
            brain = AIBrain() # This might be heavy
            prediction = brain.predict(df)
            if prediction:
                change = ((prediction - current_price) / current_price) * 100
                st.success(f"AI Predicted Price (1h): ${prediction:.2f} ({change:+.2f}%)")
        except Exception as e:
            st.error(f"Error running AI: {e}")

else:
    st.warning("No data found in database.")
