import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import requests
import json
import os
import joblib
import script_utils # Import helper

class AIBrain:
    def __init__(self, model_path='bitcoin_ai_model.keras', scaler_path='scaler.pkl', classifier_path='bitcoin_ai_classifier.keras'):
        # Resolve paths using resource_path for PyInstaller compatibility
        self.model_path = script_utils.resource_path(model_path)
        self.classifier_path = script_utils.resource_path(classifier_path)
        self.scaler_path = script_utils.resource_path(scaler_path)

        self.scaler = self._load_or_create_scaler()
        self.model = self._load_or_create_model()
        self.classifier = self._load_or_create_classifier()
        self.sequence_length = 60 # Lookback period (e.g. 60 hours)

    def _load_or_create_classifier(self):
        if os.path.exists(self.classifier_path):
            print("Loading existing AI Classifier...")
            return load_model(self.classifier_path)
        else:
            print("Initializing new AI Classifier placeholder...")
            return None

    def build_classifier(self, input_shape):
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=1, activation='sigmoid')) # Probability 0-1
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.classifier = model
        return model

    def _load_or_create_scaler(self):
        if os.path.exists(self.scaler_path):
            print("Loading existing Scaler...")
            try:
                return joblib.load(self.scaler_path)
            except:
                print("Failed to load scaler, creating new.")
                return MinMaxScaler(feature_range=(0, 1))
        else:
            return MinMaxScaler(feature_range=(0, 1))

    def _load_or_create_model(self):
        if os.path.exists(self.model_path):
            print("Loading existing AI model...")
            return load_model(self.model_path)
        else:
            print("Initializing new AI model...")
            return None

    def build_model(self, input_shape):
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=1)) # Prediction: Close price (or 1/0 for classification)
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model
        return model

    def prepare_data(self, df):
        # Feature selection (Normalized)
        # 1. Calculate Returns to fix "AI Blindness"
        df['returns'] = df['close'].pct_change()
        df.replace([np.inf, -np.inf], 0, inplace=True)
        df.fillna(0, inplace=True)
        
        features = ['returns', 'rsi', 'macd', 'bb_width', 'adx']
        data = df[features].values
        
        # Scale data
        # Note: 'returns' is small (0.01), RSI is large (50). Scaler handles this.
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        # Create sequences
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i])
            y.append(scaled_data[i, 0]) # Predicting 'close' price (index 0)
            
        return np.array(X), np.array(y)

    def train(self, df):
        if len(df) < self.sequence_length + 10:
            print("Not enough data to train.")
            return
            
        print("Preparing data for training...")
        X, y = self.prepare_data(df)
        
        if self.model is None:
            self.build_model((X.shape[1], X.shape[2]))
            
        print("Training AI model...")
        self.model.fit(X, y, epochs=5, batch_size=32) # Small epochs for demo, increase for real
        self.model.save(self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        print(f"Model saved to {self.model_path}, Scaler saved to {self.scaler_path}")

    def predict(self, df):
        if self.model is None:
            print("Model not trained yet.")
            return None
            
        # Prepare last sequence
        # Prepare last sequence
        # Calculate returns on the fly (need history)
        if 'returns' not in df.columns:
            df['returns'] = df['close'].pct_change()
            df.replace([np.inf, -np.inf], 0, inplace=True)
            df.fillna(0, inplace=True)

        features = ['returns', 'rsi', 'macd', 'bb_width', 'adx']
        data = df[features].values
        
        if len(data) < self.sequence_length:
            print(f"Not enough data for prediction. Need {self.sequence_length}, got {len(data)}")
            return None

        scaled_data = self.scaler.transform(data)
        
        last_sequence = scaled_data[-self.sequence_length:]
        last_sequence = np.reshape(last_sequence, (1, self.sequence_length, len(features)))
        
        predicted_scaled = self.model.predict(last_sequence)
        
        # Inverse transform (trickier because we need dummy for other columns)
        dummy = np.zeros((1, len(features)))
        dummy[0, 0] = predicted_scaled[0, 0]
        predicted_price = self.scaler.inverse_transform(dummy)[0, 0]
        
    def predict_batch(self, df):
        """
        Predicts for the entire dataframe in one go.
        Returns a numpy array of predictions matching the dataframe index (aligned to end).
        """
        if self.model is None:
            print("Model not trained yet.")
            return None
            
        if self.model is None:
            print("Model not trained yet.")
            return None

        # Add returns if missing
        if 'returns' not in df.columns:
            df['returns'] = df['close'].pct_change()
            df.replace([np.inf, -np.inf], 0, inplace=True)
            df.fillna(0, inplace=True)
            
        features = ['returns', 'rsi', 'macd', 'bb_width', 'adx']
        data = df[features].values
        
        # Scale
        scaled_data = self.scaler.transform(data)
        
        # Create sequences
        X = []
        # We need validation that we have enough data
        if len(scaled_data) <= self.sequence_length:
            return np.array([])

        # Vectorized sequence creation is faster but list comprehension is okay for 10k
        # Better: use stride_tricks or just loop, creating the Big X
        # For 10k rows, loop is negligible compared to inference, but let's try to be efficient
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i])
            
        X = np.array(X)
        
        # Batch Predict
        print(f"Batch Predicting on {len(X)} samples...")
        predictions_scaled = self.model.predict(X, batch_size=128, verbose=1)
        
        # Inverse Transform
        # Create dummy array with correct shape
        # predictions_scaled is (N, 1)
        # We need (N, 5) for inverse_transform, with close at index 0
        
        dummy = np.zeros((len(predictions_scaled), len(features)))
        dummy[:, 0] = predictions_scaled.flatten()
        
        predictions = self.scaler.inverse_transform(dummy)[:, 0]
        
        # The result is shorter than df by sequence_length
        # We should pad it or handle alignment in optimizer
        return predictions

    def predict_proba(self, df):
        """
        Predicts probability of a win (0.0 to 1.0).
        df should be a single sequence slice (check shape).
        """
        if self.classifier is None:
            # Try to build/load? For now return 0.5 neutral
            return 0.5
            
        # Add returns if missing (though predict_proba receives slice usually)
        # If df is DataFrame slice
        if 'returns' not in df.columns:
             # pct_change on a slice might be wrong for the first element (NaN)
             # Ideally we pass larger slice and take last
             df['returns'] = df['close'].pct_change()
             df.replace([np.inf, -np.inf], 0, inplace=True)
             df.fillna(0, inplace=True)
             
        features = ['returns', 'rsi', 'macd', 'bb_width', 'adx']
        data = df[features].values
        
        scaled_data = self.scaler.transform(data)
        
        # Check shape, we need (1, 60, 5)
        if len(scaled_data) < self.sequence_length:
             return 0.0
             
        seq = scaled_data[-self.sequence_length:]
        seq = np.reshape(seq, (1, self.sequence_length, len(features)))
        
        prob = self.classifier.predict(seq, verbose=0)[0][0]
        return prob

    def validate_signal_with_deepseek(self, current_price, predicted_price, technical_summary):
        """
        Asks Deepseek for a second opinion on the trade.
        """
        if not config.DEEPSEEK_API_KEY:
            print("Deepseek API Key not found. Skipping validation.")
            return {"approved": True, "reason": "No Deepseek Key, skipping validation."}

        # Logic to call Deepseek API
        # Mocked for now unless keys are provided
        prompt = f"""
        Analyze this Bitcoin trade setup:
        Current Price: {current_price}
        Predicted Price (AI): {predicted_price}
        Technical Indicators: {technical_summary}
        
        Should I take this trade? Reply JSON: {{ "approved": boolean, "reason": string }}
        """
        
        # Call deepseek logic here (commented out for safety/cost unless confirmed)
        # response = requests.post(...)
        
        return {"approved": True, "reason": "Simulated approval (Deepseek logic placeholder)."}

if __name__ == "__main__":
    # Test script
    from data_manager import DataManager
    from technical_analysis import TechnicalAnalysis
    
    # 1. Get Data
    dm = DataManager()
    df = dm.fetch_historical_data(limit=500)
    
    # 2. Add Features
    ta = TechnicalAnalysis(df)
    df = ta.add_all_indicators()
    df.dropna(inplace=True) # Remove NaN from indicators
    
    # 3. Train AI
    brain = AIBrain()
    brain.train(df)
    

