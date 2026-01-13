import numpy as np
import pandas as pd
import os
import joblib
import script_utils # Import helper
from sklearn.preprocessing import MinMaxScaler
import requests
import json

HAS_TENSORFLOW = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    HAS_TENSORFLOW = True
except ImportError:
    print("Warning: TensorFlow not found. AI Brain LSTM features will be disabled.")

class AIBrain:
    def __init__(self, model_filename='bitcoin_ai_model.pkl', scaler_filename='scaler.pkl'):
        # Persistent paths (cwd)
        self.model_path = os.path.abspath(model_filename)
        self.scaler_path = os.path.abspath(scaler_filename)
        
        # Bundled paths (fallback for reading initial model)
        self.bundled_model_path = script_utils.resource_path(model_filename)
        self.bundled_scaler_path = script_utils.resource_path(scaler_filename)

        self.scaler = self._load_or_create_scaler()
        self.model = None
        self.sequence_length = 60 # Lookback period
        
        # Initialize Model (TF or Sklearn)
        if HAS_TENSORFLOW:
            self.model = self._load_or_create_model_tf()
        else:
            self.model = self._load_or_create_model_sklearn()

    def _load_or_create_scaler(self):
        # 1. Try Persistent
        if os.path.exists(self.scaler_path):
            print(f"Loading existing Scaler from {self.scaler_path}...")
            try:
                return joblib.load(self.scaler_path)
            except:
                print("Failed to load persistent scaler.")
        
        # 2. Try Bundled (if different)
        if self.bundled_scaler_path != self.scaler_path and os.path.exists(self.bundled_scaler_path):
             print(f"Loading bundled Scaler from {self.bundled_scaler_path}...")
             try:
                return joblib.load(self.bundled_scaler_path)
             except:
                pass
                
        # 3. Create New
        print("Creating new Scaler.")
        return MinMaxScaler(feature_range=(0, 1))

    def _load_or_create_model_sklearn(self):
        from sklearn.ensemble import RandomForestRegressor
        # Increased complexity for better learning of subtle patterns
        
        # 1. Try Persistent
        if os.path.exists(self.model_path):
            print(f"Loading existing AI model (Sklearn) from {self.model_path}...")
            try:
                return joblib.load(self.model_path)
            except:
                print("Failed to load persistent model.")

        # 2. Try Bundled
        if self.bundled_model_path != self.model_path and os.path.exists(self.bundled_model_path):
             print(f"Loading bundled AI model from {self.bundled_model_path}...")
             try:
                return joblib.load(self.bundled_model_path)
             except:
                pass

        print("Initializing new AI model (RandomForest - Enhanced)...")
        # Increased estimators to 200 and depth to 20 for better "learning" capability
        return RandomForestRegressor(n_estimators=200, n_jobs=1, max_depth=20, random_state=42)

    def _load_or_create_model_tf(self):
        if not HAS_TENSORFLOW: return None
        if os.path.exists(self.model_path):
            print("Loading existing AI model (TF)...")
            return load_model(self.model_path)
        else:
            print("Initializing new AI model (TF)...")
            return None

    def build_model(self, input_shape):
        if not HAS_TENSORFLOW: return None
        model = Sequential()
        model.add(LSTM(units=100, return_sequences=True, input_shape=input_shape)) # Increased units
        model.add(Dropout(0.2))
        model.add(LSTM(units=100, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=1)) 
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model
        return model

    def is_model_trained(self):
        """Checks if the internal model is actually fitted/trained."""
        if self.model is None:
            return False
            
        if HAS_TENSORFLOW:
            # TF models loaded from disk are compiled/fitted. 
            # If created new (not loaded), weights might be random but structure exists.
            # Good enough proxy: if we loaded it, it's fine. 
            pass 
        else:
            # Sklearn
            try:
                from sklearn.utils.validation import check_is_fitted
                check_is_fitted(self.model)
                return True
            except:
                return False
        return True # Default assumption for TF (if loaded) or if logic passes


    def _get_feature_list(self, df):
        """Dynamically builds feature list including Multi-Timeframe columns."""
        base_features = ['returns', 'rsi', 'macd', 'bb_width', 'adx', 'ema_trend', 
                         'volatility_change', 'dist_support', 'dist_resistance', 
                         'pattern_score', 'stoch_rsi_k', 'atr', 'obv_slope',
                         'trend_score', 'pattern_triangle', 'is_pivot_high', 'is_pivot_low']
        
        # Add dynamic multi-timeframe features found in DF
        mtf_features = [c for c in df.columns if c.endswith('_5m') or c.endswith('_15m')]
        
        # Deduplicate and combine
        features = list(set(base_features + mtf_features))
        
        # Sort to ensure consistent order
        features.sort()
        
        # Enforce 'returns' as the first feature (Target)
        if 'returns' in features:
            features.remove('returns')
            features.insert(0, 'returns')
        return features

    def prepare_data(self, df):
        df = df.copy()
        df['returns'] = df['close'].pct_change()
        # Add dynamic features if missing
        if 'ema_trend' not in df.columns and 'ema_200' in df.columns:
            df['ema_trend'] = (df['close'] - df['ema_200']) / df['ema_200']
        elif 'ema_trend' not in df.columns:
            df['ema_trend'] = 0


        if 'volatility_change' not in df.columns:
            df['volatility_change'] = df['bb_width'].pct_change().fillna(0)
            
        # Add Pattern Score if patterns exist
        if 'pattern_score' not in df.columns:
            # Simple aggregation of bullish - bearish patterns
            # Check if columns exist first
            bullish = ['pattern_bullish_engulfing', 'pattern_hammer', 'pattern_marubozu']
            bearish = ['pattern_bearish_engulfing', 'pattern_shooting_star']
            
            score = pd.Series(0, index=df.index)
            for p in bullish:
                if p in df.columns: score += df[p]
            for p in bearish:
                if p in df.columns: score -= df[p]
            df['pattern_score'] = score

        # Add Support/Resistance Distance
        if 'dist_support' not in df.columns: df['dist_support'] = 0
        if 'dist_resistance' not in df.columns: df['dist_resistance'] = 0
        
        # Ensure new advanced features exist (fill 0 if not calc by TA to avoid crash)
        for col in ['stoch_rsi_k', 'atr', 'obv_slope']:
            if col not in df.columns: df[col] = 0

        # Cleanup AFTER features
        df.replace([np.inf, -np.inf], 0, inplace=True)
        df.fillna(0, inplace=True)

        # Dynamic Feature Selection
        features = self._get_feature_list(df)
        
        # Ensure base features exist
        for f in features:
            if f not in df.columns: df[f] = 0

        data = df[features].values
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        # Create sequences
        # Note: If Sklearn, we might flatten inside train/predict
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i])
            y.append(scaled_data[i, 0]) 
            
        return np.array(X), np.array(y)

    def train(self, df):
        if len(df) < self.sequence_length + 10:
            print("Not enough data to train.")
            return

        print("Preparing data for training...")
        X, y = self.prepare_data(df)
        
        if not HAS_TENSORFLOW:
            X = X.reshape(X.shape[0], -1)
            
        print(f"Training AI model ({'TensorFlow' if HAS_TENSORFLOW else 'RandomForest High-Res'})...")
        
        if HAS_TENSORFLOW:
            if self.model is None: self.build_model((X.shape[1], X.shape[2]))
            self.model.fit(X, y, epochs=10, batch_size=32, verbose=0) # Increased epochs
            self.model.save(self.model_path)
        else:
            print(f"DEBUG: Input Shape {X.shape}. Starting Fit...")
            # Automatically handles new shape by refitting
            self.model.fit(X, y)
            print("DEBUG: Fit Complete. Saving model...")
            try:
                joblib.dump(self.model, self.model_path)
                joblib.dump(self.scaler, self.scaler_path)
                print(f"Model saved to {self.model_path}")
            except Exception as e:
                print(f"WARNING: Could not save model (File locked?): {e}")

    def predict_batch(self, df):
        """
        Predicts for the entire dataframe in one go.
        """
        if self.model is None:
            print("Model not trained yet.")
            return None

        if 'returns' not in df.columns:
            df = df.copy()
            df['returns'] = df['close'].pct_change().fillna(0)
            
        # Add dynamic features if missing (for prediction)
        if 'ema_trend' not in df.columns and 'ema_200' in df.columns:
            df['ema_trend'] = (df['close'] - df['ema_200']) / df['ema_200']
        elif 'ema_trend' not in df.columns:
            df['ema_trend'] = 0

        if 'volatility_change' not in df.columns:
            df['volatility_change'] = df['bb_width'].pct_change().fillna(0)
            
        # Add Pattern Score (Simplified for prediction if cols missing, assume 0)
        if 'pattern_score' not in df.columns: df['pattern_score'] = 0
        if 'dist_support' not in df.columns: df['dist_support'] = 0
        if 'dist_resistance' not in df.columns: df['dist_resistance'] = 0
        
        # Ensure new advanced features exist
        for col in ['stoch_rsi_k', 'atr', 'obv_slope', 'trend_score', 'pattern_triangle', 'is_pivot_high', 'is_pivot_low']:
            if col not in df.columns: df[col] = 0

        # Cleanup AFTER features
        df.replace([np.inf, -np.inf], 0, inplace=True)
        df.fillna(0, inplace=True)

        features = self._get_feature_list(df)
        
        # Ensure features exist
        for f in features:
             if f not in df.columns: df[f] = 0

        data = df[features].values
        
        # Handle Scaling mismatch: If scaler expects X features but we have Y
        # If we just retrained, scaler is updated.
        # If we loaded old scaler, it might crash.
        try:
            scaled_data = self.scaler.transform(data)
        except ValueError as e:
             print(f"⚠️ Scaler Mismatch in Batch Prediction: {e}")
             self.scaler = MinMaxScaler(feature_range=(0, 1))
             self.model = None
             return np.array([])
        
        X = []
        if len(scaled_data) <= self.sequence_length:
            return np.array([])

        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i])
            
        X = np.array(X)
        
        print(f"   -> AI Predicting on {len(X)} samples via Batch...")
        if HAS_TENSORFLOW:
            predictions_scaled = self.model.predict(X, batch_size=1024, verbose=0)
            dummy = np.zeros((len(predictions_scaled), len(features)))
            dummy[:, 0] = predictions_scaled.flatten()
            predictions = self.scaler.inverse_transform(dummy)[:, 0]
        else:
            # Sklearn (Flatten first)
            X = X.reshape(X.shape[0], -1)
            predictions_scaled = self.model.predict(X)
            
            dummy = np.zeros((len(predictions_scaled), len(features)))
            dummy[:, 0] = predictions_scaled
            predictions = self.scaler.inverse_transform(dummy)[:, 0]
            
        return predictions

    def get_prediction(self, df):
        """
        Predicts the NEXT price return based on the last sequence of the dataframe.
        """
        if self.model is None or len(df) < self.sequence_length:
            return 0.0 # Neutral

        # Critical Fix: Check if scaler is fitted
        try:
           # Quick check if scaler is fitted by inspecting private attribute or just try/except
           from sklearn.utils.validation import check_is_fitted
           check_is_fitted(self.scaler)
        except:
            print("Scaler not fitted yet. Training needed.")
            # Auto-Train if possible?
            self.train(df)
            # Retest
            try:
                check_is_fitted(self.scaler)
            except:
                return 0.0

        # Prepare just the last sequence
        # We need the last 'sequence_length' rows
        # But prepare_data expects 'returns' column.
        
        if 'returns' not in df.columns:
            df = df.copy()
            df['returns'] = df['close'].pct_change().fillna(0)
            
        # Add dynamic features if missing (for single prediction)
        if 'ema_trend' not in df.columns and 'ema_200' in df.columns:
            df['ema_trend'] = (df['close'] - df['ema_200']) / df['ema_200']
        elif 'ema_trend' not in df.columns:
            df['ema_trend'] = 0

        if 'volatility_change' not in df.columns:
            df['volatility_change'] = df['bb_width'].pct_change().fillna(0)
            
        if 'pattern_score' not in df.columns: df['pattern_score'] = 0
        if 'dist_support' not in df.columns: df['dist_support'] = 0
        if 'dist_resistance' not in df.columns: df['dist_resistance'] = 0
        
        # Ensure new advanced features exist
        for col in ['stoch_rsi_k', 'atr', 'obv_slope']:
            if col not in df.columns: df[col] = 0
            
        # Cleanup
        df.replace([np.inf, -np.inf], 0, inplace=True)
        df.fillna(0, inplace=True)

        # Use dynamic feature list to match training
        features = self._get_feature_list(df)
        
        # Ensure features
        for f in features:
            if f not in df.columns: 
                # Ideally calculate them, but assuming DF is fully prepped
                df[f] = 0
                
        # Get last N rows
        input_df = df.iloc[-self.sequence_length:].copy()
        data = input_df[features].values
        
        # Scale
        try:
            scaled_data = self.scaler.transform(data)
        except ValueError as e:
            print(f"⚠️ Feature Mismatch detected (New features added?): {e}")
            print("🔄 Resetting Model and Scaler to force retraining...")
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            self.model = None # Force retrain
            # We can't predict now. Return 0.0 (Neutral) to trigger retrain on next loop.
            return 0.0
        
        # Reshape for model
        # Input shape: (1, sequence_length, features)
        X = np.array([scaled_data])
        
        if HAS_TENSORFLOW:
            pred_scaled = self.model.predict(X, verbose=0)
            val = pred_scaled[0][0]
        else:
            # Sklearn: Flatten
            X_flat = X.reshape(1, -1)
            pred_scaled = self.model.predict(X_flat)
            val = pred_scaled[0]
            
        # Inverse transform to get predicted RETURN (since target was scaled return?)
        # Wait, prepare_data y was scaled_data[i, 0] which corresponds to 'returns' (index 0).
        # So 'val' is the scaled return.
        
        dummy = np.zeros((1, len(features)))
        dummy[0, 0] = val
        predicted_return = self.scaler.inverse_transform(dummy)[0, 0]
        
        return predicted_return

    def validate_signal_with_deepseek(self, current_price, predicted_price, technical_summary, market_context=None, api_key=None):
        """
        Uses DeepSeek LLM to validate the trade signal based on technical context.
        """
        if not api_key:
             return {"approved": True, "reason": "Sem chave API fornecida, aprovado automaticamente."}
             
        # Construct Prompt
        signal_type = "COMPRA" if predicted_price > current_price else "VENDA"
        expected_return = (predicted_price - current_price) / current_price * 100
        
        # Prepare Context String
        ctx_str = ""
        if market_context:
            ctx_str = f"""
        CONTEXTO DE MERCADO ATUAL:
        - Regime: {market_context.get('regime', 'N/A')}
        - Tendência (EMA200): {market_context.get('trend', 'N/A')}
        - Dica Estratégia: {market_context.get('hint', 'N/A')}
            """
        
        prompt = f"""
        Você é um Trader Especialista em Scalping (Alta Frequência). 
        Analise esta oportunidade rápida de Bitcoin (1m Timeframe) considerando o contexto abaixo.
        {ctx_str}
        
        Sinal IA: {signal_type}
        Preço Atual: {current_price}
        Alvo Previsto: {predicted_price} (Retorno Esp.: {expected_return:.4f}%)
        
        Contexto Técnico:
        - RSI: {technical_summary.get('rsi', 50):.2f} (Neutro 40-60)
        - StochRSI K: {technical_summary.get('stoch_rsi_k', 0.5):.2f} (0-1, >0.8 ob, <0.2 os)
        - ATR (Volatilidade): {technical_summary.get('atr', 0):.4f} (Se alto, bom para scalping)
        - MACD: {technical_summary.get('macd', 0):.4f}
        - BB Width: {technical_summary.get('bb_width', 0):.4f}
        - Pontuação de Padrão: {technical_summary.get('pattern_score', 0):.1f}
        
        PRICE ACTION AVANÇADO:
        - Tendência: {market_context.get('trend', 'N/A')} (Score: {technical_summary.get('trend_score', 0)})
        - Padrão Triângulo/Squeeze? {'SIM' if technical_summary.get('pattern_triangle', 0) else 'NAO'}
        - Pivot High Recente? {'SIM' if technical_summary.get('is_pivot_high', 0) else 'NAO'}
        - Distância Fib 50%: {technical_summary.get('fib_500', 0):.2f}%
        
        Regras de Validação (SCALPING AGRESSIVO):
        1. Respeite o REGIME de mercado:
           - Se ALTA VOLATILIDADE: Aprove Trades mais agressivos se houver momentum.
           - Se BAIXA VOLATILIDADE: Exija padrões claros ou squeezes.
        2. Siga a TENDÊNCIA macro (EMA200) para aumentar winrate, mas permita contra-tendência CURTA (Reversão) se RSI estiver extremo (>80/<20).
        3. Focar em capturar movimentos curtos (0.1% - 0.5%).
        4. REJEITE se o contexto técnico contradizer fortemente o sinal (ex: Compra em Tendência de Baixa sem Oversold).
        
        Responda APENAS JSON:
        {{
            "approved": true/false,
            "reason": "Explicação curta focada no contexto e scalping"
        }}
        """
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "You are a conservative risk manager algorithm. Output JSON only."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 150,
            "temperature": 0.3
        }
        
        try:
            response = requests.post("https://api.deepseek.com/chat/completions", json=payload, headers=headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
                content = data['choices'][0]['message']['content'].strip()
                # Clean Markdown
                content = content.replace("```json", "").replace("```", "").strip()
                return json.loads(content)
            else:
                return {"approved": True, "reason": f"Erro API {response.status_code}, aprovado auto."}
        except Exception as e:
            return {"approved": True, "reason": f"Erro Validação: {e}, aprovado auto."}

if __name__ == "__main__":
    from data_manager import DataManager
    from technical_analysis import TechnicalAnalysis
    dm = DataManager()
    df = dm.fetch_historical_data(limit=500)
    if not df.empty:
        ta = TechnicalAnalysis(df)
        df = ta.add_all_indicators()
        df.dropna(inplace=True) 
        brain = AIBrain()
        brain.train(df)
