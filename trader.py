import time
import schedule
import pandas as pd
from datetime import datetime
from data_manager import DataManager
from technical_analysis import TechnicalAnalysis
from data_manager import DataManager
from technical_analysis import TechnicalAnalysis
from ai_brain import AIBrain
import config
from logger import setup_logger

class BitcoinTrader:
    def __init__(self, api_key=None, secret=None, user_settings={}):
        self.logger = setup_logger()
        self.logger.info("Initializing Bitcoin Trader...")
        self.dm = DataManager()
        
        # Connect if keys provided
        if api_key and secret:
            demo_mode = user_settings.get('demo_mode', False)
            trading_mode = user_settings.get('trading_mode', 'Spot Margin (10x)')
            
            # Set Symbol dynamically
            if "Futures" in trading_mode:
                config.SYMBOL = 'PF_XBTUSD'
            else:
                config.SYMBOL = 'BTC/USD'
                
            self.logger.info(f"Modo: {trading_mode} | Símbolo: {config.SYMBOL}")
            
            self.dm.connect_exchange(api_key, secret, demo_mode=demo_mode, trading_mode=trading_mode)
            
        self.user_settings = user_settings
        self.brain = AIBrain()
        self.last_training_time = datetime.min # Track last training for continuous learning
        
        # Ensure we have a trained model or train on startup
        self._initial_training()
        
        # Set Leverage if provided
        leverage = float(user_settings.get('leverage', config.LEVERAGE))
        if leverage > 1:
            self.dm.set_leverage(leverage)

    def train_on_historical_memory(self, limit=50000):
        """Trains the AI on a large dataset from the local DB, recreating HTF context."""
        self.logger.info(f"🧠 MEMÓRIA: Carregando histórico profundo ({limit} velas) do Banco de Dados...")
        
        # 1. Load Raw 1m Data from DB
        df_1m = self.dm.get_data_from_db(limit=limit)
        
        if len(df_1m) < 1000:
            self.logger.warning("Histórico insuficiente no DB para treino profundo.")
            return

        # 2. Resample for HTF (5m, 15m)
        # Ensure timestamp is index for resampling
        df_1m.set_index('timestamp', inplace=True)
        
        agg_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
        
        # 5m
        df_5m = df_1m.resample('5min').agg(agg_dict).dropna()
        # 15m
        df_15m = df_1m.resample('15min').agg(agg_dict).dropna()
        
        # Reset index to access timestamp column again
        df_1m.reset_index(inplace=True)
        df_5m.reset_index(inplace=True)
        df_15m.reset_index(inplace=True)

        # 3. Calculate Indicators on HTF
        ta_5m = TechnicalAnalysis(df_5m)
        df_5m = ta_5m.add_all_indicators()
        
        ta_15m = TechnicalAnalysis(df_15m)
        df_15m = ta_15m.add_all_indicators()
        
        # Select and Rename HTF columns
        cols_keep = ['timestamp', 'rsi', 'macd', 'bb_width', 'ema_200', 'supertrend']
        
        # Check if columns exist before copying (avoid KeyErrors if TA failed)
        existing_cols_5m = [c for c in cols_keep if c in df_5m.columns]
        df_5m = df_5m[existing_cols_5m].copy()
        df_5m.columns = ['timestamp'] + [f"{c}_5m" for c in existing_cols_5m if c != 'timestamp']
        
        existing_cols_15m = [c for c in cols_keep if c in df_15m.columns]
        df_15m = df_15m[existing_cols_15m].copy()
        df_15m.columns = ['timestamp'] + [f"{c}_15m" for c in existing_cols_15m if c != 'timestamp']

        # 4. Merge
        df_1m = pd.merge_asof(df_1m.sort_values('timestamp'), df_5m.sort_values('timestamp'), on='timestamp', direction='backward')
        df_1m = pd.merge_asof(df_1m.sort_values('timestamp'), df_15m.sort_values('timestamp'), on='timestamp', direction='backward')
        
        df_1m.fillna(method='ffill', inplace=True)
        df_1m.fillna(0, inplace=True)
        
        # 5. TA on 1m
        ta_1m = TechnicalAnalysis(df_1m)
        df_1m = ta_1m.add_all_indicators()
        df_1m.dropna(inplace=True)
        
        self.logger.info(f"🧠 MEMÓRIA: Dataset de treino montado. Shape: {df_1m.shape}")
        
        # 6. Train
        if not df_1m.empty:
            self.brain.train(df_1m)
            self.last_training_time = datetime.now()
            self.logger.info("🧠 MEMÓRIA: Cérebro atualizado com memória de longo prazo.")

    def _initial_training(self):
        self.logger.info("Realizando preparação inicial...")
        
        # 1. Try Deep Memory Training first (if DB has data)
        try:
            self.train_on_historical_memory(limit=50000)
        except Exception as e:
            self.logger.error(f"Erro no treino de memória profunda: {e}")

        # 2. Fetch fresh small batch just to ensure connectivity and latest data
        df = self.dm.fetch_multi_timeframe_data(limit=100)
        if not df.empty:
             self.dm.save_data(df)

    def sync_open_trades(self):
        """Checks if open trades in DB are closed on Exchange."""
        open_trades = self.dm.get_open_trades()
        if not open_trades:
            return

        self.logger.info(f"🔄 Sincronizando {len(open_trades)} trades abertos...")
        
        try:
            kraken_trades = self.dm.exchange.fetch_my_trades(symbol=config.SYMBOL, limit=50) 
            kraken_trades.sort(key=lambda x: x['timestamp'], reverse=True)
            
            for open_trade in open_trades:
                entry_side = open_trade['side']
                exit_side = 'sell' if entry_side == 'buy' else 'buy'
                
                match = None
                for kt in kraken_trades:
                    if kt['side'] == exit_side and kt['timestamp'] > open_trade['timestamp'].timestamp() * 1000:
                         match = kt
                         break
                
                if match:
                    exit_price = match['price']
                    exit_qty = match['amount']
                    if entry_side == 'buy':
                        raw_pnl = (exit_price - open_trade['price']) * exit_qty
                    else:
                        raw_pnl = (open_trade['price'] - exit_price) * exit_qty
                        
                    self.dm.update_trade_status(open_trade['id'], 'closed', pnl=raw_pnl)
                    self.logger.info(f"✅ Trade {open_trade['id']} ENCERRADO. PNL: ${raw_pnl:.2f}")

        except Exception as e:
            self.logger.error(f"Erro ao sincronizar trades: {e}")


    def job(self):
        self.logger.info(f"--- Iniciando Ciclo de Análise ---")
        
        # 0. Sync Status
        self.sync_open_trades()
        
        # 1. Fetch latest data (Multi-Timeframe)
        df = self.dm.fetch_multi_timeframe_data(limit=1000)
        
        if df.empty:
            self.logger.warning("Nenhum dado recebido.")
            return

        # 2. Save to DB
        self.dm.save_data(df)

        # 3. Analyze
        ta = TechnicalAnalysis(df)
        df = ta.add_all_indicators()
        df.dropna(inplace=True) 

        # --- MARKET REGIME ANALYSIS (Thinking) ---
        last_row = df.iloc[-1]
        volatility = last_row.get('bb_width', 0)
        atr_val = last_row.get('atr', 0)
        rsi_val = last_row.get('rsi', 50)
        
        regime = "NORMAL"
        strategy_hint = "Seguir IA"
        
        if volatility < 0.005: # Very low volatility (0.5%)
            regime = "BAIXA VOLATILIDADE (Squeeze)"
            strategy_hint = "Aguardar Rompimento ou Scalping Curto"
        elif volatility > 0.02:
            regime = "ALTA VOLATILIDADE"
            strategy_hint = "Scalping de Alta Volatilidade"
        
        self.logger.info(f"📊 CENÁRIO DE MERCADO:")
        self.logger.info(f"   > Regime: {regime}")
        self.logger.info(f"   > Volatilidade (ATR): {atr_val:.5f} | BB Width: {volatility:.4f}")
        self.logger.info(f"   > Estratégia Recomendada: {strategy_hint}")

        # 4. Continuous Learning (Throttled)
        # Train every 15 minutes (900 seconds)
        time_since_last_train = (datetime.now() - self.last_training_time).total_seconds()
        
        if time_since_last_train > 900:
            self.logger.info("🧠 CÉREBRO: Absorvendo novos dados de mercado (Retreinamento)...")
            try:
                # Use Deep Memory Training instead of just current DF
                self.train_on_historical_memory(limit=50000)
            except Exception as e:
                self.logger.error(f"Falha ao atualizar modelo IA: {e}")
        else:
            self.logger.info(f"🧠 CÉREBRO: Utilizando conhecimento atual (Próx. treino em {900 - time_since_last_train:.0f}s)")

        # 5. Get AI Prediction
        current_price = df['close'].iloc[-1]
        
        predicted_return = self.brain.get_prediction(df)
        predicted_price = current_price * (1 + predicted_return)
        
        if (predicted_return == 0.0 and predicted_price == current_price):
             predicted_price = None

        if predicted_price is None:
            self.logger.warning("IA incerta. Ativando Protocolo de Segurança (Fallback).")
            # Fallback Logic: Use current price as base and let Signals decide
            predicted_price = current_price 
            
            # Simple TA Strategy Fallback
            rsi = df['rsi'].iloc[-1]
            if rsi < 30:
                signal = "buy"
                self.logger.info("Estratégia Fallback: RSI Sobrevendido (<30) -> SINAL COMPRA")
            elif rsi > 70:
                signal = "sell"
                self.logger.info("Estratégia Fallback: RSI Sobrecomprado (>70) -> SINAL VENDA")
            else:
                 signal = "HOLD"
                 self.logger.info(f"Estratégia Fallback: RSI Neutro ({rsi:.2f}) -> HOLD")
                 return # Exit if HOLD
        else:
            # Normal AI Logic
            self.logger.info(f"Preço Atual: {current_price:.2f}")
            self.logger.info(f"Preço Previsto (Prox. Vela): {predicted_price:.2f}")
            self.logger.info(f"Retorno Previsto: {predicted_return*100:.4f}%")
             
            change_percent = predicted_return * 100
            self.logger.info(f"Mudança Esperada: {change_percent:.4f}%")
            
            # Spot Fees are ~0.26% Taker (Round trip ~0.52%).
            # Reverted to 0.45% safety threshold as per user request to rely on AI learning instead of forcing trades.
            # UPDATE: Adjusted to 0.05% for Expert Scalping Mode (High Frequency)
            threshold = 0.05 
            
            signal = "HOLD"
            if change_percent > threshold:
                signal = "buy"
            elif change_percent < -threshold:
                signal = "sell"
            
        self.logger.info(f"DECISÃO: {signal.upper()}")
        
        # 6. Deepseek Validation (Optional)
        if signal != "HOLD":
            technical_summary = df.tail(1).to_dict('records')[0]
            deepseek_key = self.user_settings.get('deepseek_key')
            
            if deepseek_key:
                self.logger.info("🤖 Validando com DeepSeek AI...")
                
                # Contexto de Mercado
                trend_direction = "ALTA" if current_price > last_row.get('ema_200', current_price) else "BAIXA"
                market_context = {
                    "regime": regime,
                    "hint": strategy_hint,
                    "trend": trend_direction
                }
                
                validation = self.brain.validate_signal_with_deepseek(
                    current_price, 
                    predicted_price, 
                    technical_summary, 
                    market_context=market_context,
                    api_key=deepseek_key
                )
                self.logger.info(f"DeepSeek: {validation['reason']}")
                
                if not validation.get('approved', True):
                    self.logger.warning("⛔ DeepSeek rejeitou a entrada. Cancelando trade.")
                    return
            else:
                self.logger.info("DeepSeek ignorado (Sem chave API).")

            # Use User Settings
            # 'amount' is treated as MARGIN (Collateral) now, per user expectation.
            margin_usd = float(self.user_settings.get('amount', 20.0))
            leverage = float(self.user_settings.get('leverage', 1.0))
            
            # Effective Position Size = Margin * Leverage
            total_position_usd = margin_usd * leverage
            
            # Calculate BTC amount based on Total Position Size
            amount_btc = total_position_usd / current_price
            
            # Ensure a minimum size
            # Kraken min is often 0.0001 or 0.0002 BTC.
            if amount_btc < 0.0001:
                self.logger.warning(f"Qtd calculada {amount_btc:.6f} BTC (Margin ${margin_usd} x {leverage}) < Mínimo. Ajustando para 0.0001")
                amount_btc = 0.0001
                
            self.logger.info(f"Alvo: Margin ${margin_usd} x {leverage} = ${total_position_usd:.2f} (~{amount_btc:.5f} BTC)")

            sl_pct = float(self.user_settings.get('sl_pct', 2.0))
            tp_pct = float(self.user_settings.get('tp_pct', 4.0))
            
            # Calculate SL/TP Prices
            # BUY: SL below, TP above
            # SELL: SL above, TP below
            
            # Calculate SL/TP Prices (BEFORE Entry to attach SL)
            # BUY: SL below, TP above
            # SELL: SL above, TP below
            
            if signal == "buy":
                sl_price = current_price * (1 - sl_pct/100)
                tp_price = current_price * (1 + tp_pct/100)
            else: # sell
                sl_price = current_price * (1 + sl_pct/100)
                tp_price = current_price * (1 - tp_pct/100)
            
            # CRITICAL: Round to 1 decimal for Kraken Spot BTC/USD
            sl_price = round(sl_price, 1)
            tp_price = round(tp_price, 1)

            # Execution Params
            # Use Conditional Close for Stop Loss (More robust on Kraken Spot)
            entry_params = {
                'leverage': leverage,
                'close': {
                    'ordertype': 'stop-loss',
                    'price': sl_price
                }
            }

            # Execute Entry (Market Order + Attached SL)
            self.logger.info(f"🚀 Executando entrada {signal.upper()} de {amount_btc:.5f} BTC...")
            entry_order = self.dm.execute_order(config.SYMBOL, signal, amount_btc, type='market', params=entry_params)
            
            if entry_order:
                self.logger.info(f"✅ Entrada Executada. ID: {entry_order['id']}")
                self.logger.info(f"   (Stop Loss anexado em {sl_price:.2f})")
                
                # Record trade to history
                self.dm.save_trade(
                    symbol=config.SYMBOL, 
                    side=signal, 
                    amount=amount_btc, 
                    price=current_price, 
                    status='open'
                )
                
                # Place Take Profit (Separate Limit Order)
                # TP matches leverage to reduce position
                tp_side = "sell" if signal == "buy" else "buy"
                # Use reduceOnly to ensure it closes position
                tp_params = {'leverage': leverage, 'reduceOnly': True}
                
                self.logger.info(f"💰 Colocando Take Profit (Limit) em {tp_price:.2f}...")
                self.dm.execute_order(config.SYMBOL, tp_side, amount_btc, type='limit', price=tp_price, params=tp_params)
                
            else:
                self.logger.error("❌ Ordem de entrada falhou.")
        
    def run(self):
        self.logger.info("Bot rodando. Pressione Ctrl+C para parar.")
        
        # Run once immediately
        self.job()
        
        # Schedule every minute (Scalping)
        schedule.every(1).minutes.do(self.job)
        
        while True:
            schedule.run_pending()
            time.sleep(1)

if __name__ == "__main__":
    bot = BitcoinTrader()
    bot.run()
