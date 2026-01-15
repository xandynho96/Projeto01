import time
import schedule
import pandas as pd
import threading
import json
from datetime import datetime
from data_manager import DataManager
from technical_analysis import TechnicalAnalysis
from ai_brain import AIBrain
from evolution import evolution_worker
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
                config.SYMBOL = 'BTC/USD:USD' # Linear Perps (USD Margin)
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

    def train_on_historical_memory(self, limit=260000):
        """Trains the AI on a large dataset from the local DB, recreating HTF context."""
        # 260,000 minutes ~= 6 months
        self.logger.info(f"🧠 MEMÓRIA: Carregando histórico profundo ({limit} velas) do Banco de Dados. Isso pode levar alguns segundos...")
        
        # 1. Load Raw 1m Data from DB
        df_1m = self.dm.get_data_from_db(limit=limit)
        
        # Auto-Backfill if insufficient data
        if len(df_1m) < (limit * 0.5): # If we have less than 50% of desired history
             self.logger.warning(f"⚠️ Histórico local insuficiente ({len(df_1m)} vs Alvo {limit}). Iniciando Backfill de 6 meses...")
             self.dm.fetch_deep_history(months=6)
             # Reload after backfill
             df_1m = self.dm.get_data_from_db(limit=limit)
        
        if len(df_1m) < 1000:
            self.logger.warning("Histórico insuficiente no DB para treino profundo, mesmo após tentativa de backfill.")
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
        
        df_1m.ffill(inplace=True)
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
        
        # Check DB size/Deep Memory availability
        try:
             # Heuristic: Check recent data count. If low, assume new setup and trigger backfill.
             # Fetch last 100 rows
             recent_data = self.dm.get_data_from_db(limit=100)
             
             if len(recent_data) < 50:
                 self.logger.info("⚠️ Banco de dados parece vazio/incompleto. Iniciando Backfill de 6 meses...")
                 self.dm.fetch_deep_history(months=6)
             else:
                 self.logger.info("Banco de dados existente detectado. Pulando backfill massivo (use script de reset se necessário).")
                 
             # INITIAL OPTIMIZATION: Check if model is already trained/loaded
             if self.brain.is_model_trained():
                 self.logger.info("🧠 CÉREBRO: Modelo validado e carregado do disco. Pulando treinamento inicial pesado.")
                 # Still update "last_training_time" so it triggers update in 15 mins
                 self.last_training_time = datetime.now()
             else:
                 self.logger.info("🧠 CÉREBRO: Modelo novo ou não treinado. Iniciando carga de memória profunda...")
                 self.train_on_historical_memory(limit=260000)
             
        except Exception as e:
            self.logger.error(f"Erro na preparação inicial: {e}")

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

        # 0.1 Check for Evolved Strategies
        try:
             best_strat = self.dm.get_best_active_strategy()
             if best_strat:
                 # Parse params
                 params = json.loads(best_strat['genes'])
                 
                 # Update internal settings if changed
                 # We simply update self.user_settings override
                 self.user_settings['sl_pct'] = params.get('sl_pct', 2.0)
                 self.user_settings['tp_pct'] = params.get('tp_pct', 4.0)
                 # self.user_settings['rsi_buy'] = params.get('rsi_buy', 30) # Used in fallback logic
                 
                 self.logger.info(f"🧬 Estratégia Evoluída Aplicada ({best_strat['origin']}): WR {best_strat['winrate']:.1f}%")
                 self.logger.info(f"   📐 SL: {self.user_settings['sl_pct']}% | TP: {self.user_settings['tp_pct']}%")
        except Exception as e:
            self.logger.error(f"Erro ao carregar estratégia evoluída: {e}")

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
            
            # --- ADVANCED SCALPING STRATEGY (Confluence) ---
            # Requires: RSI + StochRSI + BB to agree.
            
            row = df.iloc[-1]
            rsi = row.get('rsi', 50)
            stoch_k = row.get('stoch_k', 50)
            stoch_rsi_k = row.get('stoch_rsi_k', 0.5)
            close_price = row['close']
            bb_low = row['bb_low']
            bb_high = row['bb_high']
            
            # Thresholds
            RSI_BUY = 35
            RSI_SELL = 65
            STOCH_BUY = 20
            STOCH_SELL = 80
            
            signal = "HOLD"
            strategy_name = "HOLD"

            # BUY CONDITION: RSI Low AND Stoch Low AND Price near/below BB Low
            if rsi < RSI_BUY and stoch_k < STOCH_BUY:
                if close_price <= bb_low * 1.002: # Within 0.2% of BB Low
                    signal = "buy"
                    strategy_name = "Scalper Confluence (Oversold)"
            
            # SELL CONDITION: RSI High AND Stoch High AND Price near/above BB High
            elif rsi > RSI_SELL and stoch_k > STOCH_SELL:
                if close_price >= bb_high * 0.998: # Within 0.2% of BB High
                    signal = "sell"
                    strategy_name = "Scalper Confluence (Overbought)"
            
            self.logger.info(f"Estratégia Técnica ({strategy_name}): RSI={rsi:.1f}, Stoch={stoch_k:.1f}")
            
            if signal == "HOLD":
                 return # Exit if HOLD
        else:
            # Normal AI Logic
            self.logger.info(f"Preço Atual: {current_price:.2f}")
            self.logger.info(f"Preço Previsto (Prox. Vela): {predicted_price:.2f}")
            self.logger.info(f"Retorno Previsto: {predicted_return*100:.4f}%")
             
            change_percent = predicted_return * 100
            self.logger.info(f"Mudança Esperada: {change_percent:.4f}%")
            
            # Spot Fees are ~0.26% Taker.
            # LOWERING threshold to 0.02% to be more aggressive availability for scalping
            threshold = 0.02 
            
            signal = "HOLD"
            if change_percent > threshold:
                signal = "buy"
            elif change_percent < -threshold:
                signal = "sell"
            
            # --- HYBRID FALLBACK: If AI says HOLD, check Technicals ---
            if signal == "HOLD":
                 # Use current price as base
                 # --- ADVANCED SCALPING STRATEGY (Confluence) ---
                row = df.iloc[-1]
                close_price = row['close'] # Define close_price
                rsi = row.get('rsi', 50)
                stoch_k = row.get('stoch_k', 50)
                bb_low = row.get('bb_low', 0)
                bb_high = row.get('bb_high', 0)
                
                # Thresholds (Aggressive Scalping)
                RSI_BUY = 40  # Slightly relaxed
                RSI_SELL = 60
                STOCH_BUY = 25
                STOCH_SELL = 75
                
                strategy_name = "HOLD"

                # BUY CONDITION: RSI Low AND Stoch Low
                if rsi < RSI_BUY and stoch_k < STOCH_BUY:
                     # Check BB proximity (optional, relaxed)
                     if close_price <= bb_low * 1.005: 
                        signal = "buy"
                        strategy_name = "Hybrid Scalper (Oversold)"
                
                # SELL CONDITION: RSI High AND Stoch High
                elif rsi > RSI_SELL and stoch_k > STOCH_SELL:
                    if close_price >= bb_high * 0.995:
                        signal = "sell"
                        strategy_name = "Hybrid Scalper (Overbought)"
                
                if signal != "HOLD":
                     self.logger.info(f"🔄 HÍBRIDO: AI 'Hold' anulado por {strategy_name}. (RSI={rsi:.1f}, Stoch={stoch_k:.1f})")
            
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
            margin_usd = float(self.user_settings.get('amount', 5.0))
            
            # --- SAFETY CHECK: BALANCE ---
            free_balance = self.dm.get_balance(type='free')
            self.logger.info(f"Saldo Disponível: ${free_balance:.2f} | Margem Requerida: ${margin_usd:.2f}")
            
            if free_balance < margin_usd:
                self.logger.warning(f"⛔ Saldo insuficiente para abrir trade. (Livre: ${free_balance:.2f} < Req: ${margin_usd:.2f})")
                return

            leverage = int(float(self.user_settings.get('leverage', 1.0)))
            
            # Effective Position Size = Margin * Leverage
            total_position_usd = margin_usd * leverage
            
            # --- AMOUNT CALCULATION (SPOT VS FUTURES) ---
            # Kraken Futures (PF_XBTUSD) uses Inverse Contracts where 1 Contract = 1 USD (mostly)
            # Spot uses Base Currency (BTC)
            
            if "Futures" in self.user_settings.get('trading_mode', ''):
                # FUTURES: Amount in USD Contracts (Integers)
                # Ensure minimum 1 contract
                amount_to_trade = int(total_position_usd)
                if amount_to_trade < 1:
                    self.logger.warning(f"⚠️ Margem muito baixa para Futuros. Mínimo 1 Contrato ($1). Ajustando...")
                    amount_to_trade = 1
                    
                self.logger.info(f"🔮 MODO FUTUROS: Trade de {amount_to_trade} Contratos (USD) [Alavancagem {leverage}x]")
                
                # Check actual open positions
                positions = self.dm.get_open_positions()
                if positions:
                    self.logger.info(f"📊 Posições Abertas no Exchange: {len(positions)}")
                    for p in positions:
                        self.logger.info(f"   > {p['symbol']}: {p['contracts']} contratos | PNL: {p['unrealizedPnl']}")
            else:
                # SPOT: Amount in BTC
                amount_to_trade = total_position_usd / current_price
                
                # Ensure a minimum size (Kraken Min ~0.0001-0.0002 BTC)
                if amount_to_trade < 0.0001:
                    self.logger.warning(f"⚠️ Qtd {amount_to_trade:.6f} BTC < Mínimo Kraken. Ajustando para 0.0001")
                    amount_to_trade = 0.0001
                
                self.logger.info(f"🛒 MODO SPOT: Trade de {amount_to_trade:.6f} BTC (Position ${total_position_usd:.2f})")

            # --- STRATEGY: HIGH ROE (30-60%) ---
            # User Request: "Always seek 30-60% of entry value as profit"
            # --- TP / SL CALCULATION ---
            # Priority: User Settings (Price Move %) > Random High ROE
            
            # Check if User Settings provided specific moves
            # The GUI now converts ROE Input -> Price Move % before sending here
            user_tp_move = float(self.user_settings.get('tp_pct', 0.0))
            user_sl_move = float(self.user_settings.get('sl_pct', 0.0))
            
            if user_tp_move > 0 and user_sl_move > 0:
                # Use User Defined
                tp_move_pct = user_tp_move
                sl_move_pct = user_sl_move
                self.logger.info(f"🎯 Alvo Definido pelo Usuário: TP {tp_move_pct*100:.2f}% Mov | SL {sl_move_pct*100:.2f}% Mov")
            else:
                # Fallback to Random High ROE Strategy
                import random
                target_roe = random.uniform(0.30, 0.60) # Target 30% to 60%
                
                # Risk/Reward Ratio 1:2
                risk_roe = target_roe / 2.0 
                
                tp_move_pct = target_roe / leverage
                sl_move_pct = risk_roe / leverage
                
                self.logger.info(f"🎯 Alvo ROE (Auto): {target_roe*100:.1f}% (Lev {leverage}x -> Move {tp_move_pct*100:.2f}%)")
            
            if signal == "buy":
                tp_price = current_price * (1 + tp_move_pct)
                sl_price = current_price * (1 - sl_move_pct)
            else: # sell
                tp_price = current_price * (1 - tp_move_pct)
                sl_price = current_price * (1 + sl_move_pct)

            # Rounding
            sl_price = round(sl_price, 1)
            tp_price = round(tp_price, 1)

            # --- PROFIT GUARD (Redundant now but good for safety) ---
            # Ensure TP covers Fees (~0.26% Taker Entrance + ~0.16% Maker Exit + Slippage)
            MIN_PROFIT_PCT = 0.0085 # 0.85%
            
            if signal == "buy":
                min_tp = current_price * (1 + MIN_PROFIT_PCT)
                if tp_price < min_tp:
                    # Should rarely happen with High ROE, but safety first
                    tp_price = min_tp
            elif signal == "sell":
                min_tp = current_price * (1 - MIN_PROFIT_PCT)
                if tp_price > min_tp:
                    tp_price = min_tp
                    
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
            # Execute Entry (Market Order + Attached SL)
            self.logger.info(f"🚀 Executando entrada {signal.upper()} de {amount_to_trade}...")
            entry_order = self.dm.execute_order(config.SYMBOL, signal, amount_to_trade, type='market', params=entry_params)
            
            if entry_order and 'id' in entry_order:
                self.logger.info(f"✅ Entrada a Mercado Executada. ID: {entry_order['id']}")
                self.logger.info(f"   (Ordem Condicional de Stop Loss criada em {sl_price:.2f} - Ver aba 'Condicionais' na Kraken)")
                
                # Record trade to history
                self.dm.save_trade(
                    symbol=config.SYMBOL, 
                    side=signal, 
                    amount=amount_to_trade, 
                    price=current_price, 
                    status='open'
                )
                
                # Place Take Profit (Separate Limit Order)
                # TP matches leverage to reduce position
                tp_side = "sell" if signal == "buy" else "buy"
                # Use reduceOnly to ensure it closes position
                tp_params = {'leverage': leverage, 'reduceOnly': True}
                
                self.logger.info(f"💰 Colocando Take Profit (Limit) em {tp_price:.2f}...")
                self.dm.execute_order(config.SYMBOL, tp_side, amount_to_trade, type='limit', price=tp_price, params=tp_params)
                
            else:
                error_msg = entry_order.get('error', 'Unknown Error') if entry_order else 'No Response'
                self.logger.error(f"❌ Ordem de entrada falhou. Motivo: {error_msg}")
        
    def run(self):
        self.logger.info("Bot rodando. Pressione Ctrl+C para parar.")
        
        # --- Start Evolution Lab in Background Thread ---
        try:
            self.logger.info("🧪 Iniciando Laboratório de Estratégias (Background)...")
            evo_thread = threading.Thread(target=evolution_worker, daemon=True)
            evo_thread.start()
        except Exception as e:
            self.logger.error(f"Falha ao iniciar Laboratório de Evolução: {e}")

        # Run once immediately
        self.job()
        
        # Schedule every minute (Scalping)
        schedule.every(1).minutes.do(self.job)
        
        while True:
            schedule.run_pending()
            
            # Periodically check for evolved strategies (every loop approx 1s)
            # But we don't want to hammer DB. Let's do it inside job() or here with timer.
            # actually better inside job() so it's aligned with trade decisions.
            
            time.sleep(1)

if __name__ == "__main__":
    bot = BitcoinTrader()
    bot.run()
