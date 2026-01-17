import time
import schedule
import pandas as pd
import threading
import json
from datetime import datetime
from app.core.data_manager import DataManager
from app.core.technical_analysis import TechnicalAnalysis
from app.core.ai_brain import AIBrain
from app.core.evolution import evolution_worker
from app.utils import config
from app.utils.logger import setup_logger

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
        # MOVED to explicit call to prevent GUI blocking
        # self._initial_training()
        
        # Set Leverage if provided
        leverage = float(user_settings.get('leverage', config.LEVERAGE))
        if leverage > 1:
            self.dm.set_leverage(leverage)

    def train_on_historical_memory(self, limit=50000):
        """Trains the AI on a dataset from the local DB. Reduced to 50k for stability."""
        self.logger.info(f"🧠 MEMÓRIA: Carregando histórico ({limit} velas) do Banco de Dados...")
        
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
                 self.logger.info("🧠 CÉREBRO: Modelo novo ou não treinado. Iniciando carga de memória...")
                 self.train_on_historical_memory(limit=5000)
             
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
        # self.logger.info(f"--- Iniciando Ciclo de Análise ---") # Reduced Verbosity

        
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
                 self.user_settings['sl_pct'] = params.get('sl_pct', 2.0)
                 self.user_settings['tp_pct'] = params.get('tp_pct', 4.0)

                 # Only log if strategy changed
                 current_strat_id = best_strat.get('id')
                 if not hasattr(self, 'last_strat_id') or self.last_strat_id != current_strat_id:
                     self.logger.info(f"🧬 Estratégia Evoluída Aplicada ({best_strat['origin']}): WR {best_strat['winrate']:.1f}%")
                     self.logger.info(f"   📐 SL: {self.user_settings['sl_pct']}% | TP: {self.user_settings['tp_pct']}%")
                     self.last_strat_id = current_strat_id

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
        
        # Only log if regime changed or every 5 minutes
        if not hasattr(self, 'last_regime') or self.last_regime != regime or (datetime.now().minute % 5 == 0 and datetime.now().second < 5):
             self.logger.info(f"📊 CENÁRIO DE MERCADO: {regime}")
             self.logger.info(f"   > Volatilidade (ATR): {atr_val:.5f} | BB Width: {volatility:.4f}")
             self.logger.info(f"   > Estratégia Recomendada: {strategy_hint}")
             self.last_regime = regime


        # 4. Continuous Learning (Throttled)
        # Train every 15 minutes (900 seconds)
        time_since_last_train = (datetime.now() - self.last_training_time).total_seconds()
        
        if time_since_last_train > 900:
            self.logger.info("🧠 CÉREBRO: Absorvendo novos dados de mercado (Retreinamento)...")
            try:
                # Use Deep Memory Training instead of just current DF
                self.train_on_historical_memory(limit=5000)
            except Exception as e:
                self.logger.error(f"Falha ao atualizar modelo IA: {e}")
        else:
            # self.logger.info(f"🧠 CÉREBRO: Utilizando conhecimento atual (Próx. treino em {900 - time_since_last_train:.0f}s)")
            pass


        # 5. Get AI Prediction
        current_price = df['close'].iloc[-1]
        
        predicted_return = self.brain.get_prediction(df)
        predicted_price = current_price * (1 + predicted_return)
        
        if (predicted_return == 0.0 and predicted_price == current_price):
             predicted_price = None

        if predicted_price is None:
            # self.logger.warning("IA incerta. Ativando Protocolo de Segurança (Fallback).")
            # Only log fallback usage if not logged recently or if a trade is imminent?
            # Actually, let's silence it unless we are entering a trade logic log
            pass

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
            
            signal = "NEUTRO"
            strategy_name = "NEUTRO" # Initialize default to avoid unbound error

            # BUY CONDITION: RSI Low AND Stoch Low AND Price near/below BB Low

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
            
            # self.logger.info(f"Estratégia Técnica ({strategy_name}): RSI={rsi:.1f}, Stoch={stoch_k:.1f}")

            
            if signal == "NEUTRO":
                 # self.logger.info("⌛ Análise: NEUTRO (Aguardando setup ideal...)") 
                 # User wants to see activity, let's enable it but keep it clean
                 if datetime.now().second < 5: # Log only once per minute approximately
                     self.logger.info("⌛ Análise: NEUTRO (IA e Téc. aguardando oportunidade...)")
                 return

        else:
            # Normal AI Logic
            # Calculate Indicators commonly used for logging/logic
            row = df.iloc[-1]
            rsi = row.get('rsi', 50)
            stoch_k = row.get('stoch_k', 50)
            stoch_rsi_k = row.get('stoch_rsi_k', 0.5)
            bb_low = row['bb_low']
            bb_high = row['bb_high']
            close_price = row['close']

            change_percent = predicted_return * 100
            
            # Spot Fees are ~0.26% Taker.
            # LOWERING threshold to 0.02% to be more aggressive availability for scalping
            threshold = 0.02 
            
            signal = "NEUTRO"
            strategy_name = "NEUTRO" # Default

            if change_percent > threshold:
                signal = "buy"
            elif change_percent < -threshold:
                signal = "sell"
            
            # --- HYBRID FALLBACK: If AI says NEUTRO, check Technicals ---
            if signal == "NEUTRO":
                 # Use current price as base
                 # --- ADVANCED SCALPING STRATEGY (Confluence) ---
                # (Variables row, rsi, stoch_k, etc. are already defined above)
                
                # Thresholds (Aggressive Scalping)
                RSI_BUY = 40  # Slightly relaxed
                RSI_SELL = 60
                STOCH_BUY = 25
                STOCH_SELL = 75
                
                strategy_name = "NEUTRO"

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
                
            if signal != "NEUTRO":
                     # Ensure strategy_name exists if we got here via Fallback 
                     if 'strategy_name' not in locals(): strategy_name = "Hybrid/Unknown"
                     # Logic to only log Hybrid if it WAS a hybrid decision
                     if strategy_name != "NEUTRO":
                        self.logger.info(f"🔄 HÍBRIDO: AI 'Neutro' anulado por {strategy_name}. (RSI={rsi:.1f}, Stoch={stoch_k:.1f})")
            
        self.logger.info(f"DECISÃO: {signal.upper() if signal != 'NEUTRO' else 'AGUARDANDO OPORTUNIDADE (SCALPING)'}")
        
        # 6. Deepseek Validation (Optional)
        if signal != "NEUTRO":
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
                # FUTURES: 
                # Check for Linear (BTC/USD:USD - Amount in BTC) or Inverse (BTC/USD - Amount in USD contracts)
                if ':USD' in config.SYMBOL:
                    # LINEAR FUTURE
                    amount_to_trade = total_position_usd / current_price
                    # Min size approx 0.0001 BTC usually
                    if amount_to_trade < 0.0001:
                         self.logger.warning(f"⚠️ Qtd {amount_to_trade:.6f} < Mínimo. Ajustando para 0.0001")
                         amount_to_trade = 0.0001
                    self.logger.info(f"🔮 MODO FUTUROS (LINEAR): Trade de {amount_to_trade:.6f} BTC (Pos ${total_position_usd:.2f})")
                else:
                    # INVERSE FUTURE (Legacy/Standard)
                    # Amount in USD Contracts (Integers)
                    amount_to_trade = int(total_position_usd)
                    if amount_to_trade < 1:
                        self.logger.warning(f"⚠️ Margem muito baixa para Futuros. Mínimo 1 Contrato ($1). Ajustando...")
                        amount_to_trade = 1
                    
                    self.logger.info(f"🔮 MODO FUTUROS (INVERSO): Trade de {amount_to_trade} Contratos (USD) [Alavancagem {leverage}x]")
                
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

            # --- STRATEGY: DYNAMIC SL/TP (ATR BASED) ---
            # Replaces User Fixed % with Market Volatility Logic
            
            # 1. Get Current ATR (Volatility)
            atr_value = df.iloc[-1]['atr']
            current_volatility_pct = atr_value / current_price
            
            # 2. Dynamic Targets
            # SL = 1.5x ATR (To avoid noise)
            # TP = 3.0x ATR (To catch trends, Risk:Reward 1:2)
            
            sl_move_pct = current_volatility_pct * 1.5
            tp_move_pct = current_volatility_pct * 3.0
            
            # 3. Fee Protection Guard
            # Estimate Costs: Taker (0.05%) + Maker (0.02%) + Slippage (~0.05%) = ~0.12% round trip
            # Kraken Futures Fees vary, taking conservative estimate 0.2% total
            ESTIMATED_FEES_PCT = 0.002
            
            if tp_move_pct < ESTIMATED_FEES_PCT:
                self.logger.warning(f"⛔ Volatilidade muito baixa (TP {tp_move_pct*100:.3f}% < Taxas {ESTIMATED_FEES_PCT*100}%). Trade cancelado.")
                return

            self.logger.info(f"🎯 Alvo Dinâmico (ATR): Volatilidade {current_volatility_pct*100:.3f}%")
            self.logger.info(f"   > TP: {tp_move_pct*100:.2f}% ({tp_move_pct/current_volatility_pct:.1f}x ATR)")
            self.logger.info(f"   > SL: {sl_move_pct*100:.2f}% ({sl_move_pct/current_volatility_pct:.1f}x ATR)")

            if signal == "buy":
                tp_price = current_price * (1 + tp_move_pct)
                sl_price = current_price * (1 - sl_move_pct)
            else: # sell
                tp_price = current_price * (1 - tp_move_pct)
                sl_price = current_price * (1 + sl_move_pct)

            # Rounding
            sl_price = round(sl_price, 1)
            tp_price = round(tp_price, 1)
            
            # Legacy Profit Guard Removed (Handled by Fee Protection above)
            # tp_price is already set dynamically by ATR.

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
