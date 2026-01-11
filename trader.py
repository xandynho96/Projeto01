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
            self.dm.connect_exchange(api_key, secret, demo_mode=demo_mode)
            
        self.user_settings = user_settings
        self.brain = AIBrain()
        # Ensure we have a trained model or train on startup
        self._initial_training()
        
        # Set Leverage if provided
        leverage = float(user_settings.get('leverage', config.LEVERAGE))
        if leverage > 1:
            self.dm.set_leverage(leverage)

    def _initial_training(self):
        self.logger.info("Realizando busca inicial de dados e treinamento...")
        # Prefer loading from DB if we have history
        df = self.dm.get_data_from_db(limit=2000) # Quick check
        
        if df.empty or len(df) < 500:
             # Fallback to fetch if DB empty
            df = self.dm.fetch_historical_data(limit=1000)
            
        if not df.empty:
            self.dm.save_data(df)
            ta = TechnicalAnalysis(df)
            df = ta.add_all_indicators()
            df.dropna(inplace=True)
            # Only train if model doesn't exist? Or re-train?
            # Enforce training on startup to ensure Scaler is fitted
            self.logger.info("Treinando IA com dados iniciais...")
            self.brain.train(df) 
        else:
            self.logger.error("Falha ao buscar dados iniciais.")

    def sync_open_trades(self):
        """Checks if open trades in DB are closed on Exchange."""
        open_trades = self.dm.get_open_trades()
        if not open_trades:
            return

        self.logger.info(f"🔄 Sincronizando {len(open_trades)} trades abertos...")
        
        # Fetch recent trades from Kraken
        # We need enough history to cover the open trades
        try:
            # fetch_my_trades usually returns list of dicts
            kraken_trades = self.dm.exchange.fetch_my_trades(symbol=config.SYMBOL, limit=50) 
            # Sort by time desc
            kraken_trades.sort(key=lambda x: x['timestamp'], reverse=True)
            
            for open_trade in open_trades:
                # Logic: Look for a trade with OPPOSITE side that happened AFTER open_trade timestamp
                # taking into account some buffer?
                # Simple logic: If we are LONG, look for SELL.
                
                entry_side = open_trade['side']
                exit_side = 'sell' if entry_side == 'buy' else 'buy'
                
                # Filter kraken trades for potential exits
                # potential_exits = [t for t in kraken_trades if t['side'] == exit_side and t['timestamp'] > open_trade['timestamp']]
                
                # Check for Filled Orders that match
                # Often easier to check 'closed' orders if we have order ID, but we didn't save it initially.
                # Heuristic: If we find a generic 'sell' of the same amount? Or just ANY exit?
                # For this MVP, let's assume any trade with opposite side after entry is an exit relative to our position.
                # Ideally, we should track Position size.
                
                # Let's try to match by approximation logic or check position.
                
                # Better approach for Futures:
                # Check current position size from Exchange.
                # If position is 0 and we have open trades, all are closed.
                # But we might have partials.
                
                # Use fetch_positions for Futures
                # self.dm.exchange.fetch_positions()
                
                # Simpler Fallback: Match recent trades
                match = None
                for kt in kraken_trades:
                    if kt['side'] == exit_side and kt['timestamp'] > open_trade['timestamp'].timestamp() * 1000:
                         match = kt
                         break
                
                if match:
                    # Found an exit!
                    exit_price = match['price']
                    exit_qty = match['amount']
                    
                    # Calculate PNL
                    # Long: (Exit - Entry) * Qty
                    # Short: (Entry - Exit) * Qty
                    if entry_side == 'buy':
                        raw_pnl = (exit_price - open_trade['price']) * exit_qty
                    else:
                        raw_pnl = (open_trade['price'] - exit_price) * exit_qty
                        
                    self.dm.update_trade_status(open_trade['id'], 'closed', pnl=raw_pnl)
                    self.logger.info(f"✅ Trade {open_trade['id']} ENCERRADO. PNL: ${raw_pnl:.2f}")

        except Exception as e:
            self.logger.error(f"Erro ao sincronizar trades: {e}")


    def job(self):
        self.logger.info(f"--- Iniciando Análise ---")
        
        # 0. Sync Status
        self.sync_open_trades()
        
        # 1. Fetch latest data
        # Fetch slightly more than needed to calculate indicators correctly
        df = self.dm.fetch_historical_data(limit=1000)
        
        if df.empty:
            self.logger.warning("Nenhum dado recebido.")
            return

        # 2. Save to DB
        self.dm.save_data(df)

        # 3. Analyze
        ta = TechnicalAnalysis(df)
        df = ta.add_all_indicators()
        df.dropna(inplace=True) # Important for LSTM

        # 4. Continuous Learning (Retrain on latest data)
        self.logger.info("♻️ Atualizando Modelo IA com dados recentes...")
        try:
            self.brain.train(df)
        except Exception as e:
            self.logger.error(f"Falha ao atualizar modelo IA: {e}")

        # 5. Get AI Prediction
        current_price = df['close'].iloc[-1]
        
        # New API returns 'predicted_return' (e.g. 0.005 for 0.5% up)
        predicted_return = self.brain.get_prediction(df)
        
        # Calculate Target Price from Return
        predicted_price = current_price * (1 + predicted_return)
        
        if (predicted_return == 0.0 and predicted_price == current_price):
             # Try Fallback if return is exactly 0 (might mean no prediction made)
             predicted_price = None

        if predicted_price is None:
            self.logger.warning("IA não conseguiu prever. Usando Fallback de Análise Técnica.")
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
             
             # Threshold: if predicted return is > 0.1% (High Frequency Scalping)
             # User requested HFT Scalping so thresholds should be small.
             # 0.5% was for 1h Timeframe. For 1m, 0.5% is huge. Lowering to 0.05% or 0.1%?
             # Let's use 0.05% (0.0005) filter
             
             change_percent = predicted_return * 100
             self.logger.info(f"Mudança Esperada: {change_percent:.4f}%")
             
             # Spot Fees are ~0.26% Taker (Round trip ~0.52%).
             # We need a threshold > 0.5% to break even on Fees alone.
             threshold = 0.45 # Aggressive but realistic attempt (assuming some Maker or price improvement)
             
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
                validation = self.brain.validate_signal_with_deepseek(current_price, predicted_price, technical_summary, api_key=deepseek_key)
                self.logger.info(f"DeepSeek: {validation['reason']}")
                
                if not validation.get('approved', True):
                    self.logger.warning("⛔ DeepSeek rejeitou a entrada. Cancelando trade.")
                    return
            else:
                self.logger.info("DeepSeek ignorado (Sem chave API).")

            # Use User Settings
            # 'amount' is now treated as USD value (e.g., 50.0)
            amount_usd = float(self.user_settings.get('amount', 50.0))
            
            # Calculate BTC amount based on current price
            amount_btc = amount_usd / current_price
            # Ensure a minimum size (Kraken Spot min is usually around $10, e.g. 0.0001 BTC at 100k, 0.0002 at 50k)
            # Safe margin: 0.0002
            if amount_btc < 0.0002:
                self.logger.warning(f"Quantidade calculada {amount_btc:.6f} BTC ($ {amount_usd}) muito pequena. Ajustando para 0.0002")
                amount_btc = 0.0002
                
            self.logger.info(f"Alvo Entrada: ${amount_usd} (~{amount_btc:.5f} BTC)")

            sl_pct = float(self.user_settings.get('sl_pct', 2.0))
            tp_pct = float(self.user_settings.get('tp_pct', 4.0))
            
            # Calculate SL/TP Prices
            # BUY: SL below, TP above
            # SELL: SL above, TP below
            
            if signal == "buy":
                sl_price = current_price * (1 - sl_pct/100)
                tp_price = current_price * (1 + tp_pct/100)
            else: # sell
                sl_price = current_price * (1 + sl_pct/100)
                tp_price = current_price * (1 - tp_pct/100)
                
            # Execute Entry (Market Order)
            self.logger.info(f"🚀 Executando entrada {signal.upper()} de {amount_btc:.5f} BTC...")
            entry_order = self.dm.execute_order(config.SYMBOL, signal, amount_btc, type='market')
            
            if entry_order:
                self.logger.info(f"✅ Entrada Executada. ID: {entry_order['id']}")
                
                # Record trade to history
                self.dm.save_trade(
                    symbol=config.SYMBOL, 
                    side=signal, 
                    amount=amount_btc, 
                    price=current_price, 
                    status='open'
                )
                
                # --- Place Exit Orders (Reduce Only) ---
                # Stop Loss
                sl_side = "sell" if signal == "buy" else "buy"
                sl_params = {'reduceOnly': True}
                self.logger.info(f"🛡️ Colocando Stop Loss em {sl_price:.2f}...")
                self.dm.execute_order(config.SYMBOL, sl_side, amount_btc, type='stop', price=sl_price, params=sl_params)
                
                # Take Profit
                tp_side = "sell" if signal == "buy" else "buy"
                tp_params = {'reduceOnly': True}
                self.logger.info(f"💰 Colocando Take Profit em {tp_price:.2f}...")
                self.dm.execute_order(config.SYMBOL, tp_side, amount_btc, type='take_profit', price=tp_price, params=tp_params)
                
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
