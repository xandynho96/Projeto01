
import json
import time
import logging
from data_manager import DataManager
import config

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TestTrader")

def run_test():
    logger.info("🔵 INICIANDO TESTE MANUAL DE TRADE (Simulação de IA)")
    
    # 1. Load User Config
    try:
        with open("user_config.json", "r") as f:
            user_settings = json.load(f)
        logger.info("✅ Configurações carregadas.")
    except Exception as e:
        logger.error(f"❌ Erro ao ler user_config.json: {e}")
        return

    # 2. Override for TEST
    margin_usd = 1.0 # Hardcoded test amount
    leverage = 5 # Changed to 5 (Kraken Spot Max is typically 5x, 10x is Futures)
    symbol = "BTC/USD"
    trading_mode = "Spot Margin (5x)"
    
    api_key = user_settings.get('api_key')
    secret = user_settings.get('secret')
    
    if not api_key or not secret:
        logger.error("❌ API Key/Secret não encontrados user_config.json")
        return

    # 3. Connect
    dm = DataManager()
    logger.info(f"🔌 Conectando à Kraken [{trading_mode}]...")
    dm.connect_exchange(api_key, secret, demo_mode=False, trading_mode=trading_mode)
    
    # 4. Fetch Price
    logger.info("📊 Obtendo preço atual...")
    ticker = dm.exchange.fetch_ticker(symbol)
    current_price = ticker['last']
    logger.info(f"💲 Preço BTC: ${current_price:.2f}")

    # 5. Calculate Position
    total_position_usd = margin_usd * leverage
    amount_btc = total_position_usd / current_price
    
    # Enforce Min
    if amount_btc < 0.0001:
        logger.warning(f"⚠️ Qtd calculada {amount_btc:.6f} < Mínimo. Ajustando para 0.0001")
        amount_btc = 0.0001
    # Calculate SL/TP
    sl_pct = 2.0
    tp_pct = 4.0
    sl_price = round(current_price * (1 - sl_pct/100), 1)
    tp_price = round(current_price * (1 + tp_pct/100), 1)

    # 6. Execute Entry with Conditional Stop Loss
    logger.info("🚀 [Simulação IA] Enviando ordem MARKET BUY + STOP LOSS...")
    
    entry_params = {
        'leverage': leverage,
        'close': {
            'ordertype': 'stop-loss',
            'price': sl_price
        }
    }
    
    try:
        # Correct Signature: symbol, type, side, amount, price, params
        order = dm.exchange.create_order(symbol, 'market', 'buy', amount_btc, None, params=entry_params)
        
        if order and 'id' in order:
            logger.info(f"✅ ORDEM DE ENTRADA EXECUTADA! ID: {order['id']}")
            logger.info(f"   (Stop Loss anexado automaticamente em {sl_price:.2f})")
            
            # 7. Place Take Profit (Separate Limit Order)
            logger.info(f"💰 Colocando Take Profit (LIMIT) em {tp_price:.2f} com Reduce Only...")
            try:
                # TP matches leverage to reduce position
                # Trying reduceOnly=True for Spot Margin
                tp_params = {'leverage': leverage, 'reduceOnly': True}
                tp_order = dm.exchange.create_order(symbol, 'limit', 'sell', amount_btc, tp_price, params=tp_params)
                logger.info(f"✅ Take Profit (Limit) definido! ID: {tp_order['id']}")
            except Exception as e:
                logger.error(f"❌ Erro Take Profit: {e}")
            
            logger.info("✨ TESTE CONCLUÍDO COM SUCESSO! Verifique a Kraken.")
        else:
            logger.error(f"❌ Falha na ordem: {order}")
            
    except Exception as e:
        logger.error(f"❌ Erro Entrada/SL: {e}")


if __name__ == "__main__":
    run_test()
