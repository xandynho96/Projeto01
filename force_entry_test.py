import sys
import time
from data_manager import DataManager
import config

def force_test_entry():
    print("🚀 Iniciando Teste de Entrada Forçada na Kraken...")
    
    # Init DataManager
    # Note: connect_exchange is called in __init__ with config keys if present
    # But since we are using user_config.json, we load it manually here
    import json
    import os
    
    dm = DataManager()
    
    # Try loading from user_config.json
    try:
        if os.path.exists("user_config.json"):
            with open("user_config.json", "r") as f:
                cfg = json.load(f)
                api_key = cfg.get("api_key")
                secret = cfg.get("secret")
                demo_mode = cfg.get("demo_mode", True)
                if api_key and secret:
                    print("🔑 Carregando chaves de user_config.json...")
                    dm.connect_exchange(api_key, secret, demo_mode=demo_mode)
    except Exception as e:
        print(f"Erro ao ler config: {e}")

    if not dm.exchange or not dm.exchange.apiKey:
        print("❌ ERRO: Chaves de API não encontradas.")
        print("Certifique-se de que user_config.json existe e possui as chaves.")
        return

    # 1. Check Balance/Connection
    print("\n📡 Verificando Conexão e Saldo...")
    try:
        balance = dm.get_balance()
        print(f"✅ Conectado! Saldo da Conta: ${balance:.2f} (aprox.)")
    except Exception as e:
        print(f"❌ Falha ao verificar saldo: {e}")
        return

    # 2. Prepare Order
    symbol = config.SYMBOL
    side = 'buy'
    amount = 0.0001 # Minimum size for XBTUSD usually 0.0001 or 0.001 depending on contract specs
    
    print(f"\n🛒 Tentando forçar uma ordem de COMPRA (Market) de {amount} {symbol}...")
    
    # 3. Execute
    try:
        # Check market structure for min limits (optional but good practice)
        markets = dm.exchange.load_markets()
        if symbol in markets:
            min_amount = markets[symbol]['limits']['amount']['min']
            if amount < min_amount:
                print(f"⚠️ Quantidade ajustada para o mínimo da exchange: {min_amount}")
                amount = min_amount
        
        order = dm.execute_order(symbol, side, amount, type='market')
        
        if order:
            print(f"\n✅ SUCESSO! Ordem executada.")
            print(f"🆔 ID da Ordem: {order['id']}")
            print(f"📄 Status: {order['status']}")
            print("Verifique sua conta na Kraken para confirmar a posição.")
        else:
            print("\n❌ FALHA: A ordem retornou None (veja o log de erro acima).")
            
    except Exception as e:
        print(f"\n❌ ERRO CRÍTICO ao enviar ordem: {e}")

if __name__ == "__main__":
    force_test_entry()
