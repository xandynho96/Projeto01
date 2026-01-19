import sys
import os
import time

# Adicionar diretório raiz ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.data_manager import DataManager
from app.utils import config

def test_trade():
    print("🚀 Iniciando Teste de Integração de Trade ($1)...")
    
    # 1. Inicializar Gerente de Dados
    dm = DataManager()
    
    # Validar Chaves
    if not config.KRAKEN_API_KEY or not config.KRAKEN_SECRET:
        print("❌ ERRO: Chaves de API não encontradas em .env ou config/user_config.json")
        return

    # 2. Conectar (Forçar Modo Futuros se possível, pois $1 é permitido lá)
    # Tentar detectar modo preferido
    trading_mode = "Futures" 
    print(f"🔌 Tentando conectar em modo: {trading_mode}...")
    
    if dm.connect_exchange(config.KRAKEN_API_KEY, config.KRAKEN_SECRET, demo_mode=False, trading_mode=trading_mode):
        print("✅ Conexão estabelecida com sucesso!")
    else:
        print("❌ Falha na conexão com a Exchange.")
        return

    # 3. Verificar Saldo
    balance = dm.get_balance(type='free')
    print(f"💰 Saldo Disponível: ${balance:.2f}")
    
    if balance < 2.0:
        print("⚠️ Saldo muito baixo ( < $2). O teste pode falhar.")
    
    # 4. Preparar Trade de $1 em Futuros LINEARES (USD Margin)
    # Símbolo Unificado CCXT para Kraken Futures Linear: 'BTC/USD:USD'
    
    symbol = config.SYMBOL
    amount = 0
    price_approx = 0
    
    if "Futures" in trading_mode:
        # Forçar Linear se estivermos usando margem em USD
        symbol = "BTC/USD:USD" 
        
        # Obter Preço Atual para calcular Qtd em BTC
        try:
            ticker = dm.exchange.fetch_ticker(symbol)
            price_approx = ticker['last']
        except:
            price_approx = 100000 # Fallback seguro
            
        # Calcular Qtd para $2 (Mínimo seguro)
        target_usd = 2.0
        amount = target_usd / price_approx
        
        # Kraken Min Futures Size ~0.0001 BTC
        if amount < 0.0001:
            amount = 0.0001
            print(f"⚠️ Ajustando para tamanho mínimo: 0.0001 BTC (~${amount*price_approx:.2f})")
            
        amount = round(amount, 6) # Arredondar
    
    else:
        # Spot ou Outro
        amount = 0.0001 # Min Spot
        symbol = "BTC/USD"

    side = 'buy'
    
    print(f"\n🛒 Preparando ordem: COMPRA (LONG) de {amount} BTC em {symbol}")
    print(f"   Preço Aprox: ${price_approx:.0f} | Valor Nocional: ~${amount*price_approx:.2f}")
    if amount*price_approx > balance * 5: # Proteção alavancagem excessiva no teste
         print("❌ Erro: Valor do trade excede muito o saldo disponível.")
         return 
         
    print("   Isso usará dinheiro REAL da conta.")
    confirm = input("   Digite 'S' para confirmar e executar, ou 'N' para cancelar: ")
    
    if confirm.lower() != 's':
        print("🚫 Cancelado pelo usuário.")
        return

    # 5. Executar
    result = dm.execute_order(symbol, side, amount, type='market')
    
    if result and 'id' in result:
        print(f"\n✅ SUCESSO! Ordem executada.")
        print(f"   ID da Ordem: {result['id']}")
        print(f"   Status: {result.get('status', 'unknown')}")
        print("\nℹ️  Verifique sua conta na Kraken para confirmar a posição.")
        
        # Opcional: Fechar posição
        close = input("\nDeseja fechar essa posição agora para zerar o risco? (S/N): ")
        if close.lower() == 's':
            print("📉 Fechando posição...")
            dm.execute_order(symbol, 'sell', amount, type='market')
            print("✅ Posição fechada.")
    else:
        print("\n❌ ERRO ao executar ordem.")
        if result:
            print(f"   Detalhes: {result}")

if __name__ == "__main__":
    test_trade()
