import json
import os
from data_manager import DataManager
import config

def run_test():
    print("🚀 INICIANDO TESTE PRE-FLIGHT (MERCADO FUTUROS) 🚀")
    print("-" * 50)
    
    # 1. Load Config
    config_file = "user_config_roe.json"
    if not os.path.exists(config_file):
        print(f"⚠️  {config_file} não encontrado. Tentando 'user_config.json' antigo para teste...")
        config_file = "user_config.json"
        
    if not os.path.exists(config_file):
        print(f"❌ Erro: Nenhum arquivo de configuração encontrado. Salve as configs na GUI primeiro.")
        return

    try:
        with open(config_file, "r") as f:
            data = json.load(f)
            api_key = data.get("api_key")
            secret = data.get("secret")
            mode = data.get("trading_mode", "Futures (50x)")
            
        if not api_key or not secret:
            print("❌ Erro: API Key/Secret não encontrados no arquivo.")
            return

        print(f"✅ Configuração Carregada. Modo: {mode}")
        
        # 2. Connect
        dm = DataManager()
        print("🔌 Conectando à Kraken Futures...")
        # Force Futures connection for this test
        dm.connect_exchange(api_key, secret, demo_mode=False, trading_mode="Futures (50x)")
        
        if not dm.exchange:
            print("❌ Falha na conexão (Objeto exchange vazio).")
            return
            
        print("✅ Conexão Estabelecida!")

        # 3. Check Balance
        print("\n💰 Verificando Saldo...")
        try:
            balance = dm.get_balance(type='free')
            print(f"   Saldo USD Disponível (Margem): ${balance:.4f}")
            if balance > 0:
                print("   ✅ Saldo detectado com sucesso.")
            else:
                print("   ⚠️  Saldo é ZERO. Verifique se tem fundos na carteira de FUTUROS.")
        except Exception as e:
            print(f"   ❌ Erro ao ler saldo: {e}")

        # 4. Check Positions
        print("\n📊 Verificando Posições Abertas...")
        try:
            positions = dm.get_open_positions()
            if positions:
                print(f"   ✅ {len(positions)} Posições Abertas encontradas:")
                for p in positions:
                    print(f"      - {p['symbol']}: {p['contracts']} contratos | PNL: {p['unrealizedPnl']}")
            else:
                print("   ✅ Nenhuma posição aberta no momento (Retorno Vazio Correto).")
        except Exception as e:
            print(f"   ❌ Erro ao ler posições: {e}")

        print("\n" + "-" * 50)
        print("✅ TESTE DE CONECTIVIDADE CONCLUÍDO.")
        print("O sistema está pronto para operar via GUI.")

    except Exception as e:
        print(f"❌ Erro Fatal no Teste: {e}")

if __name__ == "__main__":
    run_test()
