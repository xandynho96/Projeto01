import logging
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from app.trader import BitcoinTrader
from app.core.evolutionary_strategy import EvolutionaryOptimizer
import time

# Configure logging to console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

def run_master_training():
    print("🚀 INICIANDO SESSÃO DE TREINAMENTO MESTRE...")
    print("1. Carregando Memória de Longo Prazo (50.000 velas)...")
    
    # 1. Update AI Brain (Neural Network / RandomForest)
    try:
        trader = BitcoinTrader() # This triggers initial training, but we force deep training
        trader.train_on_historical_memory(limit=50000)
        print("✅ Cérebro IA atualizado com sucesso.")
    except Exception as e:
        print(f"❌ Erro no treino da IA: {e}")

    # 2. Evolve Strategies for each Regime
    print("\n2. Evoluindo Estratégias Contextuais (Algoritmo Genético)...")
    try:
        opt = EvolutionaryOptimizer()
        # This will fetch data, analyze regimes, and evolve best strategies for Uptrend/Downtrend/Sideways
        strategies = opt.evolve() 
        print("✅ Estratégias evoluídas e salvas no banco de dados.")
    except Exception as e:
        print(f"❌ Erro na evolução de estratégias: {e}")

    print("\n🎉 SESSÃO CONCLUÍDA!")
    print("Agora você pode construir o executável e ele levará essa inteligência junto.")

if __name__ == "__main__":
    run_master_training()
