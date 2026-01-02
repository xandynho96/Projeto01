import time
import sys
import subprocess
from strategy_optimizer import StrategyOptimizer
from extract_winning_signals import WinningSignalExtractor
from train_imitation import train_classifier
from backtest import run_backtest # We need to modify backtest to return logic

from evolutionary_strategy import EvolutionaryOptimizer

# Simple orchestration script
class ContinuousOptimizer:
    def __init__(self):
        self.target_winrate = 70.0
        self.iteration = 0
        
    def start(self):
        while True:
            self.iteration += 1
            print(f"\n\n=== CICLO DE OTIMIZAÇÃO {self.iteration} ===")
            print(f"Meta de Taxa de Acerto: {self.target_winrate}%")
            
            # 1. Genetic Evolution (Create Strategy)
            print("Passo 1: Criando Estratégia via Algoritmo Genético (Evolução)...")
            ga = EvolutionaryOptimizer()
            best_strategies = ga.evolve()
            print(f"Melhores Estratégias por Regime: {best_strategies.keys()}")
            
            # 2. Extraction (Using Best Strategies - Context Aware)
            print("Passo 2: Extraindo Sinais Vencedores (Context-Aware)...")
            extractor = WinningSignalExtractor()
            extractor.extract(strategy_genome=best_strategies)
            
            # 3. Train AI
            print("Passo 3: Retreinando Cérebro da IA (Aprendendo com Erros/Acertos)...")
            train_classifier()
            
            # 4. Backtest
            print("Passo 4: Verificando Performance no Mercado (Simulação)...")
            try:
                # Direct call for executable compatibility
                from backtest import run_backtest
                result = run_backtest()
                
                # result is a dict: {'winrate': float, 'output': str, ...}
                if result:
                    # Output is already printed by run_backtest (via log function), but we can access it if needed
                    winrate = result['winrate']
                    print(f"Taxa de Acerto (Winrate): {winrate:.2f}%")
                    
                    if winrate >= self.target_winrate:
                        print(f"META ATINGIDA! Winrate: {winrate}%")
                        print("A IA está pronta para operar com alta precisão.")
                        break
                    else:
                        print(f"Meta não atingida ({winrate:.2f}% < {self.target_winrate}%). Reiniciando ciclo de aprendizado...")
                else:
                    print("Erro no Backtest: Nenhum resultado retornado.")

            except Exception as e:
                print(f"Falha no Backtest: {e}")
                import traceback
                traceback.print_exc()
            
            print("Aguardando 10s para o próximo ciclo...")
            time.sleep(10)

if __name__ == "__main__":
    opt = ContinuousOptimizer()
    opt.start()
