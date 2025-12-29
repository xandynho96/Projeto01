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
            best_strategy = ga.evolve()
            print(f"Melhor Estratégia Encontrada: {best_strategy}")
            
            # 2. Extraction (Using Best Strategy)
            print("Passo 2: Extraindo Sinais Vencedores (Gerando Dataset)...")
            extractor = WinningSignalExtractor()
            extractor.extract(strategy_genome=best_strategy)
            
            # 3. Train AI
            print("Passo 3: Retreinando Cérebro da IA (Aprendendo com Erros/Acertos)...")
            train_classifier()
            
            # 4. Backtest
            print("Passo 4: Verificando Performance no Mercado (Simulação)...")
            # run_backtest prints output, we rely on logs for now.
            try:
                # We interpret backtest.py output
                proc = subprocess.run([sys.executable, "backtest.py"], capture_output=True, text=True)
                output = proc.stdout
                error = proc.stderr
                print(output)
                if error:
                    print(f"ERRO (STDERR): {error}")
                
                # Parse Winrate from output
                import re
                match = re.search(r"Taxa de Acerto \(Winrate\): (\d+\.\d+)%", output)
                if match:
                    winrate = float(match.group(1))
                    
                    if winrate >= self.target_winrate:
                        print(f"META ATINGIDA! Winrate: {winrate}%")
                        print("A IA está pronta para operar com alta precisão.")
                        break
                    else:
                        print(f"Meta não atingida ({winrate}% < {self.target_winrate}%). Reiniciando ciclo de aprendizado...")
                        
            except Exception as e:
                print(f"Falha no Backtest: {e}")
            
            print("Aguardando 10s para o próximo ciclo...")
            time.sleep(10)

if __name__ == "__main__":
    opt = ContinuousOptimizer()
    opt.start()
