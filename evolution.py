
import random
import time
import pandas as pd
import numpy as np
import json
import requests
from data_manager import DataManager
from backtest import Backtester
import config

# --- GENETIC ALGORITHM CONFIG ---
POPULATION_SIZE = 20
GENERATIONS = 1000 
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.7

# --- PARAMETER RANGES ---
# --- PARAMETER RANGES (SCALPING FOCUSED) ---
PARAM_RANGES = {
    'sl_pct': (0.1, 1.5),      # Tight stops (Max 1.5% price move)
    'tp_pct': (0.2, 3.0),      # Scalp targets (Max 3.0% price move)
    'rsi_buy': (15, 45),       
    'rsi_sell': (55, 85),      
    # 'rsi_period': (10, 20),
}
# ...

class Genome:
    def __init__(self, params=None):
        self.params = params if params else self._random_params()
        self.fitness = 0.0
        self.winrate = 0.0
        self.trades = 0
        self.roi = 0.0
        self.id = None # DB ID

    def _random_params(self):
        params = {}
        for key, (min_val, max_val) in PARAM_RANGES.items():
            if isinstance(min_val, int):
                params[key] = random.randint(min_val, max_val)
            else:
                params[key] = round(random.uniform(min_val, max_val), 2)
        
        # Sanity check
        if params['rsi_buy'] >= params['rsi_sell']:
            params['rsi_buy'] = params['rsi_sell'] - 5
        return params

    def mutate(self):
        if random.random() < MUTATION_RATE:
            key = random.choice(list(PARAM_RANGES.keys()))
            min_val, max_val = PARAM_RANGES[key]
            
            # Small nudge
            current = self.params[key]
            nudge = (max_val - min_val) * 0.1 * random.uniform(-1, 1)
            new_val = current + nudge
            # Clamp
            new_val = max(min_val, min(max_val, new_val))
            
            if isinstance(min_val, int):
                new_val = int(round(new_val))
            else:
                new_val = round(new_val, 2)
            self.params[key] = new_val

    def crossover(self, other):
        if random.random() < CROSSOVER_RATE:
            child_params = {}
            for key in self.params:
                child_params[key] = self.params[key] if random.random() < 0.5 else other.params[key]
            return Genome(child_params)
        else:
            return Genome(self.params.copy())

def ask_deepseek_for_strategy(market_summary):
    """
    Asks DeepSeek to generate a strategy configuration based on market state.
    """
    api_key = config.DEEPSEEK_API_KEY
    if not api_key: return None
    
    prompt = f"""
    Atue como um Arquiteto de Estratégias Quant (Foco: SCALPING).
    Contexto de Mercado (Crypto 1m Data):
    {market_summary}
    
    Gere uma configuração JSON para um bot Scalper de RSI Mean Reversion.
    Ranges permitidos (SCALPING RIGOROSO):
    - sl_pct: 0.1 a 1.5 (Stop Curto)
    - tp_pct: 0.2 a 3.0 (Alvo Rápido)
    - rsi_buy: 15 a 45
    - rsi_sell: 55 a 85
    
    Objetivo: Maximizar Winrate (>60%) com entradas precisas.
    Responda APENAS o JSON: {{ "sl_pct": float, "tp_pct": float, "rsi_buy": int, "rsi_sell": int }}
    """
    
    try:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }
        resp = requests.post("https://api.deepseek.com/chat/completions", json=payload, headers=headers, timeout=10)
        if resp.status_code == 200:
            content = resp.json()['choices'][0]['message']['content']
            content = content.replace("```json", "").replace("```", "").strip()
            params = json.loads(content)
            # Validate keys
            for key in PARAM_RANGES:
                if key not in params: return None
            return params
    except Exception as e:
        print(f"DeepSeek Gen Error: {e}")
    return None

def evaluate_fitness(genome, bt):
    # Same logic as before
    df = bt.df
    if df.empty: return 0.0
    
    rsi = df['rsi']
    signals = pd.Series(0, index=df.index)
    signals[rsi < genome.params['rsi_buy']] = 1
    signals[rsi > genome.params['rsi_sell']] = -1
    
    # HACK: Fast vector calc import
    if 'fast_vector' not in globals():
        try:
            from fast_vector import calculate_outcomes_vectorized, calculate_outcomes_vectorized_short
            globals()['fast_vector_calcs'] = (calculate_outcomes_vectorized, calculate_outcomes_vectorized_short)
        except: return 0.0

    calc_long, calc_short = globals()['fast_vector_calcs']
    sl_pct = genome.params['sl_pct'] / 100
    tp_pct = genome.params['tp_pct'] / 100
    lookahead = 60
    
    long_winners = calc_long(df, tp=tp_pct, sl=sl_pct, lookahead=lookahead)
    short_winners = calc_short(df, tp=tp_pct, sl=sl_pct, lookahead=lookahead)
    
    is_long = signals == 1
    is_short = signals == -1
    
    wins = (is_long & long_winners).sum() + (is_short & short_winners).sum()
    losses = (is_long & ~long_winners).sum() + (is_short & ~short_winners).sum()
    total_trades = wins + losses
    
    if total_trades == 0:
        genome.winrate = 0; genome.trades = 0; genome.fitness = 0; return 0

    genome.winrate = (wins / total_trades) * 100
    genome.trades = total_trades
    net_pct = (wins * tp_pct) - (losses * sl_pct)
    if total_trades < 10: net_pct *= 0.5 # Penalty for low sample size
    
    genome.roi = net_pct * 100
    genome.fitness = net_pct
    return genome.fitness

def evolution_worker():
    print("🧪 Starting Strategy Laboratory (DeepSeek + Evo)...")
    bt = Backtester(symbol="BTC/USD", timeframe="1m")
    if not bt.load_data(limit=2000): return

    population = [Genome() for _ in range(POPULATION_SIZE)]
    dm = DataManager()
    
    generation = 0
    while True:
        generation += 1
        
        # 1. Ask DeepSeek for a Hypothesis (every 5 generations to save credits)
        if generation % 5 == 1:
            # Build Market Context
            last_row = bt.df.iloc[-1]
            summary = f"RSI: {last_row['rsi']:.2f}, Trend: {'Bull' if last_row['close'] > last_row.get('ema_200', 0) else 'Bear'}"
            print(f"🤖 Lab: Asking DeepSeek for strategy hypothesis ({summary})...")
            new_params = ask_deepseek_for_strategy(summary)
            if new_params:
                print(f"🤖 DeepSeek Suggested: {new_params}")
                population.append(Genome(new_params)) # Inject into population

        # 2. Evaluate
        for genome in population:
            evaluate_fitness(genome, bt)
            
        population.sort(key=lambda x: x.fitness, reverse=True)
        best = population[0]
        
        # 3. Promotion Logic
        # Lab Criteria for Promotion: Winrate > 60%, Trades > 15
        if best.winrate > 60 and best.trades > 15:
            # Check if this exact config is already promoted? (Optional)
            print(f"🏆 PROMOTING Strategy to REAL: WR {best.winrate:.1f}% | ROI {best.roi:.2f}% | Params: {best.params}")
            dm.save_strategy(best, origin='lab_deepseek' if generation % 5 == 1 else 'lab_evo', status='active')
        else:
             # Just save as Lab candidate if decent
             if best.winrate > 50 and best.trades > 5:
                  if generation % 5 == 0:
                      print(f"🧪 Saving Lab Candidate: WR {best.winrate:.1f}% (Needs Improvement)")
                      dm.save_strategy(best, origin='lab_evo', status='lab')

        # 4. Evolution
        new_pop = population[:4] # Keep top 4 (Elitism)
        while len(new_pop) < POPULATION_SIZE:
            parent_a = random.choice(population[:10])
            parent_b = random.choice(population[:10])
            child = parent_a.crossover(parent_b)
            child.mutate()
            new_pop.append(child)
        population = new_pop
        
        if generation % 5 == 0:
             print(f"🧬 Gen {generation} Best: WR {best.winrate:.1f}% | Trades {best.trades}")
             
        time.sleep(1)

if __name__ == "__main__":
    try:
        evolution_worker()
    except KeyboardInterrupt:
        print("Done.")
