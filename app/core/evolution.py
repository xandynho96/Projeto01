import random
import time
import pandas as pd
import numpy as np
import json
import requests
import os
import sys
from app.core.data_manager import DataManager
from app.core.backtest import Backtester
from app.utils import config

# --- GENETIC ALGORITHM CONFIG ---
POPULATION_SIZE = 20
GENERATIONS = 1000 
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.7

# --- PARAMETER RANGES ---
# --- PARAMETER RANGES (SCALPING FOCUSED) ---
PARAM_RANGES = {
    'sl_pct': (0.2, 1.0),      # SL: 0.2% (10% Risk) to 1.0% (50% Risk at 50x Lev)
    'tp_pct': (0.6, 3.0),      # TP: Min 0.6% (Covers Fees) up to 3% (150% ROE)
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
            from app.utils.fast_vector import calculate_outcomes_vectorized, calculate_outcomes_vectorized_short
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
    # Kraken Taker Fee: ~0.26% per side => 0.52% Roundtrip
    COMMISSION = 0.0052 
    
    # Net Profit = (Wins * (TP - Comm)) - (Losses * (SL + Comm))
    net_primary = wins * (tp_pct - COMMISSION)
    net_loss = losses * (sl_pct + COMMISSION)
    net_pct = net_primary - net_loss

    if total_trades < 10: net_pct *= 0.5 # Penalty for low sample size
    
    genome.roi = net_pct * 100
    genome.fitness = net_pct
    return genome.fitness

def cull_weak_strategies(bt, dm):
    """
    Evaluates currently ACTIVE strategies against recent data.
    If Winrate < 40% (min 5 trades), demote to ARCHIVED.
    If demoted, try to promote a replacement from LAB.
    """
    active_strats = dm.get_active_strategies()
    if not active_strats: return

    print(f"🕵️ Active Strategy Audit ({len(active_strats)} active)...")
    
    for s_data in active_strats:
        # Reconstruct Genome to evaluate
        try:
             params = json.loads(s_data['genes'])
             genome = Genome(params)
             
             # Evaluate on CURRENT backtest data (which is loaded in bt)
             # This data is the last 2000 candles (approx 33 hours)
             # So we are checking: "Is this strategy still good TODAY?"
             evaluate_fitness(genome, bt)
             
             # Criteria to Demote
             # Winrate < 40% AND Trades > 5
             if genome.trades >= 5 and genome.winrate < 40.0:
                 print(f"📉 DETECTED UNDERPERFORMANCE: ID {s_data['id']} (WR {genome.winrate:.1f}%). Demoting...")
                 dm.demote_strategy(s_data['id'], reason=f"Low Winrate {genome.winrate:.1f}% in Audit")
                 
                 # Try to Replace
                 replacement = dm.get_best_lab_strategy()
                 if replacement:
                     dm.promote_strategy(replacement['id'])
                     print(f"♻️ Replaced with Lab Candidate ID {replacement['id']}")
                 else:
                     print("⚠️ No replacement found in Lab.")
                     
             elif genome.trades < 5:
                  pass # Not enough sample size in this window
             else:
                  # Good performance or decent
                  pass
                  
        except Exception as e:
            print(f"Error auditing strategy {s_data['id']}: {e}")

        # Removed stats print to avoid NameError (variables generation/best not in scope here)
        pass
             
        # Time limit check removed from here (handled in evolution_worker)
        pass

        time.sleep(1)

def evolution_worker():
    print("🧪 Starting Strategy Laboratory (DeepSeek + Evo)...")
    
    # State File
    import sys
    if getattr(sys, 'frozen', False):
        BASE_DIR = os.path.dirname(sys.executable)
    else:
        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    STATE_FILE = os.path.join(BASE_DIR, "data", "evolution_state.json")
    MAX_DURATION = 3600 # 1 Hour in seconds
    COOLDOWN = 23 * 3600 # 23 Hours (Wait 24h total roughly, or just check next day)
    
    while True:
        # Check if we can run
        can_run = True
        last_run = 0
        
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, 'r') as f:
                    data = json.load(f)
                    last_run = data.get('last_run', 0)
            except: pass
            
        time_since = time.time() - last_run
        if time_since < 24 * 3600: # Less than 24h passed
             if time_since < 0: last_run = 0 # Clock skew fix
             else:
                 wait_time = (24 * 3600) - time_since
                 if wait_time > 60: # Only print if meaningful wait
                    print(f"⏳ Evolution Cooldown. Next run in {wait_time/3600:.1f} hours.")
                 
                 # Sleep in chunks to allow thread kill
                 sleep_chunks = int(min(wait_time, 300)) # Check every 5 mins max
                 time.sleep(sleep_chunks) 
                 can_run = False

        if not can_run:
            continue

        # START RUN
        print("🚀 Starting Daily Evolution Cycle (Max 1h)...")
        start_time = time.time()
        
        # Initialize Context for this run
        bt = Backtester(symbol="BTC/USD", timeframe="1m")
        if not bt.load_data(limit=2000): 
            time.sleep(60)
            continue

        population = [Genome() for _ in range(POPULATION_SIZE)]
        dm = DataManager()
        generation = 0
        
        # Inner Optimization Loop
        while True:
            # Check Time Limit
            if time.time() - start_time > MAX_DURATION:
                print("⏰ Daily Limit Reached. Saving State...")
                try:
                    state = {'last_run': time.time()}
                    # Ensure data dir exists
                    data_dir = os.path.dirname(STATE_FILE)
                    if not os.path.exists(data_dir): os.makedirs(data_dir)
                    
                    with open(STATE_FILE, 'w') as f:
                        json.dump(state, f)
                except Exception as e:
                    print(f"Error saving state: {e}")
                break # Break inner loop, go back to cooldown check

            generation += 1
            
            # --- IMPROVEMENT: Live Data Refresh & Stagnation Check ---
            # 1. Refresh Data every 50 gens to keep Lab "Live"
            if generation % 50 == 0:
                print("🔄 Lab: Refreshing market data for simulation...")
                bt.load_data(limit=2000)

            # 2. Check Stagnation (If best WR is 0% for too long)
            # We check the previous generation's best (stored in 'best' variable from prev loop)
            # Initialize 'best' before loop to avoid NameError on first run if needed, but 'best' is defined at end of loop.
            # actually we check at the END of loop or start of next. Let's do it here assuming 'best' exists from gen > 1
            if generation > 50 and 'best' in locals() and best.winrate == 0:
                 if not hasattr(top_level, 'stagnation_counter'): top_level_stagnation = 0 # Dummy
                 # Actually, let's track a counter locally
                 pass 

            # Simpler Stagnation Logic:
            # If we are at gen 50, 100, 150... and best.winrate is still 0, RESET population.
            if generation % 50 == 0 and 'best' in locals() and best.winrate < 1.0:
                 print("⚠️ Lab Stagnation Detected (0% Winrate). RESETTING Population for fresh ideas.")
                 population = [Genome() for _ in range(POPULATION_SIZE)]

            
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
            if best.winrate > 60 and best.trades > 15:
                print(f"🏆 PROMOTING Strategy to REAL: WR {best.winrate:.1f}% | ROI {best.roi:.2f}% | Params: {best.params}")
                dm.save_strategy(best, origin='lab_deepseek' if generation % 5 == 1 else 'lab_evo', status='active')
            else:
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
            
            if generation % 10 == 0:
                 cull_weak_strategies(bt, dm)
    
            if generation % 5 == 0:
                 print(f"🧬 Gen {generation} Best: WR {best.winrate:.1f}% | Trades {best.trades}")
                 
            time.sleep(1)

if __name__ == "__main__":
    try:
        evolution_worker()
    except KeyboardInterrupt:
        print("Done.")
