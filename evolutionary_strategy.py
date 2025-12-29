import random
import pandas as pd
import numpy as np
from technical_analysis import TechnicalAnalysis
from data_manager import DataManager

# --- CONFIG ---
POPULATION_SIZE = 20
GENERATIONS = 5
CONDITIONS_PER_STRAT = 2 # Keep it simple for now (2 conditions per strategy)

INDICATORS = [
    'rsi', 'stoch_k', 'stoch_d', 'adx', 'cci', 'mfi', 'williams_r',
    'dist_ema_200', 'dist_bb_lower',
    'pattern_bullish_engulfing', 'pattern_hammer',
    'obv_slope', 'supertrend'
]
OPERATORS = ['<', '>']
THRESHOLDS = {
    'rsi': (20, 80),
    'stoch_k': (20, 80),
    'stoch_d': (20, 80),
    'adx': (15, 50),
    'cci': (-150, 150),
    'mfi': (20, 80),
    'williams_r': (-90, -10),
    'dist_ema_200': (-0.05, 0.05), # Float
    'dist_bb_lower': (-0.02, 0.02), # Float
    'pattern_bullish_engulfing': (0.5, 1.5), 
    'pattern_hammer': (0.5, 1.5),
    'obv_slope': (-1000, 1000), # Volume change
    'supertrend': (0.5, 1.5) # Boolean 1 or 0
}

class StrategyGene:
    """Represents a single condition: Indicator Operator Threshold"""
    def __init__(self, indicator=None, operator=None, threshold=None):
        self.indicator = indicator if indicator else random.choice(INDICATORS)
        self.operator = operator if operator else random.choice(OPERATORS)
        
        if threshold is not None:
            self.threshold = threshold
        else:
            min_val, max_val = THRESHOLDS[self.indicator]
            
            # Fix for Float ranges
            if isinstance(min_val, float) or isinstance(max_val, float):
                self.threshold = random.uniform(min_val, max_val)
            else:
                self.threshold = random.randint(min_val, max_val)

    def evaluate(self, row):
        val = row.get(self.indicator)
        if val is None: return False
        
        if self.operator == '<':
            return val < self.threshold
        else:
            return val > self.threshold
            
    def __repr__(self):
        return f"{self.indicator} {self.operator} {self.threshold}"

class Genome:
    """Represents a Strategy (Collection of Genes)"""
    def __init__(self, genes=None, trend_mode=None):
        if genes:
            self.genes = genes
        else:
            self.genes = [StrategyGene() for _ in range(CONDITIONS_PER_STRAT)]
            
        # Trend Filter: 'UPTREND' (Above EMA200), 'DOWNTREND' (Below EMA200), or None (Any)
        # Randomly assign a trend bias or None
        self.trend_mode = trend_mode if trend_mode else random.choice(['UPTREND', 'DOWNTREND', None])
            
        self.fitness = 0.0
        self.winrate = 0.0
        self.trades = 0
        
    def check_signal(self, row):
        # 1. Check Trend Filter
        if self.trend_mode == 'UPTREND':
            if row['close'] <= row['ema_200']: return False
        elif self.trend_mode == 'DOWNTREND':
            if row['close'] >= row['ema_200']: return False
            
        # 2. Check Genes
        for gene in self.genes:
            if not gene.evaluate(row):
                return False
        return True
        
    def mutate(self):
        # 20% chance to mutate a gene
        if random.random() < 0.2:
            idx = random.randint(0, len(self.genes)-1)
            self.genes[idx] = StrategyGene() # Re-roll gene

    def crossover(self, other):
        # Single point crossover
        idx = random.randint(1, len(self.genes)-1) if len(self.genes)>1 else 0
        child_genes = self.genes[:idx] + other.genes[idx:]
        
        # Inherit trend from one parent
        child_trend = self.trend_mode if random.random() > 0.5 else other.trend_mode
        return Genome(child_genes, trend_mode=child_trend)
        
    def __repr__(self):
        trend_str = f"Trend:{self.trend_mode}" if self.trend_mode else "Trend:ANY"
        gene_str = " AND ".join([str(g) for g in self.genes])
        return f"[{trend_str}] {gene_str}"

class EvolutionaryOptimizer:
    def __init__(self, df=None):
        self.population = []
        if df is not None:
            self.df = df
        else:
            print("Carregando dados para Evolução...")
            dm = DataManager()
            # Increase history for robust validation (User Request)
            self.df = dm.get_data_from_db(limit=50000) 
            self.df = self.df[self.df['timeframe'] == '1m'].copy()
            # Add indicators
            ta = TechnicalAnalysis(self.df)
            self.df = ta.add_all_indicators()
            self.df.dropna(inplace=True)
            
    def initialize_population(self):
        self.population = [Genome() for _ in range(POPULATION_SIZE)]
        
    def evaluate_fitness(self, genome):
        # Backtest the genome
        # Simple Vectorized check is hard with dynamic conditions, iterating is safer/easier for logic
        # Optimize speed later
        
        data = self.df.copy()
        balance = 1000
        wins = 0
        losses = 0
        
        tp = 0.0014
        sl = 0.0004
        
        # We need to simulate
        # To be fast, we assume 'entry' at close, and checking next N candles outcome is expensive
        # Let's pre-calculate 'Future Outcome' columns for speed?
        # Ideally we only evaluate rows where the signal is True.
        
        matches = []
        for index, row in data.iterrows():
            if genome.check_signal(row):
                matches.append(index)
        
        if len(matches) < 5: # Too few trades = bad
            genome.fitness = 0
            genome.trades = len(matches)
            return
            
        # Check outcomes for matches
        # We need integer locations, index is timestamp potentially
        # Let's map timestamp index to integer location
        
        # Optimization: Pre-calculate "Is Winner" column in DF? 
        # But "Is Winner" depends on the entry? No, outcome depends on Close[i] vs Future.
        # Yes! We can pre-calculate "Potential Win" column for Longs.
        pass # See evolve method for pre-calc logic
        
    def pre_calculate_outcomes(self):
        print("Pré-calculando resultados futuros (Otimização)...")
        self.df['is_winner'] = False
        
        tp = 0.0014
        sl = 0.0004
        
        closes = self.df['close'].values
        highs = self.df['high'].values
        lows = self.df['low'].values
        
        # Numba or Vectorization would be best, but simple loop for now
        results = []
        lookahead = 60
        n = len(closes)
        
        for i in range(n - lookahead):
            entry = closes[i]
            win = False
            
            # Check window
            for j in range(1, lookahead):
                if i+j >= n: break
                
                gain = (highs[i+j] - entry) / entry
                loss = (lows[i+j] - entry) / entry
                
                if loss < -sl:
                    win = False
                    break
                if gain > tp:
                    win = True
                    break
            
            if win:
                # Store True at the index i
                results.append(True)
            else:
                results.append(False)
        
        # Fill rest
        results += [False]*lookahead
        self.df['is_winner'] = results

    def fast_evaluate(self, genome):
        # Evaluate using pre-calculated column
        # Only iterate where signal is True
        
        # Dynamic Query Construction
        query_parts = []
        
        # Add Trend Filter to Query
        if genome.trend_mode == 'UPTREND':
            query_parts.append("(close > ema_200)")
        elif genome.trend_mode == 'DOWNTREND':
            query_parts.append("(close < ema_200)")
            
        for gene in genome.genes:
            query_parts.append(f"({gene.indicator} {gene.operator} {gene.threshold})")
        
        query_str = " & ".join(query_parts)
        
        try:
            subset = self.df.query(query_str)
        except:
             genome.fitness = 0
             return

        trades = len(subset)
        if trades < 10:
            genome.fitness = 0
            genome.trades = trades
            genome.winrate = 0
            return
            
        wins = subset['is_winner'].sum()
        winrate = (wins / trades) * 100
        
        genome.trades = trades
        genome.winrate = winrate
        genome.fitness = winrate * (1 + (trades/1000)) # Bonus for more trades
        
    def evolve(self):
        print("Iniciando Evolução Genética...")
        self.pre_calculate_outcomes()
        self.initialize_population()
        
        best_ever = None
        
        for gen in range(GENERATIONS):
            # Evaluate
            for genome in self.population:
                self.fast_evaluate(genome)
            
            # Sort
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            best_gen = self.population[0]
            print(f"Gen {gen+1} Melhor: {best_gen} | WR: {best_gen.winrate:.1f}% | Trades: {best_gen.trades}")
            
            if best_ever is None or best_gen.fitness > best_ever.fitness:
                best_ever = best_gen
            
            # Select
            survivors = self.population[:int(POPULATION_SIZE*0.4)] # Top 40%
            
            # Crossover/Repopulate
            new_pop = survivors[:]
            while len(new_pop) < POPULATION_SIZE:
                p1 = random.choice(survivors)
                p2 = random.choice(survivors)
                child = p1.crossover(p2)
                child.mutate()
                new_pop.append(child)
            
            self.population = new_pop
            
        print(f"\nMelhor Estratégia Evoluída: {best_ever}")
        return best_ever

if __name__ == "__main__":
    opt = EvolutionaryOptimizer()
    best = opt.evolve()
