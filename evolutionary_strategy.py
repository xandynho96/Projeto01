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
    'obv_slope', 'supertrend',
    'pattern_marubozu', 'adx_slope',
    'dist_support', 'dist_resistance',
    'bb_width'
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
    'supertrend': (0.5, 1.5), # Boolean 1 or 0
    'pattern_marubozu': (0.5, 1.5),
    'adx_slope': (-5, 5),
    'dist_support': (0, 0.1),
    'dist_resistance': (0, 0.1),
    'bb_width': (0, 0.5)
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
    def __init__(self, genes=None):
        if genes:
            self.genes = genes
        else:
            self.genes = [StrategyGene() for _ in range(CONDITIONS_PER_STRAT)]
            
        self.fitness = 0.0
        self.winrate = 0.0
        self.trades = 0
        
    def check_signal(self, row):
        # Check Genes
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
        
        return Genome(child_genes)
        
    def __repr__(self):
        gene_str = " AND ".join([str(g) for g in self.genes])
        return f"[Genes] {gene_str}"

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
            
            if self.df.empty:
                print("⚠️ AVISO: Banco de dados local vazio ou insuficiente.")
                dm.fetch_full_history() # Ensure we have data
                self.df = dm.get_data_from_db(limit=50000)
                
            if self.df.empty:
                raise Exception("CRITICAL: Não foi possível carregar dados.")

            self.df = self.df[self.df['timeframe'] == '1m'].copy()
            # Add indicators
            ta = TechnicalAnalysis(self.df)
            self.df = ta.add_all_indicators()
            
            # --- NEW: Add Regimes ---
            from market_regime import MarketRegime
            mr = MarketRegime()
            self.df = mr.add_regime_column(self.df)
            
            self.df.dropna(inplace=True)
            
    def initialize_population(self):
        self.population = []
        # 2. Fill with Random (DeepSeek removed for simplicity in this specific scope, can be re-added)
        while len(self.population) < POPULATION_SIZE:
             self.population.append(Genome())
        
    def pre_calculate_outcomes(self):
        print("Pré-calculando resultados futuros (Otimização Vetorizada)...")
        from fast_vector import calculate_outcomes_vectorized
        
        # Calculate boolean mask of winners
        winners_mask = calculate_outcomes_vectorized(
            self.df, 
            tp=0.0014, 
            sl=0.0004, 
            lookahead=60
        )
        self.df['is_winner'] = winners_mask

    def fast_evaluate(self, genome, regime_filter=None):
        # Dynamic Query Construction
        query_parts = []
        
        # Filter by Regime if specified
        if regime_filter:
            query_parts.append(f"(regime == '{regime_filter}')")
            
        for gene in genome.genes:
            query_parts.append(f"({gene.indicator} {gene.operator} {gene.threshold})")
        
        query_str = " & ".join(query_parts)
        
        try:
            # Vectorized Boolean Query
            subset = self.df.query(query_str)
        except Exception as e:
             genome.fitness = 0
             return

        trades = len(subset)
        if trades < 5: # Minimum trades per regime
            genome.fitness = 0
            genome.trades = trades
            genome.winrate = 0
            return
            
        wins = subset['is_winner'].sum()
        winrate = (wins / trades) * 100
        
        
        # Fitness Function Improvement (User Request)
        # 1. Penalty for low winrate (Random is 50%, so below 50% is bad)
        if winrate < 50:
            penalty = 0.1 # Heavily penalize losing strategies
        else:
            penalty = 1.0
            
        # 2. Minimum Trades Threshold
        if trades < 30:
            genome.fitness = 0
        else:
            # 3. Logarithmic Scale for Trades (Diminishing returns for quantity)
            # winrate (0-100) * log(trades) * penalty
            # Example: 60% WR * log(100) * 1 ~= 60 * 4.6 = 276
            # Example: 51% WR * log(1000) * 1 ~= 51 * 6.9 = 351 (High volume slightly better)
            # Example: 45% WR * log(1000) * 0.1 ~= 4.5 * 6.9 = 31 (Punished)
            
            # Using natural log
            genome.fitness = winrate * np.log(trades) * penalty
        
    def evolve(self):
        print("Iniciando Evolução Genética Context-Aware...")
        self.pre_calculate_outcomes()
        
        regimes = ['UPTREND', 'DOWNTREND', 'SIDEWAYS']
        best_strategies = {}
        
        for regime in regimes:
            print(f"\n🌊 Evoluindo Estratégia para regime: {regime}...")
            self.initialize_population()
            best_for_regime = None
            
            for gen in range(GENERATIONS):
                # Evaluate
                for genome in self.population:
                    self.fast_evaluate(genome, regime_filter=regime)
                
                # Sort
                self.population.sort(key=lambda x: x.fitness, reverse=True)
                best_gen = self.population[0]
                
                # Print stats for top 1
                if gen % 2 == 0:
                    print(f"   Gen {gen+1}: WR {best_gen.winrate:.1f}% | Trades {best_gen.trades} | Eq: {best_gen}")
                
                if best_for_regime is None or best_gen.fitness > best_for_regime.fitness:
                    best_for_regime = best_gen
                
                # Selection & Crossover
                survivors = self.population[:int(POPULATION_SIZE*0.4)]
                new_pop = survivors[:]
                while len(new_pop) < POPULATION_SIZE:
                    p1 = random.choice(survivors)
                    p2 = random.choice(survivors)
                    child = p1.crossover(p2)
                    child.mutate()
                    new_pop.append(child)
                self.population = new_pop
                
            best_strategies[regime] = best_for_regime
            print(f"✅ Melhor para {regime}: {best_for_regime.winrate:.1f}% Winrate ({best_for_regime.trades} trades)")
            
        return best_strategies

if __name__ == "__main__":
    opt = EvolutionaryOptimizer()
    strategies = opt.evolve()
    print(strategies)

