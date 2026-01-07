import random
import pandas as pd
import numpy as np
from technical_analysis import TechnicalAnalysis
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
    'dist_bb_lower': (-0.05, 0.05), # Float
    'pattern_bullish_engulfing': (0.5, 0.5), # Fixed threshold for bool, operator determines 0 or 1
    'pattern_hammer': (0.5, 0.5),
    'obv_slope': (-100, 100), # Adjusted scale
    'supertrend': (0.5, 0.5), # Boolean
    'pattern_marubozu': (0.5, 0.5),
    'adx_slope': (-2, 2),
    'dist_support': (0, 0.05),
    'dist_resistance': (0, 0.05),
    'bb_width': (0.01, 0.5)
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
        self.dm = DataManager()
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
            
    def initialize_population(self, regime=None):
        self.population = []
        
        # 1. AI SEEDS (DeepSeek)
        # Attempt to get "Smart" seeds first
        try:
            from llm_strategy import DeepSeekStrategist
            strategist = DeepSeekStrategist()
            
            # Context description for LLM
            regime_desc = {
                'UPTREND': "Market is bullish. Look for pullbacks or breakouts.",
                'DOWNTREND': "Market is bearish. Look for short opportunities.",
                'SIDEWAYS': "Market is ranging. Look for mean reversion.",
                'HIGH_VOL': "Extreme volatility. Be careful, wide stops."
            }.get(regime, "General market conditions")
            
            # --- Gather Context for Smart Generation ---
            # 1. Market Metrics (Summary of last 500 candles roughly)
            summary_df = self.df.tail(1000)
            current_data_summary = {
                "avg_rsi": float(summary_df['rsi'].mean()),
                "avg_adx": float(summary_df['adx'].mean()),
                "volatility": float(summary_df['bb_width'].mean()),
                "recent_close": float(summary_df['close'].iloc[-1])
            }
            
            # 2. Existing Strategies (Exclusion)
            try:
                top_strats = self.dm.get_top_strategies(limit=50) # Assuming this method exists or we add it to DataManager
                existing_strategies = [s['genes'] for s in top_strats]
            except:
                existing_strategies = []
                
            # 3. Trade History (Performance Context)
            try:
                trade_history = self.dm.get_recent_trades(limit=20) # List of dicts
            except:
                trade_history = []
            
            # Ask for 3 smart strategies
            ai_strategies = strategist.generate_strategies(
                INDICATORS, 
                count=3, 
                market_regime=regime if regime else "SIDEWAYS",
                description=regime_desc,
                current_data_summary=current_data_summary,
                existing_strategies=existing_strategies,
                trade_history=trade_history
            )
            
            for strat_dict in ai_strategies:
                genes = []
                for cond in strat_dict.get('conditions', []):
                    # Map JSON to Gene
                    # "indicator": "rsi", "operator": "<", "threshold": 30
                    ind = cond['indicator']
                    op = cond['operator']
                    th = cond['threshold']
                    
                    # Basic validation to ensure indicator exists in our list
                    if ind in INDICATORS:
                        genes.append(StrategyGene(ind, op, th))
                
                if genes:
                    # Create Genome from AI Genes
                    genome = Genome(genes)
                    self.population.append(genome)
                    print(f"🤖 Estratégia IA Adicionada: {genome}")
                    
        except Exception as e:
            print(f"⚠️ Falha ao obter estratégias da IA: {e}")

        # 2. Add HARDCODED SEEDS (Fallback/Baseline)
        # Prevents "0% Winrate" stagnation by providing viable parents
        seeds = [
            # RSI Oversold
            Genome([StrategyGene('rsi', '<', 30)]),
            # Trend Pullback
            Genome([StrategyGene('dist_ema_200', '>', 0.001), StrategyGene('rsi', '<', 40)]),
            # Bollinger Breakout
            Genome([StrategyGene('bb_width', '>', 0.05), StrategyGene('dist_bb_lower', '<', 0.001)]),
            # Stochastic Reversion
            Genome([StrategyGene('stoch_k', '<', 20), StrategyGene('adx', '>', 20)]),
            # Momentum
            Genome([StrategyGene('rsi', '>', 55), StrategyGene('adx', '>', 25)])
        ]
        
        self.population.extend(seeds)
        
        # 3. Fill rest with VARIATIONS of Seeds (Smart Initialization)
        # User Constraint: "Never create random strategies"
        # Solution: Fill the rest of the population by mutating the intelligent seeds
        while len(self.population) < POPULATION_SIZE:
             if self.population:
                 # Pick a parent from the existing seeds using a weighted choice (simulate 'survival of the fittest' bias even at start)
                 # or just simple random choice from the good seeds
                 parent = random.choice(self.population[:len(seeds) + len(ai_strategies) if 'ai_strategies' in locals() else len(seeds)])
                 
                 # Clone and Mutate
                 child = Genome(genes=[StrategyGene(g.indicator, g.operator, g.threshold) for g in parent.genes])
                 child.mutate() # Slight variation
                 self.population.append(child)
             else:
                 # Fallback if NO seeds exist (should unlikely happen)
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
            self.initialize_population(regime=regime)
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
            best_strategies[regime] = best_for_regime
            print(f"✅ Melhor para {regime}: {best_for_regime.winrate:.1f}% Winrate ({best_for_regime.trades} trades)")
            
            # Save Best for Regime (or all if desired, user asked for cataloging)
            # Let's save the Top 5 of the final generation for each regime to the catalog
            for genome in self.population[:5]:
                if genome.trades > 0:
                   self.dm.save_strategy(genome, origin='evolution', regime=regime)

        return best_strategies

if __name__ == "__main__":
    opt = EvolutionaryOptimizer()
    strategies = opt.evolve()
    print(strategies)

