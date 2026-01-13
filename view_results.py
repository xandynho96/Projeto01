from data_manager import DataManager
import json

def show_results():
    dm = DataManager()
    print("🔎 Buscando melhores estratégias aprendidas...")
    
    # Fetch top strategies
    strats = dm.get_top_strategies(limit=10)
    
    if not strats:
        print("Nenhuma estratégia encontrada no banco de dados.")
        return

    print(f"\n🏆 TOP 10 Estratégias Evoluídas:")
    print(f"{'ID':<5} | {'Origem':<12} | {'WinRate':<8} | {'Trades':<6} | {'Params (SL/TP/RSI)'}")
    print("-" * 80)
    
    for s in strats:
        try:
            params = json.loads(s['genes'])
            p_str = f"SL: {params.get('sl_pct')}% TP: {params.get('tp_pct')}% RSI: {params.get('rsi_buy')}/{params.get('rsi_sell')}"
        except:
            p_str = s['genes']
            
        try:
            # Ensure trade count is int
            trades_count = int(s['trades']) if s['trades'] is not None else 0
            winrate_val = float(s['winrate']) if s['winrate'] is not None else 0.0
            print(f"{str(s['id']):<5} | {str(s['origin']):<12} | {winrate_val:.1f}%   | {trades_count:<6} | {p_str}")
        except Exception as e:
            print(f"Error printing row: {e}")

    # Check Active Strategy
    active = dm.get_best_active_strategy()
    if active:
        print(f"\n🌟 Estratégia Ativa Atual (Sendo usada pelo Bot):")
        try:
            params = json.loads(active['genes'])
            print(f"   Origem: {active['origin']}")
            print(f"   WinRate: {active['winrate']:.1f}% ({active['trades']} trades)")
            print(f"   Config: SL {params.get('sl_pct')}% | TP {params.get('tp_pct')}%")
        except:
             print(f"   Genes: {active['genes']}")
    else:
        print("\n⚠️ Nenhuma estratégia marcada como 'active' no momento.")

if __name__ == "__main__":
    show_results()
