
import sys
import os
import json

# Setup Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from app.core.data_manager import DataManager

def analyze_strategies():
    dm = DataManager()
    
    # 1. Active Strategies
    active = dm.get_active_strategies()
    print(f"\n✅ ESTRATÉGIAS ATIVAS ({len(active)}):")
    print("-" * 60)
    for s in active:
        try:
            params = json.loads(s['genes'])
            print(f"[ID {s['id']}] WR: {s['winrate']:.1f}% | Trades: {s['trades']} | Origin: {s['origin']}")
            print(f"   Config: SL {params.get('sl_pct')}% | TP {params.get('tp_pct')}% | RSI {params.get('rsi_buy')}/{params.get('rsi_sell')}")
        except:
            print(f"[ID {s['id']}] (Error parsing genes)")
    
    if not active:
        print("   (Nenhuma estratégia ativa no momento)")

    # 2. Lab Candidates
    # Need to manually query as there is no specific method in DM for 'lab' status publicly
    session = dm.Session()
    try:
        from app.core.data_manager import StrategyModel
        lab = session.query(StrategyModel).filter(StrategyModel.status == 'lab').order_by(StrategyModel.fitness.desc()).limit(10).all()
        
        print(f"\n🧪 CANDIDATOS NO LABORATÓRIO (Top {len(lab)}):")
        print("-" * 60)
        for s in lab:
             print(f"[ID {s.id}] WR: {s.winrate:.1f}% | Trades: {s.trades} | Origin: {s.origin}")
             
        if not lab:
             print("   (Laboratório vazio)")
             
    except Exception as e:
        print(f"Error checking lab: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    analyze_strategies()
