import re
import pandas as pd
import sqlite3
import os
from datetime import datetime, timedelta

# Configurações
LOG_FILE = r"c:\Users\Alexandre\Documents\Testes\Projeto01\dist\logs\bot.log"
DB_FILE = r"c:\Users\Alexandre\Documents\Testes\Projeto01\dist\crypto_data.db"

# Padrões de Regex (Adaptados ao que vi no log e user request)
# Ex: 2026-01-14 22:30:43,799 - BitcoinAI - INFO - DECISÃO: BUY
# Ex: 2026-01-14 22:30:43,798 - BitcoinAI - INFO - Preço Atual: 96467.00
# Ex: 2026-01-14 22:31:54,709 - BitcoinAI - INFO -    📐 SL: 0.18% | TP: 0.24%
# Ex: 2026-01-14 22:32:00,058 - BitcoinAI - WARNING - ⛔ DeepSeek rejeitou a entrada. Cancelando trade.

# Regex
re_date = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
re_price = re.compile(r"Preço Atual:\s*([\d\.]+)")
re_decision = re.compile(r"DECISÃO:\s*(BUY|SELL)")
re_rejection = re.compile(r"DeepSeek.*?rejeitou|DeepSeek.*?Cancelando trade")
re_sl_tp = re.compile(r"SL:\s*([\d\.]+)%.*TP:\s*([\d\.]+)%")

def parse_log():
    print(f"Lendo log: {LOG_FILE}...")
    events = []
    
    current_event = {}
    
    try:
        with open(LOG_FILE, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Timestamp
            match_date = re_date.search(line)
            if match_date:
                ts_str = match_date.group(1)
                ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
            else:
                continue

            # Context Collecting (Price, SL/TP)
            # We look nicely backwards/forwards usually, but sequential scan is easier.
            # We assume Price and SL/TP appear slightly BEFORE Decision.
            
            match_price = re_price.search(line)
            if match_price:
                current_event['price'] = float(match_price.group(1))
                
            match_sl_tp = re_sl_tp.search(line)
            if match_sl_tp:
                current_event['sl'] = float(match_sl_tp.group(1))
                current_event['tp'] = float(match_sl_tp.group(2))

            match_dec = re_decision.search(line)
            if match_dec:
                current_event['decision'] = match_dec.group(1).lower()
                current_event['timestamp'] = ts
                # Wait for rejection or execution in next lines
                
                # Look ahead for rejection in next 20 lines
                rejected = False
                for j in range(1, 40):
                    if i + j >= len(lines): break
                    next_line = lines[i+j]
                    
                    if re_rejection.search(next_line):
                        rejected = True
                        break
                    if "Executando entrada" in next_line:
                        rejected = False # Executed
                        break
                        
                if rejected:
                    if 'price' in current_event and 'decision' in current_event:
                        events.append(current_event.copy())
                    else:
                        pass # Incomplete data
                    
                current_event = {} # Reset

    except Exception as e:
        print(f"Erro ao ler log: {e}")
        return []

    return events

def analyze_events(events):
    if not events:
        print("Nenhum evento de recusa encontrado.")
        return

    print(f"Encontrados {len(events)} eventos de recusa pelo DeepSeek.")
    
    # Connect DB
    conn = sqlite3.connect(DB_FILE)
    
    results = {
        'total': 0,
        'saved_from_loss': 0, # DeepSeek rejected a losing trade (Good)
        'missed_win': 0,      # DeepSeek rejected a winning trade (Bad)
        'neutral': 0
    }
    
    for ev in events:
        ts = ev['timestamp']
        price = ev['price']
        side = ev['decision'] # buy/sell
        # Default SL/TP based on User Request:
        # TP 40% ROE @ 50x = 0.8% Price Move (0.008)
        # SL 20% ROE @ 50x = 0.4% Price Move (0.004)
        sl_pct = 0.004
        tp_pct = 0.008
        
        # Calculate Targets
        if side == 'buy':
            tp_price = price * (1 + tp_pct)
            sl_price = price * (1 - sl_pct)
        else:
            tp_price = price * (1 - tp_pct)
            sl_price = price * (1 + sl_pct)
            
        # Fetch candles after timestamp
        # Look for 4 hours ahead
        start_ts = ts
        end_ts = ts + timedelta(hours=4)
        
        query = f"""
            SELECT timestamp, open, high, low, close 
            FROM market_data 
            WHERE timeframe='1m' 
            AND timestamp >= '{start_ts}' 
            AND timestamp <= '{end_ts}'
            ORDER BY timestamp ASC
        """
        
        df = pd.read_sql_query(query, conn)
        
        if df.empty:
            print(f"[{ts}] Sem dados futuros para validar.")
            results['neutral'] += 1
            continue
            
        # Simular
        outcome = "neutral"
        
        for _, row in df.iterrows():
            curr_high = row['high']
            curr_low = row['low']
            
            if side == 'buy':
                # Check SL first (conservative) or checking both in same candle implies wicks
                # Let's assume Worst Case: Hit SL first if in range
                if curr_low <= sl_price:
                    outcome = "loss"
                    break
                if curr_high >= tp_price:
                    outcome = "win"
                    break
            else: # sell
                if curr_high >= sl_price:
                    outcome = "loss"
                    break
                if curr_low <= tp_price:
                    outcome = "win"
                    break
                    
        print(f"[{ts}] {side.upper()} @ {price} | TP: {tp_price:.2f} SL: {sl_price:.2f} | Resultado: {outcome.upper()}")
        
        if outcome == 'win':
            results['missed_win'] += 1 # Bad rejection
        elif outcome == 'loss':
            results['saved_from_loss'] += 1 # Good rejection
        else:
            results['neutral'] += 1
            
    # Relatório
    print("\n--- Relatório de Auditoria do DeepSeek ---")
    print(f"Total Recusas Analisadas: {len(events)}")
    print(f"✅ DeepSeek Salvou de Loss: {results['saved_from_loss']} (Acertou em recusar)")
    print(f"❌ DeepSeek Impediu Win: {results['missed_win']} (Errou em recusar)")
    print(f"🤷 Indeterminado (Sem dados/Lateral): {results['neutral']}")

if __name__ == "__main__":
    events = parse_log()
    analyze_events(events)
