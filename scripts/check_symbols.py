
import sys
import os
import ccxt

def list_futures_markets():
    try:
        print("🔌 Conectando ao Kraken Futures...")
        kraken = ccxt.krakenfutures()
        markets = kraken.load_markets()
        
        print("\n🔎 Símbolos BTC Disponíveis:")
        for symbol, market in markets.items():
            if 'BTC' in symbol or 'XBT' in symbol:
                if market.get('linear') or market.get('inverse'):
                    settle = market.get('settle', 'Unknown')
                    quote = market.get('quote', 'Unknown')
                    margin = market.get('margin', 'Unknown')
                    print(f"   - {symbol} (ID: {market['id']}) | Tipo: {'Linear' if market.get('linear') else 'Inverse'} | Settle: {settle}")
                    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    list_futures_markets()
