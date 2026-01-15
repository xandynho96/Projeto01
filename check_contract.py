import ccxt
import sys

def check_contract():
    print("Conectando ao Kraken Futures...")
    exchange = ccxt.krakenfutures()
    exchange.load_markets()
    
    symbol = 'BTC/USD:USD'
    
    if symbol in exchange.markets:
        market = exchange.markets[symbol]
        print(f"--- Detalhes do Contrato: {symbol} ---")
        print(f"ID: {market['id']}")
        print(f"Type: {market['type']}")
        print(f"Linear: {market.get('linear')}")
        print(f"Inverse: {market.get('inverse')}")
        print(f"Contract Size: {market.get('contractSize')}")
        print(f"Spot: {market.get('spot')}")
        print(f"Margin: {market.get('margin')}")
        print(f"Future: {market.get('future')}")
        print(f"Swap: {market.get('swap')}")
        print(f"Base: {market.get('base')}")
        print(f"Quote: {market.get('quote')}")
        print(f"Settle: {market.get('settle')}")
    else:
        print(f"Símbolo {symbol} não encontrado.")
        print("Símbolos disponíveis (Amostra):", list(exchange.markets.keys())[:10])

if __name__ == "__main__":
    check_contract()
