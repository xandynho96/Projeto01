
import sys
import os
import time
import json

# Add root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.data_manager import DataManager
from app.utils import config

def debug_balance():
    print("💰 DEBUGGANDO SALDO KRAKEN (FUTURES)...")
    dm = DataManager()
    
    # print("💰 DEBUGGANDO SALDO KRAKEN (FUTURES - SPOT CONFIG)...")
    # dm = DataManager()
    
    # # Load keys from user_config.json (Spot keys)
    # try:
    #     with open('data/user_config.json', 'r') as f:
    #         cfg = json.load(f)
    #         key = cfg.get('api_key')
    #         secret = cfg.get('secret')
    #         mode = "Futures (50x)" # Force Futures
            
    #     print(f"Loading config from SPOT: Mode={mode}")
    #     dm.connect_exchange(key, secret, trading_mode=mode)
        
    # except Exception as ex:
    #     print(f"❌ Failed to load SPOT config: {ex}")
    #     return
    
    # if not dm.exchange:
    #     print("❌ Exchange não conectado.")
    #     return

    try:
        print("Fetching Balance Raw...")
        balance = dm.exchange.fetch_balance()
        
        print("\n--- BALANCE RAW STRUCTURE (KEYS) ---")
        print(balance.keys())
        
        print("\n--- 'total' KEYS (FULL DUMP) ---")
        if 'total' in balance:
            print(json.dumps(balance['total'], indent=2))
        else:
            print("No 'total' key found.")

        print("\n--- 'free' KEYS ---")
        if 'free' in balance:
            print(json.dumps(balance['free'], indent=2))
        else:
            print("No 'free' key found.")
            
        print("\n--- SPECIFIC VALUES ---")
        print(f"USD (Total): {balance.get('total', {}).get('USD', 'N/A')}")
        print(f"USD (Free): {balance.get('free', {}).get('USD', 'N/A')}")
        print(f"ZUSD: {balance.get('total', {}).get('ZUSD', 'N/A')}")
        print(f"USDT: {balance.get('total', {}).get('USDT', 'N/A')}")
        print(f"PF_USD: {balance.get('total', {}).get('PF_USD', 'N/A')}")
        
        # Check Info/Flex
        print("\n--- INFO / FLEX CHECK ---")
        if 'info' in balance and 'accounts' in balance['info']:
            accounts = balance['info']['accounts']
            print("Accounts keys:", accounts.keys())
            if 'flex' in accounts:
                print("Flex Wallet:", accounts['flex'])
            else:
                 print("No Flex wallet in accounts.")
        
        # Check for Futures specific keys in top level
        print(f"\n--- FUTURES KEYS CHECK ---")
        print(f"PF_USD: {balance.get('total', {}).get('PF_USD', 'N/A')}")
        print(f"PF_XBT/USD: {balance.get('total', {}).get('PF_XBTUSD', 'N/A')}")
        print(f"USDT.B: {balance.get('total', {}).get('USDT.B', 'N/A')}")

        print("\n--- DATA MANAGER CALCULATION ---")
        # Force a re-calc or inspect the loop
        calc_free = dm.get_balance('free')
        calc_total = dm.get_balance('total')
        print(f"DM Calculated Free: {calc_free}")
        print(f"DM Calculated Total: {calc_total}")
        
        print(f"\n--- MODE: {dm.user_settings.get('trading_mode', 'Unknown')} ---")

        
    except Exception as e:
        print(f"❌ Erro ao buscar saldo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_balance()
