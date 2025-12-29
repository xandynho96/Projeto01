import script_utils as su
# su.setup_tensorflow_gpu() # Uncomment if GPU needed and available, handles dlls

from trader import BitcoinTrader
import time
import sys

# Hook to ensure dependencies are loaded for PyInstaller
import pandas
import numpy
import ta
import ccxt
import tensorflow
import sqlalchemy
import schedule

def main():
    print("Starting Bitcoin AI Trader (Executable Mode)...")
    try:
        bot = BitcoinTrader()
        bot.run()
    except KeyboardInterrupt:
        print("Stopping bot...")
    except Exception as e:
        print(f"Critical Error: {e}")
        input("Press Enter to exit...") # Keep window open on error

if __name__ == "__main__":
    main()
