import script_utils as su
# su.setup_tensorflow_gpu() # Uncomment if GPU needed and available, handles dlls

# from trader import BitcoinTrader # Moved inside main/try block for safety
import time
import sys
import logging
import os

# --- Setup Logging for Debugging ---
log_file = "launcher_debug.txt"
logging.basicConfig(
    filename=log_file,
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_print(msg):
    print(msg)
    logging.info(msg)

log_print("=== LAUNCHER STARTED ===")
log_print(f"Current Directory: {os.getcwd()}")
# Hook to ensure dependencies are loaded for PyInstaller (Analysis time only)
def _pyinstaller_hooks():
    import pandas
    import numpy
    import ta
    import ccxt
    import tensorflow
    import sqlalchemy
    import schedule

log_print("Launcher initialized.")

def main():
    while True:
        print("\n=== BITCOIN AI TRADER ===")
        print("1. Iniciar Robô de Trading (Live)")
        print("2. Iniciar Otimização Contínua (Treino/Backtest)")
        print("3. Sair")
        
        choice = input("Escolha uma opção: ")
        
        if choice == '1':
            log_print("Starting Bitcoin AI Trader (Executable Mode)...")
            try:
                from trader import BitcoinTrader
                bot = BitcoinTrader()
                bot.run()
            except KeyboardInterrupt:
                log_print("Stopping bot...")
            except Exception as e:
                log_print(f"Critical Error in Trader: {e}")
                # Print full traceback to log
                import traceback
                logging.error(traceback.format_exc())
                input("Press Enter to continue...")
                
        elif choice == '2':
            log_print("Starting Optimization Loop...")
            try:
                from continuous_optimizer import ContinuousOptimizer
                opt = ContinuousOptimizer()
                # Run content directly
                opt.start()
            except KeyboardInterrupt:
                log_print("Stopping optimization...")
            except Exception as e:
                log_print(f"Error in optimization: {e}")
                import traceback
                logging.error(traceback.format_exc())
                input("Press Enter to continue...")
                
        elif choice == '3':
            log_print("Saindo...")
            sys.exit()
        else:
            print("Opção inválida!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_print(f"FATAL ERROR in main loop: {e}")
        import traceback
        logging.error(traceback.format_exc())
        input("Press Enter to crash gracefully...")
