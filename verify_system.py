from trader import BitcoinTrader
import sys

def test_system():
    print("Starting system verification...")
    try:
        # Initialize
        bot = BitcoinTrader()
        
        # Run one job cycle
        print("Running single job cycle...")
        bot.job()
        
        print("\nSUCCESS: System performed analysis cycle without errors.")
        return True
    except Exception as e:
        print(f"\nFAILURE: System verification failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_system()
    if not success:
        sys.exit(1)
