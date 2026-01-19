import os
import glob

def clean_models():
    paths = [
        "data/models/*.pkl",
        "dist/data/models/*.pkl"
    ]
    
    print("🧹 Cleaning old model files...")
    for pattern in paths:
        files = glob.glob(pattern)
        for f in files:
            try:
                os.remove(f)
                print(f"   Deleted: {f}")
            except Exception as e:
                print(f"   Error deleting {f}: {e}")
                
if __name__ == "__main__":
    clean_models()
