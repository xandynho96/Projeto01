import PyInstaller.__main__
import os
import shutil

def build_executable():
    print("Starting Build Process...")
    
    # Define the main script
    main_script = 'gui_app.py'
    executable_name = 'BitcoinAI'
    
    # Clean previous builds
    if os.path.exists('build'):
        shutil.rmtree('build')
    if os.path.exists('dist'):
        shutil.rmtree('dist')

    # PyInstaller arguments
    args = [
        main_script,
        '--name=%s' % executable_name,
        '--onefile',       # Create a single executable
        '--windowed',      # No console window
        '--clean',
        '--paths=%s' % os.getcwd(),
        
        # Hidden Imports (Crucial for pandas, sklearn, tensorflow, etc.)
        '--hidden-import=pandas',
        '--hidden-import=numpy',
        '--hidden-import=sklearn',
        '--hidden-import=sklearn.utils._cython_blas',
        '--hidden-import=sklearn.neighbors.typedefs',
        '--hidden-import=sklearn.neighbors.quad_tree',
        '--hidden-import=sklearn.tree',
        '--hidden-import=sklearn.tree._utils',
        # '--hidden-import=tensorflow', # TensorFlow not available on Py3.14
        '--hidden-import=ta',
        '--hidden-import=ccxt',
        '--hidden-import=sqlalchemy',
        '--hidden-import=schedule',
        '--hidden-import=plotly',
        '--hidden-import=streamlit',
        '--hidden-import=custom_strategies', # If any
        
        # Explicit Local Modules (Fixes ModuleNotFoundError)
        '--hidden-import=trader',
        '--hidden-import=ai_brain',
        '--hidden-import=technical_analysis',
        '--hidden-import=logger',
        '--hidden-import=data_manager',
        '--hidden-import=config',
        '--hidden-import=evolution',
        
        # Data files
        '--add-data=dashboard.py;.', 
        # '--add-data=crypto_data.db;.',  <-- REMOVED: Do not bundle DB inside Exe. Use external file.
        # '--add-data=bitcoin_ai_model.pkl;.', <-- REMOVED: Model should be external for persistence
        # '--add-data=scaler.pkl;.',           <-- REMOVED: Scaler should be external
        '--add-data=app_icon.png;.',

        # Force Include Source Files (Fixes ModuleNotFoundError)
        '--add-data=ai_brain.py;.',
        '--add-data=trader.py;.',
        '--add-data=technical_analysis.py;.',
        '--add-data=data_manager.py;.',
        '--add-data=config.py;.',
        '--add-data=logger.py;.',
        '--add-data=evolution.py;.',
    ]
    
    # Run PyInstaller
    PyInstaller.__main__.run(args)
    
    print("Build Complete.")
    dist_dir = os.path.abspath('dist')
    print(f"Executable is located in: {dist_dir}")
    
    # Post-Build: Copy external resources to dist folder
    print("Copying external resources (DB, Models, Config) to dist/...")
    
    files_to_copy = [
        'crypto_data.db',
        '.env', 
        'user_config.json',
        'bitcoin_ai_model.pkl',
        'scaler.pkl'
    ]
    
    for f in files_to_copy:
        if os.path.exists(f):
            try:
                shutil.copy(f, dist_dir)
                print(f"   [+] Copied {f}")
            except Exception as e:
                print(f"   [!] Failed to copy {f}: {e}")
        else:
             print(f"   [-] Skipped {f} (Not found)")

if __name__ == "__main__":
    build_executable()
