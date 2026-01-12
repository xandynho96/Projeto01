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
        
        # Data files
        '--add-data=dashboard.py;.', 
        '--add-data=crypto_data.db;.', 
        '--add-data=bitcoin_ai_model.pkl;.', 
        '--add-data=scaler.pkl;.', 
        '--add-data=app_icon.png;.',

        # Force Include Source Files (Fixes ModuleNotFoundError)
        '--add-data=ai_brain.py;.',
        '--add-data=trader.py;.',
        '--add-data=technical_analysis.py;.',
        '--add-data=data_manager.py;.',
        '--add-data=config.py;.',
        '--add-data=logger.py;.',
    ]
    
    # Run PyInstaller
    PyInstaller.__main__.run(args)
    
    print("Build Complete.")
    print(f"Executable is located in: {os.path.abspath('dist')}")
    
    # Instructions for the user
    print("\nIMPORTANT: Copy .env and crypto_data.db to the dist folder before running if they are not included/bundled correctly for your specific use case (e.g. if you want them editable outside the exe).")

if __name__ == "__main__":
    build_executable()
