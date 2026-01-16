import PyInstaller.__main__
import os
import shutil

def build_executable():
    print("Starting Build Process...")
    
    # Define the main script
    main_script = 'main_launcher.py'
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
        # Explicit Local Modules (Fixes ModuleNotFoundError)
        '--hidden-import=app.trader',
        '--hidden-import=app.core.ai_brain',
        '--hidden-import=app.core.technical_analysis',
        '--hidden-import=app.utils.logger',
        '--hidden-import=app.core.data_manager',
        '--hidden-import=app.utils.config',
        '--hidden-import=app.core.evolution',
        
        # Data files
        '--add-data=app/dashboard.py;app', 
        # '--add-data=crypto_data.db;.',  <-- REMOVED: Do not bundle DB inside Exe. Use external file.
        # '--add-data=bitcoin_ai_model.pkl;.', <-- REMOVED: Model should be external for persistence
        # '--add-data=scaler.pkl;.',           <-- REMOVED: Scaler should be external
        '--add-data=app_icon.png;.',

        # Force Include Source Files (Fixes ModuleNotFoundError)
        '--add-data=app;app',
    ]
    
    # Run PyInstaller
    PyInstaller.__main__.run(args)
    
    print("Build Complete.")
    dist_dir = os.path.abspath('dist')
    print(f"Executable is located in: {dist_dir}")
    
    # Post-Build: Copy external resources to dist folder
    print("Copying external resources (DB, Models, Config) to dist/...")
    
    files_to_copy = [
        os.path.join('data', 'crypto_data.db'),
        os.path.join('config', '.env'), 
        os.path.join('config', 'user_config.json'),
        os.path.join('data', 'models', 'bitcoin_ai_model.pkl'),
        os.path.join('data', 'models', 'scaler.pkl')
    ]
    
    for f in files_to_copy:
        if os.path.exists(f):
            try:
                # Maintain folder structure
                dest_path = os.path.join(dist_dir, f)
                dest_dir = os.path.dirname(dest_path)
                
                if not os.path.exists(dest_dir):
                    os.makedirs(dest_dir)
                    
                shutil.copy(f, dest_path)
                print(f"   [+] Copied {f} -> {dest_path}")
            except Exception as e:
                print(f"   [!] Failed to copy {f}: {e}")
        else:
             print(f"   [-] Skipped {f} (Not found)")

if __name__ == "__main__":
    build_executable()
