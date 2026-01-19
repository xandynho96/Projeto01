import PyInstaller.__main__
import os
import shutil

def build_executable():
    print("Starting Build Process...")
    
    # Define the main script
    main_script = 'main_launcher.py'
    executable_name = 'BitcoinAI'
    
    # Backup existing database to prevent data loss
    db_backup_path = None
    existing_db = os.path.join('dist', 'data', 'crypto_data.db')
    if os.path.exists(existing_db):
        print(f"🛡️  Preserving existing database found at: {existing_db}")
        db_backup_path = 'crypto_data.db.backup'
        try:
            shutil.copy2(existing_db, db_backup_path)
        except Exception as e:
            print(f"Warning: Could not backup DB: {e}")
            db_backup_path = None

    # Clean previous builds
    if os.path.exists('build'):
        shutil.rmtree('build')
    if os.path.exists('dist'):
        try:
            shutil.rmtree('dist')
        except:
             print("Warning: Could not clean dist folder.")

    # PyInstaller arguments
    args = [
        main_script,
        '--name=%s' % executable_name,
        '--onefile',       # Create a single executable
        '--windowed',      # No console window
        '--clean',
        '--paths=%s' % os.getcwd(),
        '--distpath=dist',
        
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
    
    # Restore preserved database if it existed
    if db_backup_path and os.path.exists(db_backup_path):
        target_db_dir = os.path.join(dist_dir, 'data')
        if not os.path.exists(target_db_dir):
            os.makedirs(target_db_dir)
        target_db_path = os.path.join(target_db_dir, 'crypto_data.db')
        try:
            shutil.move(db_backup_path, target_db_path)
            print(f"✅ Restored preserved database to: {target_db_path}")
        except Exception as e:
            print(f"❌ Failed to restore database: {e}")

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
                
                # Check for existing DB (from restore)
                if 'crypto_data.db' in f and os.path.exists(dest_path):
                    print(f"   [!] Skipping {f} (Preserved DB exists)")
                    continue
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
