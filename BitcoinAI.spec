# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['gui_app.py'],
    pathex=['C:\\Users\\Alexandre\\Documents\\Testes\\Projeto01'],
    binaries=[],
    datas=[('dashboard.py', '.'), ('crypto_data.db', '.'), ('bitcoin_ai_model.pkl', '.'), ('scaler.pkl', '.'), ('app_icon.png', '.'), ('ai_brain.py', '.'), ('trader.py', '.'), ('technical_analysis.py', '.'), ('data_manager.py', '.'), ('config.py', '.'), ('logger.py', '.')],
    hiddenimports=['pandas', 'numpy', 'sklearn', 'sklearn.utils._cython_blas', 'sklearn.neighbors.typedefs', 'sklearn.neighbors.quad_tree', 'sklearn.tree', 'sklearn.tree._utils', 'ta', 'ccxt', 'sqlalchemy', 'schedule', 'plotly', 'streamlit', 'custom_strategies', 'trader', 'ai_brain', 'technical_analysis', 'logger', 'data_manager', 'config'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='BitcoinAI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
