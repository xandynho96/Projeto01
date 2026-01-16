# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main_launcher.py'],
    pathex=['C:\\Users\\Alexandre\\Documents\\Testes\\Projeto01'],
    binaries=[],
    datas=[('app/dashboard.py', 'app'), ('app_icon.png', '.'), ('app', 'app')],
    hiddenimports=['pandas', 'numpy', 'sklearn', 'sklearn.utils._cython_blas', 'sklearn.neighbors.typedefs', 'sklearn.neighbors.quad_tree', 'sklearn.tree', 'sklearn.tree._utils', 'ta', 'ccxt', 'sqlalchemy', 'schedule', 'plotly', 'streamlit', 'custom_strategies', 'app.trader', 'app.core.ai_brain', 'app.core.technical_analysis', 'app.utils.logger', 'app.core.data_manager', 'app.utils.config', 'app.core.evolution'],
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
