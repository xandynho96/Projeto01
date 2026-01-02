@echo off
echo Building BitcoinAI Executable...
python -m PyInstaller --onefile --clean --name BitcoinAI_v1 --add-data "bitcoin_ai_model.keras;." --add-data "scaler.pkl;." --add-data "bitcoin_ai_classifier.keras;." --collect-all tensorflow --collect-all pandas --collect-all numpy --collect-all ta --collect-all ccxt --collect-all yfinance --hidden-import="sqlalchemy.sql.default_comparator" --hidden-import="pkg_resources.py2_warn" launcher.py
echo Build complete. Check dist/ folder.
pause
