@echo off
echo Building BitcoinAI Executable...
python -m PyInstaller --onefile --clean --name BitcoinAI_v1 --add-data "bitcoin_ai_model.keras;." --collect-all tensorflow --collect-all pandas --collect-all numpy --collect-all ta --collect-all ccxt --collect-all yfinance launcher.py
echo Build complete. Check dist/ folder.
pause
