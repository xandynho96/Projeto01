@echo off
echo Building BitcoinAI Executable (Optimized)...
echo.
echo Make sure you are NOT running this inside VS Code terminal if you have low RAM.
echo.
python -m PyInstaller BitcoinAI_optimized.spec --clean
echo.
echo Build complete. Check dist/ folder.

