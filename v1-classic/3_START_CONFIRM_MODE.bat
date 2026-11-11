@echo off
title PocketBot MT5 - Interactive Confirm Mode (YES/NO)
cd /d "%~dp0"

if not exist "%~dp0.venv\Scripts\python.exe" goto :no_venv

echo ============================================================================
echo   POCKETBOT MT5 - INTERACTIVE CONFIRM MODE (--mode confirm)
echo   - M5 Tight-Spread  : EURUSD, GBPUSD, USDJPY, XAUUSD, SPX, NQ
echo   - M15 Momentum     : BTCUSD, ETHUSD, #TSLA, #NVDA
echo   - Prompts YES/NO before sending any order to MetaTrader 5
echo   - Press Ctrl+C at any time to stop scanning.
echo ============================================================================
echo.

call "%~dp0.venv\Scripts\activate.bat"
"%~dp0.venv\Scripts\python.exe" "%~dp0pocketbot_auto.py" --mode confirm

echo.
pause


goto :eof

:no_venv
echo ============================================================================
echo   [!] FIRST-TIME SETUP REQUIRED
echo   Please double-click "1_FIRST_TIME_SETUP.bat" once before starting PocketBot.
echo ============================================================================
echo.
pause
exit /b 1
