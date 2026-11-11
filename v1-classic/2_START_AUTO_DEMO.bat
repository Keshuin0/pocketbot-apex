@echo off
title PocketBot MT5 - Autonomous Demo Mode (demo-auto)
cd /d "%~dp0"

if not exist "%~dp0.venv\Scripts\python.exe" goto :no_venv

echo ============================================================================
echo   POCKETBOT MT5 - AUTONOMOUS DEMO MODE (--mode demo-auto)
echo   - M5 Tight-Spread  : EURUSD, GBPUSD, USDJPY, XAUUSD, SPX, NQ
echo   - M15 Momentum     : BTCUSD, ETHUSD, #TSLA, #NVDA
echo   - Max Risk / Trade : 0.5%% ($50.00) ^| Daily Kill-Switch: -2.0%% (-$200.00)
echo   - Press Ctrl+C at any time to stop scanning.
echo ============================================================================
echo.

call "%~dp0.venv\Scripts\activate.bat"
"%~dp0.venv\Scripts\python.exe" "%~dp0pocketbot_auto.py" --mode demo-auto

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [!] PocketBot stopped with status code %ERRORLEVEL%.
)
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
