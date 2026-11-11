@echo off
title PocketBot MT5 - Live Account & Performance Stats
cd /d "%~dp0"

if not exist "%~dp0.venv\Scripts\python.exe" goto :no_venv

call "%~dp0.venv\Scripts\activate.bat"
"%~dp0.venv\Scripts\python.exe" "%~dp0stats.py"

echo.
pause


goto :eof

:no_venv
echo ============================================================================
echo   [!] FIRST-TIME SETUP REQUIRED
echo   Please double-click "1_FIRST_TIME_SETUP.bat" once before checking stats.
echo ============================================================================
echo.
pause
exit /b 1
