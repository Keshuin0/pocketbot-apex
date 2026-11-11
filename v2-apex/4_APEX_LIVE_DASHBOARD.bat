@echo off
setlocal
title PocketBot Apex - Live TUI Cockpit Dashboard (shadow-ab)
cd /d "%~dp0"

if not exist "%~dp0.venv\Scripts\python.exe" goto :no_venv

echo ============================================================================
echo   POCKETBOT APEX -- Walk-Forward Shadow A/B ^& Live Rich Cockpit Dashboard
echo ============================================================================
"%~dp0.venv\Scripts\python.exe" "%~dp0apex_runner.py" --mode shadow-ab --dashboard %*
pause
exit /b 0

:no_venv
echo ============================================================================
echo   [ERROR] PocketBot Apex Virtual Environment .venv Not Found!
echo ============================================================================
echo   Please double-click 0_FIRST_TIME_SETUP.bat first to set up Python and
echo   all required dependencies automatically on your computer.
echo ============================================================================
echo.
pause
exit /b 1
