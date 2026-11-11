@echo off
setlocal
title PocketBot Apex - Autonomous DEMO Trading (demo-auto)
cd /d "%~dp0"

if not exist "%~dp0.venv\Scripts\python.exe" goto :no_venv

echo ============================================================================
echo   POCKETBOT APEX -- Autonomous DEMO Execution Mode (--mode demo-auto)
echo ============================================================================
"%~dp0.venv\Scripts\python.exe" "%~dp0apex_runner.py" --mode demo-auto --dashboard %*
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
