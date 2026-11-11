@echo off
setlocal
title PocketBot Apex - Configure 3-Tier AI Router
cd /d "%~dp0"

if not exist "%~dp0.venv\Scripts\python.exe" goto :no_venv

echo ============================================================================
echo   POCKETBOT APEX -- Interactive 3-Tier AI Mode ^& Key Switcher (Hot-Reload)
echo ============================================================================
"%~dp0.venv\Scripts\python.exe" "%~dp0tools\configure_ai_mode.py" %*
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
