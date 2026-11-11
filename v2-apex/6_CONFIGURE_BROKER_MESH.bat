@echo off
setlocal
title PocketBot Apex - Configure Hydra Broker Mesh & SOR
cd /d "%~dp0"

if not exist "%~dp0.venv\Scripts\python.exe" goto :no_venv

echo ============================================================================
echo   POCKETBOT APEX -- Apex Hydra Dual-Venue Broker Mesh ^& SOR Configuration
echo ============================================================================
"%~dp0.venv\Scripts\python.exe" "%~dp0tools\configure_broker_mesh.py" %*
if not "%1"=="" exit /b %ERRORLEVEL%
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
