@echo off
title PocketBot Apex - 60s Live 0.01-Lot MT5 Demo Lifecycle Drill
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] .venv not found. Run 0_FIRST_TIME_SETUP.bat first.
    pause
    exit /b 1
)
echo Running 60s Live 0.01-Lot MT5 Demo Lifecycle Drill...
".venv\Scripts\python.exe" tools\live_lifecycle_drill.py --hold-seconds 15 %*
echo.
pause
