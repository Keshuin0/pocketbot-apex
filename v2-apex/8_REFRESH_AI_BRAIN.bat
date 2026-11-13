@echo off
setlocal
title PocketBot Apex - Refresh AI Brain (Sleep-Wake Consolidation)
cd /d "%~dp0"
color 0a

echo ============================================================================
echo   POCKETBOT APEX -- CIRCADIAN SLEEP-WAKE MEMORY CONSOLIDATION
echo ============================================================================
echo.
"%~dp0.venv\Scripts\python.exe" "%~dp0tools\run_sleep_cycle.py"
echo.
pause
