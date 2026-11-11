@echo off
setlocal EnableDelayedExpansion
title PocketBot MT5 - First Time Setup (Run Once)
cd /d "%~dp0"

echo ==============================================================================
echo   POCKETBOT MT5 - 1-CLICK FIRST-TIME SETUP
echo   Working Folder: "%~dp0"
echo ==============================================================================
echo.

set "PY_EXE="

echo [Step 1/4] Checking for 64-bit Python 3.12 (or compatible 64-bit Python)...

:: 1. Check Python Launcher (py.exe) for 64-bit Python 3.12 first
where py >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    py -3.12-64 -c "import sys, struct; assert struct.calcsize('P')*8 == 64" >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        for /f "delims=" %%I in ('py -3.12-64 -c "import sys; print(sys.executable)"') do set "PY_EXE=%%I"
    )
)

:: 2. Check standard Windows user/system install locations for Python 3.12 (64-bit)
if not defined PY_EXE (
    if exist "%LocalAppData%\Programs\Python\Python312\python.exe" (
        set "PY_EXE=%LocalAppData%\Programs\Python\Python312\python.exe"
    ) else if exist "%ProgramFiles%\Python312\python.exe" (
        set "PY_EXE=%ProgramFiles%\Python312\python.exe"
    )
)

:: 3. If Python 3.12 is not installed, auto-install Python.Python.3.12 via winget
if not defined PY_EXE (
    echo   64-bit Python 3.12 not found yet. Attempting automatic install via winget...
    where winget >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
        if exist "%LocalAppData%\Programs\Python\Python312\python.exe" (
            set "PY_EXE=%LocalAppData%\Programs\Python\Python312\python.exe"
        ) else if exist "%ProgramFiles%\Python312\python.exe" (
            set "PY_EXE=%ProgramFiles%\Python312\python.exe"
        )
    )
)

:: 4. Fallback: check any existing 64-bit Python 3.10 - 3.13 on the PC if 3.12 wasn't found
if not defined PY_EXE (
    for %%V in (312 313 311 310) do (
        if not defined PY_EXE if exist "%LocalAppData%\Programs\Python\Python%%V\python.exe" (
            set "PY_EXE=%LocalAppData%\Programs\Python\Python%%V\python.exe"
        )
        if not defined PY_EXE if exist "%ProgramFiles%\Python%%V\python.exe" (
            set "PY_EXE=%ProgramFiles%\Python%%V\python.exe"
        )
    )
)
if not defined PY_EXE (
    where python >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        python -c "import sys, struct; assert sys.version_info >= (3, 10) and struct.calcsize('P')*8 == 64" >nul 2>&1
        if !ERRORLEVEL! EQU 0 (
            for /f "delims=" %%I in ('python -c "import sys; print(sys.executable)"') do set "PY_EXE=%%I"
        )
    )
)

if not defined PY_EXE (
    echo.
    echo [!] ERROR: Could not find or auto-install 64-bit Python 3.12.
    echo     Please install Python 3.12 (64-bit) from https://www.python.org/downloads/
    echo     (Check the box "Add python.exe to PATH" during installation), then run this file again.
    echo.
    pause
    exit /b 1
)

echo   [OK] Using 64-bit Python: "!PY_EXE!"
echo.

echo [Step 2/4] Creating fresh local virtual environment (.venv) inside "%~dp0.venv"...
if exist "%~dp0.venv" (
    rmdir /s /q "%~dp0.venv"
)
"!PY_EXE!" -m venv "%~dp0.venv"
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo.
    echo [!] ERROR: Failed to create local .venv folder.
    pause
    exit /b 1
)
echo   [OK] Created "%~dp0.venv"
echo.

echo [Step 3/4] Installing MetaTrader5 and numpy packages...
"%~dp0.venv\Scripts\python.exe" -m pip install --upgrade pip
"%~dp0.venv\Scripts\python.exe" -m pip install MetaTrader5 numpy
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [!] ERROR: Failed to install MetaTrader5 and numpy. Check your internet connection and try again.
    pause
    exit /b 1
)
echo.

"%~dp0.venv\Scripts\python.exe" "%~dp0check_mt5_setup.py"
echo.
pause
