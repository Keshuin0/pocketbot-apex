@echo off
setlocal EnableDelayedExpansion
title PocketBot Apex - 1-Click First-Time Setup Wizard
cd /d "%~dp0"
color 0b

echo ============================================================================
echo   POCKETBOT APEX -- CHILD-PROOF 1-CLICK INSTALLER ^& SETUP WIZARD
echo ============================================================================
echo   This script will automatically set up Python 3.12, create your private
echo   isolated environment, install all institutional quant libraries, run 25
echo   automated safety tests, and connect to MetaTrader 5.
echo ============================================================================
echo.

:: -----------------------------------------------------------------------------
:: STEP A: Verify 64-Bit Windows
:: -----------------------------------------------------------------------------
echo [Step A/F] Checking Windows Operating System architecture...
set "IS_64BIT=0"
if "%PROCESSOR_ARCHITECTURE%"=="AMD64" set "IS_64BIT=1"
if "%PROCESSOR_ARCHITECTURE%"=="ARM64" set "IS_64BIT=1"
if defined PROCESSOR_ARCHITEW6432 set "IS_64BIT=1"

if "%IS_64BIT%"=="0" goto :fail_32bit
echo   [OK] 64-bit Windows verified.
echo.

:: -----------------------------------------------------------------------------
:: STEP B: Locate or Automatically Install Python 3.12 x64
:: -----------------------------------------------------------------------------
echo [Step B/F] Locating 64-bit Python 3.12...
set "PYTHON_EXE="

if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
)

if not defined PYTHON_EXE (
    if exist "C:\Program Files\Python312\python.exe" (
        set "PYTHON_EXE=C:\Program Files\Python312\python.exe"
    )
)

if not defined PYTHON_EXE (
    py -3.12 -c "import sys; sys.exit(0 if sys.maxsize > 2**32 else 1)" >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        for /f "delims=" %%I in ('py -3.12 -c "import sys; print(sys.executable)"') do set "PYTHON_EXE=%%I"
    )
)

if not defined PYTHON_EXE (
    python -c "import sys; sys.exit(0 if (sys.version_info[:2]==(3,12) and sys.maxsize > 2**32) else 1)" >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        for /f "delims=" %%I in ('python -c "import sys; print(sys.executable)"') do set "PYTHON_EXE=%%I"
    )
)

if defined PYTHON_EXE goto :python_found

echo   Python 3.12 64-bit was not found on your system.
echo   Downloading and installing official Python 3.12 via Windows Package Manager winget...
echo   Please wait 30-60 seconds...
echo.
winget install -e --id Python.Python.3.12 --architecture x64 --accept-package-agreements --accept-source-agreements
if !ERRORLEVEL! NEQ 0 goto :fail_winget

if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
) else if exist "C:\Program Files\Python312\python.exe" (
    set "PYTHON_EXE=C:\Program Files\Python312\python.exe"
)

if not defined PYTHON_EXE goto :fail_python_locate

:python_found
echo   [OK] Found Python 3.12: "!PYTHON_EXE!"
echo.

:: -----------------------------------------------------------------------------
:: STEP C: Create Fresh Isolated Local Virtual Environment (.venv)
:: -----------------------------------------------------------------------------
echo [Step C/F] Setting up isolated local virtual environment in .venv...
if exist "%~dp0.venv\Scripts\python.exe" goto :venv_exists

echo   Creating local virtual environment .venv...
"!PYTHON_EXE!" -m venv "%~dp0.venv"
if !ERRORLEVEL! NEQ 0 goto :fail_venv

:venv_exists
echo   Upgrading pip in .venv...
"%~dp0.venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
echo   [OK] Virtual environment ready.
echo.

:: -----------------------------------------------------------------------------
:: STEP D: Install Required Quantitative, Broker and AI Libraries
:: -----------------------------------------------------------------------------
echo [Step D/F] Installing required packages into .venv...
echo   Installing MetaTrader5, numpy, scipy, ctrader-open-api, twisted, pyOpenSSL,
echo   service_identity, aiohttp, python-dotenv, rich, pytest...
echo.
"%~dp0.venv\Scripts\python.exe" -m pip install MetaTrader5 numpy scipy ctrader-open-api twisted pyOpenSSL service_identity aiohttp python-dotenv rich pytest --quiet
if !ERRORLEVEL! NEQ 0 goto :fail_packages
echo   [OK] All institutional packages successfully installed.
echo.

:: -----------------------------------------------------------------------------
:: STEP E: Run Full 25-Test Automated Verification Suite
:: -----------------------------------------------------------------------------
echo [Step E/F] Running full 25-test institutional safety and quant verification suite...
echo.
"%~dp0.venv\Scripts\python.exe" -m pytest "%~dp0tests" -q
if !ERRORLEVEL! NEQ 0 goto :fail_tests
echo   [OK] All 25 safety, quant, AI-waterfall, and Hydra-mesh tests PASSED!
echo.

:: -----------------------------------------------------------------------------
:: STEP F: Probe Live MetaTrader 5 Connection and Demo Telemetry
:: -----------------------------------------------------------------------------
echo [Step F/F] Probing MetaTrader 5 IPC connection...
echo.
"%~dp0.venv\Scripts\python.exe" "%~dp0tools\verify_first_time_setup.py"
echo.
echo ============================================================================
echo   SETUP COMPLETE! PocketBot Apex is ready on this PC.
echo   Press any key to close this setup window.
echo ============================================================================
pause
exit /b 0

:fail_32bit
echo [ERROR] PocketBot Apex requires a 64-bit version of Windows.
echo 32-bit Windows is not supported by MetaTrader 5 or quantitative C extensions.
goto :fail

:fail_winget
echo [ERROR] Automatic winget installation failed or was cancelled.
echo Please manually download and install Python 3.12 64-bit from:
echo   https://www.python.org/downloads/release/python-31210/
echo Make sure to check Add Python to PATH during installation.
goto :fail

:fail_python_locate
echo [ERROR] Could not resolve python.exe location after install.
echo Please launch Python from your Start Menu once or add it to PATH.
goto :fail

:fail_venv
echo [ERROR] Failed to create .venv virtual environment.
goto :fail

:fail_packages
echo [ERROR] Failed to install one or more required packages.
echo Check your internet connection and retry.
goto :fail

:fail_tests
echo [ERROR] One or more verification tests failed!
echo Please inspect the test output above.
goto :fail

:fail
echo.
echo ============================================================================
echo   [SETUP FAILED] Please review the error message above.
echo   For detailed troubleshooting, open START_HERE_OPEN_ME.html in your browser.
echo ============================================================================
pause
exit /b 1
