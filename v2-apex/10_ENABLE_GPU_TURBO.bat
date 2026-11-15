@echo off
setlocal
title PocketBot Apex - Enable Hyperion GPU & Numba LLVM Turbo
cd /d "%~dp0"
color 0b

echo ============================================================================
echo   POCKETBOT APEX v2.7.0 -- HYPERION GPU ^& NUMBA LLVM TURBO SETUP
echo ============================================================================
echo   Auto-detecting NVIDIA CUDA / DirectML GPU / Multi-Core Numba LLVM AVX...
echo.

if not exist "%~dp0.venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment .venv not found. Run 0_FIRST_TIME_SETUP.bat first!
    pause
    exit /b 1
)

"%~dp0.venv\Scripts\python.exe" "%~dp0tools\setup_gpu_acceleration.py"
echo.
pause
exit /b 0
