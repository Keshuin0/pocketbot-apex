@echo off
setlocal EnableDelayedExpansion
title PocketBot Apex - GitHub Live Updater
cd /d "%~dp0"

echo ==============================================================================
echo   POCKETBOT APEX - AUTOMATIC UPDATE FROM GITHUB
echo   Repository: Keshuin0/pocketbot-apex
echo   Target Folder: "%~dp0"
echo ==============================================================================
echo.

where git >nul 2>&1
if !ERRORLEVEL! EQU 0 (
    if exist "%~dp0..\.git" (
        echo [Git Mode] Local repository detected. Pulling latest commits...
        pushd "%~dp0.."
        git pull --rebase origin main
        popd
        echo.
        echo [OK] Successfully updated via git pull!
        goto :done
    )
    if exist "%~dp0.git" (
        echo [Git Mode] Local repository detected. Pulling latest commits...
        git pull --rebase origin main
        echo.
        echo [OK] Successfully updated via git pull!
        goto :done
    )
)

echo [ZIP Mode] Downloading latest source release from GitHub...
set "TEMP_ZIP=%TEMP%\pocketbot_apex_latest.zip"
set "TEMP_DIR=%TEMP%\pocketbot_apex_update_%RANDOM%"

powershell -NoProfile -Command ^
    "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; " ^
    "Write-Host 'Downloading https://github.com/Keshuin0/pocketbot-apex/archive/refs/heads/main.zip ...'; " ^
    "Invoke-WebRequest -Uri 'https://github.com/Keshuin0/pocketbot-apex/archive/refs/heads/main.zip' -OutFile '%TEMP_ZIP%';"
if %ERRORLEVEL% NEQ 0 (
    echo [!] ERROR: Failed to download update zip from GitHub.
    goto :error
)

echo [ZIP Mode] Extracting archive...
powershell -NoProfile -Command ^
    "Expand-Archive -Path '%TEMP_ZIP%' -DestinationPath '%TEMP_DIR%' -Force"
if %ERRORLEVEL% NEQ 0 (
    echo [!] ERROR: Failed to extract update zip.
    goto :error
)

echo [ZIP Mode] Syncing updated files into "%~dp0" (strictly preserving .env, .venv, and databases)...
powershell -NoProfile -Command ^
    "$src = Join-Path '%TEMP_DIR%' 'pocketbot-apex-main\v2-apex'; " ^
    "$dst = '%~dp0'; " ^
    "$excludes = @('.env', '.venv', 'trader', 'apex_ledger.db', 'sor_ledger.db', 'test_ledger.db', '__pycache__', '.pytest_cache'); " ^
    "Get-ChildItem -Path $src -Recurse | ForEach-Object { " ^
    "    $rel = $_.FullName.Substring($src.Length).TrimStart('\'); " ^
    "    $skip = $false; " ^
    "    foreach ($exc in $excludes) { if ($rel -eq $exc -or $rel.StartsWith($exc + '\')) { $skip = $true; break } }; " ^
    "    if (-not $skip) { " ^
    "        $targetPath = Join-Path $dst $rel; " ^
    "        if ($_.PSIsContainer) { " ^
    "            if (-not (Test-Path $targetPath)) { New-Item -ItemType Directory -Path $targetPath -Force | Out-Null } " ^
    "        } else { " ^
    "            Copy-Item -Path $_.FullName -Destination $targetPath -Force; " ^
    "        } " ^
    "    } " ^
    "}"

if exist "%TEMP_ZIP%" del /f /q "%TEMP_ZIP%" >nul 2>&1
if exist "%TEMP_DIR%" rmdir /s /q "%TEMP_DIR%" >nul 2>&1

echo.
echo ==============================================================================
echo   [OK] UPDATE COMPLETE!
echo   Your code files and configurations have been synced to latest GitHub main.
echo   Your local .env credentials, .venv runtime, and trade database were kept 100%% safe.
echo ==============================================================================
goto :done

:error
echo.
echo [!] Update failed. Please check your internet connection or download manually from:
echo     https://github.com/Keshuin0/pocketbot-apex
if exist "%TEMP_ZIP%" del /f /q "%TEMP_ZIP%" >nul 2>&1
if exist "%TEMP_DIR%" rmdir /s /q "%TEMP_DIR%" >nul 2>&1

:done
echo.
pause
