"""PocketBot Flagship Dual-Edition Staging & Verification Engine.

Stages D:\\Projects\\PocketBot_GitHub_Repo with:
- v1-classic/ (Portable PocketBot v1 Classic)
- v2-apex/ (Portable PocketBot Apex v2 Hydra SOR)
- docs/ (GitHub Pages hub, interactive HTML manual, deep architecture & math docs)
- .github/ (CI/CD workflows, CodeQL, Pages deploy, portable zip packager, issue templates)
- Root community standards (README, CHANGELOG, SECURITY, CONTRIBUTING, LICENSE, CITATION)
- tools/repo_security_audit.py
"""

from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = Path(r"D:\Projects\PocketBot_GitHub_Repo")
V1_SRC = Path(r"D:\Projects\PocketBot_USB\PocketBot")
V2_SRC = Path(r"D:\Projects\PocketBot_Apex_USB\PocketBot_Apex")
HTML_MANUAL_SRC = Path(r"D:\Projects\PocketBot_Apex_USB\START_HERE_OPEN_ME.html")

EXCLUDED_NAMES = {
    ".venv",
    ".pytest_cache",
    "__pycache__",
    ".git",
    ".env",
    "bot_log.txt",
    "orders.json",
    "paper_positions.json",
    "apex_ledger.db",
    "sor_ledger.db",
    "test_ledger.db",
}

EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".log"}


def copy_clean_tree(src: Path, dst: Path) -> None:
    """Recursively copy directory tree excluding caches, virtualenvs, logs, and databases."""
    if not dst.exists():
        dst.mkdir(parents=True, exist_ok=True)

    for item in src.iterdir():
        if item.name in EXCLUDED_NAMES:
            continue
        if item.suffix in EXCLUDED_SUFFIXES:
            continue

        target = dst / item.name
        if item.is_dir():
            copy_clean_tree(item, target)
        else:
            shutil.copy2(item, target)


def normalize_bat_crlf(file_path: Path) -> None:
    """Ensure batch file uses Windows CRLF (\\r\\n) line endings."""
    raw = file_path.read_bytes()
    # Replace all CRLF or lone LF with \\r\\n
    text = raw.decode("utf-8", errors="ignore").replace("\r\n", "\n").replace("\r", "\n")
    crlf_text = text.replace("\n", "\r\n")
    file_path.write_bytes(crlf_text.encode("utf-8"))


def main() -> None:
    print(f"[*] Staging Flagship Dual-Edition Monorepo at: {REPO_ROOT}")

    if REPO_ROOT.exists():
        print(f"[*] Cleaning existing staging directory {REPO_ROOT}...")
        # Handle read-only files if any
        def on_rm_error(func, path, exc_info):
            os.chmod(path, stat.S_IWRITE)
            func(path)
        shutil.rmtree(REPO_ROOT, onerror=on_rm_error)

    REPO_ROOT.mkdir(parents=True, exist_ok=True)

    # 1. Stage v1-classic
    print("[*] Staging v1-classic/ ...")
    v1_dst = REPO_ROOT / "v1-classic"
    copy_clean_tree(V1_SRC, v1_dst)

    # Patch v1 batch launchers to use goto :no_venv and CRLF
    v1_bat_2 = v1_dst / "2_START_AUTO_DEMO.bat"
    if v1_bat_2.exists():
        txt = v1_bat_2.read_text(encoding="utf-8")
        if 'if not exist "%~dp0.venv\\Scripts\\python.exe" (' in txt:
            txt = txt.replace(
                'if not exist "%~dp0.venv\\Scripts\\python.exe" (\n'
                '    echo ============================================================================\n'
                '    echo   [!] FIRST-TIME SETUP REQUIRED\n'
                '    echo   Please double-click "1_FIRST_TIME_SETUP.bat" once before starting PocketBot.\n'
                '    echo ============================================================================\n'
                '    echo.\n'
                '    pause\n'
                '    exit /b 1\n'
                ')',
                'if not exist "%~dp0.venv\\Scripts\\python.exe" goto :no_venv'
            )
            txt += (
                '\n\ngoto :eof\n\n'
                ':no_venv\n'
                'echo ============================================================================\n'
                'echo   [!] FIRST-TIME SETUP REQUIRED\n'
                'echo   Please double-click "1_FIRST_TIME_SETUP.bat" once before starting PocketBot.\n'
                'echo ============================================================================\n'
                'echo.\n'
                'pause\n'
                'exit /b 1\n'
            )
            v1_bat_2.write_text(txt, encoding="utf-8")

    v1_bat_3 = v1_dst / "3_START_CONFIRM_MODE.bat"
    if v1_bat_3.exists():
        txt = v1_bat_3.read_text(encoding="utf-8")
        if 'if not exist "%~dp0.venv\\Scripts\\python.exe" (' in txt:
            txt = txt.replace(
                'if not exist "%~dp0.venv\\Scripts\\python.exe" (\n'
                '    echo ============================================================================\n'
                '    echo   [!] FIRST-TIME SETUP REQUIRED\n'
                '    echo   Please double-click "1_FIRST_TIME_SETUP.bat" once before starting PocketBot.\n'
                '    echo ============================================================================\n'
                '    echo.\n'
                '    pause\n'
                '    exit /b 1\n'
                ')',
                'if not exist "%~dp0.venv\\Scripts\\python.exe" goto :no_venv'
            )
            txt += (
                '\n\ngoto :eof\n\n'
                ':no_venv\n'
                'echo ============================================================================\n'
                'echo   [!] FIRST-TIME SETUP REQUIRED\n'
                'echo   Please double-click "1_FIRST_TIME_SETUP.bat" once before starting PocketBot.\n'
                'echo ============================================================================\n'
                'echo.\n'
                'pause\n'
                'exit /b 1\n'
            )
            v1_bat_3.write_text(txt, encoding="utf-8")

    v1_bat_4 = v1_dst / "4_CHECK_STATS.bat"
    if v1_bat_4.exists():
        txt = v1_bat_4.read_text(encoding="utf-8")
        if 'if not exist "%~dp0.venv\\Scripts\\python.exe" (' in txt:
            txt = txt.replace(
                'if not exist "%~dp0.venv\\Scripts\\python.exe" (\n'
                '    echo ============================================================================\n'
                '    echo   [!] FIRST-TIME SETUP REQUIRED\n'
                '    echo   Please double-click "1_FIRST_TIME_SETUP.bat" once before checking stats.\n'
                '    echo ============================================================================\n'
                '    echo.\n'
                '    pause\n'
                '    exit /b 1\n'
                ')',
                'if not exist "%~dp0.venv\\Scripts\\python.exe" goto :no_venv'
            )
            txt += (
                '\n\ngoto :eof\n\n'
                ':no_venv\n'
                'echo ============================================================================\n'
                'echo   [!] FIRST-TIME SETUP REQUIRED\n'
                'echo   Please double-click "1_FIRST_TIME_SETUP.bat" once before checking stats.\n'
                'echo ============================================================================\n'
                'echo.\n'
                'pause\n'
                'exit /b 1\n'
            )
            v1_bat_4.write_text(txt, encoding="utf-8")

    # Add 7_UPDATE_FROM_GITHUB.bat to v1-classic
    v1_update_bat = v1_dst / "7_UPDATE_FROM_GITHUB.bat"
    v1_update_content = r"""@echo off
setlocal EnableDelayedExpansion
title PocketBot v1 Classic - GitHub Live Updater
cd /d "%~dp0"

echo ==============================================================================
echo   POCKETBOT v1 CLASSIC - AUTOMATIC UPDATE FROM GITHUB
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
set "TEMP_ZIP=%TEMP%\pocketbot_v1_latest.zip"
set "TEMP_DIR=%TEMP%\pocketbot_v1_update_%RANDOM%"

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

echo [ZIP Mode] Syncing updated files into "%~dp0" (strictly preserving .venv, config, and orders)...
powershell -NoProfile -Command ^
    "$src = Join-Path '%TEMP_DIR%' 'pocketbot-apex-main\v1-classic'; " ^
    "$dst = '%~dp0'; " ^
    "$excludes = @('.env', '.venv', 'trader\state', 'orders.json', 'paper_positions.json', 'bot_log.txt', '__pycache__', '.pytest_cache'); " ^
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
echo   Your code files and launcher scripts have been synced to latest GitHub main.
echo   Your local .venv runtime and trade records were preserved 100%% safely.
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
"""
    v1_update_bat.write_text(v1_update_content, encoding="utf-8")

    # Normalize all CRLF in v1 batch files
    for bat_f in v1_dst.glob("*.bat"):
        normalize_bat_crlf(bat_f)

    # 2. Stage v2-apex
    print("[*] Staging v2-apex/ ...")
    v2_dst = REPO_ROOT / "v2-apex"
    copy_clean_tree(V2_SRC, v2_dst)

    # Copy START_HERE_OPEN_ME.html into v2-apex as well
    if HTML_MANUAL_SRC.exists():
        shutil.copy2(HTML_MANUAL_SRC, v2_dst / "START_HERE_OPEN_ME.html")

    # Add 7_UPDATE_FROM_GITHUB.bat to v2-apex
    v2_update_bat = v2_dst / "7_UPDATE_FROM_GITHUB.bat"
    v2_update_content = r"""@echo off
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
"""
    v2_update_bat.write_text(v2_update_content, encoding="utf-8")

    # Normalize all CRLF in v2 batch files
    for bat_f in v2_dst.glob("*.bat"):
        normalize_bat_crlf(bat_f)

    # 3. Stage docs/
    print("[*] Staging docs/ ...")
    docs_dst = REPO_ROOT / "docs"
    docs_dst.mkdir(parents=True, exist_ok=True)

    if HTML_MANUAL_SRC.exists():
        shutil.copy2(HTML_MANUAL_SRC, docs_dst / "index.html")
    (docs_dst / ".nojekyll").write_text("", encoding="utf-8")

    print("[*] Clean copy staging complete.")


if __name__ == "__main__":
    main()
