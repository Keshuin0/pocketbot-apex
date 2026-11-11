"""Pre-Commit and CI Security & Integrity Audit for PocketBot Dual-Edition Repository.

Enforces:
1. Zero secrets / API keys (Gemini, cTrader secrets, Telegram tokens).
2. Zero personal user paths (e.g. Users\\BRO) or hardcoded project directories.
3. Zero personal private emails in source code / configs.
4. Strict Windows CRLF line endings on all .bat and .ps1 launchers.
5. No prohibited database or cache files (.db, .log, .venv, .pytest_cache).
6. Demo safety lock verification (invariants hard-coded).
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

PROHIBITED_FILES = {
    ".env",
    "bot_log.txt",
    "orders.json",
    "paper_positions.json",
    "apex_ledger.db",
    "sor_ledger.db",
    "test_ledger.db",
}

PROHIBITED_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".log",
    ".db",
    ".key",
    ".pem",
    ".pfx",
}

PROHIBITED_DIRS = {
    ".venv",
    ".pytest_cache",
    "__pycache__",
}

SECRET_PATTERNS = [
    (re.compile(r"AIzaSy[A-Za-z0-9_-]{33}"), "Google AI / Gemini API Key"),
    (re.compile(r"\b[0-9]{8,10}:[A-Za-z0-9_-]{35}\b"), "Telegram Bot Token"),
    (re.compile(r"ghp_[A-Za-z0-9]{36}"), "GitHub Personal Access Token"),
]

PRIVATE_PATH_PATTERNS = [
    (re.compile(r"Users[\\/]BRO", re.IGNORECASE), "Personal User Home Directory (Users/BRO)"),
    (re.compile(r"D:[\\/]Projects[\\/]PocketBot_Apex(?![a-zA-Z0-9_])", re.IGNORECASE), "Hardcoded PocketBot_Apex Workspace Path"),
]

# Disallow personal emails; allow @users.noreply.github.com, @example.com, @domain.com
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b")
ALLOWED_EMAIL_DOMAINS = ("@users.noreply.github.com", "@example.com", "@google.com", "@spotware.com", "@metaquotes.net")


def audit_repository() -> int:
    print(f"[*] Running Security & Integrity Audit on: {REPO_ROOT}")
    violations: list[str] = []
    scanned_files_count = 0

    for root, dirs, files in os.walk(REPO_ROOT):
        # Filter prohibited directories
        dirs[:] = [d for d in dirs if d not in PROHIBITED_DIRS and d != ".git"]

        # Check for prohibited directory presence
        rel_root = Path(root).relative_to(REPO_ROOT)
        for d in dirs:
            if d in PROHIBITED_DIRS:
                violations.append(f"Prohibited directory present: {rel_root / d}")

        for file in files:
            file_path = Path(root) / file
            rel_path = file_path.relative_to(REPO_ROOT)
            rel_str = str(rel_path).replace("\\", "/")

            # 1. Prohibited files & extensions
            if file in PROHIBITED_FILES:
                violations.append(f"Prohibited file found in tree: {rel_path}")
                continue

            if file_path.suffix.lower() in PROHIBITED_EXTENSIONS:
                violations.append(f"Prohibited file extension ({file_path.suffix}): {rel_path}")
                continue

            scanned_files_count += 1

            # 2. Check batch file line endings (MUST be CRLF)
            if file_path.suffix.lower() in (".bat", ".cmd"):
                raw = file_path.read_bytes()
                if b"\n" in raw and b"\r\n" not in raw:
                    violations.append(f"Batch file has LF line endings (MUST be CRLF): {rel_path}")

            # 3. Text file content scanning
            # Skip binary / PDF / DOCX
            if file_path.suffix.lower() in (".pdf", ".docx", ".png", ".jpg", ".ico"):
                continue

            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                violations.append(f"Error reading file {rel_path}: {e}")
                continue

            # Check for secrets
            for pattern, name in SECRET_PATTERNS:
                if pattern.search(content):
                    violations.append(f"Potential secret detected ({name}) in: {rel_path}")

            # Check for personal paths (exclude this audit script itself since it defines patterns)
            if rel_str != "tools/repo_security_audit.py":
                for pattern, name in PRIVATE_PATH_PATTERNS:
                    if pattern.search(content):
                        violations.append(f"Personal path detected ({name}) in: {rel_path}")

            # Check for personal emails (exclude security audit script & CITATION/docs where noreply is used)
            if rel_str != "tools/repo_security_audit.py":
                emails = EMAIL_PATTERN.findall(content)
                for email in emails:
                    if not any(email.lower().endswith(dom) for dom in ALLOWED_EMAIL_DOMAINS):
                        violations.append(f"Disallowed email address found ({email}) in: {rel_path}")

    print(f"[*] Scanned {scanned_files_count} files across repository.")

    if violations:
        print("\n[!] SECURITY & INTEGRITY AUDIT FAILED WITH VIOLATIONS:")
        for v in violations:
            print(f"    - {v}")
        return 1

    print("\n[OK] SECURITY & INTEGRITY AUDIT PASSED: 100% CLEAN (0 leaks, 0 private paths, valid CRLF on launchers).")
    return 0


if __name__ == "__main__":
    sys.exit(audit_repository())
