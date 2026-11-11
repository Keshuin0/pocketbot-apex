# Contributing to PocketBot Apex

Thank you for your interest in contributing to PocketBot Apex! We welcome contributions from quantitative researchers, algorithmic traders, and software engineers.

---

## Code of Conduct

All contributors and participants are expected to adhere to our [Code of Conduct](CODE_OF_CONDUCT.md).

---

## Development Setup

1. **Fork and Clone the Repository:**
   ```bash
   git clone https://github.com/<your-username>/pocketbot-apex.git
   cd pocketbot-apex
   ```

2. **Initialize Local Virtual Environment:**
   Run the 1-click setup script inside `v2-apex/`:
   ```cmd
   cd v2-apex
   0_FIRST_TIME_SETUP.bat
   ```
   Or manually create a virtual environment with Python 3.12 (64-bit):
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   pip install pytest
   ```

3. **Run the Verification Test Suite:**
   ```bash
   pytest tests/ -v
   ```
   All 25 automated unit tests must pass.

4. **Run the Security & Integrity Audit:**
   ```bash
   python tools/repo_security_audit.py
   ```

---

## Contribution Guidelines

1. **Architecture Boundaries:**
   - **Hot Path ($0 Token Cost):** Real-time tick ingestion, Level 2 DOM math, regime filters, Monte Carlo simulations, and order routing must remain purely mathematical with zero LLM API calls.
   - **AI Layer:** Invoked strictly on pre-trade risk vetoes (~0–5 times per day) when all quantitative barriers have already passed.
2. **Safety Locks:**
   - Never disable the Demo safety lock in PRs.
   - Ensure server-side SL and TP parameters are strictly verified on all order requests.
3. **No Mock Data in Production Paths:**
   - Code paths must use real data or realistic synthetic market structures.
4. **Style Conventions:**
   - Use strict Python type annotations (`from __future__ import annotations`).
   - Format batch files with Windows CRLF line endings (`\r\n`).
   - Use `Path(__file__).resolve()` and `%~dp0` for portable relative path resolution.

---

## Submitting Pull Requests

1. Create a feature branch: `git checkout -b feat/your-feature-name`.
2. Follow Conventional Commits format (`feat:`, `fix:`, `docs:`, `test:`, `perf:`).
3. Ensure CI passes on `windows-latest`.
4. Submit a Pull Request targeting the `main` branch with a clear description of your changes.
