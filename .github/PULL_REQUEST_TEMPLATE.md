## Description of Changes

Please provide a clear and concise summary of the quantitative, architectural, or launcher improvements implemented in this PR.

---

## Architectural Compliance Checklist

- [ ] **$0 Hot Path Invariant:** No LLM or cloud API calls added to tick, candle, or DOM loops.
- [ ] **Demo Safety Lock Invariant:** Demo-locked default behavior is strictly preserved.
- [ ] **Server-Side Protection Invariant:** All order submissions enforce hard server-side SL and TP.
- [ ] **A/B Testing Toggle:** Any new filter exposes an independent `"enabled": true | false` flag in `config/apex_config.json`.
- [ ] **Zero Secrets:** No `.env`, API keys, broker account numbers, or personal paths are introduced.
- [ ] **Windows CRLF on Launchers:** All `.bat` and `.cmd` scripts retain CRLF line endings.
- [ ] **Automated Tests:** All unit tests pass cleanly (`pytest v2-apex/tests/ -v`).
- [ ] **Security Audit:** `python tools/repo_security_audit.py` passes with zero violations.
