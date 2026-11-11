# Security Policy

## Supported Versions

| Version | Supported |
| :--- | :--- |
| 2.0.x (Apex Hydra SOR) | :white_check_mark: |
| 1.0.x (v1 Classic) | :white_check_mark: |
| < 1.0.0 | :x: |

---

## Safety Invariants

PocketBot Apex is engineered with strict cryptographic and operational safety guarantees:

1. **Hard-Locked Demo Execution:**
   Production code paths are hard-locked to Demo accounts by default. Connecting to live brokerage accounts will raise an unhandled runtime error unless **both**:
   - The `--unlock-live` CLI flag is explicitly passed at runtime, **and**
   - The environment variable `APEX_UNLOCK_LIVE=I_ACCEPT_REAL_RISK` is set.
2. **Server-Side Protective Stops:**
   Every order submitted to MetaTrader 5 or cTrader Open API must carry valid server-side hard Stop-Loss (SL) and Take-Profit (TP) parameters. Orders missing either are rejected prior to socket transmission.
3. **Zero Secrets Commitment:**
   Personal API keys, passwords, broker account numbers, tokens, and private paths must never be committed to source control. Automated pre-push audits and GitHub Secret Push Protection enforce this policy.

---

## Reporting a Vulnerability

If you discover a security vulnerability or sensitive data leak in this repository:
1. **Do NOT open a public GitHub issue.**
2. Send a confidential report via GitHub Private Vulnerability Reporting under the **Security** tab of this repository.
3. Alternatively, contact the maintainer directly through GitHub Security Advisories.
4. Include detailed steps to reproduce the issue and affected components.

We commit to acknowledging reports within 48 hours and deploying patches promptly.
