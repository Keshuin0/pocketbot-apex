# PocketBot Apex — Non-Negotiable Architecture Rules

These architecture rules are strictly enforced across all phases of **PocketBot Apex** in this repository.

## 1. Python Interpreter
- **ALWAYS** use the workspace virtual environment interpreter: `.\.venv\Scripts\python.exe` (relative/workspace `.venv\Scripts\python.exe`).
- **NEVER** invoke global Python or hardcode user-specific home directory paths in scripts, configs, or tests.

## 2. Strict Separation of Concerns ($0 Real-Time Hot Path vs. Event-Driven AI)
- **Hot Path (`apex/` Async / Twisted Python):** Handles 100% of real-time market ingestion (cTrader Protobuf TCP & MetaTrader 5 IPC), Level 2 Depth of Market (DOM) math (Order-Book Imbalance, liquidity walls, VWAP fill & slippage modeling), regime classification, 10,000-run Monte Carlo / stochastic simulations, and low-latency order routing at **$0 AI token cost**.
- **Event-Driven LLM / MCP Layer:** Strictly event-driven. Invoked **only** for pre-trade risk veto checks on candidate setups that pass all quantitative filters, post-trade diagnostics, and interactive cockpit queries. Never place LLM calls in tick, DOM, or candle loops.

## 3. Non-Negotiable Safety Invariants
- **Hard-Locked DEMO Default:** Execution is hard-locked to **DEMO** accounts by default. Live account routing or connection is strictly prohibited and will raise a hard runtime error unless **BOTH**:
  1. The `--unlock-live` CLI flag is passed at launch, **AND**
  2. The environment variable `APEX_UNLOCK_LIVE=I_ACCEPT_REAL_RISK` is explicitly set.
- **Mandatory Server-Side Protection:** Every single order submitted to cTrader or MT5 **MUST** carry a valid server-side hard Stop-Loss (`SL`) and Take-Profit (`TP`). Orders missing either parameter are rejected prior to wire transmission.
- **Zero Placeholders / Zero Mock Data:** Production code paths must never contain dummy/random prices, stubbed functions, or mock responses.

## 4. Modular Toggle Configuration (`config/apex_config.json`)
- Every quantitative and safety filter in `config/apex_config.json` **MUST** expose an independent `"enabled": true | false` boolean flag to support systematic walk-forward A/B testing and ablation studies:
  - `regime` (`config["filters"]["regime"]["enabled"]`)
  - `dom_obi` (`config["filters"]["dom_obi"]["enabled"]`)
  - `correlation_cap` (`config["filters"]["correlation_cap"]["enabled"]`)
  - `ai_veto` (`config["filters"]["ai_veto"]["enabled"]`)
  - `dual_feed_guard` (`config["filters"]["dual_feed_guard"]["enabled"]`)
