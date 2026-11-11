# Changelog

All notable changes to the PocketBot Dual-Edition project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.1.0] - 2026-11-11

### Added
- **Full Broker Catalog Auto-Discovery (251 Symbols across MT5 & cTrader):**
  - Dynamic broker catalog enumeration via `mt5.symbols_get()` and cTrader `ProtoOASymbolsListReq` with bidirectional smart alias mapping (`SmartAliasMapper`).
  - Auto-categorization of 251 tradable instruments into 7 distinct asset classes: `US_EQUITIES` (109), `FOREX` (86), `INDICES` (16), `CRYPTO` (14), `METALS` (12), `COMMODITIES` (11), `ENERGY` (3).
  - Smart ticker normalization for Fusion Markets equity CFDs (e.g., `NVIDIA` $\leftrightarrow$ `NVDA`, `Tesla` $\leftrightarrow$ `TSLA`, `Alphabet-A` $\leftrightarrow$ `GOOGL`, `JPMorganChase` $\leftrightarrow$ `JPM`) and standard aliases (`US500`/`SPX500`/`SPX`, `NAS100`/`USTEC`/`NQ`, `XTIUSD`/`WTI`/`USOUSD`).
- **Two-Stage Full-Market Funnel Scanner (`apex/quant/universe_scanner.py`):**
  - **Stage A (Sub-Second Universe Sweep):** High-speed filter measuring spread friction, session open/close status, and momentum/Kaufman Efficiency Ratio (ER) mover scores across all 251 symbols.
  - **Stage B (Deep Quantitative Evaluation):** Top ranked movers feed into Hurst regime classification, Momentum/DOM OBI gating, 10,000-path Monte Carlo Time Machine, and factor exposure checks.
  - **TUI Cockpit Universe Header:** Real-time sweep progress visualization (`Universe: X/251 Symbols Scanned (Y Closed)`).
- **Institutional Correlation Shield Factor Decomposition:**
  - Expanded factor decomposition mapping all 251 universe instruments into canonical factors: signed currency legs (+base / -quote), commodities, energy, crypto beta, broad market/tech beta, and single-stock factor caps (`Stock_<Ticker>`).
  - Strict 1.0% single-factor risk capping across all active positions, preventing cross-pair concentration traps and sector/stock overexposure.
- **Verification Suite Expansion:**
  - Added `tests/test_phase6_universe_scanner.py`, expanding automated institutional test coverage from 25 to 32 tests (100% passing).

---

## [2.0.0] - 2026-11-11

### Added
- **Apex Hydra Active-Active Smart Order Router (SOR):**
  - Consolidated Synthetic NBBO engine comparing real-time best bid and offer across cTrader Open API and MetaTrader 5.
  - Effective All-In Cost formula factoring raw spread, round-turn commissions ($4.50/lot vs $0.00), DOM VWAP depth slippage, and latency penalties.
  - Multi-venue 50/50 risk split routing (`SPLIT_EXECUTION`).
  - Single-Broker Spike Rejector blocking artificial broker stop-hunts when cross-venue basis exceeds 2.5x normal.
- **Microstructure & Level 2 Depth of Market (DOM):**
  - Full Level 2 depth ladder processing via cTrader Protobuf `ProtoOADepthEvent`.
  - Order-Book Imbalance (OBI) calculator with resting liquidity wall detection.
  - VWAP depth slippage estimator.
- **Institutional Quant Core:**
  - Real-time rolling Hurst Exponent and Kaufman Efficiency Ratio (ER) market regime classifier.
  - Time Machine v2: 10,000-path stationary block-bootstrap Monte Carlo simulator with fractional Kelly sizing and VaR95 metrics.
  - Correlation Shield computing rolling Pearson and Kendall Tau cross-asset factor exposure caps.
- **3-Tier Hybrid AI Risk Veto Waterfall:**
  - Event-driven pre-trade committee with hot-reloadable waterfall: Google AI Studio REST (`studio_api`) $\to$ Google AI Pro CLI (`agy_pro`) $\to$ Deterministic Pure Math (`pure_math`).
- **Telemetry & Cockpit:**
  - Full-screen Rich Terminal UI (TUI) live dashboard.
  - Remote smartphone control via Telegram bot (`/status`, `/positions`, `/daily`, `/panic`, `/mode`).
- **Safety & Persistence:**
  - Hard-locked Demo account safety guard.
  - SQLite Walk-Forward Auto-Quarantine Ledger (`WalkForwardLedgerDB`) with 24-hour symbol quarantine and cooldown recovery loops.
  - Portable 1-click batch launchers with `%~dp0` isolation, Windows CRLF line endings, and automated Python 3.12 installer.

---

## [1.0.0] - 2026-11-01

### Added
- Initial public release of **PocketBot v1 Classic**:
  - Standalone MetaTrader 5 IPC automated trading engine.
  - M5 tight-spread breakout strategies for major FX and indices.
  - M15 momentum strategies for crypto and single-stock equities.
  - Fixed 0.5% ($50) risk management with 2R asymmetric profit targets.
  - JSON trade state persistence and terminal performance statistics.
  - 1-click batch launcher suite (`1_FIRST_TIME_SETUP.bat` to `4_CHECK_STATS.bat`).
