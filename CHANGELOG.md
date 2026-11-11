# Changelog

All notable changes to the PocketBot Dual-Edition project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
