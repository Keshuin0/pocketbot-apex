# Changelog

All notable changes to the PocketBot Dual-Edition project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.5.1] - 2026-11-13

### Added
- **Unified 11-Agent Canonical Pipeline (`apex/engine/canonical_pipeline.py`, `apex/cockpit/replay_engine.py`, `tools/live_lifecycle_drill.py`):**
  - Routes Live Scanning, Historical Replay (`20x`/`50x`/`100x` across `2d`/`5d`/`10d`), and `60s Live 0.01-Lot MT5 Drill` (`strict` | `plumbing` mode) through one shared `Canonical11AgentPipeline` emitting a 11-stage `AgentPipelineTrace` with explicit `symbol` and `timeframe` metadata.
  - Historical Replay enforces `min_net_avg_R >= 0.10`, real Z3 gates, and steps simulated trades bar-by-bar through `AutonomousPositionOrchestrator`.
- **251-Symbol Asset Picker, Replay Trade Log & 11-Agent X-Ray Modal (`apex/cockpit/static/index.html`, `apex/cockpit/web_cockpit.py`, `apex_runner.py`):**
  - Searchable 251-symbol dropdown grouped by asset class (`Auto Top-12 Active Basket`, `Auto Best Open Mover`, `FOREX`, `INDICES`, `METALS`, `CRYPTO`, `US_EQUITIES`, `ENERGY`, `COMMODITIES`) plus Replay Window/Speed & Drill Mode selectors and a clickable **11-Agent X-Ray Scorecard** modal.

### Fixed
- **6 Live Cockpit & Cost Forensic Bugs (`universe_scanner.py`, `mt5_feed.py`, `walkforward_db.py`, `model_selector.py`, `veto_committee.py`):**
  - Clear top KPI titles (`ACCOUNT EQUITY`, `OPEN POSITIONS & MESH`, `AI BRAIN & QUOTA`, `CONTAGION & Z3 INTEGRITY`) and non-`UNKNOWN` `active_model_label` before the first AI call.
  - Persisted AI model-family quarantine cooldowns in SQLite (`model_family_cooldowns`) across bot restarts.
  - Closed-market weekend exclusion requiring BOTH `trade_mode == SYMBOL_TRADE_MODE_FULL` AND tick age `<= 120s` (eliminating `SUGAR`/`UKCOCOA` weekend leaks).
  - Asset-class commission (`$4.50/lot` FOREX/METALS only, `$0.00` CRYPTO/INDICES/US_EQUITIES/ENERGY/COMMODITIES) + `ContractSpec` risk scaling (fixing `ADAUSD` skew).
  - Atomic `effective_cost_pips >= spread_pips` clamping, forced `Q-KELLY = 0.00%` on filtered/chop rows, and actionable-first sorting in `scan_cycle()`.

---

## [2.5.0] - 2026-11-13

### Added
- **Higher-Timeframe (`H1`) Structural Alignment & Anti-Chase Filter (`apex/quant/mtf_confluence.py`, `apex/quant/universe_scanner.py`):**
  - Blocks `M5`/`M15` breakouts that conflict with `H1` EMA/structure slope or sit within `0.30 * ATR_M5` of a 20-bar `H1` swing high ceiling / swing low floor.
  - Enforces an Anti-Chase Overextension Guard rejecting breakouts stretched `> 2.2 * ATR(14)` past the prior range without pullback.
- **Autonomous Multi-Agent In-Flight Position Orchestrator (`apex/execution/position_manager.py`, `apex/ai/aegis_verifier.py`, `apex_runner.py`):**
  - **Rule A (Anti-Panic Grace Window & Shield 8 Z3):** Forbids early exit or SL tightening during the first `min_hold_bars = 2` completed bars and `±0.65 * ATR` noise cone (enforced in Python + `AegisVerifier.verify_z3_inflight_trajectory`), unless extreme flash-crash `Contagion Shock Z > 4.2` triggers.
  - **Rule B (Continuous Conditional Forward Time Machine):** Re-runs `RegimeClassifier` + `BlockBootstrapTimeMachine` from current mark price toward remaining `SL` and `TP` on every closed bar, triggering early thesis exit only with `>= 3` confirming signals and Z3 `SAT` proof.
  - **Rule C (Dynamic Profit-Ratchet & Super-Momentum Runner Mode):** Moves SL to fee-adjusted `+0.05R` free-roll at `+1.0R` (never widening SL backward); at `+1.5R`, banks `+1.0R` on decaying momentum or expands TP to `3.5R` with ATR/swing trailing SL when `H >= 0.60` & `ER >= 0.55`.
  - **Rule D (Zombie Time-Decay & Friday Weekend-Gap Shield):** Exits stagnant trades (`bars >= 18`, `|R| < 0.20`, `ER < 0.15`, `forward_ev_R <= 0`) and flattens non-crypto positions after Friday `20:45 UTC`.
- **Closed-Deal Reconciliation & `MFE/MAE` Autopsy (`apex/ledger/walkforward_db.py`, `apex/ai/sleep_consolidation.py`, `apex/cockpit/static/index.html`):**
  - Tracks live `MFE_R`/`MAE_R`, reconciles closed MT5 deals into `trade_ledger`, consolidates `mfe_mae_stats` into `semantic_memory.json`, and displays `In-Flight Trajectory (Stage | Fwd P(TP) | MFE/MAE)` in the Chrome Cockpit.
- **Verification Suite:** Added `tests/test_phase11_inflight_orchestrator_and_mtf.py` (**72/72** total tests passing).

---

## [2.4.1] - 2026-11-13

### Fixed
- **`cmd.exe` Rich `Live` Height-Overflow Scroll-Spam (`apex_runner.py`, `apex/cockpit/tui_dashboard.py`):**
  - Eliminated multi-line Rich table printing in default Web Cockpit mode (`--dashboard` used by `2_START_APEX_AUTO_DEMO.bat` and `4_APEX_LIVE_DASHBOARD.bat`). Prints a clean 5-line startup banner once with the `http://127.0.0.1:8765` URL and updates a single `<100`-char carriage-return heartbeat line throttled every 5s.
  - Updated `--tui-only` fallback in `tui_dashboard.py` to `POCKETBOT APEX v2.4.1`, set `no_wrap=True` on all table cells (capped `<=22` lines), and passed `screen=True, vertical_overflow="crop"` to `rich.live.Live`.
- **Reliable Windows Chrome/Edge App Window Auto-Launch (`apex/cockpit/web_cockpit.py`):**
  - Expanded browser discovery across `%ProgramFiles%`, `%ProgramFiles(x86)%`, `%LocalAppData%`, and Windows Registry `App Paths\chrome.exe` / `msedge.exe`, launching detached via `--app=<url> --new-window` with `os.startfile(url)` fallback.
- **Verification Suite:** **66/66** automated tests passing + live desktop verification (`chrome.exe --app=http://127.0.0.1:8765` launched, `/ws` streamed live `FusionMarkets-Demo` state, zero multi-line table spam).

---

## [2.4.0] - 2026-11-12

### Added
- **Real-Time WebSocket Chrome App Cockpit (`apex/cockpit/web_cockpit.py`, `apex/cockpit/static/index.html`, `apex_runner.py`):**
  - Embedded async `aiohttp` HTTP + WebSocket server bound strictly to `127.0.0.1:8765` (auto-falling back across `8765..8785` if busy) with zero external CDN dependencies.
  - Auto-launches a standalone frameless Chrome/Edge Desktop App (`chrome.exe --app=...` / `msedge.exe --app=...`, skipped automatically under `--no-gui`, `GITHUB_ACTIONS=true`, or `PYTEST_CURRENT_TEST`).
  - Live `/ws` JSON pushes update KPI cards, sortable Top-12 Matrix with price flash highlights, Open Positions floating CAD P/L & `SL <-> Entry <-> TP` progress bars, Quarantined Model Families, Contagion Shock Z, and Execution/Veto/RetCode Feed in place without page reloads.
  - Added localhost CSRF-protected (`X-Apex-CSRF`) interactive controls: `[Pause / Resume Scan]`, `[Run 60s Live 0.01-Lot MT5 Drill]`, `[Run 100x Historical Replay]`, `[Refresh AI Brain]`, and `[PANIC FLATTEN ALL DEMO POSITIONS]`.
- **100x Speed Historical Market Replay (`apex/cockpit/replay_engine.py`):**
  - Streams simulated entries, Aegis Z3 vetoes, SL/TP closes, Win Rate, Profit Factor, and live SVG equity curves over `/ws`, strictly isolated from live `trade_ledger` stats.
- **60s Live 0.01-Lot MT5 Demo Lifecycle Drill (`tools/live_lifecycle_drill.py`, `9_TEST_FULL_LIFECYCLE_NOW.bat`, `apex/ledger/walkforward_db.py`):**
  - Strictly enforces `ACCOUNT_TRADE_MODE_DEMO`, runs Aegis Z3 axiom check, submits one `0.01`-lot bracket order with mandatory SL/TP, streams floating CAD P/L, verifies `TRADE_ACTION_SLTP`, closes via opposite `TRADE_ACTION_DEAL`, reads settled deal history, and isolates rows via `is_drill=1` in SQLite.
- **Verification Suite:** Added `tests/test_phase10_web_cockpit_drill_and_replay.py` (**65/65** total tests passing) and live-verified against `FusionMarkets-Demo` (`568159`).

---

## [2.3.1] - 2026-11-12

### Fixed
- **Per-Model-Family Quota Quarantine & Cross-Family Cascade (`apex/ai/model_selector.py`, `apex/ai/veto_committee.py`):**
  - Grouped discovered `agy_pro` and `studio_api` models into billing/quota families (`argon`, `hydrogen`, `flash`, etc.) via `classify_model_family()`.
  - Quarantined the **entire model family** for 60 minutes whenever any variant returns credit/quota exhaustion (`"credits balance is too low"`, `"quota"`, `429`), preventing wasted retries on `-low`/`-medium` siblings or bare `--model`-less `agy -p` defaults.
  - Added `rank_cross_family_candidates()` to cascade cleanly across distinct families (`argon` -> `hydrogen` -> `flash`) and updated `ApexTUIDashboard` to display the actual live-verified working model.
- **Contagion Oracle Post-Rollover False Positives (`apex/quant/contagion_oracle.py`, `apex/quant/universe_scanner.py`):**
  - Recomputed `ContagionOracle.evaluate()` fresh on every `scan_cycle()` instead of caching the first cycle's result, ignored stale maintenance bars (`>900s`), and applied a robust MAD/floor baseline on the 7 liquid core benchmarks (`EURUSD`, `GBPUSD`, `USDJPY`, `XAUUSD`, `US500`, `NAS100`, `BTCUSD`).
- **Stage A Funnel Ranking (`apex/quant/universe_scanner.py`):**
  - Penalized `MEAN_REVERTING_CHOP` (`H < 0.50` or `ER < 0.18`) and wide-spread/high-spread-to-ATR exotics so `TRENDING_EXPANSION` majors, indices, and equities rank into the Top 12.
- **In-Place Rich Live TUI & Audit Event Logging (`apex/cockpit/tui_dashboard.py`, `apex_runner.py`):**
  - Rendered the dashboard loop in-place via `rich.live.Live` and logged `VETO` events (with model used and Z3 status) into the bottom `Last 5 Execution & Aegis Audit Events` table.
- **Verification Suite:** Added `tests/test_phase9_hotfixes.py` (**60/60** total tests passing).

---

## [2.3.0] - 2026-11-12

### Added
- **Circadian Sleep-Wake Memory Consolidation & Context Hygiene Engine (`apex/ai/sleep_consolidation.py`):**
  - **Phase A (Working Memory Purge):** Flushes per-candle AI verdict caches, recent AI event logs, expired model cooldowns, oversized `HydraNBBOEngine.spike_rejector` buffers (`>50`), and orphaned `%TEMP%\pocketbot_agy_isolated` temp files; runs `gc.collect()` and refreshes MT5 heartbeats.
  - **Phase B (Episodic-to-Semantic Memory Consolidation):** Distills `trader/apex_ledger.db` into atomic SHA-256-verified `trader/semantic_memory.json` and SQLite `consolidation_checkpoints` (realized slippage calibration multipliers feeding `TimeMachineCostModel` & `HydraNBBOEngine`, UTC session-hour expectancy priors, Split-Conformal quantiles, and `<120`-token Aegis numeric priors).
  - **Phase C (Morning Wake-Up Self-Audit):** Refreshes `AdaptiveModelSelector` discovery & daily quota budgets, syncs conformal trust scores, runs `run_canary_integrity_probe()`, and logs `[MORNING WAKE-UP READY]`.
- **Automated & 1-Click Triggers (`apex_runner.py`, `config/apex_config.json`, `tools/run_sleep_cycle.py`, `8_REFRESH_AI_BRAIN.bat`):**
  - Automatic daily trigger at `22:05 UTC` (`sleep_wake` config), startup trigger if `>16.0h` elapsed, and portable 1-click `8_REFRESH_AI_BRAIN.bat` launcher.
- **TUI Cockpit (`apex/cockpit/tui_dashboard.py`):** Upgraded header to `POCKETBOT APEX v2.3` with `Mind State: FRESH (Cycle #<N> | Wisdom Checkpoints: <M> | Last Refresh: <HH:MM>)`.
- **Verification Suite:** Added `tests/test_phase8_sleep_consolidation.py` (**56/56** total tests passing).

---

## [2.2.0] - 2026-11-12

### Added
- **Apex Aegis 7-Shield Formal Neuro-Symbolic Anti-Hallucination Firewall (`apex/ai/aegis_verifier.py` + `z3-solver`):**
  - Subtract-Only Authority Diode, Ticker-Blinded Canonicalization + Closed-World Contract, Proof-Carrying Atomic Claim AST Grounding, `z3-solver` Formal SMT Consistency & Axiom Prover, Counter-Abductive Framing + `ABSTAIN`, Split-Conformal & Brier Epistemic Trust Calibration, and Canary Poison Probe.
- **Live 251-Symbol Cross-Asset Contagion & Session Rollover Oracle (`apex/quant/contagion_oracle.py`):** Deterministic rollover illiquidity & cross-asset shock detection (`>2.5 sigma`).
- **FusionMarkets-Demo Execution Reliability (`apex/feeds/mt5_feed.py`, `apex/execution/router.py`):** `mt5.order_check()` filling-mode probing (`IOC`/`FOK`/`RETURN`) + `10030` retry and `trade_tick_value_loss` CAD lot-sizing (**53/53** tests passing).

---

## [2.1.1] - 2026-11-12

### Added
- **Adaptive Multi-Factor Gemini Model & Quota Selector (`apex/ai/model_selector.py`):**
  - Dynamic model discovery via Google AI Studio `/v1beta/models` and `agy.exe models` (in an isolated empty temp directory with `stdin=subprocess.DEVNULL`), automatically classifying models into `PRO_REASONING`, `FLASH_FAST`, and `FLASH_LITE`.
  - Real-time multi-factor scoring across **Remaining Quota & Budget Pacing** (RPM/RPD, HTTP 429 exponential cooldowns, Pro quota conservation), **Setup Complexity & Stakes** (volatility percentile, borderline `net_avg_R`, risk %), and **Rolling EWMA Latency vs Timeframe Urgency** (penalizing >6s models on fast `M5` breakouts).
  - Intra-provider model cascade + per-candle `(symbol, side, timeframe, bar_timestamp)` verdict caching.
  - Added option `[6] Inspect Discovered Gemini Models, Quota Health & Latency` to `tools/configure_ai_mode.py` (`5_CONFIGURE_AI_MODE.bat`).
- **Verification Suite:**
  - Added `tests/test_phase7_adaptive_model_selector.py` and 3 regression tests in `tests/test_phase6_universe_scanner.py` (39/39 total tests passing).

### Fixed
- **Live Order Dispatch & Fusion Markets CAD Sizing (`apex_runner.py`, `apex/feeds/mt5_feed.py`, `apex/execution/router.py`):**
  - Wired `_dispatch_approved_trades()` in `ApexOrchestrator.run_cycle()` for `--mode demo-auto` and `--mode shadow-ab`, routing AI-approved setups through all 6 pre-trade circuit breakers directly to MT5/cTrader execution.
  - Added `compute_compliant_lot_size()` using exact broker `trade_tick_value` (in account currency such as `CAD` or `USD`), `trade_tick_size`, `volume_min`, `volume_step`, and `volume_max`.
  - Synced live MT5 account equity, currency (`CAD`/`USD`), floating P/L, and open positions count (`get_open_positions()`, `get_open_symbols()`) to prevent duplicate symbol orders.
- **TUI Cockpit Execution Telemetry (`apex/cockpit/tui_dashboard.py`):**
  - Added the **Last Execution / Broker Events** panel displaying the last 5 order attempts, ticket IDs, lot sizes, SL/TP, venue/status, and exact MT5 retcodes/rejection details.
  - Dynamically formatted equity and P/L badges using the live account currency (`CAD`, `USD`, etc.).
- **`agy_pro` Subprocess Timeout & Per-Candle Caching (`apex/ai/veto_committee.py`):**
  - Executed `agy.exe -p` in an isolated empty temporary directory (`pocketbot_agy_isolated`) with `stdin=subprocess.DEVNULL` to avoid workspace `.agents/mcp_config.json` offline timeouts.
  - Added per-candle verdict caching `(symbol, side, timeframe, bar_timestamp)` and a 10-minute cooldown to `pure_math` after 2 consecutive failures.
- **Minimum Expected Value Gate (`config/apex_config.json`, `apex/quant/universe_scanner.py`, `apex/ai/veto_committee.py`):**
  - Enforced `min_net_avg_R: 0.10` floor across scanner and AI pure-math fallback evaluations.
- **Verification Suite:**
  - Added 3 regression tests in `tests/test_phase6_universe_scanner.py` (35/35 total tests passing).

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
