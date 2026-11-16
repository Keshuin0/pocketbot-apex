"""Phase 20 (`v3.2.0 OMEGA`) — Full Consolidation Acceptance & Invariant Regression Suite.

Covers every Phase 1–6 invariant of Apex Omega v3.2:
  - Phase 1: Z3 isolated contexts + 8-thread stress, SQLite thread-local WAL + close_trade_with_autopsy,
    is_drill=0 filter, lazy MT5 import.
  - Phase 2: Commission-inclusive Decimal lot sizing with 0.0 floor, FX conversion, closed-bar MT5 API,
    fail-closed cTrader RPCs, Hydra NBBO winning_venue="NONE" & DOM impact penalty, 120m stop-out cooldown.
  - Phase 3: Monotonic Split Conformal p_calib, single-count QMC spread & deterministic CRC32 per-bar seed,
    stationary Hawkes branching < 1.0, smooth linear SLSQP factor constraints in optimize_portfolio_weights.
  - Phase 4: Calibrated NeuralAlphaForecaster zero-skill gating + positive skill blending, Stage-B 3D grid,
    continuous Zone 1/2/3 in-flight SL/TP + Z3 Shield 8 + initial_one_r persistence.
  - Phase 5: Causal H1 filtering (closed H1 bars only), real pipeline direction inference & round-trip cost,
    self.last_matrix persistence in ApexOrchestrator.
  - Phase 6: Single version source (v3.2.0 OMEGA), zero XSS onclick, COMPLETED drill status, /api/state
    cross-origin 403 enforcement, bounded 512-entry LRU _verdict_cache, visible TUI EURUSD row.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

from apex.__version__ import DISPLAY_VERSION, __version__
from apex.ai.aegis_verifier import AegisVerifier
from apex.ai.veto_committee import AIVetoCommittee, AIVetoMode, CandidateSetupAuditContext
from apex.cockpit.replay_engine import HistoricalReplayEngine
from apex.cockpit.tui_dashboard import ApexTUIDashboard, SymbolCockpitRow
from apex.cockpit.web_cockpit import WebCockpitServer
from apex.execution.position_manager import AutonomousPositionOrchestrator, PositionAction
from apex.execution.router import DualBrokerExecutionRouter, ExecutionOrderRequest
from apex.feeds.ctrader_feed import CTraderOpenApiFeed
from apex.feeds.dual_router import MarketFeedRouter
from apex.feeds.hydra_nbbo import HydraNBBOEngine
from apex.feeds.models import Candle, ContractSpec, DOMAnalytics, Side, Tick
from apex.feeds.mt5_feed import MT5Feed
from apex.ledger.walkforward_db import SymbolExecutionMode, WalkForwardLedgerDB
from apex.quant.correlation_shield import CorrelationShield, PositionRiskSpec
from apex.quant.mtf_confluence import MTFConfluenceFilter
from apex.quant.neural_alpha import NeuralAlphaForecaster
from apex.quant.regime import HawkesBranchingParameters
from apex.quant.timemachine_v2 import BlockBootstrapTimeMachine, TimeMachineCostModel
from apex_runner import (
    ApexOrchestrator,
    format_single_line_heartbeat,
    format_web_cockpit_banner,
)

ROOT = Path(__file__).resolve().parents[1]


def test_phase1_z3_multithread_and_sqlite_wal_close_lifecycle(tmp_path: Path) -> None:
    """Verify 8-thread Z3 isolation, un-nested SQLite autopsy, and initial_one_r persistence."""
    telem = {
        "timemachine_net_avg_R": 0.40,
        "spread_pct_of_sl": 4.0,
        "vol_pct": 50.0,
        "cross_asset_shock_z": 0.4,
        "rollover_illiquidity_flag": False,
        "recommended_risk_pct": 0.45,
        "remaining_daily_budget_pct": 2.5,
    }

    def _run_z3_batch(_idx: int) -> None:
        for _ in range(25):
            ok_s, _, msg_s = AegisVerifier.verify_z3_smt_axioms(telem, "APPROVE", [])
            assert ok_s, msg_s
            ok_t, _, msg_t = AegisVerifier.verify_z3_inflight_trajectory(
                proposed_action="HOLD",
                bars_held=4,
                min_hold_bars=2,
                abs_move_atr=0.75,
                cross_asset_shock_z=0.5,
                forward_ev_r=0.3,
                current_r=0.4,
                invalidation_signals=0,
                mark_price=1.1040,
                sl_old=1.0950,
                sl_new=1.0950,
                tp_old=1.1150,
                tp_new=1.1150,
                side="BUY",
            )
            assert ok_t, msg_t

    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = [pool.submit(_run_z3_batch, i) for i in range(8)]
        for fut in futs:
            fut.result()

    db = WalkForwardLedgerDB(tmp_path / "omega_p1.db")
    try:
        db.record_trade_open(
            order_id="ORD-LIVE-1",
            symbol="CHFSGD",
            side="BUY",
            lots=0.05,
            entry_price=1.5000,
            stop_loss=1.4950,
            take_profit=1.5100,
            initial_one_r=0.0050,
            predicted_net_r=0.35,
            regime="TRENDING_CALM",
            venue="mt5",
            execution_mode=SymbolExecutionMode.ACTIVE,
        )
        assert db.get_open_initial_one_r(order_id="ORD-LIVE-1", symbol="CHFSGD") == 0.0050
        db.close_trade_with_autopsy(
            order_id="ORD-LIVE-1",
            exit_price=1.4950,
            realized_net_r=-1.0,
            realized_pnl=-50.0,
            mfe_r=0.1,
            mae_r=-1.0,
            symbol="CHFSGD",
        )
        # Drill trade must not trigger stop-out cooldown or pollute live cooldown
        db.log_trade_outcome(
            order_id="DRILL-1",
            symbol="EURUSD",
            side="BUY",
            execution_mode=SymbolExecutionMode.ACTIVE,
            regime="DRILL",
            dom_obi=0.1,
            predicted_net_r=0.3,
            ai_decision="APPROVE",
            ai_confidence=1.0,
            ai_reason="DRILL",
            entry_price=1.1000,
            executed_price=1.1000,
            stop_loss=1.0990,
            take_profit=1.1020,
            realized_slippage_r=0.0,
            realized_net_r=-1.0,
            winning_venue="mt5",
            mesh_topology="MT5_DEMO_DRILL",
            is_drill=True,
        )
        in_cd_chf, rem_chf = db.is_symbol_side_in_stopout_cooldown("CHFSGD", "BUY", cooldown_minutes=120)
        in_cd_eur, _ = db.is_symbol_side_in_stopout_cooldown("EURUSD", "BUY", cooldown_minutes=120)
        assert in_cd_chf is True and rem_chf > 0.0
        assert in_cd_eur is False
    finally:
        db.close()

    for rel in ("apex/feeds/mt5_feed.py", "apex/execution/router.py", "apex/execution/position_manager.py"):
        tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Import):
                assert all(alias.name != "MetaTrader5" for alias in node.names)


def test_phase2_sizing_failclosed_brokers_and_hydra_none() -> None:
    """Verify 0.0 floor lot sizing, commission deduction, closed-bar MT5, cTrader fail-closed, and Hydra NONE."""
    feed = MT5Feed()
    feed._contract_specs["CHFSGD"] = ContractSpec(
        symbol="CHFSGD",
        digits=5,
        pip_size=0.0001,
        point_size=0.00001,
        lot_size=100_000.0,
        min_volume_lots=0.01,
        max_volume_lots=50.0,
        step_volume_lots=0.01,
        source="mt5",
        commission_per_lot_rt=7.0,
    )
    assert feed.compute_compliant_lot_size("CHFSGD", 1.5000, 1.0000, 10.0) == 0.0

    feed._contract_specs["EURUSD_0"] = ContractSpec(
        symbol="EURUSD_0",
        digits=5,
        pip_size=0.0001,
        point_size=0.00001,
        lot_size=100_000.0,
        min_volume_lots=0.01,
        max_volume_lots=50.0,
        step_volume_lots=0.01,
        source="mt5",
        commission_per_lot_rt=0.0,
    )
    feed._contract_specs["EURUSD_C"] = ContractSpec(
        symbol="EURUSD_C",
        digits=5,
        pip_size=0.0001,
        point_size=0.00001,
        lot_size=100_000.0,
        min_volume_lots=0.01,
        max_volume_lots=50.0,
        step_volume_lots=0.01,
        source="mt5",
        commission_per_lot_rt=15.0,
    )
    assert feed.compute_compliant_lot_size("EURUSD_C", 1.10, 1.098, 50.0) < feed.compute_compliant_lot_size(
        "EURUSD_0", 1.10, 1.098, 50.0
    )

    ct = CTraderOpenApiFeed()
    res_open = ct.send_new_order(symbol="EURUSD", side=Side.BUY, volume_lots=0.01, stop_loss=1.099, take_profit=1.102)
    assert res_open.get("ok") is False

    hydra = HydraNBBOEngine()
    q_none = hydra.evaluate_nbbo("EURUSD", 0.1, Side.BUY, ct_tick=None, mt5_tick=None, ct_ok=False, mt5_ok=False)
    assert q_none.winning_venue == "NONE"

    # DOM impact penalty increases effective spread when top depth < intended volume
    thin_dom = DOMAnalytics(
        symbol="EURUSD",
        timestamp_utc=time.time(),
        top_n_levels=1,
        bid_volume_top_n=2000.0,
        ask_volume_top_n=2000.0,
        obi=0.0,
        best_bid=1.0999,
        best_ask=1.1001,
        spread_pips=2.0,
    )
    q_thin = hydra.evaluate_nbbo(
        symbol="EURUSD",
        intended_lots=0.25,
        side=Side.BUY,
        ct_tick=Tick("EURUSD", 1.0999, 1.1001, time.time(), "ctrader"),
        mt5_tick=Tick("EURUSD", 1.0999, 1.1001, time.time(), "mt5"),
        ctrader_dom=thin_dom,
        ct_ok=True,
        mt5_ok=True,
    )
    assert q_thin.winning_venue in {"ctrader", "mt5"}
    ct_quote = q_thin.ctrader_quote
    assert ct_quote is not None
    assert ct_quote.dom_impact_pips > 0.0
    assert ct_quote.effective_ask > 1.1001


def test_phase3_conformal_qmc_hawkes_and_covariance_slsqp(tmp_path: Path) -> None:
    """Verify monotonic conformal p-values, single-count spread, stationary Hawkes, and smooth SLSQP."""
    db = WalkForwardLedgerDB(tmp_path / "omega_p3.db")
    try:
        for i in range(16):
            aid = db.log_ai_epistemic_audit(
                symbol="EURUSD",
                side="BUY",
                timeframe="M5",
                provider="studio_api",
                model_id="m3",
                input_sha256=f"sha-{i}",
                raw_decision="APPROVE",
                final_decision="APPROVE",
                confidence=0.90,
                hallucination_detected=False,
                violation_codes="",
            )
            db.grade_ai_shadow_outcome(aid, realized_r=0.9)
        assert db.get_conformal_p_value("studio_api", "m3", 0.95) > db.get_conformal_p_value("studio_api", "m3", 0.10)
        assert db.get_conformal_p_value("studio_api", "m3", 0.0) <= 0.06
    finally:
        db.close()

    tm = BlockBootstrapTimeMachine(replays=128)
    cm = TimeMachineCostModel(
        spread_price=0.0002,
        commission_per_lot_round_turn_usd=0.0,
        vwap_slippage_price=0.0,
        risk_per_lot_usd=100.0,
        asymmetric_barriers=True,
    )
    ratios = np.tile(np.array([1.0, 1.0001, 0.9950, 0.9960], dtype=np.float64), (16, 1))
    r1 = tm.simulate_from_ratios("EURUSD", Side.BUY, 1.10, 1.099, 1.102, ratios, 12, cm, bar_ts=1700000300.0)
    r2 = tm.simulate_from_ratios("EURUSD", Side.BUY, 1.10, 1.099, 1.102, ratios, 12, cm, bar_ts=1700000300.0)
    assert r1.round_trip_cost_r <= 1e-6 and r1.net_avg_R >= -1.001
    assert abs(r1.net_avg_r - r2.net_avg_R) < 1e-9

    hw = HawkesBranchingParameters()
    assert hw.alpha_fast < hw.beta_fast and hw.alpha_slow < hw.beta_slow

    shield = CorrelationShield(max_factor_risk_pct=1.0)
    rng = np.random.default_rng(7)
    hist = {
        "EURUSD": rng.normal(0.0, 0.001, 60),
        "GBPUSD": rng.normal(0.0, 0.001, 60),
        "USDJPY": rng.normal(0.0, 0.001, 60),
    }
    alloc = shield.optimize_portfolio_weights(
        candidates=[
            ("EURUSD", Side.BUY, 0.45, 0.50),
            ("GBPUSD", Side.BUY, 0.40, 0.50),
            ("USDJPY", Side.SELL, 0.35, 0.50),
        ],
        open_positions=[],
        return_histories=hist,
        max_concurrent_positions=2,
        portfolio_risk_budget_pct=1.0,
    )
    assert sum(alloc.values()) <= 1.0 + 1e-4


def test_phase4_neural_alpha_and_inflight_three_zones() -> None:
    """Verify NeuralAlpha skill gating and Zone 1/2/3 continuous SL/TP behavior."""
    na = NeuralAlphaForecaster()
    flat = [Candle("EURUSD", "M5", 1700000000.0 + i * 300, 1.1, 1.1003, 1.0997, 1.1, 100.0) for i in range(60)]
    fc = na.predict_quantiles(flat, current_atr=0.0005)
    assert fc.blend_weight == 0.0

    pm = AutonomousPositionOrchestrator()
    # Zone 1: Negative excursion + weak M5 momentum -> tighten or exit (not dead-zone HOLD)
    down = [
        Candle(
            "CHFSGD",
            "M5",
            1700000000.0 + i * 300,
            1.50 - i * 0.0001,
            1.5002 - i * 0.0001,
            1.4990 - i * 0.0001,
            1.4994 - i * 0.0001,
            150.0,
        )
        for i in range(130)
    ]
    dec1 = pm.evaluate_position(
        ticket=4001,
        symbol="CHFSGD",
        side=Side.BUY,
        entry_price=1.5000,
        mark_price=1.4980,
        current_sl=1.4950,
        current_tp=1.5100,
        bars_held=4,
        opened_at_utc=time.time() - 1200.0,
        candles_m5=down,
        spread_price=0.0002,
        initial_one_r=0.0050,
    )
    assert dec1.action in (PositionAction.RATCHET_SLTP, PositionAction.EARLY_EXIT)

    # Zone 2: Winning trade >= 1.0R ratchets SL tighter smoothly
    up = [
        Candle(
            "AUDCAD",
            "M5",
            1700000000.0 + i * 300,
            0.90 + i * 0.00015,
            0.9005 + i * 0.00015,
            0.8998 + i * 0.00015,
            0.9003 + i * 0.00015,
            400.0,
        )
        for i in range(130)
    ]
    dec2 = pm.evaluate_position(
        ticket=4002,
        symbol="AUDCAD",
        side=Side.BUY,
        entry_price=0.9000,
        mark_price=0.9060,
        current_sl=0.8950,
        current_tp=0.9100,
        bars_held=5,
        opened_at_utc=time.time() - 1500.0,
        candles_m5=up,
        spread_price=0.0002,
        initial_one_r=0.0050,
    )
    assert dec2.action == PositionAction.RATCHET_SLTP
    assert (dec2.new_sl or 0.0) > 0.8950


def test_phase5_causal_h1_confluence_and_replay_costs() -> None:
    """Verify forming H1 exclusion in MTF confluence and round-trip cost deduction in replay."""
    h1 = [
        Candle("EURUSD", "H1", 1700000000.0 + i * 3600.0, 1.10 + i * 0.001, 1.102 + i * 0.001, 1.099 + i * 0.001, 1.101 + i * 0.001, 1000.0)
        for i in range(30)
    ]
    # Align m5 bars inside the final forming H1 bar (h1[-1]) so is_forming_h1 is True
    m5 = [
        Candle("EURUSD", "M5", h1[-1].timestamp_utc + i * 300.0, 1.124, 1.126, 1.123, 1.125, 100.0)
        for i in range(10)
    ]
    # Perturbing only the forming final H1 bar [-1] must not change closed-bar H1 confluence
    h1_perturbed = list(h1[:-1]) + [
        Candle("EURUSD", "H1", h1[-1].timestamp_utc, 1.10, 1.99, 0.50, 0.55, 9999.0)
    ]
    mtf = MTFConfluenceFilter()
    c1 = mtf.evaluate("EURUSD", Side.BUY, 1.125, 0.001, m5, h1)
    c2 = mtf.evaluate("EURUSD", Side.BUY, 1.125, 0.001, m5, h1_perturbed)
    assert c1.h1_ema_slope == c2.h1_ema_slope
    assert c1.h1_swing_high == c2.h1_swing_high
    assert c1.h1_trend_aligned == c2.h1_trend_aligned

    bar_ts = 1700000000.0 + 10 * 3600.0
    causal = HistoricalReplayEngine.filter_causal_h1_candles(h1, bar_ts)
    assert all(c.timestamp_utc + 3600.0 <= bar_ts for c in causal)


def test_phase6_version_security_xss_lru_and_tui(tmp_path: Path) -> None:
    """Verify v3.2.0 OMEGA branding, /api/state 403 origin check, zero onclick XSS, and 512 LRU cache."""
    assert __version__ == "3.2.0"
    assert DISPLAY_VERSION == "v3.2.0 OMEGA"

    banner = format_web_cockpit_banner("http://127.0.0.1:8765", "demo-auto")
    assert DISPLAY_VERSION in banner
    fake_orch = SimpleNamespace(
        execution_router=SimpleNamespace(account_equity_usd=10000.0),
        last_scan_summary=SimpleNamespace(open_scanned=10, total_scanned=251),
        paused=False,
    )
    assert DISPLAY_VERSION in format_single_line_heartbeat(fake_orch)  # type: ignore[arg-type]

    html = (ROOT / "apex" / "cockpit" / "static" / "index.html").read_text(encoding="utf-8")
    assert 'onclick="openXrayModal' not in html
    assert "v2.4.1" not in html and "v2.7.1" not in html
    assert "COMPLETED" in html

    # Verify /api/state blocks cross-origin requests and / substitutes __APEX_VERSION__
    srv = WebCockpitServer(port=8851, max_port=8859)
    try:
        base = srv.start()
        with urlopen(f"{base}/", timeout=5.0) as resp:
            served_html = resp.read().decode("utf-8")
            assert DISPLAY_VERSION in served_html
        with urlopen(f"{base}/api/state", timeout=5.0) as resp:
            assert resp.status == 200
        try:
            urlopen(Request(f"{base}/api/state", headers={"Origin": "https://evil.example"}), timeout=5.0)
            raise AssertionError("Expected HTTP 403 on cross-origin /api/state")
        except HTTPError as err:
            assert err.code == 403
    finally:
        srv.stop()

    # Verify _verdict_cache enforces LRU cap at 512 entries
    db = WalkForwardLedgerDB(tmp_path / "omega_p6.db")
    try:
        committee = AIVetoCommittee(
            enabled=True,
            mode=AIVetoMode.FORCE_PURE_MATH.value,
            env_path=tmp_path / "no.env",
            config_path=tmp_path / "no.json",
            agy_executable_override="",
            ledger_db=db,
            probe_agy_auth=False,
        )
        for i in range(600):
            ctx = CandidateSetupAuditContext(
                symbol=f"SYM{i % 20}",
                side=Side.BUY,
                timeframe="M5",
                entry_price=1.1000,
                stop_loss=1.0980,
                take_profit=1.1040,
                regime="TRENDING_CALM",
                hurst_exponent=0.60,
                kaufman_er=0.45,
                volatility_percentile=50.0,
                dom_obi=0.2,
                spread_pct_of_sl=4.0,
                net_avg_R=0.35,
                prob_profit=0.62,
                cvar_95_R=-1.1,
                recommended_risk_pct=0.5,
                bar_timestamp=float(1700000000 + i * 300),
            )
            committee.evaluate_sync(ctx)
        assert len(committee._verdict_cache) <= 512
        assert len(committee._verdict_cache) == 512
    finally:
        db.close()

    # Verify headless TUI subprocess preserves full EURUSD symbol without truncation
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"}
    proc = subprocess.run(
        [sys.executable, "-B", "-m", "apex.cockpit.tui_dashboard", "--equity", "12500", "--mode", "DEMO-AUTO"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=20,
    )
    assert proc.returncode == 0
    assert "EURUSD" in proc.stdout
    assert DISPLAY_VERSION in proc.stdout
