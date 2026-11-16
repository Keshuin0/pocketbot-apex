r"""Mechanical Self-Checker for Apex Omega v3.2 Runbook Phases (Gates 0..6).

Usage:
    .\.venv\Scripts\python.exe tools\verify_omega_gates.py --gate 0
    .\.venv\Scripts\python.exe tools\verify_omega_gates.py --gate all
"""

from __future__ import annotations

import argparse
import ast
import inspect
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

sys.dont_write_bytecode = True
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


def _fail(gate: int, msg: str) -> None:
    print(f"[GATE {gate} FAIL] {msg}", file=sys.stderr)
    raise AssertionError(f"Gate {gate} failed: {msg}")


def _pass(gate: int, msg: str) -> None:
    print(f"[GATE {gate} OK] {msg}")


def _read_text(rel_path: str) -> str:
    return (WORKSPACE_ROOT / rel_path).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Gate 0: Workspace Sync, Numba Cache Isolation, MCP Intact & Baseline Pytest
# ---------------------------------------------------------------------------


def verify_gate_0(run_pytest: bool = True) -> None:
    src = _read_text("apex/compute/hyperion_fabric.py")
    numba_cache_idx = src.find("NUMBA_CACHE_DIR")
    import_numba_idx = src.find("import numba")
    if numba_cache_idx < 0:
        _fail(0, "NUMBA_CACHE_DIR is not set in apex/compute/hyperion_fabric.py")
    if import_numba_idx < 0:
        _fail(0, "import numba not found in apex/compute/hyperion_fabric.py")
    if numba_cache_idx > import_numba_idx:
        _fail(0, "NUMBA_CACHE_DIR must be configured BEFORE import numba in apex/compute/hyperion_fabric.py")

    mcp_cfg = WORKSPACE_ROOT / ".agents" / "mcp_config.json"
    if not mcp_cfg.exists():
        _fail(0, f"Missing required MCP config: {mcp_cfg}")

    skills_dir = WORKSPACE_ROOT / ".agents" / "skills"
    if not skills_dir.exists():
        _fail(0, f"Missing required skills dir: {skills_dir}")

    # Import hyperion_fabric and verify no .nbc/.nbi written into repo __pycache__
    import apex.compute.hyperion_fabric as hf  # noqa: F401

    pycache_dir = WORKSPACE_ROOT / "apex" / "compute" / "__pycache__"
    if pycache_dir.exists():
        numba_artifacts = [p for p in pycache_dir.iterdir() if p.suffix in {".nbc", ".nbi"}]
        if numba_artifacts:
            _fail(0, f"Numba wrote cache artifacts into repo __pycache__: {numba_artifacts}")

    if run_pytest:
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        res = subprocess.run(
            [sys.executable, "-B", "-m", "pytest", "tests/", "-q", "-p", "no:cacheprovider"],
            cwd=str(WORKSPACE_ROOT),
            env=env,
        )
        if res.returncode != 0:
            _fail(0, f"Baseline pytest exited with code {res.returncode}")

    _pass(0, "NUMBA_CACHE_DIR set pre-import, .agents/ intact, zero .nbc/.nbi pollution, baseline pytest green.")


# ---------------------------------------------------------------------------
# Gate 1: Concurrent Z3 Contexts, SQLite Lock/Ledger, Single-Instance & Imports
# ---------------------------------------------------------------------------


def verify_gate_1() -> None:
    aegis_src = _read_text("apex/ai/aegis_verifier.py")
    tree = ast.parse(aegis_src)
    z3_ctors = {"Solver", "Real", "Int", "Bool", "RealVal", "BoolVal"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "z3"
                and node.func.attr in z3_ctors
            ):
                has_ctx = any(kw.arg == "ctx" for kw in node.keywords)
                if not has_ctx:
                    _fail(
                        1,
                        f"aegis_verifier.py line {node.lineno}: z3.{node.func.attr}(...) missing explicit ctx=ctx",
                    )

    if "_Z3_LOCK" not in aegis_src:
        _fail(1, "aegis_verifier.py missing module-level _Z3_LOCK = threading.RLock()")

    # 8-thread concurrent Z3 stress test (50 iterations per thread)
    from apex.ai.aegis_verifier import AegisVerifier

    telem = {
        "timemachine_net_avg_R": 0.35,
        "spread_pct_of_sl": 4.0,
        "vol_pct": 52.0,
        "cross_asset_shock_z": 0.6,
        "rollover_illiquidity_flag": False,
        "recommended_risk_pct": 0.50,
        "remaining_daily_budget_pct": 3.0,
    }

    def _worker(tid: int) -> None:
        for i in range(50):
            ok_s, _, msg_s = AegisVerifier.verify_z3_smt_axioms(telem, "APPROVE", [])
            if not ok_s:
                raise AssertionError(f"Thread {tid} iter {i} verify_z3_smt_axioms failed: {msg_s}")
            ok_t, _, msg_t = AegisVerifier.verify_z3_inflight_trajectory(
                proposed_action="HOLD",
                bars_held=4,
                min_hold_bars=2,
                abs_move_atr=0.8,
                cross_asset_shock_z=0.5,
                forward_ev_r=0.35,
                current_r=0.45,
                invalidation_signals=0,
            )
            if not ok_t:
                raise AssertionError(f"Thread {tid} iter {i} verify_z3_inflight_trajectory failed: {msg_t}")

    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = [pool.submit(_worker, t) for t in range(8)]
        for fut in futs:
            fut.result()

    # Double close_trade_with_autopsy on CHFSGD without sqlite3.OperationalError: database is locked
    from apex.ledger.walkforward_db import SymbolExecutionMode, WalkForwardLedgerDB

    with tempfile.TemporaryDirectory() as td:
        db = WalkForwardLedgerDB(Path(td) / "gate1_test.db")
        try:
            if not hasattr(db, "record_trade_open"):
                _fail(1, "WalkForwardLedgerDB missing record_trade_open()")
            for idx in range(2):
                oid = f"CHFSGD-{idx}"
                db.record_trade_open(
                    order_id=oid,
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
                db.close_trade_with_autopsy(
                    order_id=oid,
                    exit_price=1.4950,
                    realized_net_r=-1.0,
                    realized_pnl=-50.0,
                    mfe_r=0.2,
                    mae_r=-1.0,
                    symbol="CHFSGD",
                )
        finally:
            if hasattr(db, "close"):
                db.close()

    # import os in apex_runner.py
    runner_tree = ast.parse(_read_text("apex_runner.py"))
    has_import_os = any(
        isinstance(n, ast.Import) and any(a.name == "os" for a in n.names)
        for n in runner_tree.body
    )
    if not has_import_os:
        _fail(1, "apex_runner.py missing top-level `import os`")

    # Lazy try/except ImportError MetaTrader5 in mt5_feed.py, router.py, position_manager.py
    for rel in ("apex/feeds/mt5_feed.py", "apex/execution/router.py", "apex/execution/position_manager.py"):
        mod_tree = ast.parse(_read_text(rel))
        for n in mod_tree.body:
            if isinstance(n, ast.Import) and any(a.name == "MetaTrader5" for a in n.names):
                _fail(1, f"{rel} has unguarded top-level `import MetaTrader5`")

    _pass(1, "Z3 local contexts 8-thread clean, SQLite un-nested & record_trade_open OK, import os & lazy MT5 verified.")


# ---------------------------------------------------------------------------
# Gate 2: Broker Execution, Closed Bars Only, Risk Cap, Breakers & NBBO
# ---------------------------------------------------------------------------


def verify_gate_2() -> None:
    mt5_src = _read_text("apex/feeds/mt5_feed.py")
    if re.search(r"copy_rates_from_pos\([^)]*,\s*0\s*,", mt5_src):
        _fail(2, "apex/feeds/mt5_feed.py still calls copy_rates_from_pos(..., 0, ...) (forming bar lookahead)")
    if '"time"' not in mt5_src and "'time'" not in mt5_src:
        _fail(2, "apex/feeds/mt5_feed.py get_open_positions missing 'time' field")

    from apex.feeds.models import ContractSpec, Side
    from apex.feeds.mt5_feed import MT5Feed

    feed = MT5Feed()
    spec = ContractSpec(
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
    feed._contract_specs["CHFSGD"] = spec
    # Huge SL + low max risk -> lots_raw < vol_min MUST return 0.0 (never clamp up to 0.01!)
    too_small = feed.compute_compliant_lot_size(
        symbol="CHFSGD",
        entry_price=1.5000,
        stop_loss=1.0000,
        risk_usd=10.0,
    )
    if too_small != 0.0:
        _fail(2, f"compute_compliant_lot_size clamped small volume UP to {too_small} instead of returning 0.0")

    # Commission must reduce maximum affordable lot size vs zero commission
    spec_nocomm = ContractSpec(
        symbol="EURUSD",
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
    spec_comm = ContractSpec(
        symbol="EURUSD_COMM",
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
    feed._contract_specs["EURUSD"] = spec_nocomm
    feed._contract_specs["EURUSD_COMM"] = spec_comm
    l_no = feed.compute_compliant_lot_size("EURUSD", 1.1000, 1.0980, 50.0)
    l_co = feed.compute_compliant_lot_size("EURUSD_COMM", 1.1000, 1.0980, 50.0)
    if l_co >= l_no:
        _fail(2, f"Commission not deducted in compute_compliant_lot_size: comm={l_co} vs nocomm={l_no}")

    router_src = _read_text("apex/execution/router.py")
    if "ORDER_TYPE_BUY_LIMIT" not in router_src or "ORDER_TYPE_SELL_LIMIT" not in router_src:
        _fail(2, "router.py missing ORDER_TYPE_BUY_LIMIT / ORDER_TYPE_SELL_LIMIT for pending orders")

    # Verify _try_execute_ctrader has no fire-and-forget "FILLED" else branch
    if "awaiting event ACK" in router_src or "fire-and-forget" in router_src:
        _fail(2, "router.py still contains fire-and-forget FILLED fallback in _try_execute_ctrader")

    from apex.feeds.ctrader_feed import CTraderOpenApiFeed
    from apex.feeds.hydra_nbbo import HydraNBBOEngine

    ct = CTraderOpenApiFeed()
    ct_res = ct.send_new_order(
        symbol="EURUSD",
        side=Side.BUY,
        volume_lots=0.01,
        stop_loss=1.0990,
        take_profit=1.1020,
    )
    if ct_res.get("ok") is True:
        _fail(2, f"ctrader_feed.send_new_order returned fake ok=True offline: {ct_res}")

    hydra = HydraNBBOEngine()
    q_blocked = hydra.evaluate_nbbo(
        symbol="EURUSD",
        intended_lots=0.10,
        side=Side.BUY,
        ct_tick=None,
        mt5_tick=None,
        ct_ok=False,
        mt5_ok=False,
    )
    if q_blocked.winning_venue != "NONE":
        _fail(2, f"hydra_nbbo returned winning_venue={q_blocked.winning_venue!r} instead of 'NONE' when both down")

    _pass(2, "Closed bars only, commission-inclusive 0.0 floor sizing, fail-closed cTrader, and Hydra NONE verified.")


# ---------------------------------------------------------------------------
# Gate 3: Conformal Monotonicity, QMC Single-Count Spread, Hawkes & Covariance
# ---------------------------------------------------------------------------


def verify_gate_3() -> None:
    from apex.ledger.walkforward_db import WalkForwardLedgerDB
    from apex.quant.regime import HawkesBranchingParameters
    from apex.quant.timemachine_v2 import BlockBootstrapTimeMachine, TimeMachineCostModel

    with tempfile.TemporaryDirectory() as td:
        db = WalkForwardLedgerDB(Path(td) / "gate3_conf.db")
        try:
            for i in range(15):
                aid = db.log_ai_epistemic_audit(
                    symbol="EURUSD",
                    side="BUY",
                    timeframe="M5",
                    provider="studio_api",
                    model_id="test-m",
                    input_sha256=f"h{i}",
                    raw_decision="APPROVE",
                    final_decision="APPROVE",
                    confidence=0.90,
                    hallucination_detected=False,
                    violation_codes="",
                )
                db.grade_ai_shadow_outcome(aid, realized_r=0.8)
            p_hi = db.get_conformal_p_value("studio_api", "test-m", 0.95)
            p_lo = db.get_conformal_p_value("studio_api", "test-m", 0.10)
            p_zero = db.get_conformal_p_value("studio_api", "test-m", 0.0)
            if not (p_hi > p_lo and p_zero <= 0.05):
                _fail(3, f"Conformal p_calib non-monotonic: p(0.95)={p_hi}, p(0.10)={p_lo}, p(0.0)={p_zero}")
        finally:
            if hasattr(db, "close"):
                db.close()

    # Verify timemachine_v2 asymmetric_barriers does not double-count spread_price
    import numpy as np
    from apex.feeds.models import Side

    tm = BlockBootstrapTimeMachine(replays=128)
    cm_no_comm = TimeMachineCostModel(
        spread_price=0.0002,
        commission_per_lot_round_turn_usd=0.0,
        vwap_slippage_price=0.0,
        risk_per_lot_usd=100.0,
        asymmetric_barriers=True,
    )
    # Every bar drops sharply so SL hits immediately with zero commission/slippage
    ratios = np.tile(np.array([1.0, 1.0001, 0.9950, 0.9960], dtype=np.float64), (16, 1))
    res = tm.simulate_from_ratios(
        symbol="EURUSD",
        side=Side.BUY,
        entry_price=1.1000,
        stop_price=1.0990,
        target_price=1.1020,
        bar_ratios=ratios,
        horizon_bars=12,
        cost_model=cm_no_comm,
        bar_ts=1_700_000_000.0,
    )
    if res.round_trip_cost_r > 1e-6 or res.net_avg_R < -1.001:
        _fail(
            3,
            f"TimeMachine double-counted spread on stop-out (cost_r={res.round_trip_cost_r:.4f}, net_avg_R={res.net_avg_R:.4f})",
        )

    hw = HawkesBranchingParameters()
    if hw.alpha_slow >= hw.beta_slow:
        _fail(3, f"Hawkes slow branching ratio >= 1.0 ({hw.alpha_slow} / {hw.beta_slow})")
    if hw.alpha_fast >= hw.beta_fast:
        _fail(3, f"Hawkes fast branching ratio >= 1.0 ({hw.alpha_fast} / {hw.beta_fast})")

    _pass(3, "Monotonic Conformal p_calib, single-count QMC spread, and stationary Hawkes branching verified.")


# ---------------------------------------------------------------------------
# Gate 4: Calibrated NeuralAlpha, 3D Entry Grid & Continuous Sliding SL/TP
# ---------------------------------------------------------------------------


def verify_gate_4() -> None:
    pipe_src = _read_text("apex/engine/canonical_pipeline.py")
    if "NeuralAlphaForecaster" not in pipe_src:
        _fail(4, "canonical_pipeline.py does not call NeuralAlphaForecaster")
    if "optimize_sl_tp_grid_batch" not in pipe_src:
        _fail(4, "canonical_pipeline.py does not call optimize_sl_tp_grid_batch")

    from apex.execution.position_manager import AutonomousPositionOrchestrator, PositionAction
    from apex.feeds.models import Candle, Side
    from apex.quant.neural_alpha import NeuralAlphaForecaster

    forecaster = NeuralAlphaForecaster()
    candles = [
        Candle("EURUSD", "M5", 1700000000.0 + i * 300, 1.1, 1.1005, 1.0995, 1.1, 100.0)
        for i in range(60)
    ]
    fc = forecaster.predict_quantiles(candles, current_atr=0.0005)
    if getattr(fc, "blend_weight", -1.0) != 0.0:
        _fail(4, f"Untrained/flat NeuralAlphaForecaster has non-zero blend_weight={fc.blend_weight}")

    # Zone 1 defensive compression check (negative excursion after grace window with low prob_tp)
    pos_mgr = AutonomousPositionOrchestrator()
    # Flat/downtrending M5 candles after BUY entry so current_r is negative (-0.35R) & P(TP) low
    down_candles = [
        Candle("CHFSGD", "M5", 1700000000.0 + i * 300, 1.5000 - i * 0.00008, 1.5002 - i * 0.00008, 1.4990 - i * 0.00008, 1.4995 - i * 0.00008, 100.0)
        for i in range(130)
    ]
    decision_loss = pos_mgr.evaluate_position(
        ticket=1001,
        symbol="CHFSGD",
        side=Side.BUY,
        entry_price=1.5000,
        mark_price=1.4982,  # -0.36R given 0.0050 initial_one_r
        current_sl=1.4950,
        current_tp=1.5100,
        bars_held=4,
        opened_at_utc=time.time() - 1200.0,
        candles_m5=down_candles,
        spread_price=0.0002,
        initial_one_r=0.0050,
    )
    if decision_loss.action not in (PositionAction.RATCHET_SLTP, PositionAction.EARLY_EXIT):
        _fail(
            4,
            f"Negative excursion dying trade remained in dead zone ({decision_loss.action}, "
            f"p_tp={decision_loss.forward_prob_tp:.1f}%, ev={decision_loss.forward_ev_r:+.2f}R)",
        )

    # Zone 2 continuous winning SL(t)/TP(t) slide at +1.10R using initial_one_r
    up_candles = [
        Candle("AUDCAD", "M5", 1700000000.0 + i * 300, 0.9000 + i * 0.00012, 0.9004 + i * 0.00012, 0.8998 + i * 0.00012, 0.9002 + i * 0.00012, 500.0)
        for i in range(130)
    ]
    decision_win = pos_mgr.evaluate_position(
        ticket=1002,
        symbol="AUDCAD",
        side=Side.BUY,
        entry_price=0.9000,
        mark_price=0.9055,  # +1.10R with initial_one_r=0.0050
        current_sl=0.8950,
        current_tp=0.9100,
        bars_held=5,
        opened_at_utc=time.time() - 1500.0,
        candles_m5=up_candles,
        spread_price=0.0002,
        initial_one_r=0.0050,
    )
    if decision_win.action != PositionAction.RATCHET_SLTP or (decision_win.new_sl or 0.0) <= 0.8950:
        _fail(4, f"Winning +1.10R trade did not slide SL tighter smoothly: {decision_win}")

    _pass(4, "NeuralAlpha zero-skill gate, Zone 1 defensive compression, and Zone 2 continuous SL/TP curve verified.")


# ---------------------------------------------------------------------------
# Gate 5: Causal Replay Engine, real pipeline/costs & last_matrix persistence
# ---------------------------------------------------------------------------


def verify_gate_5() -> None:
    runner_src = _read_text("apex_runner.py")
    if "self.last_matrix = list(rows)" not in runner_src:
        _fail(5, "apex_runner.py does not assign `self.last_matrix = list(rows)` in run_cycle")

    replay_src = _read_text("apex/cockpit/replay_engine.py")
    if "hist[-1].close >= hist[-5].close" in replay_src:
        _fail(5, "replay_engine.py still contains toy `hist[-1].close >= hist[-5].close` shortcut")
    if "bars_held >= 7" in replay_src:
        _fail(5, "replay_engine.py still contains artificial `bars_held >= 7` forced close")

    from apex.cockpit.replay_engine import HistoricalReplayEngine
    from apex.feeds.models import Candle

    # Test causal H1 filtering at bar_ts: future H1 bars (+3600 > bar_ts) must be excluded!
    h1_bars = [
        Candle("EURUSD", "H1", 1700000000.0 + i * 3600.0, 1.1, 1.11, 1.09, 1.10, 1000.0)
        for i in range(10)
    ]
    bar_ts = 1700000000.0 + 5 * 3600.0
    causal = HistoricalReplayEngine.filter_causal_h1_candles(h1_bars, bar_ts)
    if any(c.timestamp_utc + 3600.0 > bar_ts for c in causal):
        _fail(5, "Replay H1 causal filter leaked incomplete/future H1 candle!")
    if len(causal) != 5:
        _fail(5, f"Expected 5 completed H1 candles at bar_ts, got {len(causal)}")

    _pass(5, "self.last_matrix persisted, causal H1 replay filter & real pipeline/costs verified.")


# ---------------------------------------------------------------------------
# Gate 6: Single Version Source, XSS Elimination, Origin /api/state & TUI
# ---------------------------------------------------------------------------


def verify_gate_6() -> None:
    import apex.__version__ as ver_mod

    if ver_mod.__version__ != "3.2.0" or ver_mod.DISPLAY_VERSION != "v3.2.0 OMEGA":
        _fail(6, f"Unexpected version metadata: {ver_mod.__version__!r} / {ver_mod.DISPLAY_VERSION!r}")

    for rel in ("apex_runner.py", "apex/cockpit/tui_dashboard.py", "apex/cockpit/web_cockpit.py"):
        txt = _read_text(rel)
        if "__version__" not in txt:
            _fail(6, f"{rel} does not import apex.__version__")
        if "v2.4.1" in txt:
            _fail(6, f"{rel} still contains stale hardcoded 'v2.4.1' string")

    html = _read_text("apex/cockpit/static/index.html")
    if 'onclick="openXrayModal' in html:
        _fail(6, "static/index.html still contains inline XSS onclick=\"openXrayModal(...)\"")
    if "v2.4.1" in html or "v2.7.1" in html:
        _fail(6, "static/index.html still contains stale hardcoded version string")

    drill_src = _read_text("tools/live_lifecycle_drill.py")
    if "COMPLETED" not in drill_src or "COMPLETED" not in html:
        _fail(6, "COMPLETED status mismatch between live_lifecycle_drill.py and index.html")

    # Verify /api/state enforces _is_same_origin on foreign Origin
    from apex.cockpit.web_cockpit import WebCockpitServer
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    srv = WebCockpitServer(port=8841, max_port=8849)
    try:
        base = srv.start()
        bad_req = Request(f"{base}/api/state", headers={"Origin": "https://evil.example"})
        try:
            urlopen(bad_req, timeout=5.0)
            _fail(6, "GET /api/state allowed cross-origin request from https://evil.example")
        except HTTPError as err:
            if err.code != 403:
                _fail(6, f"GET /api/state returned HTTP {err.code} instead of 403 for foreign Origin")
    finally:
        srv.stop()

    # Verify TUI renders EURUSD visibly without blanking Symbol column
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"}
    proc = subprocess.run(
        [sys.executable, "-B", "-m", "apex.cockpit.tui_dashboard", "--equity", "12500", "--mode", "DEMO-AUTO"],
        cwd=str(WORKSPACE_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=20,
    )
    if proc.returncode != 0:
        _fail(6, f"tui_dashboard CLI exited {proc.returncode}: {proc.stderr}")
    if "EURUSD" not in proc.stdout:
        _fail(6, "tui_dashboard CLI blanked out Symbol column (EURUSD missing from stdout)")

    _pass(6, "Single v3.2.0 OMEGA version, zero XSS onclick, /api/state Origin check, and visible TUI Symbol verified.")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="PocketBot Apex Omega v3.2 Gate Inspector")
    parser.add_argument(
        "--gate",
        choices=["0", "1", "2", "3", "4", "5", "6", "all"],
        required=True,
        help="Gate number to inspect (or 'all')",
    )
    args = parser.parse_args()

    gates: dict[int, Callable[[], None]] = {
        0: verify_gate_0,
        1: verify_gate_1,
        2: verify_gate_2,
        3: verify_gate_3,
        4: verify_gate_4,
        5: verify_gate_5,
        6: verify_gate_6,
    }

    targets = [0, 1, 2, 3, 4, 5, 6] if args.gate == "all" else [int(args.gate)]
    for g in targets:
        try:
            gates[g]()
        except Exception as exc:
            print(f"[!] GATE {g} VERIFICATION FAILED: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
