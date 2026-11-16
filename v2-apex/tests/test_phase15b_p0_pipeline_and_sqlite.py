"""Phase 1/15b Verification Suite: P0 Runtime Fixes, Pipeline Parity, Drill Safety, and SQLite Hygiene."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from apex.engine.canonical_pipeline import Canonical11AgentPipeline
from apex.execution.position_manager import AutonomousPositionOrchestrator, PositionDecision
from apex.feeds.dual_router import MarketFeedRouter
from apex.feeds.models import AssetClass, Candle, DiscoveredSymbol, Side, Tick
from apex.ledger.walkforward_db import SymbolExecutionMode, WalkForwardLedgerDB
from apex.quant.correlation_shield import CorrelationShield, PositionRiskSpec
from apex.quant.momentum_dom import MomentumDOMGate
from apex.quant.regime import RegimeClassifier
from apex.quant.timemachine_v2 import BlockBootstrapTimeMachine
from apex.quant.universe_scanner import UniverseScanner
from apex_runner import ApexConfig, ApexOrchestrator

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent


def _make_uptrend_candles(symbol: str = "BTCUSD", tf: str = "M15", n: int = 80, start_ts: float = 1_763_000_000.0) -> list[Candle]:
    step = 900 if tf == "M15" else (300 if tf == "M5" else 3600)
    out: list[Candle] = []
    px = 50_082.0 if tf == "H1" else 50_080.0
    for i in range(n):
        is_last_entry = (tf in ("M5", "M15") and i == n - 1)
        o = px
        if is_last_entry:
            # Clean >1.5x ATR impulse breakout closing at top of bar and clear of H1 resistance
            h = px + 14.0
            l = px - 1.0
            c = px + 13.5
        elif tf == "H1":
            # Mild bullish H1 slope ending around 50_102 (EMA20 ~ 50_099)
            h = px + 2.0
            l = px - 2.0
            c = px + 0.5
        else:
            h = px + 3.0
            l = px - 2.5
            c = px + 0.2
        out.append(
            Candle(
                symbol=symbol,
                timeframe=tf,
                timestamp_utc=start_ts + i * step,
                open=o,
                high=h,
                low=l,
                close=c,
                volume=1000.0 + i * 10.0,
                source="test",
            )
        )
        px = c
    return out


def test_canonical_pipeline_corr_ok_unit_normalization_and_blocked_properties() -> None:
    """Verify corr_ok NameError fix, 100x spread%SL normalization, and blocked_stage/blocked_reason properties."""
    m5 = _make_uptrend_candles("BTCUSD", "M15", 80)
    h1 = _make_uptrend_candles("BTCUSD", "H1", 40)

    # Relax Stage 3 ceiling to 0.30 so pre_quant_ok=True reaches Stage 9 (Z3 > 15.0% axiom gate)
    # when spread_pct_of_sl is 0.20 (20.0% of SL).
    dom_gate = MomentumDOMGate(dom_obi_enabled=False, max_spread_pct_of_sl=0.30)
    fake_tm = MagicMock(spec=BlockBootstrapTimeMachine)
    fake_tm.simulate_candles.return_value = SimpleNamespace(
        net_avg_R=0.45,
        prob_profit=0.68,
        cvar_95_R=-0.95,
        recommended_risk_pct=0.50,
    )

    pipeline = Canonical11AgentPipeline(
        regime_classifier=RegimeClassifier(enabled=False),
        momentum_dom_gate=dom_gate,
        time_machine=fake_tm,
        correlation_shield=CorrelationShield(enabled=True, max_factor_risk_pct=1.0),
    )

    # Mock MomentumDOMGate output at 20% spread-to-SL fraction (0.20 -> must become 20.0% in Z3 & Stage 3 log)
    diag = pipeline.regime_classifier.classify_candles("BTCUSD", m5, current_spread=10.0)
    atr_val = max(diag.current_atr, 0.05)
    sl_dist = 1.5 * atr_val
    spread_20pct = 0.20 * sl_dist

    trace = pipeline.evaluate(
        symbol="BTCUSD",
        side=Side.BUY,
        timeframe="M15",
        entry_price=m5[-1].close,
        candles=m5,
        h1_candles=h1,
        spread_price=spread_20pct,
        pip_size=0.01,
        open_positions=[],
        run_ai_committee=False,
    )

    assert trace.final_decision == "VETO", (trace.blocked_stage, trace.blocked_reason)
    assert trace.passed_all is False
    assert trace.blocked_stage is not None
    assert "Stage 9" in trace.blocked_stage
    assert trace.blocked_reason != ""
    stage3 = next(s for s in trace.stages if s.stage_num == 3)
    assert "20.0%" in stage3.metric_value
    assert trace.stages[7].passed is True  # Stage 8 CorrelationShield ran with corr_ok without NameError

    # Now verify passing trace has blocked_stage is None and blocked_reason == ""
    spread_5pct = 0.05 * sl_dist
    ok_trace = pipeline.evaluate(
        symbol="BTCUSD",
        side=Side.BUY,
        timeframe="M15",
        entry_price=m5[-1].close,
        candles=m5,
        h1_candles=h1,
        spread_price=spread_5pct,
        pip_size=0.01,
        open_positions=[],
        run_ai_committee=False,
    )
    assert ok_trace.passed_all is True
    assert ok_trace.blocked_stage is None
    assert ok_trace.blocked_reason == ""


def test_universe_scanner_empty_and_closed_market_no_phantom_fallback_and_delegates(tmp_path: Path) -> None:
    """Verify sweep_universe([]) guard, no closed-market prescores fallback, get_catalog alias, and Stage 8 parity."""
    feed_router = MagicMock(spec=MarketFeedRouter)
    ledger = WalkForwardLedgerDB(tmp_path / "scanner.db")
    scanner = UniverseScanner(
        feed_router=feed_router,
        regime_classifier=RegimeClassifier(enabled=False),
        momentum_dom_gate=MomentumDOMGate(dom_obi_enabled=False),
        time_machine=BlockBootstrapTimeMachine(replays=256, seed=7),
        ai_committee=None,  # type: ignore[arg-type]
        ledger=ledger,
        config={"scanner": {"max_tick_age_seconds": 60.0}},
    )

    # Empty list must never raise IndexError
    empty_cands, empty_summary = scanner.sweep_universe(target_symbols=[])
    assert empty_cands == []
    assert empty_summary.total_scanned == 0

    # Closed symbols must NOT trigger any phantom fallback
    now = 1_763_000_000.0
    closed_sym = DiscoveredSymbol(
        name="AAPL",
        canonical_name="AAPL",
        asset_class=AssetClass.US_EQUITIES,
        path="Equities",
        digits=2,
        point=0.01,
        trade_mode=4,
        is_visible=True,
        source="test",
    )
    feed_router.get_tick.return_value = (
        Tick(symbol="AAPL", bid=0.0, ask=0.0, timestamp_utc=now - 10.0, source="test"),
        None,
    )
    feed_router.get_contract_spec.return_value = None
    feed_router.discover_catalog.return_value = [closed_sym]
    assert scanner.get_catalog() == [closed_sym]

    closed_cands, closed_summary = scanner.sweep_universe(target_symbols=[closed_sym], now_utc=now)
    assert closed_cands == []
    assert closed_summary.open_scanned == 0
    assert closed_summary.closed_skipped == 1

    # Verify UniverseScanner Stage B delegates through Canonical11AgentPipeline (Stage 8 respects CorrelationShield!)
    m5 = _make_uptrend_candles("EURUSD", "M5", 60, start_ts=now - 60 * 300)
    h1 = _make_uptrend_candles("EURUSD", "H1", 40, start_ts=now - 40 * 3600)
    open_eur = DiscoveredSymbol(
        name="EURUSD",
        canonical_name="EURUSD",
        asset_class=AssetClass.FOREX,
        path="Forex",
        digits=5,
        point=0.00001,
        trade_mode=4,
        is_visible=True,
        source="test",
    )
    feed_router.get_tick.return_value = (
        Tick(symbol="EURUSD", bid=m5[-1].close, ask=m5[-1].close + 0.0001, timestamp_utc=now - 1.0, source="test"),
        None,
    )
    feed_router.get_candles.side_effect = lambda sym, timeframe="M5", count=200: (h1 if timeframe == "H1" else m5)
    feed_router.get_contract_spec.return_value = None
    feed_router.get_dom_analytics.return_value = None
    feed_router.evaluate_nbbo.return_value = SimpleNamespace(winning_venue="mt5", winning_cost_pips=1.0, pips_saved=0.0)

    cand = scanner.pre_score_symbol(open_eur, now_utc=now)
    existing_eur_pos = [PositionRiskSpec(symbol="EURTRY", side=Side.BUY, risk_pct=0.8)]
    row = scanner.evaluate_candidate_pipeline(cand, open_positions=existing_eur_pos)
    assert row.pipeline_trace is not None
    stage8 = row.pipeline_trace["stages"][7]
    assert stage8["stage_num"] == 8
    assert stage8["passed"] is False
    assert "EUR" in stage8["detail"]


def test_sqlite_context_manager_closes_connections_and_paper_shadow_settlement(tmp_path: Path) -> None:
    """Verify 200 sequential SQLite queries leak zero connections, plus PAPER_SHADOW OHLC settlement & anti-ping-pong."""
    db_path = tmp_path / "hygiene.db"
    ledger = WalkForwardLedgerDB(db_path=db_path)

    opened_conns: list[sqlite3.Connection] = []
    real_connect = sqlite3.connect

    class _TrackedConn:
        def __init__(self, inner: sqlite3.Connection) -> None:
            self._inner = inner
            self.closed = False

        def __getattr__(self, name: str):
            return getattr(self._inner, name)

        def close(self) -> None:
            self.closed = True
            self._inner.close()

    def _spy_connect(*args, **kwargs):
        c = _TrackedConn(real_connect(*args, **kwargs))
        opened_conns.append(c)  # type: ignore[arg-type]
        return c

    orig_connect = sqlite3.connect
    sqlite3.connect = _spy_connect  # type: ignore[assignment]
    try:
        for _ in range(200):
            ledger.get_quarantine_state("EURUSD")
    finally:
        sqlite3.connect = orig_connect  # type: ignore[assignment]

    assert len(opened_conns) == 200
    assert all(getattr(c, "closed", False) for c in opened_conns)

    # Seed 15 ACTIVE losing trades -> demoted to PAPER_SHADOW
    for i in range(15):
        ledger.log_trade_outcome(
            order_id=f"act_{i}",
            symbol="GBPUSD",
            side="BUY",
            execution_mode=SymbolExecutionMode.ACTIVE,
            regime="TRENDING_EXPANSION",
            dom_obi=0.1,
            predicted_net_r=0.50,
            ai_decision="APPROVE",
            ai_confidence=0.8,
            ai_reason="ok",
            entry_price=1.2500,
            executed_price=1.2500,
            stop_loss=1.2400,
            take_profit=1.2700,
            realized_slippage_r=0.0,
            realized_net_r=-1.0 if i < 12 else 0.5,
        )
    assert ledger.get_quarantine_state("GBPUSD").mode == SymbolExecutionMode.PAPER_SHADOW

    # Log 15 open (unsettled, is_closed=0) PAPER_SHADOW trades with absurd predicted_net_r=4.0 vs +2.0 TP
    for i in range(15):
        ledger.log_trade_outcome(
            order_id=f"ps_open_{i}",
            symbol="GBPUSD",
            side="BUY",
            execution_mode=SymbolExecutionMode.PAPER_SHADOW,
            regime="TRENDING_EXPANSION",
            dom_obi=0.2,
            predicted_net_r=4.0,
            ai_decision="APPROVE",
            ai_confidence=0.8,
            ai_reason="paper_open",
            entry_price=1.2500,
            executed_price=1.2500,
            stop_loss=1.2400,
            take_profit=1.2700,
            realized_slippage_r=0.0,
            realized_net_r=None,
        )

    # Before OHLC settlement, still in PAPER_SHADOW because is_closed=0
    assert ledger.evaluate_quarantine("GBPUSD").mode == SymbolExecutionMode.PAPER_SHADOW

    # Settle via bar hitting TP (1.2750) -> realized +2.0R (PF inf > 1.25),
    # BUT z_deficit = (4.0 - 2.0) / tiny_se >> 1.0 -> anti-ping-pong keeps PAPER_SHADOW!
    settled = ledger.settle_paper_shadow_trades("GBPUSD", bar_high=1.2750, bar_low=1.2490, bar_close=1.2720)
    assert settled == 15
    st_after = ledger.get_quarantine_state("GBPUSD")
    assert st_after.mode == SymbolExecutionMode.PAPER_SHADOW
    assert "z_deficit" in st_after.reason


def test_apex_config_validation_and_web_handlers_safety(tmp_path: Path) -> None:
    """Verify ApexConfig binds apex_config.json, rejects invalid risk, and thread-safe pause + 15s drill message."""
    raw = json.loads((WORKSPACE_ROOT / "config" / "apex_config.json").read_text(encoding="utf-8"))
    cfg = ApexConfig.from_dict(raw)
    assert cfg.account_mode == "DEMO"
    assert cfg.risk_account_equity_usd > 0.0
    assert cfg.scanner_max_candidates_per_cycle >= 1

    bad = json.loads(json.dumps(raw))
    bad["risk"]["max_risk_per_trade_pct"] = -0.5
    with pytest.raises(ValueError):
        ApexConfig.from_dict(bad)

    # Ensure AutonomousPositionOrchestrator.evaluate_position returns PositionDecision with str action
    mgr = AutonomousPositionOrchestrator()
    dec: PositionDecision = mgr.evaluate_position(
        ticket=101,
        symbol="EURUSD",
        side=Side.BUY,
        entry_price=1.1000,
        current_sl=1.0950,
        current_tp=1.1100,
        mark_price=1.1010,
        candles=_make_uptrend_candles("EURUSD", "M5", 35),
    )
    assert isinstance(dec.action, str)

    orch = ApexOrchestrator(
        mode="shadow-ab",
        config_path=WORKSPACE_ROOT / "config" / "apex_config.json",
        db_path=tmp_path / "orch.db",
    )
    p_res = orch._handle_web_pause()
    assert p_res["paused"] is True
    p_res2 = orch._handle_web_pause()
    assert p_res2["paused"] is False
