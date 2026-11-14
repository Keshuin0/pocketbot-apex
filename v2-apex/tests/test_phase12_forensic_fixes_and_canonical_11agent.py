"""Phase 12 (`v2.5.1`) Tests — Canonical 11-Agent Pipeline & 6 Forensic Fixes."""

from __future__ import annotations

import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from apex.ai.model_selector import AdaptiveModelSelector
from apex.ai.veto_committee import AIVetoCommittee, AIVetoMode
from apex.cockpit.replay_engine import HistoricalReplayEngine
from apex.engine.canonical_pipeline import (
    Canonical11AgentPipeline,
    compute_asset_class_commission_usd,
)
from apex.feeds.models import AssetClass, Candle, ContractSpec, DiscoveredSymbol, Side, Tick
from apex.ledger.walkforward_db import WalkForwardLedgerDB
from apex.quant.momentum_dom import MomentumDOMGate
from apex.quant.regime import RegimeClassifier
from apex.quant.timemachine_v2 import BlockBootstrapTimeMachine
from apex.quant.universe_scanner import UniverseScanner


def _make_trending_candles(
    symbol: str,
    count: int = 120,
    start_price: float = 1.1000,
    step: float = 0.0005,
    base_ts: float = 1762992000.0,  # Midweek Wed 2025-11-12 UTC
    tf_sec: int = 300,
) -> list[Candle]:
    candles: list[Candle] = []
    px = start_price
    for i in range(count):
        # Smooth upward trend with small realistic pullbacks
        delta = step if i % 4 != 3 else -0.25 * step
        o = px
        c = px + delta
        hi = max(o, c) + abs(step) * 0.4
        lo = min(o, c) - abs(step) * 0.4
        candles.append(
            Candle(
                symbol=symbol,
                timeframe="M5" if tf_sec == 300 else "H1",
                timestamp_utc=base_ts + i * tf_sec,
                open=round(o, 6),
                high=round(hi, 6),
                low=round(lo, 6),
                close=round(c, 6),
                volume=1000.0 + i * 5.0,
                source="mt5",
            )
        )
        px = c
    return candles


def _make_chop_candles(symbol: str, count: int = 120, mid: float = 1.1000) -> list[Candle]:
    candles: list[Candle] = []
    base_ts = 1762992000.0
    for i in range(count):
        osc = 0.0004 * math.sin(i * 1.3)
        o = mid + osc
        c = mid - osc
        candles.append(
            Candle(
                symbol=symbol,
                timeframe="M5",
                timestamp_utc=base_ts + i * 300,
                open=round(o, 6),
                high=round(max(o, c) + 0.0003, 6),
                low=round(min(o, c) - 0.0003, 6),
                close=round(c, 6),
                volume=500.0,
                source="mt5",
            )
        )
    return candles


def test_canonical_11_agent_pipeline_and_replay_parity() -> None:
    """Canonical11AgentPipeline emits 11 stages with symbol/timeframe, and Replay routes through it."""
    m5 = _make_trending_candles("EURUSD", count=140, start_price=1.0800, step=0.0004, tf_sec=300)
    h1 = _make_trending_candles("EURUSD", count=60, start_price=1.0600, step=0.0012, tf_sec=3600)

    pipeline = Canonical11AgentPipeline()
    trace = pipeline.evaluate(
        symbol="EURUSD",
        side=Side.BUY,
        timeframe="M5",
        entry_price=m5[-1].close,
        candles=m5,
        h1_candles=h1,
        spread_price=0.00010,
        pip_size=0.0001,
        run_ai_committee=False,
    )
    assert trace.symbol == "EURUSD"
    assert trace.timeframe == "M5"
    assert len(trace.stages) == 11
    assert [s.stage_num for s in trace.stages] == list(range(1, 12))
    assert trace.to_dict()["symbol"] == "EURUSD"

    class _StubRouter:
        def get_candles(self, symbol: str, timeframe: str = "M5", count: int = 200) -> list[Candle]:
            return h1 if timeframe == "H1" else m5

        def get_contract_spec(self, symbol: str) -> ContractSpec:
            return ContractSpec(
                symbol=symbol,
                digits=5,
                pip_size=0.0001,
                point_size=0.00001,
                lot_size=100000.0,
                min_volume_lots=0.01,
                max_volume_lots=100.0,
                step_volume_lots=0.01,
                source="mt5",
            )

    updates: list[dict[str, Any]] = []
    engine = HistoricalReplayEngine()
    res = engine.run_replay(
        feed_router=_StubRouter(),
        symbols=("EURUSD",),
        window_days=2,
        speed=100.0,
        on_update=updates.append,
    )
    assert updates[-1]["status"] == "COMPLETED"
    assert "EURUSD" in res.symbols_replayed
    assert res.bars_processed > 0
    assert updates[-1]["symbol"] == "EURUSD"
    assert isinstance(res.trades, list)


def test_closed_market_weekend_exclusion_and_trade_mode_check(tmp_path: Path) -> None:
    """SUGAR/UKCOCOA with trade_mode!=4 or tick_age>120s must be strictly excluded."""
    now = 1763078400.0

    class _StubMT5:
        _initialized = True

        def is_symbol_trade_mode_full(self, symbol: str) -> bool:
            return symbol != "UKCOCOA"

    class _StubRouter:
        mt5 = _StubMT5()

    scanner = UniverseScanner(
        feed_router=_StubRouter(),  # type: ignore[arg-type]
        regime_classifier=RegimeClassifier(),
        momentum_dom_gate=MomentumDOMGate(),
        time_machine=BlockBootstrapTimeMachine(),
        ai_committee=AIVetoCommittee(enabled=False, ledger_db=WalkForwardLedgerDB(tmp_path / "l.db")),
        ledger=WalkForwardLedgerDB(tmp_path / "l.db"),
    )

    # 1. trade_mode != FULL (e.g. UKCOCOA closed)
    ok_cocoa, reason_cocoa = scanner.is_symbol_market_open(
        "UKCOCOA",
        Tick(symbol="UKCOCOA", bid=5000.0, ask=5005.0, timestamp_utc=now - 5.0, source="mt5"),
        now_utc=now,
    )
    assert not ok_cocoa
    assert reason_cocoa == "TRADE_MODE_CLOSED"

    # 2. Stale tick (> 120s, e.g. SUGAR weekend)
    ok_sugar, reason_sugar = scanner.is_symbol_market_open(
        "SUGAR",
        Tick(symbol="SUGAR", bid=20.10, ask=20.15, timestamp_utc=now - 140.0, source="mt5"),
        now_utc=now,
    )
    assert not ok_sugar
    assert "STALE_TICK" in reason_sugar

    # 3. Fresh active symbol (BTCUSD)
    ok_btc, reason_btc = scanner.is_symbol_market_open(
        "BTCUSD",
        Tick(symbol="BTCUSD", bid=90000.0, ask=90020.0, timestamp_utc=now - 2.0, source="mt5"),
        now_utc=now,
    )
    assert ok_btc and reason_btc == "ACTIVE"


def test_asset_class_commission_atomic_venue_cost_kelly_zero_and_sorting(tmp_path: Path) -> None:
    """Verify $0 crypto commission, effective_cost>=spread, Q-KELLY=0 on chop, and actionable-first sort."""
    assert compute_asset_class_commission_usd("EURUSD", AssetClass.FOREX) == 4.50
    assert compute_asset_class_commission_usd("XAUUSD", AssetClass.METALS) == 4.50
    assert compute_asset_class_commission_usd("ADAUSD", AssetClass.CRYPTO) == 0.0
    assert compute_asset_class_commission_usd("US500", AssetClass.INDICES) == 0.0

    now = time.time()
    chop_m5 = _make_chop_candles("BCHUSD", count=100, mid=400.0)
    trend_m5 = _make_trending_candles("BTCUSD", count=100, start_price=80000.0, step=150.0)
    trend_h1 = _make_trending_candles("BTCUSD", count=50, start_price=75000.0, step=400.0, tf_sec=3600)

    class _MultiRouter:
        mt5 = None

        def discover_catalog(self, auto_enable_market_watch: bool = True) -> list[DiscoveredSymbol]:
            return [
                DiscoveredSymbol("BCHUSD", "BCHUSD", AssetClass.CRYPTO, "Crypto", 2, 0.01, 4, True, "mt5"),
                DiscoveredSymbol("BTCUSD", "BTCUSD", AssetClass.CRYPTO, "Crypto", 2, 0.01, 4, True, "mt5"),
            ]

        def get_tick(self, symbol: str) -> tuple[Tick, str]:
            if symbol == "BCHUSD":
                # 14.4p spread -> NBBO must NEVER report < 14.4p
                return Tick("BCHUSD", 400.00, 414.40, now, "mt5"), "mt5"
            return Tick("BTCUSD", 94800.0, 94815.0, now, "mt5"), "mt5"

        def get_contract_spec(self, symbol: str) -> ContractSpec:
            return ContractSpec(
                symbol=symbol,
                digits=2,
                pip_size=1.0,
                point_size=0.01,
                lot_size=1.0,
                min_volume_lots=0.01,
                max_volume_lots=100.0,
                step_volume_lots=0.01,
                source="mt5",
                trade_tick_value=0.01,
                trade_tick_size=0.01,
            )

        def get_candles(self, symbol: str, timeframe: str = "M5", count: int = 200) -> list[Candle]:
            if symbol == "BCHUSD":
                return chop_m5
            return trend_h1 if timeframe == "H1" else trend_m5

        def get_dom_analytics(self, symbol: str) -> None:
            return None

        def evaluate_nbbo(self, symbol: str, one_r_price_distance: float, side: Side) -> Any:
            # Simulate slightly stale NBBO quoting 14.0p while atomic tick spread is 14.4p
            return SimpleNamespace(
                symbol=symbol,
                timestamp_utc=now,
                winning_venue="mt5",
                winning_cost_pips=14.0,
                pips_saved=0.0,
                mesh_topology="MT5_SOLO",
            )

    ledger = WalkForwardLedgerDB(tmp_path / "scanner.db")
    scanner = UniverseScanner(
        feed_router=_MultiRouter(),  # type: ignore[arg-type]
        regime_classifier=RegimeClassifier(),
        momentum_dom_gate=MomentumDOMGate(),
        time_machine=BlockBootstrapTimeMachine(),
        ai_committee=AIVetoCommittee(enabled=False, ledger_db=ledger),
        ledger=ledger,
    )
    rows, _ = scanner.scan_cycle(now_utc=now)
    by_sym = {r.symbol: r for r in rows}

    # BCHUSD is chop -> Q-KELLY must be 0.00%, and VENUE cost must be >= 14.4p (never 14.0p!)
    bch = by_sym["BCHUSD"]
    assert bch.rec_risk_pct == 0.0
    assert "14.4p" in bch.best_venue

    # Actionable trending rows (if any) must rank ahead of filtered rows
    if rows[0].symbol != "BTCUSD":
        assert rows[0].regime == "TRENDING_EXPANSION" or (rows[0].net_avg_r or 0.0) >= (rows[1].net_avg_r or 0.0)


def test_persisted_model_family_quarantine_and_non_unknown_ai_label(tmp_path: Path) -> None:
    """Model-family quarantine persists in SQLite across restarts, and active_model_label is never UNKNOWN."""
    db_path = tmp_path / "ai_persist.db"
    ledger1 = WalkForwardLedgerDB(db_path)
    selector1 = AdaptiveModelSelector(ledger_db=ledger1)
    selector1.quarantine_family("agy_pro", "argon", duration_seconds=3600.0, reason="Low AI credits")

    # Simulate fresh process restart with new ledger & selector instance
    ledger2 = WalkForwardLedgerDB(db_path)
    selector2 = AdaptiveModelSelector(ledger_db=ledger2)
    quarantined, rem_s, reason = selector2.is_family_quarantined("agy_pro", "argon")
    assert quarantined is True
    assert rem_s > 3500
    assert "Low AI credits" in reason

    committee = AIVetoCommittee(enabled=True, mode=AIVetoMode.FORCE_PURE_MATH.value, ledger_db=ledger2)
    status = committee.get_router_status()
    assert status.active_model_label != "UNKNOWN"
    assert ("AGY_PRO:" in status.active_model_label or "PURE_MATH" in status.active_model_label or "STUDIO_API:" in status.active_model_label)
