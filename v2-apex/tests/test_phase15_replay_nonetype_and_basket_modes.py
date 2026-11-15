"""Phase 15 tests: Replay NoneType iterable fixes, Auto Top-12 basket modes, and null-boundary hardening."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

from apex.cockpit.replay_engine import DEFAULT_REPLAY_SYMBOLS, HistoricalReplayEngine
from apex.engine.canonical_pipeline import Canonical11AgentPipeline
from apex.execution.position_manager import AutonomousPositionOrchestrator
from apex.feeds.models import Candle, Side
from apex.quant.correlation_shield import (
    CorrelationShield,
    PositionRiskSpec,
    compute_ledoit_wolf_and_tail_correlation,
)
from apex.quant.mtf_confluence import MTFConfluenceFilter
from apex_runner import ApexOrchestrator


def _make_candles(symbol: str, count: int = 80, tf: str = "M15", base: float = 1.1000) -> list[Candle]:
    out: list[Candle] = []
    px = base
    for i in range(count):
        drift = 0.0004 * math.sin(i / 4.0) + 0.00015
        o = px
        c = px + drift
        h = max(o, c) + 0.0003
        l = min(o, c) - 0.0003
        out.append(
            Candle(
                symbol=symbol,
                timeframe=tf,
                timestamp_utc=1_700_000_000.0 + i * 900.0,
                open=o,
                high=h,
                low=l,
                close=c,
                volume=1500.0 + (i % 5) * 200.0,
                source="test",
            )
        )
        px = c
    return out


def test_replay_handles_none_symbols_and_null_candles_from_feed() -> None:
    class _NullMixedFeed:
        def get_candles(self, symbol: str, timeframe: str = "M15", count: int = 200):
            if symbol == "EURUSD" and timeframe in {"M15", "H1"}:
                return _make_candles("EURUSD", count=140, tf=timeframe)
            return None  # Simulate MT5 copy_rates_from_pos returning None

    eng = HistoricalReplayEngine(step_delay_seconds=0.0)
    updates: list[dict] = []
    summary = eng.run_replay(
        feed_router=_NullMixedFeed(),
        symbols=None,
        window_days=2,
        speed=100,
        on_update=updates.append,
    )
    assert updates
    assert updates[-1]["status"] == "COMPLETED"
    assert summary.symbols_replayed == ["EURUSD"]
    assert summary.bars_processed > 0
    assert len(summary.equity_curve) >= 2
    assert summary.last_trace is not None


def test_orchestrator_resolve_replay_symbols_all_modes() -> None:
    orch = ApexOrchestrator.__new__(ApexOrchestrator)
    orch.last_matrix = [
        SimpleNamespace(symbol="BTCUSD", net_avg_r=0.14),
        SimpleNamespace(symbol="ETHUSD", net_avg_r=-0.22),
    ]

    basket, label_basket = ApexOrchestrator._resolve_replay_symbols(orch, "AUTO_TOP12")
    assert "BTCUSD" in basket
    assert "EURUSD" in basket
    assert len(basket) <= 12
    assert label_basket == "Auto Top-12 Active Basket"

    best_syms, best_label = ApexOrchestrator._resolve_replay_symbols(orch, "AUTO_BEST_MOVER")
    assert best_syms == ("ETHUSD",)
    assert "ETHUSD" in best_label

    single_syms, single_label = ApexOrchestrator._resolve_replay_symbols(orch, "US500")
    assert single_syms == ("US500",)
    assert single_label == "US500"


def test_correlation_shield_pipeline_mtf_and_pos_mgr_null_iterables() -> None:
    rho_lw, rho_tail = compute_ledoit_wolf_and_tail_correlation(None)
    assert rho_lw.shape == (1, 1)
    assert rho_tail.shape == (1, 1)

    shield = CorrelationShield()
    shield.register_return_series("EURUSD", None)
    verdict = shield.evaluate_proposed_trade(
        proposed=PositionRiskSpec(symbol="EURUSD", side=Side.BUY, risk_pct=0.5),
        open_positions=None,
        return_histories={"EURUSD": None},
    )
    assert verdict.allowed is True

    mtf = MTFConfluenceFilter()
    mtf_v = mtf.evaluate("EURUSD", Side.BUY, 1.10, 0.001, m5_candles=None, h1_candles=None)  # type: ignore[arg-type]
    assert mtf_v.passed is True

    candles = _make_candles("EURUSD", count=50)
    pipe = Canonical11AgentPipeline()
    trace = pipe.evaluate(
        symbol="EURUSD",
        side=Side.BUY,
        timeframe="M15",
        entry_price=candles[-1].close,
        candles=candles,
        h1_candles=None,
        open_positions=None,  # type: ignore[arg-type]
        run_ai_committee=False,
    )
    assert len(trace.stages) == 11

    pm = AutonomousPositionOrchestrator()
    dec = pm.evaluate_position(
        ticket=1,
        symbol="EURUSD",
        side=Side.BUY,
        entry_price=1.10,
        current_sl=1.095,
        current_tp=1.11,
        mark_price=1.101,
        candles=None,  # type: ignore[arg-type]
        force_evaluate=True,
    )
    assert dec.action in {"HOLD", "RATCHET_SLTP", "EARLY_EXIT"}
    assert pm.reconcile_closed_deals(None, deals_fetcher=lambda _t: None) == []  # type: ignore[arg-type]


def test_cockpit_html_updates_control_msg_on_replay_completion() -> None:
    html = (Path(__file__).resolve().parents[1] / "apex" / "cockpit" / "static" / "index.html").read_text(encoding="utf-8")
    assert "rp.status === 'COMPLETED' || rp.status === 'ERROR' || rp.status === 'NO_DATA'" in html
    assert "ctrlMsgEl.textContent = rp.message" in html
