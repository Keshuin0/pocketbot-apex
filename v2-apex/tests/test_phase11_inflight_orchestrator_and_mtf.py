"""Phase 10 (v2.5.0) Tests: Autonomous In-Flight Position Orchestrator, Shield 8 Z3, MTF H1 Filter & MFE/MAE."""

from __future__ import annotations

import datetime
from pathlib import Path
from types import SimpleNamespace

from apex.ai.aegis_verifier import AegisVerifier
from apex.ai.sleep_consolidation import SleepWakeConsolidator
from apex.execution.position_manager import AutonomousPositionOrchestrator
from apex.feeds.models import Candle, Side
from apex.ledger.walkforward_db import SymbolExecutionMode, WalkForwardLedgerDB
from apex.quant.mtf_confluence import MTFConfluenceFilter
from apex.quant.regime import RegimeClassifier


def _make_candles(
    symbol: str,
    timeframe: str,
    closes: list[float],
    spread_bar: float = 0.0004,
    start_ts: float = 1_700_000_000.0,
    step_s: float = 300.0,
) -> list[Candle]:
    out: list[Candle] = []
    for i, c in enumerate(closes):
        prev = closes[i - 1] if i > 0 else c - 0.0001
        out.append(
            Candle(
                symbol=symbol,
                timeframe=timeframe,
                timestamp_utc=start_ts + i * step_s,
                open=round(prev, 5),
                high=round(max(prev, c) + spread_bar * 0.5, 5),
                low=round(min(prev, c) - spread_bar * 0.5, 5),
                close=round(c, 5),
                volume=100.0 + i,
                source="synthetic",
            )
        )
    return out


def test_grace_window_blocks_panic_exit_and_z3_enforces_lock() -> None:
    candles = _make_candles("EURUSD", "M5", [1.0800 + i * 0.00015 for i in range(40)])
    mgr = AutonomousPositionOrchestrator(min_hold_bars=2, noise_cone_atr=0.65)

    # 5 seconds after entry (bars_held=0) with small adverse noise inside 0.65 ATR
    dec = mgr.evaluate_position(
        ticket=1001,
        symbol="EURUSD",
        side=Side.BUY,
        entry_price=1.0860,
        current_sl=1.0840,
        current_tp=1.0900,
        mark_price=1.0858,
        candles=candles,
        opened_at_utc=1_700_012_000.0,
        now_utc=1_700_012_005.0,
        bars_held_override=0,
    )
    assert dec.action == "HOLD"
    assert dec.ratchet_stage == "GRACE_LOCK"
    assert dec.grace_locked is True

    # Z3 Shield 8 must formally reject EARLY_EXIT during grace lock
    z3_ok, lat_ms, _ = AegisVerifier.verify_z3_inflight_trajectory(
        proposed_action="EARLY_EXIT",
        bars_held=1,
        min_hold_bars=2,
        abs_move_atr=0.35,
        cross_asset_shock_z=0.5,
        forward_ev_r=-0.40,
        current_r=-0.20,
        invalidation_signals=4,
    )
    assert z3_ok is False
    assert lat_ms < 50.0


def test_profit_ratchet_1r_free_roll_and_never_widens_sl() -> None:
    midweek_ts = 1_700_049_600.0  # Wednesday 2023-11-15 12:00 UTC
    candles = _make_candles("EURUSD", "M5", [1.0800 + i * 0.0001 for i in range(45)])
    sltp_calls: list[tuple[int, float, float]] = []
    mgr = AutonomousPositionOrchestrator(
        sltp_sender=lambda t, s, sl, tp: sltp_calls.append((t, sl, tp)) or 10009,
    )

    # Entry 1.0800, initial SL 1.0780 (1R = 0.0020), TP 1.0840. Mark at 1.0822 (+1.10R)
    dec = mgr.evaluate_position(
        ticket=2002,
        symbol="EURUSD",
        side=Side.BUY,
        entry_price=1.0800,
        current_sl=1.0780,
        current_tp=1.0840,
        mark_price=1.0822,
        candles=candles,
        spread_price=0.0001,
        commission_price_equiv=0.00005,
        now_utc=midweek_ts,
        bars_held_override=4,
        force_evaluate=True,
        execute_broker_actions=True,
    )
    assert dec.action == "RATCHET_SLTP"
    assert dec.ratchet_stage == "FREE_ROLL_1R"
    assert dec.new_sl is not None
    # SL moved above entry (fee buffer + 0.05R = 1.0800 + 0.00015 + 0.00010 = 1.08025)
    assert dec.new_sl >= 1.0802
    assert len(sltp_calls) == 1

    # Even if a subsequent calculation tried to lower SL, _never_widen_sl prevents backward widening
    assert mgr._never_widen_sl(Side.BUY, 1.0815, 1.0802) == 1.0815
    assert mgr._never_widen_sl(Side.SELL, 1.0790, 1.0805) == 1.0790


def test_15r_super_momentum_runner_vs_decay_bank() -> None:
    midweek_ts = 1_700_049_600.0
    # Strong monotonic trend => high Hurst & high ER
    strong_closes = [1.0800 + i * 0.0004 for i in range(60)]
    strong_candles = _make_candles("EURUSD", "M5", strong_closes, spread_bar=0.0002)
    mgr_runner = AutonomousPositionOrchestrator()
    dec_run = mgr_runner.evaluate_position(
        ticket=3003,
        symbol="EURUSD",
        side=Side.BUY,
        entry_price=1.0900,
        current_sl=1.0880,
        current_tp=1.0940,  # 2.0R
        mark_price=1.0934,  # +1.70R
        candles=strong_candles,
        now_utc=midweek_ts,
        bars_held_override=5,
        force_evaluate=True,
    )
    assert dec_run.action == "RATCHET_SLTP"
    assert dec_run.ratchet_stage == "RUNNER_3_5R"
    assert dec_run.new_tp is not None and dec_run.new_tp >= 1.0969  # expanded to 3.5R (1.0970)
    assert dec_run.new_sl is not None and dec_run.new_sl >= 1.0920  # >= +1.0R lock

    # Choppy decaying closes => low ER => BANK_1R instead of Runner
    chop_closes = [1.0800 + (0.0004 if i % 2 == 0 else -0.0003) for i in range(60)]
    chop_candles = _make_candles("EURUSD", "M5", chop_closes, spread_bar=0.0002)
    mgr_bank = AutonomousPositionOrchestrator()
    dec_bank = mgr_bank.evaluate_position(
        ticket=3004,
        symbol="EURUSD",
        side=Side.BUY,
        entry_price=1.0800,
        current_sl=1.0780,
        current_tp=1.0840,
        mark_price=1.0832,  # +1.60R
        candles=chop_candles,
        now_utc=midweek_ts,
        bars_held_override=5,
        force_evaluate=True,
    )
    assert dec_bank.action == "RATCHET_SLTP"
    assert dec_bank.ratchet_stage == "BANK_1R"
    assert dec_bank.new_sl == 1.0820


def test_confirmed_3_signal_thesis_collapse_exits_cleanly() -> None:
    midweek_ts = 1_700_049_600.0
    # Strong downtrend candles + large bearish last bar so forward EV < -0.25R & ER/regime break
    down_closes = [1.1000 - i * 0.0005 for i in range(59)] + [1.0680]
    candles = _make_candles("EURUSD", "M5", down_closes, spread_bar=0.0003)
    close_calls: list[int] = []
    mgr = AutonomousPositionOrchestrator(
        close_sender=lambda t, s, side, lots, px: close_calls.append(t) or 10009,
    )
    dec = mgr.evaluate_position(
        ticket=4004,
        symbol="EURUSD",
        side=Side.BUY,
        entry_price=1.0950,
        current_sl=1.0650,
        current_tp=1.1250,
        mark_price=1.0680,
        candles=candles,
        dom_obi=-0.60,
        cross_asset_shock_z=3.4,
        now_utc=midweek_ts,
        bars_held_override=4,
        force_evaluate=True,
        execute_broker_actions=True,
    )
    assert dec.action == "EARLY_EXIT"
    assert dec.ratchet_stage == "THESIS_EXIT"
    assert close_calls == [4004]


def test_mtf_h1_brick_wall_and_anti_chase_filter() -> None:
    mtf = MTFConfluenceFilter()
    m5 = _make_candles("EURUSD", "M5", [1.0800 + i * 0.0001 for i in range(30)], spread_bar=0.0002)

    # H1 strongly bearish => block M5 BUY
    h1_bear = _make_candles("EURUSD", "H1", [1.1200 - i * 0.0010 for i in range(30)], spread_bar=0.0005, step_s=3600.0)
    v_bear = mtf.evaluate("EURUSD", Side.BUY, entry_price=1.0830, atr_m5=0.0003, m5_candles=m5, h1_candles=h1_bear)
    assert v_bear.passed is False
    assert v_bear.h1_trend_aligned is False

    # H1 bullish overall, but entry sits within 0.30*ATR below 20-bar H1 swing high ceiling
    h1_bull = _make_candles("EURUSD", "H1", [1.0700 + i * 0.0005 for i in range(30)], spread_bar=0.0004, step_s=3600.0)
    swing_high = max(c.high for c in h1_bull[-20:])
    m5_near = _make_candles("EURUSD", "M5", [swing_high - 0.0010 + i * 0.00003 for i in range(30)], spread_bar=0.0002)
    v_ceil = mtf.evaluate("EURUSD", Side.BUY, entry_price=swing_high - 0.00005, atr_m5=0.0005, m5_candles=m5_near, h1_candles=h1_bull)
    assert v_ceil.passed is False
    assert v_ceil.anti_chase_ok is True
    assert "SWING_HIGH_CEILING" in v_ceil.reason

    # Anti-chase: entry stretched > 2.2 * ATR beyond prior 12-bar M5 range
    v_chase = mtf.evaluate("EURUSD", Side.BUY, entry_price=1.0850, atr_m5=0.0003, m5_candles=m5, h1_candles=None)
    assert v_chase.passed is False
    assert v_chase.anti_chase_ok is False


def test_closed_deal_reconciliation_mfe_mae_and_consolidation(tmp_path: Path) -> None:
    db_path = tmp_path / "autopsy.db"
    mem_path = tmp_path / "semantic_memory.json"
    ledger = WalkForwardLedgerDB(db_path=db_path)

    ledger.log_trade_outcome(
        order_id="5005",
        symbol="EURUSD",
        side="BUY",
        execution_mode=SymbolExecutionMode.ACTIVE,
        regime="TRENDING_EXPANSION",
        dom_obi=0.30,
        predicted_net_r=0.35,
        ai_decision="APPROVE",
        ai_confidence=0.90,
        ai_reason="test",
        entry_price=1.0800,
        executed_price=1.0800,
        stop_loss=1.0780,
        take_profit=1.0840,
        realized_slippage_r=0.0,
        realized_net_r=None,
    )

    candles = _make_candles("EURUSD", "M5", [1.0800 + i * 0.0001 for i in range(35)])
    mgr = AutonomousPositionOrchestrator(ledger=ledger)
    # Mark goes up to 1.0830 (+1.5R MFE), then dips to 1.0792 (-0.4R MAE)
    mgr.evaluate_position(
        ticket=5005, symbol="EURUSD", side=Side.BUY, entry_price=1.0800,
        current_sl=1.0780, current_tp=1.0840, mark_price=1.0830,
        candles=candles, bars_held_override=0,
    )
    mgr.evaluate_position(
        ticket=5005, symbol="EURUSD", side=Side.BUY, entry_price=1.0800,
        current_sl=1.0780, current_tp=1.0840, mark_price=1.0792,
        candles=candles, bars_held_override=0,
    )

    # Broker closes ticket 5005 at 1.0820 (+1.0R)
    reconciled = mgr.reconcile_closed_deals(
        open_tickets=set(),
        deals_fetcher=lambda t: [SimpleNamespace(entry=1, price=1.0820, profit=52.0, commission=-1.0, swap=0.0)],
    )
    assert len(reconciled) == 1
    assert reconciled[0]["mfe_r"] == 1.5
    assert reconciled[0]["mae_r"] == -0.4

    stats = ledger.compute_mfe_mae_stats()
    assert stats["closed_count"] == 1
    assert stats["avg_mfe_r"] == 1.5
    assert stats["avg_mae_r"] == -0.4

    consolidator = SleepWakeConsolidator(ledger_db=ledger, memory_path=mem_path)
    snap, _ = consolidator.phase_b_consolidate_semantic_memory("TEST_AUTOPSY", purged_cache_entries=0, canary_passed=True)
    assert snap.mfe_mae_stats["avg_mfe_r"] == 1.5
    assert snap.mfe_mae_stats["avg_mae_r"] == -0.4
