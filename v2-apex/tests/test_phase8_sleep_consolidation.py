"""Phase 8 verification: Circadian Sleep-Wake Memory Consolidation & Context Hygiene Engine."""

from __future__ import annotations

import datetime
import json
import tempfile
import time
from collections import deque
from pathlib import Path

from apex.ai.model_selector import DiscoveredModel, ModelTier
from apex.ai.sleep_consolidation import SleepWakeConsolidator
from apex.ai.veto_committee import (
    AIVetoCommittee,
    AIVetoDecision,
    CandidateSetupAuditContext,
)
from apex.cockpit.tui_dashboard import ApexTUIDashboard
from apex.feeds.hydra_nbbo import HydraNBBOEngine
from apex.ledger.walkforward_db import SymbolExecutionMode, WalkForwardLedgerDB
from apex.quant.timemachine_v2 import TimeMachineCostModel


def _make_canary_safe_evaluator():
    """Deterministic canary evaluator that vetoes toxic setups and approves healthy ones."""

    def _eval(ctx: CandidateSetupAuditContext) -> AIVetoDecision:
        if ctx.net_avg_R < 0.0 or ctx.spread_pct_of_sl > 20.0:
            return AIVetoDecision(
                decision="VETO",
                confidence=0.95,
                reason="Negative expectancy or excessive spread cost",
                risk_multiplier=0.0,
                latency_ms=5.0,
            )
        return AIVetoDecision(
            decision="APPROVE",
            confidence=0.80,
            reason="Positive expectancy",
            risk_multiplier=1.0,
            latency_ms=5.0,
        )

    return _eval


def test_phase_a_purges_working_memory_and_zero_prompt_bloat(tmp_path: Path) -> None:
    """Phase A flushes caches, cooldowns, buffers, temp files, and emits compact priors (<120 tokens)."""
    db = WalkForwardLedgerDB(tmp_path / "ledger.db")
    consolidator = SleepWakeConsolidator(
        ledger_db=db,
        memory_path=tmp_path / "semantic_memory.json",
    )

    committee = AIVetoCommittee(enabled=True, ledger_db=db)
    # Populate short-term working memory caches & expired cooldown
    committee._verdict_cache[("EURUSD", "BUY", "M5", "1700000000")] = AIVetoDecision(
        decision="APPROVE",
        confidence=0.85,
        reason="cached",
        risk_multiplier=1.0,
        latency_ms=10.0,
    )
    committee._recent_ai_events.append({"symbol": "EURUSD", "decision": "APPROVE"})
    models = committee.model_selector.get_models()
    if not models:
        committee.model_selector._models["studio_api:gemini-2.5-flash"] = DiscoveredModel(
            model_id="gemini-2.5-flash",
            provider="studio_api",
            tier=ModelTier.FLASH_FAST,
        )
        models = committee.model_selector.get_models()
    assert len(models) > 0
    models[0].cooldown_until_utc = time.time() - 10.0
    models[0].consecutive_errors = 3

    # Orphaned temp file in pocketbot_agy_isolated
    iso_dir = Path(tempfile.gettempdir()) / "pocketbot_agy_isolated"
    iso_dir.mkdir(parents=True, exist_ok=True)
    orphan = iso_dir / "orphan_test_phase8.tmp"
    orphan.write_text("stale", encoding="utf-8")

    # Oversized spike rejector buffer inside mock feed_router
    hydra = HydraNBBOEngine()
    hydra.spike_rejector._basis_history["EURUSD"] = deque([0.0001] * 80, maxlen=200)

    class _DummyMT5:
        is_connected = False

    class _DummyRouter:
        nbbo_engine = hydra
        mt5 = _DummyMT5()

    stats = consolidator.phase_a_purge_working_memory(
        ai_committee=committee,
        feed_router=_DummyRouter(),  # type: ignore[arg-type]
    )

    assert len(committee._verdict_cache) == 0
    assert len(committee._recent_ai_events) == 0
    assert models[0].cooldown_until_utc == 0.0
    assert models[0].consecutive_errors == 0
    assert not orphan.exists()
    assert len(hydra.spike_rejector._basis_history["EURUSD"]) == 50
    assert stats["purged_verdicts"] == 1
    assert stats["purged_events"] == 1
    assert stats["cleared_cooldowns"] >= 1
    assert stats["trimmed_buffers"] == 30

    # Compact Aegis priors must be concise numeric telemetry (<120 tokens, zero raw chat history)
    priors = consolidator.get_compact_aegis_priors("EURUSD", utc_hour=14)
    serialized = json.dumps(priors, separators=(",", ":"))
    assert len(serialized) < 480  # ~<120 tokens


def test_phase_b_c_persistence_across_restart_and_slippage_feedback(tmp_path: Path) -> None:
    """Realized slippage, session priors, & conformal trust survive full restart and feed TimeMachineCostModel."""
    db_path = tmp_path / "apex_ledger.db"
    mem_path = tmp_path / "semantic_memory.json"

    db1 = WalkForwardLedgerDB(db_path)
    ts_base = datetime.datetime(2026, 11, 12, 14, 0, tzinfo=datetime.timezone.utc).timestamp()

    # Log 5 closed trades with elevated realized slippage on XAUUSD / mt5
    for idx in range(5):
        db1.log_trade_outcome(
            order_id=f"ORD-{idx}",
            symbol="XAUUSD",
            side="BUY",
            execution_mode=SymbolExecutionMode.ACTIVE,
            regime="TRENDING_VOLATILE",
            dom_obi=0.45,
            predicted_net_r=0.35,
            ai_decision="APPROVE",
            ai_confidence=0.80,
            ai_reason="Valid setup",
            entry_price=2650.0,
            executed_price=2650.5,
            stop_loss=2640.0,
            take_profit=2670.0,
            realized_slippage_r=0.05,
            realized_net_r=0.30,
            timestamp_utc=ts_base + idx * 60,
            winning_venue="mt5",
        )

    # Log AI epistemic audits for conformal calibration
    for idx in range(10):
        db1.log_ai_epistemic_audit(
            symbol="XAUUSD",
            side="BUY",
            timeframe="M5",
            provider="gemini_api",
            model_id="gemini-2.5-flash",
            input_sha256=f"sha-{idx}",
            raw_decision="APPROVE",
            final_decision="APPROVE" if idx < 9 else "FALLBACK_PURE_MATH",
            confidence=0.82,
            risk_multiplier=1.0,
            hallucination_detected=(idx == 9),
            violation_codes="NUMERIC_MISMATCH" if idx == 9 else "",
            reason="Audit test",
            timestamp_utc=ts_base + idx * 30,
        )

    consolidator1 = SleepWakeConsolidator(ledger_db=db1, memory_path=mem_path)
    committee1 = AIVetoCommittee(enabled=True, ledger_db=db1)
    hydra = HydraNBBOEngine()

    res = consolidator1.run_full_cycle(
        trigger_reason="TEST_RUN",
        ai_committee=committee1,
        hydra_nbbo=hydra,
        canary_evaluator=_make_canary_safe_evaluator(),
        now_utc=ts_base + 3600,
    )

    assert res.cycle_number == 1
    assert res.canary_passed is True
    assert "[MORNING WAKE-UP READY]" in res.status_message
    assert mem_path.exists()
    assert res.snapshot.trades_consolidated == 5
    assert res.snapshot.audits_consolidated == 11

    # Slippage multiplier should be > 1.0 (realized_slippage_r=0.05 is high) and applied to Hydra
    mult = consolidator1.get_slippage_multiplier_for_symbol("XAUUSD", venue="mt5")
    assert mult > 1.0
    assert hydra.symbol_slippage_multipliers["XAUUSD"]["mt5"] == mult

    # Verify TimeMachineCostModel scales VWAP slippage accurately using consolidated multiplier
    base_cost_model = TimeMachineCostModel(
        commission_per_lot_round_turn_usd=4.50,
        lots=1.0,
        risk_per_lot_usd=100.0,
        spread_price=0.20,
        vwap_slippage_price=0.10,
        slippage_calibration_multiplier=1.0,
    )
    calibrated_cost_model = TimeMachineCostModel(
        commission_per_lot_round_turn_usd=4.50,
        lots=1.0,
        risk_per_lot_usd=100.0,
        spread_price=0.20,
        vwap_slippage_price=0.10,
        slippage_calibration_multiplier=mult,
    )
    base_r = base_cost_model.compute_round_trip_cost_r(one_r_price_distance=10.0)
    cal_r = calibrated_cost_model.compute_round_trip_cost_r(one_r_price_distance=10.0)
    assert cal_r > base_r

    # Simulate FULL PROCESS RESTART: new DB handle + new SleepWakeConsolidator + new AIVetoCommittee
    db2 = WalkForwardLedgerDB(db_path)
    consolidator2 = SleepWakeConsolidator(ledger_db=db2, memory_path=mem_path)
    assert consolidator2.get_slippage_multiplier_for_symbol("XAUUSD", venue="mt5") == mult
    priors = consolidator2.get_compact_aegis_priors("XAUUSD", utc_hour=14)
    assert priors["semantic_slippage_mult"] == mult
    assert priors["session_hour_prior_r"] == 0.3
    assert "conformal_quantile" in priors

    latest_ckpt = db2.get_latest_consolidation_checkpoint()
    assert latest_ckpt is not None
    assert latest_ckpt["cycle_number"] == 1
    assert bool(latest_ckpt["canary_passed"]) is True

    committee2 = AIVetoCommittee(enabled=True, ledger_db=db2)
    passed, _ = consolidator2.phase_c_morning_wakeup_audit(
        ai_committee=committee2,
        canary_evaluator=_make_canary_safe_evaluator(),
    )
    assert passed is True


def test_circadian_triggers_and_dashboard_mind_state_badge(tmp_path: Path) -> None:
    """Verify startup/22:05 UTC triggers and ApexTUIDashboard Mind State: FRESH header."""
    db = WalkForwardLedgerDB(tmp_path / "trigger_ledger.db")
    consolidator = SleepWakeConsolidator(
        ledger_db=db,
        memory_path=tmp_path / "semantic_memory.json",
        rollover_utc_hour=22,
        rollover_utc_minute=5,
        stale_hours_threshold=16.0,
    )

    # Initial startup with no checkpoints -> must trigger
    due, reason = consolidator.should_run_sleep_cycle(is_startup=True)
    assert due is True
    assert reason == "STARTUP_INITIAL_CONSOLIDATION"

    # Run at 22:06 UTC
    t_2206 = datetime.datetime(2026, 11, 12, 22, 6, tzinfo=datetime.timezone.utc).timestamp()
    consolidator.run_full_cycle(trigger_reason=reason, now_utc=t_2206)

    # 2 hours later on startup -> fresh (no trigger)
    due_fresh, reason_fresh = consolidator.should_run_sleep_cycle(
        now_utc=t_2206 + 2 * 3600,
        is_startup=True,
    )
    assert due_fresh is False
    assert reason_fresh == "STARTUP_FRESH"

    # Same UTC day at 22:10 UTC -> already consolidated today
    due_same_day, reason_same = consolidator.should_run_sleep_cycle(
        now_utc=t_2206 + 4 * 60,
        is_startup=False,
    )
    assert due_same_day is False
    assert reason_same == "ALREADY_CONSOLIDATED_TODAY"

    # 17 hours later on startup -> stale (>16h) -> must trigger
    due_stale, reason_stale = consolidator.should_run_sleep_cycle(
        now_utc=t_2206 + 17 * 3600,
        is_startup=True,
    )
    assert due_stale is True
    assert "STARTUP_STALE_" in reason_stale

    # Next day at 22:07 UTC -> daily circadian window triggers
    t_next_2207 = datetime.datetime(2026, 11, 13, 22, 7, tzinfo=datetime.timezone.utc).timestamp()
    due_next, reason_next = consolidator.should_run_sleep_cycle(
        now_utc=t_next_2207,
        is_startup=False,
    )
    assert due_next is True
    assert reason_next == "DAILY_CIRCADIAN_2205_UTC"

    # Dashboard badge formatting & rendering
    badge = consolidator.get_mind_state_badge()
    assert "Mind State: FRESH (Cycle #1 | Wisdom Checkpoints: 1 | Last Refresh: 22:06)" == badge

    dashboard = ApexTUIDashboard()
    panel = dashboard.build_renderable(
        run_mode="DEMO",
        active_feed="MT5",
        ctrader_connected=False,
        mt5_connected=True,
        account_equity_usd=10000.0,
        daily_pnl_usd=0.0,
        open_positions_count=0,
        rows=[],
        mind_state_badge=badge,
    )
    assert panel is not None
