"""Phase 3 End-to-End Integration & Final Assembly Verification Suite for PocketBot Apex."""

from __future__ import annotations

from pathlib import Path

import pytest

from apex.ai.veto_committee import (
    AIVetoCommittee,
    CandidateSetupAuditContext,
)
from apex.cockpit.telegram_remote import TelegramRemoteController
from apex.cockpit.tui_dashboard import ApexTUIDashboard
from apex.execution.router import (
    DualBrokerExecutionRouter,
    ExecutionOrderRequest,
)
from apex.feeds.dual_router import MarketFeedRouter
from apex.feeds.models import DOMAnalytics, LiquidityWall, Side
from apex.ledger.walkforward_db import (
    SymbolExecutionMode,
    WalkForwardLedgerDB,
)
from apex.quant.correlation_shield import PositionRiskSpec
from apex_runner import ApexOrchestrator

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent


def test_six_hard_circuit_breakers_and_paper_shadow_fallback() -> None:
    """Validate all 6 pre-trade circuit breakers (a-f), ProtoOANewOrderReq builder, and Paper-Shadow fallback."""
    feed_router = MarketFeedRouter(config_path=WORKSPACE_ROOT / "config" / "apex_config.json")
    exec_router = DualBrokerExecutionRouter(
        feed_router=feed_router,
        account_equity_usd=10_000.0,
        max_daily_drawdown_pct=2.0,
        max_concurrent_positions=2,
        max_factor_exposure_pct=1.0,
        max_spread_slippage_sl_ratio=0.15,
    )

    valid_order = ExecutionOrderRequest(
        symbol="EURUSD",
        side=Side.BUY,
        lots=0.50,
        entry_price=1.10000,
        stop_loss=1.09900,  # 10 pips SL (0.00100)
        take_profit=1.10200,  # 20 pips TP (0.00200)
        risk_pct=0.5,
        risk_usd=50.0,
        force_paper_shadow=True,
    )

    # 1. Clean execution passes all 6 circuit breakers and routes to Paper-Shadow
    clean_dom = DOMAnalytics(
        symbol="EURUSD",
        timestamp_utc=1700000000.0,
        top_n_levels=5,
        bid_volume_top_n=650_000.0,
        ask_volume_top_n=350_000.0,
        obi=0.30,
        best_bid=1.09995,
        best_ask=1.10005,
        spread_pips=1.0,
        liquidity_walls=(),
    )
    receipt = exec_router.execute_order(
        order=valid_order,
        closed_pnl_today_usd=-50.0,
        floating_pnl_usd=0.0,
        open_positions=[],
        current_spread_price=0.00008,
        vwap_slippage_price=0.00002,
        dom_analytics=clean_dom,
    )
    assert receipt.breaker_report.all_passed is True
    assert len(receipt.breaker_report.checks) == 6
    assert receipt.status == "PAPER_SHADOW_FILLED"
    assert receipt.venue == "PAPER_SHADOW"

    # 2. Breaker (b): -2.0% Daily Drawdown Kill-Switch (-$210 on $10,000 = 2.1% >= 2.0%)
    dd_report = exec_router.evaluate_circuit_breakers(
        order=valid_order,
        closed_pnl_today_usd=-150.0,
        floating_pnl_usd=-60.0,
        open_positions=[],
        current_spread_price=0.00008,
        vwap_slippage_price=0.00002,
    )
    assert dd_report.all_passed is False
    assert any("B_DAILY_DRAWDOWN" in r for r in dd_report.failed_reasons)

    # 3. Breaker (c): Max 2 Concurrent Positions
    open_two = [
        PositionRiskSpec(symbol="EURGBP", side=Side.BUY, risk_pct=0.4),
        PositionRiskSpec(symbol="AUDJPY", side=Side.BUY, risk_pct=0.4),
    ]
    pos_report = exec_router.evaluate_circuit_breakers(
        order=valid_order,
        open_positions=open_two,
        current_spread_price=0.00008,
        vwap_slippage_price=0.00002,
    )
    assert pos_report.all_passed is False
    assert any("C_POSITIONS_AND_FACTOR_CAP" in r for r in pos_report.failed_reasons)

    # 4. Breaker (d): Spread + VWAP Slippage > 15% of SL distance (0.00020 / 0.00100 = 20% > 15%)
    slip_report = exec_router.evaluate_circuit_breakers(
        order=valid_order,
        current_spread_price=0.00014,
        vwap_slippage_price=0.00006,
    )
    assert slip_report.all_passed is False
    assert any("D_SPREAD_VWAP_SLIPPAGE" in r for r in slip_report.failed_reasons)

    # 5. Breaker (e): Opposing DOM wall within 0.35R
    wall_dom = DOMAnalytics(
        symbol="EURUSD",
        timestamp_utc=1700000000.0,
        top_n_levels=5,
        bid_volume_top_n=650_000.0,
        ask_volume_top_n=350_000.0,
        obi=0.30,
        best_bid=1.09995,
        best_ask=1.10005,
        spread_pips=1.0,
        liquidity_walls=(
            LiquidityWall(
                side=Side.SELL,
                price=1.10025,  # +0.25R above 1.10000 (<= 0.35R)
                volume_units=900_000.0,
                ratio_to_median=4.2,
                distance_from_mid_pips=2.5,
            ),
        ),
    )
    dom_report = exec_router.evaluate_circuit_breakers(
        order=valid_order,
        current_spread_price=0.00008,
        vwap_slippage_price=0.00002,
        dom_analytics=wall_dom,
    )
    assert dom_report.all_passed is False
    assert any("E_LEVEL2_DOM_GATE" in r for r in dom_report.failed_reasons)

    # 6. Breaker (f): Missing SL/TP rejected prior to wire transmission
    missing_sl_order = ExecutionOrderRequest(
        symbol="EURUSD",
        side=Side.BUY,
        lots=0.50,
        entry_price=1.10000,
        stop_loss=0.0,
        take_profit=1.10200,
    )
    sl_report = exec_router.evaluate_circuit_breakers(order=missing_sl_order)
    assert sl_report.all_passed is False
    assert any("F_MANDATORY_SL_TP" in r for r in sl_report.failed_reasons)

    # 7. Verify cTrader ProtoOANewOrderReq carries server-side relative & absolute SL/TP
    pb_req = DualBrokerExecutionRouter.build_ctrader_Order_req(
        account_id=123456,
        symbol_id=1,
        order=valid_order,
        lot_size_units=100_000.0,
    )
    assert pb_req.ctidTraderAccountId == 123456
    assert pb_req.relativeStopLoss == 100  # 0.00100 * 100,000
    assert pb_req.relativeTakeProfit == 200  # 0.00200 * 100,000
    assert pytest.approx(pb_req.stopLoss, abs=1e-6) == 1.09900
    assert pytest.approx(pb_req.takeProfit, abs=1e-6) == 1.10200


def test_ai_veto_committee_graceful_bypass_and_structured_json_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate event-driven AI Veto Committee graceful bypass when GEMINI_API_KEY and agy_pro are unset and JSON schema parsing."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    committee = AIVetoCommittee(
        enabled=True,
        api_key="",
        timeout_seconds=4.0,
        agy_executable_override="",
    )

    ctx = CandidateSetupAuditContext(
        symbol="EURUSD",
        side=Side.BUY,
        timeframe="M5",
        entry_price=1.10000,
        stop_loss=1.09900,
        take_profit=1.10200,
        regime="TRENDING_EXPANSION",
        hurst_exponent=0.61,
        kaufman_er=0.45,
        volatility_percentile=68.0,
        dom_obi=0.28,
        spread_pct_of_sl=0.08,
        net_avg_R=0.42,
        prob_profit=0.58,
        cvar_95_R=-1.12,
        recommended_risk_pct=0.35,
    )

    dec_bypass = committee.evaluate_sync(ctx)
    assert dec_bypass.decision == "APPROVE"
    assert dec_bypass.bypassed is True
    assert dec_bypass.latency_ms < 4000.0

    # Verify structured JSON parsing for both APPROVE and VETO
    raw_veto = '{"decision": "VETO", "confidence": 0.88, "reason": "Macro Auditor: London-NY handoff whipsaw risk"}'
    parsed_veto = AIVetoCommittee.parse_structured_decision(raw_veto, latency_ms=210.5)
    assert parsed_veto.decision == "VETO"
    assert pytest.approx(parsed_veto.confidence) == 0.88
    assert "Macro Auditor" in parsed_veto.reason


def test_sqlite_ledger_auto_quarantine_demotion_and_recovery(tmp_path: Path) -> None:
    """Validate rolling 20-trade Auto-Quarantine demotion (>1.5 SE below predicted EV or PF < 0.90) & PF > 1.25 recovery."""
    db_file = tmp_path / "test_apex_ledger.db"
    ledger = WalkForwardLedgerDB(db_path=db_file)

    # 1. Log 15 trades on GBPUSD where Time Machine predicted +0.55R, but realized outcomes bleed (-0.50R average, PF < 0.90)
    realized_losses = [-1.0, -1.0, +0.8, -1.0, -1.0, +0.6, -1.0, -1.0, -1.0, +0.5, -1.0, -1.0, +0.4, -1.0, -1.0]
    state = ledger.get_quarantine_state("GBPUSD")
    assert state.mode == SymbolExecutionMode.ACTIVE

    for idx, r_val in enumerate(realized_losses):
        state = ledger.log_trade_outcome(
            order_id=f"ord_{idx}",
            symbol="GBPUSD",
            side="BUY",
            execution_mode=state.mode,
            regime="TRENDING_EXPANSION",
            dom_obi=0.20,
            predicted_net_r=0.55,
            ai_decision="APPROVE",
            ai_confidence=0.80,
            ai_reason="ok",
            entry_price=1.27000,
            executed_price=1.27002,
            stop_loss=1.26900,
            take_profit=1.27200,
            realized_slippage_r=0.02,
            realized_net_r=r_val,
        )

    # After 15 losing trades, GBPUSD must be auto-demoted to PAPER_SHADOW!
    assert state.mode == SymbolExecutionMode.PAPER_SHADOW
    assert state.z_deficit > 1.5
    assert state.profit_factor < 0.90
    assert "AUTO_QUARANTINE" in state.reason

    # 2. Now log 15 strong PAPER_SHADOW trades on GBPUSD with PF > 1.25 (e.g., 10 wins of +1.5R, 5 losses of -1.0R -> PF = 15/5 = 3.0)
    paper_recovery_outcomes = [+1.5, +1.5, -1.0, +1.5, +1.5, -1.0, +1.5, +1.5, -1.0, +1.5, +1.5, -1.0, +1.5, +1.5, -1.0]
    for idx, r_val in enumerate(paper_recovery_outcomes):
        state = ledger.log_trade_outcome(
            order_id=f"paper_{idx}",
            symbol="GBPUSD",
            side="BUY",
            execution_mode=SymbolExecutionMode.PAPER_SHADOW,
            regime="TRENDING_EXPANSION",
            dom_obi=0.25,
            predicted_net_r=0.50,
            ai_decision="APPROVE",
            ai_confidence=0.85,
            ai_reason="paper_shadow_recovery",
            entry_price=1.27000,
            executed_price=1.27001,
            stop_loss=1.26900,
            take_profit=1.27200,
            realized_slippage_r=0.01,
            realized_net_r=r_val,
        )

    # After 15 paper trades with PF = 3.0 (> 1.25), GBPUSD automatically recovers to ACTIVE!
    assert state.mode == SymbolExecutionMode.ACTIVE
    assert "RECOVERED_TO_ACTIVE" in state.reason


def test_telegram_remote_commands_and_panic_flatten(tmp_path: Path) -> None:
    """Validate Telegram Remote Controller commands (/status, /quarantine, /pause, /panic_flatten) and .bat launchers."""
    feed_router = MarketFeedRouter(config_path=WORKSPACE_ROOT / "config" / "apex_config.json")
    exec_router = DualBrokerExecutionRouter(feed_router=feed_router)
    ledger = WalkForwardLedgerDB(db_path=tmp_path / "tg_ledger.db")
    tg = TelegramRemoteController(execution_router=exec_router, ledger_db=ledger)

    # Seed one shadow position
    exec_router.execute_order(
        ExecutionOrderRequest(
            symbol="EURUSD",
            side=Side.BUY,
            lots=0.10,
            entry_price=1.10000,
            stop_loss=1.09900,
            take_profit=1.10200,
            force_paper_shadow=True,
        )
    )
    assert len(exec_router._shadow_positions) == 1

    status_msg = tg.handle_command("/status")
    assert "PocketBot Apex Status" in status_msg

    q_msg = tg.handle_command("/quarantine")
    assert "Quarantine" in q_msg

    pause_msg = tg.handle_command("/pause")
    assert tg.scanning_paused is True
    assert "PAUSED" in pause_msg

    flatten_msg = tg.handle_command("/panic_flatten")
    assert tg.scanning_halted is True
    assert len(exec_router._shadow_positions) == 0
    assert "PANIC FLATTEN EXECUTED" in flatten_msg

    # Verify all 4 portable relative-path (%~dp0) Windows batch launchers exist
    for bat_name in (
        "1_SETUP_CTRADER_KEYS.bat",
        "2_START_APEX_AUTO_DEMO.bat",
        "3_START_APEX_CONFIRM.bat",
        "4_APEX_LIVE_DASHBOARD.bat",
    ):
        bat_path = WORKSPACE_ROOT / bat_name
        assert bat_path.exists(), f"Missing launcher {bat_name}"
        bat_text = bat_path.read_text(encoding="utf-8")
        assert "%~dp0" in bat_text, f"{bat_name} must use portable %~dp0 paths"


def test_end_to_end_orchestrator_dry_run(tmp_path: Path) -> None:
    """Run an end-to-end dry-run pass across the full PocketBot Apex pipeline."""
    orch = ApexOrchestrator(
        mode="shadow-ab",
        config_path=WORKSPACE_ROOT / "config" / "apex_config.json",
        db_path=tmp_path / "e2e_ledger.db",
    )
    try:
        rows = orch.run_cycle(symbols=["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "BTCUSD"])
        assert orch.last_scan_summary is not None
        assert orch.last_scan_summary.total_scanned == 5
        assert len(rows) <= 5
    finally:
        orch.shutdown()
