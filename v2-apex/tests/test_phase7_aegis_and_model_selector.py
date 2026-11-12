"""Phase 7b Verification Suite: Apex Aegis 6-Shield Anti-Hallucination Firewall & Adaptive Model Selector."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from apex.ai.aegis_verifier import AegisVerifier
from apex.ai.model_selector import AdaptiveModelSelector, ModelTier, SetupStakes
from apex.ai.veto_committee import (
    AIVetoCommittee,
    AIVetoDecision,
    AIVetoMode,
    CandidateSetupAuditContext,
)
from apex.cockpit.tui_dashboard import ApexTUIDashboard
from apex.execution.router import DualBrokerExecutionRouter, ExecutionOrderRequest
from apex.feeds.dual_router import MarketFeedRouter
from apex.feeds.models import Side
from apex.ledger.walkforward_db import WalkForwardLedgerDB


def _make_ctx(**overrides: object) -> CandidateSetupAuditContext:
    defaults: dict[str, object] = {
        "symbol": "US500",
        "side": Side.BUY,
        "timeframe": "M5",
        "entry_price": 5950.0,
        "stop_loss": 5935.0,
        "take_profit": 5980.0,
        "regime": "TRENDING_EXPANSION",
        "hurst_exponent": 0.62,
        "kaufman_er": 0.44,
        "volatility_percentile": 55.0,
        "dom_obi": 0.25,
        "spread_pct_of_sl": 0.04,
        "net_avg_R": 0.35,
        "prob_profit": 0.61,
        "cvar_95_R": -1.02,
        "recommended_risk_pct": 0.50,
        "session_utc_iso": "2026-11-12T18:15:00Z",
        "bar_timestamp": 1700000000.0,
    }
    defaults.update(overrides)
    return CandidateSetupAuditContext(**defaults)  # type: ignore[arg-type]


def _valid_proof_payload(decision: str = "APPROVE", reason: str = "") -> dict[str, object]:
    return {
        "bull_case_claim": {
            "field": "timemachine_net_avg_R",
            "op": ">=",
            "cited_value": 0.35,
            "limit": 0.10,
        },
        "bear_trap_claim": {
            "field": "spread_pct_of_sl",
            "op": "<=",
            "cited_value": 4.0,
            "limit": 15.0,
        },
        "evidence_claims": [
            {
                "field": "timemachine_net_avg_R",
                "op": ">=",
                "cited_value": 0.35,
                "limit": 0.10,
            },
            {
                "field": "spread_pct_of_sl",
                "op": "<=",
                "cited_value": 4.0,
                "limit": 15.0,
            },
        ],
        "decision": decision,
        "confidence": 0.85,
        "risk_multiplier": 0.80 if decision == "APPROVE" else 0.0,
        "reason": reason or "timemachine_net_avg_R 0.35 >= 0.10 and spread_pct_of_sl 4.0 <= 15.0",
    }


def test_shield1_subtract_only_diode_enforces_bounds() -> None:
    """Shield 1: AI cannot trigger quant-rejected setups, alter SL/TP, or exceed 0.50% risk."""
    # Quant rejected -> AI APPROVE cannot override
    d_rej = AegisVerifier.apply_subtract_only_diode(
        quant_approved=False,
        ai_decision="APPROVE",
        requested_risk_multiplier=2.5,
        recommended_risk_pct=0.50,
        sl_price=5935.0,
        tp_price=5980.0,
    )
    assert d_rej.final_approved is False
    assert d_rej.final_risk_pct == 0.0
    assert d_rej.stop_loss == 5935.0
    assert d_rej.take_profit == 5980.0

    # Quant approved + AI APPROVE with excessive risk_multiplier=3.0 -> clamped to 1.0 and 0.50%
    d_app = AegisVerifier.apply_subtract_only_diode(
        quant_approved=True,
        ai_decision="APPROVE",
        requested_risk_multiplier=3.0,
        recommended_risk_pct=0.80,
        sl_price=5935.0,
        tp_price=5980.0,
    )
    assert d_app.final_approved is True
    assert d_app.risk_multiplier == 1.0
    assert d_app.final_risk_pct == 0.50
    assert d_app.stop_loss == 5935.0
    assert d_app.take_profit == 5980.0


def test_shield2_ticker_blinding_and_leak_prevention() -> None:
    """Shield 2: Strips ticker and raw price levels; rejects any ticker leak in reason."""
    ctx = _make_ctx(symbol="AAPL", entry_price=225.40, stop_loss=224.40, take_profit=227.40)
    blinded = AegisVerifier.normalize_ticker_blinded_telemetry(ctx)
    prompt = AegisVerifier.build_aegis_prompt(blinded)

    assert "AAPL" not in prompt
    assert "225.4" not in prompt
    assert "224.4" not in prompt
    assert "227.4" not in prompt
    assert blinded.telemetry["asset_class"] == "US_EQUITIES"
    assert blinded.telemetry["normalized_entry"] == 100.0
    assert blinded.telemetry["normalized_stop_loss"] == 99.0
    assert blinded.telemetry["normalized_take_profit"] == 102.0
    assert blinded.telemetry["session_utc_hhmm"] == "18:15"
    assert blinded.telemetry["closed_world_contract"] is True

    verifier = AegisVerifier()
    leaked_payload = _valid_proof_payload(reason="AAPL looks strong at 0.35 >= 0.10")
    res = verifier.verify_response(leaked_payload, blinded)
    assert res.hallucination_detected is True
    assert "TICKER_LEAK_DETECTED" in res.violation_codes


def test_shield3_rejects_fabricated_numbers_external_news_and_math_contradictions() -> None:
    """Shield 3: Deterministically catches (a) fabricated numbers, (b) Fed/earnings news, (c) operator contradictions."""
    ctx = _make_ctx()
    blinded = AegisVerifier.normalize_ticker_blinded_telemetry(ctx)
    verifier = AegisVerifier()

    # (a) Fabricated cited_value in evidence_claims (actual net_avg_R is 0.35, LLM cites 0.92)
    fab_payload = _valid_proof_payload()
    fab_payload["evidence_claims"] = [
        {"field": "timemachine_net_avg_R", "op": ">=", "cited_value": 0.92, "limit": 0.10}
    ]
    res_fab = verifier.verify_response(fab_payload, blinded)
    assert res_fab.hallucination_detected is True
    assert "NUMERIC_HALLUCINATION" in res_fab.violation_codes
    assert res_fab.fallback_to_pure_math is True

    # (a2) Unprovided number invented inside reason
    inv_num_payload = _valid_proof_payload(reason="timemachine_net_avg_R 0.35 >= 0.10 and RSI is 78.4")
    res_inv = verifier.verify_response(inv_num_payload, blinded)
    assert res_inv.hallucination_detected is True
    assert "UNPROVIDED_NUMBER_HALLUCINATION" in res_inv.violation_codes

    # (b) Invented external Fed / FOMC / earnings news
    news_payload = _valid_proof_payload(reason="Veto ahead of FOMC rate cut and Powell speech")
    res_news = verifier.verify_response(news_payload, blinded)
    assert res_news.hallucination_detected is True
    assert "EXTERNAL_NEWS_OR_DATE_HALLUCINATION" in res_news.violation_codes

    # (c) Operator math contradiction (claims 0.35 < 0.10 which is False in Python!)
    math_payload = _valid_proof_payload()
    math_payload["evidence_claims"] = [
        {"field": "timemachine_net_avg_R", "op": "<", "cited_value": 0.35, "limit": 0.10}
    ]
    res_math = verifier.verify_response(math_payload, blinded)
    assert res_math.hallucination_detected is True
    assert "OPERATOR_MATH_CONTRADICTION" in res_math.violation_codes

    # Valid proof-carrying response passes cleanly
    ok_payload = _valid_proof_payload()
    res_ok = verifier.verify_response(ok_payload, blinded)
    assert res_ok.passed is True
    assert res_ok.hallucination_detected is False
    assert res_ok.final_decision == "APPROVE"
    assert res_ok.risk_multiplier == 0.80


def test_shield5_canary_trap_quarantines_compromised_model(tmp_path: Path) -> None:
    """Shield 5: Model approving toxic canary setup (net_avg_R=-0.45, spread=27.5%) is quarantined."""
    db = WalkForwardLedgerDB(db_path=tmp_path / "aegis_canary.db")
    selector = AdaptiveModelSelector()
    selector.register_model("agy_pro", "bad-model-v1", ModelTier.PRO_REASONING)
    selector.register_model("agy_pro", "good-model-v2", ModelTier.FLASH_FAST)

    verifier = AegisVerifier(ledger_db=db, model_selector=selector)

    def compromised_eval(toxic_ctx: CandidateSetupAuditContext) -> AIVetoDecision:
        return AIVetoDecision(
            decision="APPROVE",
            confidence=0.95,
            reason="Looks fine",
            latency_ms=120.0,
            bypassed=False,
            provider_used="agy_pro",
            model_used="bad-model-v1",
        )

    probe = verifier.run_canary_integrity_probe(compromised_eval, provider="agy_pro", model_id="bad-model-v1")
    assert probe.passed is False
    assert probe.quarantined is True

    # Quarantined model must be excluded from rank_models
    ranked = selector.rank_models("agy_pro", SetupStakes(0.6, True, False))
    ranked_ids = [r.model.model_id for r in ranked]
    assert "bad-model-v1" not in ranked_ids
    assert "good-model-v2" in ranked_ids


def test_shield6_shadow_grading_brier_and_4factor_demotion(tmp_path: Path) -> None:
    """Shield 6: Logs audits, grades counterfactual vetoes, and auto-demotes if hallucination_rate > 10%."""
    db = WalkForwardLedgerDB(db_path=tmp_path / "aegis_shield6.db")
    selector = AdaptiveModelSelector()
    selector.register_model("studio_api", "gemini-2.5-flash", ModelTier.FLASH_FAST, initial_latency_ms=900.0)
    selector.register_model("studio_api", "gemini-2.5-pro", ModelTier.PRO_REASONING, initial_latency_ms=1200.0)

    # Log 20 audits for gemini-2.5-flash with 3 hallucinations (15% > 10% threshold)
    for i in range(20):
        aid = db.log_ai_epistemic_audit(
            symbol="EURUSD",
            side="BUY",
            timeframe="M5",
            provider="studio_api",
            model_id="gemini-2.5-flash",
            input_sha256=f"sha_{i}",
            raw_decision="VETO" if i % 2 == 0 else "APPROVE",
            final_decision="ABSTAIN" if i < 3 else ("VETO" if i % 2 == 0 else "APPROVE"),
            confidence=0.80,
            hallucination_detected=(i < 3),
            violation_codes="NUMERIC_HALLUCINATION" if i < 3 else "",
        )
        # Grade counterfactual: vetoed trade lost -1.2R -> True Save (+1.2R veto_value_added_r)
        if i % 2 == 0:
            db.grade_ai_shadow_outcome(aid, realized_r=-1.2)
        else:
            db.grade_ai_shadow_outcome(aid, realized_r=0.8)

    stats = db.get_model_epistemic_stats("studio_api", "gemini-2.5-flash", window=20)
    assert stats.total_audits == 20
    assert stats.hallucination_count == 3
    assert stats.hallucination_rate == 0.15
    assert stats.veto_value_added_r == 1.2
    assert stats.should_demote_to_pure_math is True

    # Apply Factor D trust update -> gemini-2.5-flash is quarantined, gemini-2.5-pro wins
    selector.update_epistemic_trust(
        "studio_api",
        "gemini-2.5-flash",
        stats.epistemic_trust_score,
        quarantined=stats.should_demote_to_pure_math,
        reason=stats.demote_reason,
    )
    ranked = selector.rank_models("studio_api", SetupStakes(0.3, False, True))
    assert [r.model.model_id for r in ranked] == ["gemini-2.5-pro"]


def test_committee_hallucination_falls_back_to_pure_math(tmp_path: Path) -> None:
    """When LLM returns a hallucinated claim, AIVetoCommittee discards it and falls back to pure_math."""
    db = WalkForwardLedgerDB(db_path=tmp_path / "aegis_fallback.db")
    cfg_file = tmp_path / "apex_config.json"
    cfg_file.write_text(
        json.dumps({"filters": {"ai_veto": {"enabled": True, "mode": "force_studio"}}}),
        encoding="utf-8",
    )
    committee = AIVetoCommittee(
        api_key="AIzaFakeKey",
        mode=AIVetoMode.FORCE_STUDIO.value,
        env_path=tmp_path / ".env",
        config_path=cfg_file,
        agy_executable_override="",
        ledger_db=db,
    )
    committee.model_selector.register_model("studio_api", "gemini-2.5-flash", ModelTier.FLASH_FAST)

    hallucinated_api_body = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": json.dumps(
                                _valid_proof_payload(
                                    decision="VETO",
                                    reason="Veto due to unexpected CPI inflation report 9.9%",
                                )
                            )
                        }
                    ]
                }
            }
        ]
    }
    fake_resp = MagicMock()
    fake_resp.status = 200
    fake_resp.read.return_value = json.dumps(hallucinated_api_body).encode("utf-8")
    fake_resp.__enter__.return_value = fake_resp

    with patch("urllib.request.urlopen", return_value=fake_resp):
        dec = committee.evaluate_sync(_make_ctx(net_avg_R=0.35, spread_pct_of_sl=0.04))

    assert dec.hallucination_detected is True
    assert dec.provider_used == "pure_math"
    assert dec.decision == "APPROVE"  # pure_math deterministically approves net_avg_R=0.35 >= 0.10
    summary = db.get_aegis_integrity_summary()
    assert summary["hallucination_count"] == 1


def test_mt5_order_check_filling_mode_retry_prevents_silent_drop(tmp_path: Path) -> None:
    """Verify MT5 router probes order_check across filling modes and retries INVALID_FILL (10030)."""
    cfg_file = tmp_path / "apex_config.json"
    cfg_file.write_text(json.dumps({"account_mode": "DEMO"}), encoding="utf-8")
    feed_router = MarketFeedRouter(config_path=cfg_file)
    feed_router.mt5._connected = True  # type: ignore[attr-defined]
    feed_router.mt5._initialized = True
    feed_router.mt5._symbol_map["US500"] = "US500"

    router = DualBrokerExecutionRouter(feed_router=feed_router, account_equity_usd=10_000.0)
    order = ExecutionOrderRequest(
        symbol="US500",
        side=Side.BUY,
        lots=0.92,
        entry_price=5950.0,
        stop_loss=5935.0,
        take_profit=5980.0,
        risk_pct=0.5,
        risk_usd=50.0,
    )

    fake_term = MagicMock(trade_allowed=True)
    fake_info = MagicMock(
        digits=2,
        point=0.01,
        filling_mode=1,  # advertises FOK, but order_check only accepts IOC
        trade_tick_size=0.01,
        trade_tick_value=1.0,
        trade_tick_value_loss=1.0,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
        trade_contract_size=1.0,
    )
    fake_tick = MagicMock(bid=5949.5, ask=5950.0, time=1700000000, time_msc=1700000000000)

    def fake_order_check(req: dict[str, object]) -> MagicMock:
        # Reject FOK (0), accept IOC (1)
        if req["type_filling"] == 0:
            return MagicMock(retcode=10030)
        return MagicMock(retcode=0)

    fake_send_res = MagicMock(retcode=10009, order=998877, price=5950.0, comment="Done")

    with (
        patch("MetaTrader5.terminal_info", return_value=fake_term),
        patch("MetaTrader5.symbol_info", return_value=fake_info),
        patch("MetaTrader5.symbol_info_tick", return_value=fake_tick),
        patch("MetaTrader5.order_check", side_effect=fake_order_check),
        patch("MetaTrader5.order_send", return_value=fake_send_res) as mock_send,
        patch("MetaTrader5.TRADE_RETCODE_DONE", 10009),
        patch("MetaTrader5.ORDER_FILLING_FOK", 0),
        patch("MetaTrader5.ORDER_FILLING_IOC", 1),
        patch("MetaTrader5.ORDER_FILLING_RETURN", 2),
    ):
        receipt = router.execute_order(order, current_spread_price=0.5)

    assert receipt.status == "FILLED"
    assert receipt.venue == "mt5"
    assert mock_send.call_args[0][0]["type_filling"] == 1
    assert any("998877" in ev.details for ev in router.recent_broker_events)


def test_live_isolated_agy_pro_aegis_execution(tmp_path: Path) -> None:
    """Verify live agy_pro runs in isolated temp dir with 30s timeout, 0 MCP hangs, and Aegis proof."""
    agy_path = AIVetoCommittee.locate_agy_executable()
    if not agy_path:
        pytest.skip("agy.exe not installed on this machine")

    db = WalkForwardLedgerDB(db_path=tmp_path / "aegis_live.db")
    cfg_file = tmp_path / "apex_config.json"
    cfg_file.write_text(
        json.dumps(
            {
                "filters": {
                    "ai_veto": {
                        "enabled": True,
                        "mode": "force_pro",
                        "agy_pro_timeout_seconds": 30.0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    committee = AIVetoCommittee(
        mode=AIVetoMode.FORCE_PRO.value,
        env_path=tmp_path / ".env",
        config_path=cfg_file,
        probe_agy_auth=True,
        ledger_db=db,
    )
    status = committee.get_router_status(probe_agy_auth=True)
    if not status.agy_pro_status.ready:
        pytest.skip(f"agy_pro not authenticated: {status.agy_pro_status.detail}")

    t0 = time.perf_counter()
    dec = committee.evaluate_sync(_make_ctx())
    elapsed_s = time.perf_counter() - t0

    assert dec.decision in ("APPROVE", "VETO", "ABSTAIN")
    assert elapsed_s < 60.0
    if dec.provider_used == "agy_pro":
        assert dec.aegis_verified is True
    else:
        assert dec.provider_used == "pure_math"
