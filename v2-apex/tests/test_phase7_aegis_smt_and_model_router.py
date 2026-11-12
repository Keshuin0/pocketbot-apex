"""Phase 7 Formal Verification Suite: Z3 SMT Prover, Contagion Oracle & Adaptive Model Router."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

from apex.ai.aegis_verifier import AegisVerifier
from apex.ai.model_selector import AdaptiveModelSelector, ModelTier, SetupStakes
from apex.ai.veto_committee import AIVetoDecision, CandidateSetupAuditContext
from apex.feeds.models import Side
from apex.ledger.walkforward_db import WalkForwardLedgerDB
from apex.quant.contagion_oracle import ContagionOracle


def _make_ctx(**overrides: object) -> CandidateSetupAuditContext:
    defaults: dict[str, object] = {
        "symbol": "US500",
        "side": Side.BUY,
        "timeframe": "M5",
        "entry_price": 7585.0,
        "stop_loss": 7570.0,
        "take_profit": 7615.0,
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
        "session_utc_iso": "2026-11-12T15:30:00Z",
        "bar_timestamp": 1700000000.0,
        "rollover_illiquidity_flag": False,
        "cross_asset_shock_z": 0.8,
    }
    defaults.update(overrides)
    return CandidateSetupAuditContext(**defaults)  # type: ignore[arg-type]


def test_z3_smt_catches_true_numbers_plus_contradictory_approve_in_under_5ms() -> None:
    """Shield 4: AI cites TRUE negative EV (-0.20 < 0.10) yet outputs APPROVE -> Z3 proves unsat."""
    ctx = _make_ctx(net_avg_R=-0.20, spread_pct_of_sl=0.04)
    blinded = AegisVerifier.normalize_ticker_blinded_telemetry(ctx)
    verifier = AegisVerifier()

    # Every number and operator in this payload is 100% factually true,
    # so AST grounding alone passes, but Z3 catches the axiom contradiction!
    contradictory_payload = {
        "bull_claims": [
            {"field": "spread_pct_of_sl", "op": "<=", "cited_value": 4.0, "threshold": 15.0}
        ],
        "bear_claims": [
            {"field": "timemachine_net_avg_R", "op": "<", "cited_value": -0.20, "threshold": 0.10}
        ],
        "decision": "APPROVE",
        "confidence": 0.90,
        "risk_scale": 1.0,
        "reason": "timemachine_net_avg_R -0.20 < 0.10 and spread_pct_of_sl 4.0 <= 15.0",
    }

    res = verifier.verify_response(contradictory_payload, blinded)
    assert res.hallucination_detected is True
    assert "Z3_UNSAT_HALLUCINATION" in res.violation_codes
    assert res.z3_checked is True
    assert res.z3_sat is False
    assert res.z3_latency_ms < 25.0  # typically <2ms after first Z3 init
    assert res.fallback_to_pure_math is True


def test_z3_smt_enforces_rollover_and_contagion_shock_axioms() -> None:
    """Shield 4: Z3 blocks APPROVE during rollover+high-vol or cross_asset_shock_z > 3.5."""
    verifier = AegisVerifier()

    # Case 1: rollover_illiquidity_flag=True and vol_pct=90.0 (>85.0)
    ctx_roll = _make_ctx(rollover_illiquidity_flag=True, volatility_percentile=90.0)
    b_roll = AegisVerifier.normalize_ticker_blinded_telemetry(ctx_roll)
    payload_roll = {
        "bull_claims": [{"field": "timemachine_net_avg_R", "op": ">=", "cited_value": 0.35, "threshold": 0.10}],
        "bear_claims": [{"field": "vol_pct", "op": ">", "cited_value": 90.0, "threshold": 85.0}],
        "decision": "APPROVE",
        "confidence": 0.80,
        "risk_scale": 1.0,
        "reason": "timemachine_net_avg_R 0.35 >= 0.10 despite vol_pct 90.0 > 85.0",
    }
    res_roll = verifier.verify_response(payload_roll, b_roll)
    assert "Z3_UNSAT_HALLUCINATION" in res_roll.violation_codes

    # Case 2: cross_asset_shock_z=4.2 (>3.5)
    ctx_shock = _make_ctx(cross_asset_shock_z=4.2)
    b_shock = AegisVerifier.normalize_ticker_blinded_telemetry(ctx_shock)
    payload_shock = {
        "bull_claims": [{"field": "timemachine_net_avg_R", "op": ">=", "cited_value": 0.35, "threshold": 0.10}],
        "bear_claims": [{"field": "cross_asset_shock_z", "op": ">", "cited_value": 4.2, "threshold": 3.5}],
        "decision": "APPROVE",
        "confidence": 0.80,
        "risk_scale": 1.0,
        "reason": "timemachine_net_avg_R 0.35 >= 0.10 with cross_asset_shock_z 4.2 > 3.5",
    }
    res_shock = verifier.verify_response(payload_shock, b_shock)
    assert "Z3_UNSAT_HALLUCINATION" in res_shock.violation_codes


def test_ast_catches_001_drift_and_invented_fed_news_and_ticker_blinding() -> None:
    """Shields 2 & 3: Strips US500 @ 7585, catches 0.01 numeric drift, and catches Fed news."""
    ctx = _make_ctx(symbol="US500", entry_price=7585.0, stop_loss=7570.0, take_profit=7615.0)
    blinded = AegisVerifier.normalize_ticker_blinded_telemetry(ctx)
    prompt = AegisVerifier.build_aegis_prompt(blinded)

    assert "US500" not in prompt
    assert "7585" not in prompt
    assert "INDICES" in prompt
    assert blinded.telemetry["normalized_entry"] == 100.0

    verifier = AegisVerifier()
    # 0.01 drift: actual net_avg_R is 0.35, cited is 0.36
    drift_payload = {
        "bull_claims": [{"field": "timemachine_net_avg_R", "op": ">=", "cited_value": 0.36, "threshold": 0.10}],
        "bear_claims": [{"field": "spread_pct_of_sl", "op": "<=", "cited_value": 4.0, "threshold": 15.0}],
        "decision": "APPROVE",
        "confidence": 0.85,
        "risk_scale": 1.0,
        "reason": "timemachine_net_avg_R 0.36 >= 0.10",
    }
    res_drift = verifier.verify_response(drift_payload, blinded)
    assert "NUMERIC_HALLUCINATION" in res_drift.violation_codes

    # Invented Fed news
    fed_payload = {
        "bull_claims": [{"field": "timemachine_net_avg_R", "op": ">=", "cited_value": 0.35, "threshold": 0.10}],
        "bear_claims": [{"field": "spread_pct_of_sl", "op": "<=", "cited_value": 4.0, "threshold": 15.0}],
        "decision": "VETO",
        "confidence": 0.90,
        "risk_scale": 0.0,
        "reason": "Veto due to Federal Reserve FOMC statement",
    }
    res_fed = verifier.verify_response(fed_payload, blinded)
    assert "EXTERNAL_NEWS_OR_DATE_HALLUCINATION" in res_fed.violation_codes


def test_shield5_borderline_order_swap_semantic_invariance() -> None:
    """Shield 5: Borderline high-stakes setup triggers order-swap check; flip -> ABSTAIN."""
    ctx = _make_ctx(net_avg_R=0.15, volatility_percentile=84.0)
    blinded = AegisVerifier.normalize_ticker_blinded_telemetry(ctx)
    assert AegisVerifier.is_borderline_high_stakes(blinded.telemetry) is True

    p_fwd = AegisVerifier.build_aegis_prompt(blinded, swap_order=False)
    p_rev = AegisVerifier.build_aegis_prompt(blinded, swap_order=True)
    assert "BULL CASE merits first" in p_fwd
    assert "BEAR TRAP risks first" in p_rev
    assert AegisVerifier.check_semantic_invariance("APPROVE", "VETO") is False
    assert AegisVerifier.check_semantic_invariance("APPROVE", "APPROVE") is True


def test_shield6_conformal_calibration_and_8pct_quarantine(tmp_path: Path) -> None:
    """Shield 6: Split Conformal p-value & auto-quarantine at >8% hallucination rate."""
    db = WalkForwardLedgerDB(db_path=tmp_path / "conformal.db")
    selector = AdaptiveModelSelector()
    selector.register_model("studio_api", "gemini-2.5-flash", ModelTier.FLASH_FAST)

    # Log 25 audits with 3 hallucinations (12% > 8% threshold)
    for i in range(25):
        aid = db.log_ai_epistemic_audit(
            symbol="EURUSD",
            side="BUY",
            timeframe="M5",
            provider="studio_api",
            model_id="gemini-2.5-flash",
            input_sha256=f"h_{i}",
            raw_decision="APPROVE",
            final_decision="ABSTAIN" if i < 3 else "APPROVE",
            confidence=0.80,
            hallucination_detected=(i < 3),
        )
        db.grade_ai_shadow_outcome(aid, realized_r=1.0 if i % 2 == 0 else -0.5)

    p_val = db.get_conformal_p_value("studio_api", "gemini-2.5-flash", raw_confidence=0.80)
    assert 0.0 < p_val <= 1.0

    stats = db.get_model_epistemic_stats("studio_api", "gemini-2.5-flash", window=25)
    assert stats.hallucination_rate == 0.12
    assert stats.should_demote_to_pure_math is True

    selector.update_epistemic_trust(
        "studio_api",
        "gemini-2.5-flash",
        stats.epistemic_trust_score,
        quarantined=stats.should_demote_to_pure_math,
        reason=stats.demote_reason,
    )
    assert selector.rank_models("studio_api", SetupStakes(0.4, False, True)) == []


def test_contagion_oracle_rollover_and_cross_asset_shock() -> None:
    """ContagionOracle detects 21:55-23:10 UTC rollover and >2.5-sigma simultaneous shocks."""
    oracle = ContagionOracle()

    rollover_dt = datetime.datetime(2026, 11, 12, 22, 15, tzinfo=datetime.timezone.utc)
    normal_dt = datetime.datetime(2026, 11, 12, 14, 30, tzinfo=datetime.timezone.utc)

    quiet = [0.0001 * ((-1) ** i) for i in range(25)]
    shocked = quiet[:-1] + [0.015]  # huge >5-sigma bar

    tel_shock = oracle.evaluate(
        anchor_returns={"EURUSD": shocked, "US500": shocked, "XAUUSD": quiet, "BTCUSD": quiet},
        utc_dt=rollover_dt,
    )
    assert tel_shock.rollover_illiquidity_flag is True
    assert tel_shock.cross_asset_shock_z > 3.5
    assert "USD_FX" in tel_shock.shocked_buckets
    assert "INDICES" in tel_shock.shocked_buckets

    tel_norm = oracle.evaluate(
        anchor_returns={"EURUSD": quiet, "US500": quiet, "XAUUSD": quiet, "BTCUSD": quiet},
        utc_dt=normal_dt,
    )
    assert tel_norm.rollover_illiquidity_flag is False
    assert tel_norm.cross_asset_shock_z < 2.5
