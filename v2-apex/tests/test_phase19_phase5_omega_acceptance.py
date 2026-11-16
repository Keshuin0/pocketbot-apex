"""Phase 19 — Apex Omega Phase 5 Acceptance Suite.

Covers Symbolic Z3 Counterexamples & In-Flight Trajectories, Split Conformal Prediction,
Local vLLM / Native Async Agy Timeout Kill / Header-Auth Studio API / Shield 5 Invariance,
Sleep-Wake Ordering & Sample Thresholds, Hardened Cockpit CSRF/Origin Security,
Telegram Auth/MarkdownV2/Async Polling, TUI CLI Entrypoint, Replay CPCV/PBO/DSR & Chronology,
Enriched Open Position Snapshots, and Interactive/Web `--mode confirm` Flow.
"""

from __future__ import annotations

import asyncio
import json
import math
import subprocess
import sys
from dataclasses import replace
from http.cookiejar import CookieJar
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import HTTPCookieProcessor, Request, build_opener

import aiohttp
import pytest
import z3
from aiohttp import web

from apex.ai.aegis_verifier import AegisVerifier
from apex.ai.model_selector import AdaptiveModelSelector, ModelTier
from apex.ai.sleep_consolidation import SleepWakeConsolidator
from apex.ai.veto_committee import (
    AIVetoCommittee,
    AIProviderName,
    CandidateSetupAuditContext,
)
from apex.cockpit.replay_engine import (
    HistoricalReplayEngine,
    compute_cpcv_metrics,
    compute_deflated_sharpe_ratio,
)
from apex.cockpit.telegram_remote import TelegramRemoteController
from apex.cockpit.tui_dashboard import SymbolCockpitRow
from apex.cockpit.web_cockpit import WebCockpitServer
from apex.execution.router import DualBrokerExecutionRouter
from apex.feeds.dual_router import MarketFeedRouter
from apex.feeds.models import Candle, Side
from apex.ledger.walkforward_db import SymbolExecutionMode, WalkForwardLedgerDB
from apex.quant.correlation_shield import PositionRiskSpec
from apex_runner import ApexOrchestrator

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent


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
        "session_utc_iso": "2026-11-12T15:30:00Z",
        "bar_timestamp": 1700000000.0,
        "rollover_illiquidity_flag": False,
        "cross_asset_shock_z": 0.6,
    }
    defaults.update(overrides)
    return CandidateSetupAuditContext(**defaults)  # type: ignore[arg-type]


def _make_trending_candles(
    symbol: str,
    count: int = 120,
    tf: str = "M5",
    base: float = 1.1000,
    step: float = 0.0003,
    base_ts: float = 1762992000.0,
) -> list[Candle]:
    tf_sec = 300.0 if tf == "M5" else (900.0 if tf == "M15" else 3600.0)
    out: list[Candle] = []
    px = base
    for i in range(count):
        delta = step if i % 5 != 3 else -0.3 * step
        o = px
        c = px + delta
        hi = max(o, c) + abs(step) * 0.5
        lo = min(o, c) - abs(step) * 0.5
        out.append(
            Candle(
                symbol=symbol,
                timeframe=tf,
                timestamp_utc=base_ts + i * tf_sec,
                open=round(o, 6),
                high=round(hi, 6),
                low=round(lo, 6),
                close=round(c, 6),
                volume=1200.0 + i * 10.0,
                source="mt5",
            )
        )
        px = c
    return out


# ---------------------------------------------------------------------------
# 1. Z3 Symbolic Counterexamples, Timeout/NaN Fail-Closed & In-Flight Proofs
# ---------------------------------------------------------------------------


def test_omega5_z3_symbolic_counterexamples_and_fail_closed() -> None:
    safe_telem = {
        "timemachine_net_avg_R": 0.35,
        "spread_pct_of_sl": 4.0,
        "vol_pct": 50.0,
        "cross_asset_shock_z": 0.6,
        "rollover_illiquidity_flag": False,
        "recommended_risk_pct": 0.50,
        "remaining_daily_budget_pct": 3.0,
    }
    ok_safe, lat_safe, _ = AegisVerifier.verify_z3_smt_axioms(safe_telem, "APPROVE", [], risk_scale=1.0)
    assert ok_safe is True
    assert lat_safe >= 0.0

    # Budget breach via Exists([d_shock, eps_slip], Loss > remaining_daily_budget)
    budget_breach_telem = {**safe_telem, "recommended_risk_pct": 1.0, "remaining_daily_budget_pct": 1.2}
    ok_breach, lat_breach, msg_breach = AegisVerifier.verify_z3_smt_axioms(
        budget_breach_telem, "APPROVE", [], risk_scale=1.0
    )
    assert ok_breach is False
    assert lat_breach >= 0.0
    assert "Z3_COUNTEREXAMPLE_BUDGET_BREACH" in msg_breach

    # Non-finite NaN / Inf / bool inputs must fail closed
    for bad_key, bad_val in (
        ("timemachine_net_avg_R", float("nan")),
        ("vol_pct", float("inf")),
        ("spread_pct_of_sl", True),
    ):
        bad_telem = {**safe_telem, bad_key: bad_val}
        ok_bad, lat_bad, _ = AegisVerifier.verify_z3_smt_axioms(bad_telem, "APPROVE", [], risk_scale=1.0)
        assert ok_bad is False
        assert lat_bad >= 0.0

    # z3.unknown timeout must fail closed
    with patch.object(z3.Solver, "check", return_value=z3.unknown):
        ok_unk, lat_unk, _ = AegisVerifier.verify_z3_smt_axioms(safe_telem, "APPROVE", [], risk_scale=1.0)
        assert ok_unk is False
        assert lat_unk >= 0.0

    # Shield 8 In-Flight trajectory proofs:
    # (a) HOLD / RATCHET / RATCHET_SLTP blocked under shock > 4.2
    for action in ("HOLD", "RATCHET", "RATCHET_SLTP"):
        ok_shock, _, r_shock = AegisVerifier.verify_z3_inflight_trajectory(
            proposed_action=action,
            bars_held=4,
            min_hold_bars=2,
            abs_move_atr=0.8,
            cross_asset_shock_z=4.5,
            forward_ev_r=0.35,
            current_r=0.40,
            invalidation_signals=0,
        )
        assert ok_shock is False
        assert "Z3_SHIELD8_UNSAT" in r_shock

    # (b) Weekend close blocks RATCHET
    ok_wknd, _, r_wknd = AegisVerifier.verify_z3_inflight_trajectory(
        proposed_action="RATCHET",
        bars_held=6,
        min_hold_bars=2,
        abs_move_atr=1.1,
        cross_asset_shock_z=0.5,
        forward_ev_r=0.40,
        current_r=0.60,
        invalidation_signals=0,
        is_weekend_close=True,
    )
    assert ok_wknd is False
    assert "Z3_SHIELD8_UNSAT" in r_wknd

    # (c) Widening SL on BUY or SELL fails Exists([P_T]) counterexample proof
    ok_widen_buy, _, r_buy = AegisVerifier.verify_z3_inflight_trajectory(
        proposed_action="RATCHET_SLTP",
        bars_held=4,
        min_hold_bars=2,
        abs_move_atr=1.0,
        cross_asset_shock_z=0.4,
        forward_ev_r=0.30,
        current_r=0.40,
        invalidation_signals=0,
        side="BUY",
        entry_price=1.1000,
        sl_old=1.0990,
        sl_new=1.0980,
    )
    assert ok_widen_buy is False
    assert "Z3_SHIELD8_RATCHET_UNSAT" in r_buy

    ok_widen_sell, _, r_sell = AegisVerifier.verify_z3_inflight_trajectory(
        proposed_action="RATCHET_SLTP",
        bars_held=4,
        min_hold_bars=2,
        abs_move_atr=1.0,
        cross_asset_shock_z=0.4,
        forward_ev_r=0.30,
        current_r=0.40,
        invalidation_signals=0,
        side="SELL",
        entry_price=1.1000,
        sl_old=1.1010,
        sl_new=1.1025,
    )
    assert ok_widen_sell is False
    assert "Z3_SHIELD8_RATCHET_UNSAT" in r_sell

    # (d) Valid BUY SL tightening succeeds
    ok_tight, _, _ = AegisVerifier.verify_z3_inflight_trajectory(
        proposed_action="RATCHET_SLTP",
        bars_held=5,
        min_hold_bars=2,
        abs_move_atr=1.3,
        cross_asset_shock_z=0.4,
        forward_ev_r=0.35,
        current_r=0.55,
        invalidation_signals=0,
        side="BUY",
        entry_price=1.1000,
        sl_old=1.0990,
        sl_new=1.0996,
    )
    assert ok_tight is True


# ---------------------------------------------------------------------------
# 2. Shield 3 Structural Allowlist, Split Conformal Prediction & Canary
# ---------------------------------------------------------------------------


def test_omega5_shield3_conformal_and_canary_randomization(tmp_path: Path) -> None:
    ctx = _make_ctx()
    blinded = AegisVerifier.normalize_ticker_blinded_telemetry(ctx)
    verifier = AegisVerifier()

    base_claims = [
        {"field": "timemachine_net_avg_R", "op": ">=", "cited_value": 0.35, "limit": 0.10},
        {"field": "spread_pct_of_sl", "op": "<=", "cited_value": 4.0, "limit": 15.0},
    ]

    # Structural count phrase ("2 verified claims") must pass without UNPROVIDED_NUMBER_HALLUCINATION
    res_struct = verifier.verify_response(
        {
            "bull_case_claim": base_claims[0],
            "bear_trap_claim": base_claims[1],
            "evidence_claims": base_claims,
            "decision": "APPROVE",
            "confidence": 0.85,
            "risk_multiplier": 1.0,
            "reason": "Checked 2 verified claims: timemachine_net_avg_R 0.35 >= 0.10 and spread_pct_of_sl 4.0 <= 15.0",
        },
        blinded,
    )
    assert res_struct.passed is True
    assert "UNPROVIDED_NUMBER_HALLUCINATION" not in res_struct.violation_codes

    # Unrelated fabricated number still fails
    res_fab = verifier.verify_response(
        {
            "bull_case_claim": base_claims[0],
            "bear_trap_claim": base_claims[1],
            "evidence_claims": base_claims,
            "decision": "APPROVE",
            "confidence": 0.85,
            "risk_multiplier": 1.0,
            "reason": "Checked 2 verified claims and RSI is 78.4 with timemachine_net_avg_R 0.35 >= 0.10",
        },
        blinded,
    )
    assert res_fab.hallucination_detected is True
    assert "UNPROVIDED_NUMBER_HALLUCINATION" in res_fab.violation_codes

    # Split Conformal prediction: high nonconformity scores (bad model) -> low p-value for high-conf Hallucination
    db = WalkForwardLedgerDB(tmp_path / "conformal.db")
    for i in range(15):
        aid = db.log_ai_epistemic_audit(
            symbol="US500",
            side="BUY",
            timeframe="M5",
            provider="studio_api",
            model_id="cal-model",
            input_sha256=f"s_{i}",
            raw_decision="APPROVE",
            final_decision="APPROVE",
            confidence=0.80,
            hallucination_detected=False,
            violation_codes="",
        )
        # Non-hallucinating wrong/right calibration rows: realized +0.8R vs APPROVE (y=1 -> alpha = 0.20)
        db.grade_ai_shadow_outcome(aid, realized_r=0.8 if i < 12 else -1.0)

    p_accurate = db.get_conformal_p_value("studio_api", "cal-model", raw_confidence=0.90)  # cand_alpha=0.10 -> conforms well (higher p)
    p_lowconf = db.get_conformal_p_value("studio_api", "cal-model", raw_confidence=0.05)  # cand_alpha=0.95 -> poor conformity (lower p)
    assert p_accurate > p_lowconf
    quantiles = db.compute_conformal_quantiles(alpha=0.15)
    assert "studio_api:cal-model" in quantiles
    assert "conformal_p_calib_85" in quantiles["studio_api:cal-model"]

    # Conformalized Kelly sizing on diode: low conformal_p=0.05 scales multiplier down to floor 0.25
    d_low = AegisVerifier.apply_subtract_only_diode(
        quant_approved=True,
        ai_decision="APPROVE",
        requested_risk_multiplier=1.0,
        recommended_risk_pct=0.50,
        sl_price=5935.0,
        tp_price=5980.0,
        conformal_p_value=0.05,
    )
    assert d_low.final_approved is True
    assert pytest.approx(d_low.risk_multiplier, abs=1e-6) == 0.25

    d_mid = AegisVerifier.apply_subtract_only_diode(
        quant_approved=True,
        ai_decision="APPROVE",
        requested_risk_multiplier=1.0,
        recommended_risk_pct=0.50,
        sl_price=5935.0,
        tp_price=5980.0,
        conformal_p_value=0.10,
    )
    assert pytest.approx(d_mid.risk_multiplier, abs=1e-6) == 0.50

    # Canary toxic context must randomize across calls
    canaries = [AegisVerifier.build_canary_toxic_context() for _ in range(8)]
    unique_signatures = {(c.symbol, c.entry_price, c.spread_pct_of_sl, c.net_avg_R) for c in canaries}
    assert len(unique_signatures) > 1


# ---------------------------------------------------------------------------
# 3. Veto Committee: Header Auth, ABSTAIN, Auth Check, Local vLLM, Async Kill & Shield 5
# ---------------------------------------------------------------------------


def test_omega5_ai_committee_vllm_header_auth_timeout_and_shield5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # (a) Missing/empty ~/.gemini returns False from check_agy_pro_authenticated
    empty_home = tmp_path / "empty_home"
    empty_home.mkdir()
    fake_agy = tmp_path / "agy.exe"
    fake_agy.write_text("stub", encoding="utf-8")
    monkeypatch.setenv("HOME", str(empty_home))
    monkeypatch.setenv("USERPROFILE", str(empty_home))
    ok_missing, _ = AIVetoCommittee.check_agy_pro_authenticated(str(fake_agy), probe_cli=False)
    assert ok_missing is False

    (empty_home / ".gemini").mkdir()
    ok_empty, _ = AIVetoCommittee.check_agy_pro_authenticated(str(fake_agy), probe_cli=False)
    assert ok_empty is False

    (empty_home / ".gemini" / "settings.json").write_text('{"auth":"ok"}', encoding="utf-8")
    ok_present, _ = AIVetoCommittee.check_agy_pro_authenticated(str(fake_agy), probe_cli=False)
    assert ok_present is True

    # (b) parse_structured_decision defaults missing "decision" to ABSTAIN
    dec_missing = AIVetoCommittee.parse_structured_decision('{"confidence": 0.9, "reason": "no decision field"}', 5.0)
    assert dec_missing.decision == "ABSTAIN"

    # (c) AdaptiveModelSelector retains non-Gemini models like claude-sonnet-5-5-medium
    sel = AdaptiveModelSelector()
    fake_cli = SimpleNamespace(
        returncode=0,
        stdout="Available models:\n  claude-sonnet-5-5-medium\n  gemini-2.5-flash\n  text-embedding-004\n",
        stderr="",
    )
    with patch("subprocess.run", return_value=fake_cli):
        discovered = sel.discover_agy_models("agy.exe")
    discovered_ids = {m.model_id for m in discovered}
    assert "claude-sonnet-5-5-medium" in discovered_ids
    assert "gemini-2.5-flash" in discovered_ids
    assert "text-embedding-004" not in discovered_ids

    # (d) Studio REST uses x-goog-api-key header (never ?key= in URL) & /models does too
    captured_studio: dict[str, object] = {}

    def _fake_urlopen(req: Request, timeout: float = 5.0):
        captured_studio["url"] = req.full_url
        captured_studio["headers"] = {k.lower(): v for k, v in req.header_items()}
        body = json.dumps(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": json.dumps(
                                        {
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
                                            "decision": "APPROVE",
                                            "confidence": 0.85,
                                            "risk_multiplier": 1.0,
                                            "reason": "timemachine_net_avg_R 0.35 >= 0.10 and spread_pct_of_sl 4.0 <= 15.0",
                                        }
                                    )
                                }
                            ]
                        }
                    }
                ]
            }
        ).encode("utf-8")

        class _Resp:
            status = 200

            def read(self) -> bytes:
                return body

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        return _Resp()

    ctx = _make_ctx()
    cm = AIVetoCommittee(
        enabled=True,
        api_key="test-secret-key-123",
        priority_order=("studio_api", "pure_math"),
        env_path=tmp_path / ".env",
        config_path=tmp_path / "no.json",
    )
    with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        dec_studio = cm._call_studio_api_sync(ctx, ("studio_api",))
    assert dec_studio.decision == "APPROVE"
    assert "key=" not in str(captured_studio["url"])
    assert captured_studio["headers"].get("x-goog-api-key") == "test-secret-key-123"

    # (e) Async local vLLM server + Shield 5 borderline semantic entropy swap
    async def _run_async_checks() -> None:
        vllm_requests: list[dict] = []

        async def _handle_vllm(request: web.Request) -> web.Response:
            body = await request.json()
            vllm_requests.append(body)
            user_prompt = body["messages"][-1]["content"]
            # Simulate flip on swapped order for Shield 5 borderline test
            is_swapped = "BEAR" in user_prompt.splitlines()[0] or "order_swap" in str(body).lower()
            if len(vllm_requests) == 2 and is_borderline_run["active"]:
                dec_str = "VETO"
            else:
                dec_str = "APPROVE"
            _ = user_prompt
            _ = is_swapped
            net_val = 0.14 if is_borderline_run["active"] else 0.35
            reply = {
                "bull_case_claim": {
                    "field": "timemachine_net_avg_R",
                    "op": ">=",
                    "cited_value": net_val,
                    "limit": 0.10,
                },
                "bear_trap_claim": {
                    "field": "spread_pct_of_sl",
                    "op": "<=",
                    "cited_value": 4.0,
                    "limit": 15.0,
                },
                "evidence_claims": [
                    {"field": "timemachine_net_avg_R", "op": ">=", "cited_value": net_val, "limit": 0.10},
                    {"field": "spread_pct_of_sl", "op": "<=", "cited_value": 4.0, "limit": 15.0},
                ],
                "decision": dec_str,
                "confidence": 0.72,
                "risk_multiplier": 1.0 if dec_str == "APPROVE" else 0.0,
                "reason": f"timemachine_net_avg_R {net_val} >= 0.10 and spread_pct_of_sl 4.0 <= 15.0",
            }
            return web.json_response({"choices": [{"message": {"content": json.dumps(reply)}}]})

        is_borderline_run = {"active": False}
        app = web.Application()
        app.router.add_post("/v1/chat/completions", _handle_vllm)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = runner.addresses[0][1]

        try:
            vllm_cm = AIVetoCommittee(
                enabled=True,
                api_key="",
                priority_order=(AIProviderName.LOCAL_VLLM.value, AIProviderName.PURE_MATH.value),
                vllm_endpoint=f"http://127.0.0.1:{port}/v1/chat/completions",
                vllm_enabled=True,
                env_path=tmp_path / ".env",
                config_path=tmp_path / "no.json",
                timeout_seconds=3.0,
            )
            dec_vllm = await vllm_cm._call_local_vllm_async(ctx, (AIProviderName.LOCAL_VLLM.value,))
            assert dec_vllm.decision == "APPROVE"
            assert dec_vllm.provider_used == AIProviderName.LOCAL_VLLM.value
            assert vllm_requests[0]["response_format"]["type"] == "json_schema"

            # Borderline context triggers Shield 5 swap -> flipped response causes SHIELD5_SEMANTIC_ENTROPY / ABSTAIN
            vllm_requests.clear()
            is_borderline_run["active"] = True
            borderline_ctx = _make_ctx(net_avg_R=0.14, volatility_percentile=85.0)
            assert AegisVerifier.is_borderline_high_stakes(
                AegisVerifier.normalize_ticker_blinded_telemetry(borderline_ctx).telemetry
            )
            dec_flip = await vllm_cm._call_local_vllm_async(borderline_ctx, (AIProviderName.LOCAL_VLLM.value,))
            assert len(vllm_requests) == 2
            assert dec_flip.bypassed is True
            assert dec_flip.provider_used == AIProviderName.PURE_MATH.value
            assert "SHIELD5_SEMANTIC_ENTROPY" in dec_flip.reason

            # (f) Native asyncio.create_subprocess_exec timeout kills child process cleanly
            spawned_procs: list[asyncio.subprocess.Process] = []
            real_create = asyncio.create_subprocess_exec

            async def _sleep_subproc(*args, **kwargs):
                proc = await real_create(
                    sys.executable,
                    "-c",
                    "import time; time.sleep(15)",
                    stdout=kwargs.get("stdout", asyncio.subprocess.PIPE),
                    stderr=kwargs.get("stderr", asyncio.subprocess.PIPE),
                    cwd=kwargs.get("cwd"),
                )
                spawned_procs.append(proc)
                return proc

            agy_cm = AIVetoCommittee(
                enabled=True,
                api_key="",
                priority_order=("agy_pro", "pure_math"),
                agy_executable_override="agy.exe",
                timeout_seconds=0.3,
                env_path=tmp_path / ".env",
                config_path=tmp_path / "no.json",
            )
            agy_cm.agy_pro_timeout_seconds = 0.3
            agy_cm._cached_agy_path = "agy.exe"
            agy_cm._cached_agy_auth_ok = True
            agy_cm.model_selector._models.clear()
            agy_cm.model_selector.register_model("agy_pro", "gemini-2.5-flash", ModelTier.FLASH_FAST)

            with patch("asyncio.create_subprocess_exec", side_effect=_sleep_subproc):
                with pytest.raises((TimeoutError, asyncio.TimeoutError, RuntimeError)):
                    await agy_cm._call_agy_pro_async(ctx, ("agy_pro",))
            assert spawned_procs
            assert spawned_procs[0].returncode is not None
        finally:
            await runner.cleanup()

    asyncio.run(_run_async_checks())


# ---------------------------------------------------------------------------
# 4. Sleep-Wake Ordering & Slippage Calibration Minimum N >= 10 Threshold
# ---------------------------------------------------------------------------


def test_omega5_sleep_wake_order_and_slippage_min_samples(tmp_path: Path) -> None:
    db = WalkForwardLedgerDB(tmp_path / "sleep_omega.db")

    # 9 closed trades (< 10) must NOT emit slippage multipliers
    for i in range(9):
        db.log_trade_outcome(
            order_id=f"T9-{i}",
            symbol="EURUSD",
            side="BUY",
            execution_mode=SymbolExecutionMode.ACTIVE,
            regime="TRENDING_EXPANSION",
            dom_obi=0.25,
            predicted_net_r=0.40,
            ai_decision="APPROVE",
            ai_confidence=0.80,
            ai_reason="ok",
            entry_price=1.1000,
            executed_price=1.1001,
            stop_loss=1.0990,
            take_profit=1.1020,
            realized_slippage_r=0.05,
            realized_net_r=0.50,
            winning_venue="mt5",
        )
    assert db.compute_slippage_calibration_multipliers() == {}

    # Adding the 10th closed trade activates EURUSD/mt5 calibration
    db.log_trade_outcome(
        order_id="T9-9",
        symbol="EURUSD",
        side="BUY",
        execution_mode=SymbolExecutionMode.ACTIVE,
        regime="TRENDING_EXPANSION",
        dom_obi=0.25,
        predicted_net_r=0.40,
        ai_decision="APPROVE",
        ai_confidence=0.80,
        ai_reason="ok",
        entry_price=1.1000,
        executed_price=1.1001,
        stop_loss=1.0990,
        take_profit=1.1020,
        realized_slippage_r=0.05,
        realized_net_r=0.50,
        winning_venue="mt5",
    )
    mults = db.compute_slippage_calibration_multipliers()
    assert "EURUSD" in mults and "mt5" in mults["EURUSD"]

    # Verify SleepWakeConsolidator executes Phase A -> Phase B -> Phase C in strict order
    consolidator = SleepWakeConsolidator(ledger_db=db, memory_path=tmp_path / "sem.json")
    call_order: list[str] = []

    def _pA(*args, **kwargs):
        call_order.append("A")
        return {"total_purged": 0, "gc_collected": 0}

    def _pB(*args, **kwargs):
        call_order.append("B")
        return SimpleNamespace(cycle_number=1), 1

    def _pC(*args, **kwargs):
        call_order.append("C")
        return True, "Canary passed"

    with (
        patch.object(consolidator, "phase_a_purge_working_memory", side_effect=_pA),
        patch.object(consolidator, "phase_b_consolidate_semantic_memory", side_effect=_pB),
        patch.object(consolidator, "phase_c_morning_wakeup_audit", side_effect=_pC),
    ):
        consolidator.run_full_cycle()
    assert call_order == ["A", "B", "C"]


# ---------------------------------------------------------------------------
# 5. Hardened WebCockpit (No CSRF in /api/state or /ws, Cookie CSRF, Cross-Origin 403)
# ---------------------------------------------------------------------------


def test_omega5_web_cockpit_csrf_cookie_and_cross_origin_403() -> None:
    srv = WebCockpitServer(port=8811, max_port=8818)
    try:
        base = srv.start()
        jar = CookieJar()
        opener = build_opener(HTTPCookieProcessor(jar))

        with opener.open(f"{base}/api/state", timeout=5.0) as resp:
            state_json = json.loads(resp.read().decode("utf-8"))
            assert "csrf_token" not in state_json

        with opener.open(f"{base}/api/csrf", timeout=5.0) as resp:
            assert resp.status == 200
            set_cookie = resp.headers.get("Set-Cookie", "")
            assert "apex_csrf=" in set_cookie
            assert "HttpOnly" in set_cookie
            assert "SameSite=Strict" in set_cookie
            csrf_token = json.loads(resp.read().decode("utf-8"))["csrf_token"]

        # Valid same-origin POST works with cookie + header
        ok_req = Request(
            f"{base}/api/control/pause",
            data=b"{}",
            headers={"X-Apex-CSRF": csrf_token, "Origin": base},
            method="POST",
        )
        with opener.open(ok_req, timeout=5.0) as resp:
            assert resp.status == 200

        # Cross-origin Origin on /api/csrf, /api/control/pause, and /ws returns 403
        for path, method in (("/api/csrf", "GET"), ("/api/control/pause", "POST")):
            bad_req = Request(
                f"{base}{path}",
                data=(b"{}" if method == "POST" else None),
                headers={"Origin": "https://evil.example", "X-Apex-CSRF": csrf_token},
                method=method,
            )
            with pytest.raises(HTTPError) as exc_info:
                opener.open(bad_req, timeout=5.0)
            assert exc_info.value.code == 403

        async def _check_ws_security() -> None:
            ws_url = base.replace("http://", "ws://") + "/ws"
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(ws_url, headers={"Origin": base}) as ws:
                    first = await ws.receive_json()
                    assert "csrf_token" not in first

                with pytest.raises(aiohttp.WSServerHandshakeError) as ws_err:
                    await session.ws_connect(ws_url, headers={"Origin": "https://evil.example"})
                assert ws_err.value.status == 403

        asyncio.run(_check_ws_security())
    finally:
        srv.stop()


# ---------------------------------------------------------------------------
# 6. Telegram Security, MarkdownV2 Escaping, Commands & Async Long Polling
# ---------------------------------------------------------------------------


def test_omega5_telegram_security_markdownv2_and_polling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    feed_router = MarketFeedRouter(config_path=WORKSPACE_ROOT / "config" / "apex_config.json")
    exec_router = DualBrokerExecutionRouter(feed_router=feed_router)
    ledger = WalkForwardLedgerDB(db_path=tmp_path / "tg_omega.db")

    escaped = TelegramRemoteController.escape_markdown_v2("US500 +0.35R [v3.0]!")
    for special in ("+", ".", "[", "]", "!"):
        assert f"\\{special}" in escaped

    modes_seen: list[str] = []
    tg = TelegramRemoteController(
        execution_router=exec_router,
        ledger_db=ledger,
        bot_token="fake-token",
        chat_id="999001",
        mode_change_callback=modes_seen.append,
    )

    # Unauthorized chat_id rejected
    assert (
        tg.process_update(
            {"update_id": 1, "message": {"chat": {"id": 123456}, "text": "/panic"}}
        )
        is None
    )
    assert tg.scanning_halted is False

    # Authorized commands & @bot suffix stripping
    assert "No open positions" in (
        tg.process_update(
            {"update_id": 2, "message": {"chat": {"id": 999001}, "text": "/positions@ApexBot"}}
        )
        or ""
    )
    assert "Daily Risk & Performance" in (
        tg.process_update({"update_id": 3, "message": {"chat": {"id": 999001}, "text": "/daily"}}) or ""
    )
    assert "Execution mode updated" in (
        tg.process_update(
            {"update_id": 4, "message": {"chat": {"id": 999001}, "text": "/mode confirm"}}
        )
        or ""
    )
    assert modes_seen == ["confirm"]
    assert "Remote Commands" in (
        tg.process_update({"update_id": 5, "message": {"chat": {"id": 999001}, "text": "/help"}}) or ""
    )

    # Local aiohttp Telegram Bot API poll_updates_loop test
    async def _run_poll_test() -> None:
        sent_messages: list[dict] = []

        async def _get_updates(request: web.Request) -> web.Response:
            return web.json_response(
                {
                    "ok": True,
                    "result": [
                        {"update_id": 100, "message": {"chat": {"id": 777}, "text": "/panic"}},
                        {"update_id": 101, "message": {"chat": {"id": 999001}, "text": "/resume"}},
                    ],
                }
            )

        async def _send_msg(request: web.Request) -> web.Response:
            sent_messages.append(await request.json())
            return web.json_response({"ok": True, "result": {}})

        app = web.Application()
        app.router.add_get("/botfake-token/getUpdates", _get_updates)
        app.router.add_post("/botfake-token/sendMessage", _send_msg)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = runner.addresses[0][1]
        try:
            monkeypatch.setattr(
                TelegramRemoteController,
                "API_BASE",
                f"http://127.0.0.1:{port}/bot{{token}}/{{method}}",
            )
            async with aiohttp.ClientSession() as sess:
                await tg.poll_updates_loop(poll_timeout=1, max_iterations=1, session=sess)
            assert len(sent_messages) == 1
            assert sent_messages[0]["parse_mode"] == "MarkdownV2"
        finally:
            await runner.cleanup()

    asyncio.run(_run_poll_test())


# ---------------------------------------------------------------------------
# 7. TUI Entrypoint, Replay CPCV/PBO/DSR & Chronology, Enriched Positions & Confirm Mode
# ---------------------------------------------------------------------------


def test_omega5_tui_cli_replay_cpcv_positions_and_confirm_mode(tmp_path: Path) -> None:
    # (a) Standalone TUI CLI preview entrypoint exits cleanly (both --help and snapshot render)
    for tui_args in (["--help"], ["--equity", "12500", "--mode", "DEMO-CONFIRM"]):
        proc = subprocess.run(
            [sys.executable, "-m", "apex.cockpit.tui_dashboard", *tui_args],
            cwd=str(WORKSPACE_ROOT),
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert proc.returncode == 0

    # (b) Replay Engine: tier-aware M5 preference, AUTO_TOP12 handling, CPCV/PBO/DSR, finite PF, non-overlapping trades
    seen_tfs: list[tuple[str, str]] = []

    class _TierFeed:
        def get_candles(self, symbol: str, timeframe: str = "M15", count: int = 200):
            seen_tfs.append((symbol, timeframe))
            if symbol == "EURUSD":
                return _make_trending_candles("EURUSD", count=140, tf=timeframe)
            return None

    eng = HistoricalReplayEngine(step_delay_seconds=0.0)
    updates: list[dict] = []
    summary = eng.run_replay(
        feed_router=_TierFeed(),
        symbols="AUTO_TOP12",
        window_days=2,
        speed=100,
        on_update=updates.append,
    )
    assert ("EURUSD", "M5") in seen_tfs
    assert updates[-1]["status"] == "COMPLETED"
    assert math.isfinite(summary.profit_factor)
    assert isinstance(summary.cpcv_paths, int) and summary.cpcv_paths >= 0
    assert 0.0 <= summary.pbo_probability <= 1.0
    assert math.isfinite(summary.deflated_sharpe_ratio)

    # Zero-loss bounded PF & non-overlapping chronological entry/exit bar indices
    cpcv_paths, cpcv_mean_sharpe, pbo_prob = compute_cpcv_metrics(
        [0.8, 0.9, 0.7, 1.1, 0.6, 0.85, 0.95, 1.0, 0.75, 0.8, 0.9, 1.0]
    )
    assert cpcv_paths >= 1
    assert math.isfinite(cpcv_mean_sharpe)
    assert 0.0 <= pbo_prob <= 1.0
    dsr = compute_deflated_sharpe_ratio([0.8, -0.4, 0.9, 1.1, -0.3, 0.7, 0.85, 0.6], num_trials=12)
    assert math.isfinite(dsr)

    for tr in summary.trades:
        if "entry_bar" in tr and "exit_bar" in tr:
            assert int(tr["exit_bar"]) > int(tr["entry_bar"])

    # (c) Enriched PositionRiskSpec -> WebCockpit snapshot propagation & `--mode confirm`
    orch = ApexOrchestrator(
        mode="confirm",
        config_path=WORKSPACE_ROOT / "config" / "apex_config.json",
        db_path=tmp_path / "confirm_omega.db",
    )
    cockpit = WebCockpitServer(port=8821, max_port=8828)
    try:
        cockpit.start()
        orch.attach_web_cockpit(cockpit)

        pos = PositionRiskSpec(
            symbol="EURUSD",
            side="BUY",
            risk_pct=0.50,
            lots=0.10,
            entry_price=1.10000,
            current_price=1.10120,
            stop_loss=1.09900,
            take_profit=1.10300,
            profit_usd=12.0,
            ticket=98765,
            venue="mt5",
        )
        orch._push_web_cockpit_snapshot(
            account_currency="USD",
            daily_pnl=12.0,
            open_positions=[pos],
            rows=[],
            ai_status=None,
            aegis_badge="OK",
            contagion_z=0.5,
            mind_badge="READY",
            combined_events=[],
        )
        snap = cockpit.get_state_snapshot()
        assert len(snap["open_positions"]) == 1
        snap_pos = snap["open_positions"][0]
        assert snap_pos["lots"] == 0.10
        assert snap_pos["entry_price"] == pytest.approx(1.10000)
        assert snap_pos["current_price"] == pytest.approx(1.10120)
        assert snap_pos["stop_loss"] == pytest.approx(1.09900)
        assert snap_pos["take_profit"] == pytest.approx(1.10300)
        assert snap_pos["profit_usd"] == pytest.approx(12.0)

        row = SymbolCockpitRow(
            symbol="EURUSD",
            tier="TIGHT_SPREAD_M5",
            timeframe="M5",
            bid=1.10000,
            ask=1.10005,
            spread_pips=0.5,
            regime="TRENDING_EXPANSION",
            hurst=0.62,
            kaufman_er=0.45,
            obi=0.30,
            net_avg_r=0.42,
            rec_risk_pct=0.50,
            ai_verdict="APPROVE",
            quarantine_mode="ACTIVE",
            profit_factor=1.8,
            best_venue="mt5",
            side=Side.BUY,
            entry_price=1.10005,
            stop_loss=1.09905,
            take_profit=1.10205,
            pipeline_trace={"ai_risk_multiplier": 1.0},
        )

        # Interactive `[y/N]` prompt executes order when user answers "y"
        prompts: list[str] = []
        orch.confirm_input_fn = lambda msg: (prompts.append(msg) or "y")
        orch._process_confirm_mode_candidates([row], [])
        assert len(prompts) == 1
        assert "[y/N]" in prompts[0]
        assert len(orch.execution_router.get_open_positions()) == 1

        # Non-interactive web confirm queue + WebCockpit `/api/control/confirm`
        orch.confirm_input_fn = None
        row2 = replace(row, symbol="GBPUSD", entry_price=1.27005, stop_loss=1.26905, take_profit=1.27205)
        orch._process_confirm_mode_candidates([row2], orch.execution_router.get_open_positions())
        pending = cockpit.get_state_snapshot()["pending_confirmations"]
        assert len(pending) == 1
        cid = pending[0]["confirm_id"]
        confirm_res = orch._handle_web_confirm({"confirm_id": cid, "decision": "APPROVE"})
        assert confirm_res["executed"] is True
        assert len(orch.execution_router.get_open_positions()) == 2
    finally:
        cockpit.stop()
        orch.shutdown()
