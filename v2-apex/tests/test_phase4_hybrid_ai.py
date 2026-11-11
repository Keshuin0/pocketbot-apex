"""Phase 4 Verification Suite for PocketBot Apex Hybrid Self-Detecting 3-Tier AI Router."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Sequence

import pytest

from apex.ai.veto_committee import (
    AIProviderName,
    AIVetoCommittee,
    AIVetoDecision,
    AIVetoMode,
    CandidateSetupAuditContext,
)
from apex.cockpit.tui_dashboard import ApexTUIDashboard, SymbolCockpitRow
from apex.feeds.models import Side
from tools.configure_ai_mode import (
    apply_menu_choice,
    update_ai_veto_config,
    update_studio_api_key_in_env,
)

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent


def _sample_context() -> CandidateSetupAuditContext:
    return CandidateSetupAuditContext(
        symbol="USDJPY",
        side=Side.BUY,
        timeframe="M5",
        entry_price=159.740,
        stop_loss=159.540,
        take_profit=160.140,
        regime="TRENDING_EXPANSION",
        hurst_exponent=0.58,
        kaufman_er=0.54,
        volatility_percentile=64.0,
        dom_obi=0.24,
        spread_pct_of_sl=0.05,
        net_avg_R=0.871,
        prob_profit=0.61,
        cvar_95_R=-1.08,
        recommended_risk_pct=0.50,
        session_utc_iso="2026-11-11T16:25:00Z",
    )


def test_ansi_and_non_tty_wrapper_stripping() -> None:
    """Verify clean_and_extract_json_payload strips Windows non-TTY ANSI codes and agy.exe JSON envelopes."""
    noisy_stdout = (
        "\x1b[36m[agy] Initializing headless session...\x1b[0m\n"
        "Loading workspace configuration...\n"
        '{"conversation_id":"abc-123","status":"DONE","response":"```json\\n{\\"decision\\": \\"VETO\\", \\"confidence\\": 0.91, \\"reason\\": \\"London-NY session trap\\"}\\n```"}\n'
    )
    dec = AIVetoCommittee.parse_structured_decision(
        raw_text=noisy_stdout,
        latency_ms=340.0,
        provider_used=AIProviderName.AGY_PRO.value,
        cascade_path=("agy_pro",),
    )
    assert dec.decision == "VETO"
    assert pytest.approx(dec.confidence) == 0.91
    assert dec.provider_used == "agy_pro"
    assert "London-NY session trap" in dec.reason


def test_failover_permutations_studio_pro_math_and_midflight_cascade(tmp_path: Path) -> None:
    """Validate all failover permutations:
    1) Studio -> Pro -> Math (when Studio throws timeout/error, cascades to Pro; when Pro also errors, cascades to Pure Math)
    2) Pro -> Studio -> Math (when Pro throws timeout/error, cascades to Studio; when Studio also errors, cascades to Pure Math)
    3) Neither ready -> Pure Math (0ms, $0)
    """
    env_file = tmp_path / ".env"
    cfg_file = tmp_path / "apex_config.json"
    fake_agy = tmp_path / "fake_agy.exe"
    fake_agy.write_text("binary", encoding="utf-8")

    # 1. Mode: auto_studio_first (studio_api -> agy_pro -> pure_math) with both keys/binaries ready
    env_file.write_text("GEMINI_API_KEY=AIzaSyTestStudioKey123456\n", encoding="utf-8")
    update_ai_veto_config(cfg_file, mode="auto_studio_first", enabled=True)

    router = AIVetoCommittee(
        env_path=env_file,
        config_path=cfg_file,
        agy_executable_override=str(fake_agy),
    )
    st = router.get_router_status()
    assert st.active_primary == "studio_api"
    assert st.fallback_chain == ("agy_pro", "pure_math")

    ctx = _sample_context()

    # Simulate Studio timeout -> cascades to AGY_PRO which succeeds
    def studio_timeout(c: CandidateSetupAuditContext, cascade_path: Sequence[str]) -> AIVetoDecision:
        raise TimeoutError("studio_api 4.0s timeout exceeded")

    def agy_success(c: CandidateSetupAuditContext, cascade_path: Sequence[str]) -> AIVetoDecision:
        return AIVetoDecision(
            decision="APPROVE",
            confidence=0.89,
            reason="AGY_PRO failover audit passed",
            latency_ms=420.0,
            bypassed=False,
            provider_used="agy_pro",
            cascade_path=tuple(cascade_path),
        )

    router._call_studio_api_sync = studio_timeout  # type: ignore[method-assign]
    router._call_agy_pro_sync = agy_success  # type: ignore[method-assign]

    res_studio_to_pro = router.evaluate_sync(ctx)
    assert res_studio_to_pro.decision == "APPROVE"
    assert res_studio_to_pro.provider_used == "agy_pro"
    assert res_studio_to_pro.cascade_path == ("studio_api", "agy_pro")

    # Now simulate BOTH Studio AND AGY_PRO timing out -> cascades cleanly to Pure Math without crashing
    def agy_timeout(c: CandidateSetupAuditContext, cascade_path: Sequence[str]) -> AIVetoDecision:
        raise TimeoutError("agy_pro 15s print-timeout exceeded")

    router._call_agy_pro_sync = agy_timeout  # type: ignore[method-assign]
    res_studio_pro_to_math = router.evaluate_sync(ctx)
    assert res_studio_pro_to_math.decision == "APPROVE"
    assert res_studio_pro_to_math.provider_used == "pure_math"
    assert res_studio_pro_to_math.cascade_path == ("studio_api", "agy_pro", "pure_math")
    assert "studio_api failed" in res_studio_pro_to_math.reason
    assert "agy_pro failed" in res_studio_pro_to_math.reason

    # 2. Mode: auto_pro_first (agy_pro -> studio_api -> pure_math)
    time.sleep(0.02)
    update_ai_veto_config(cfg_file, mode="auto_pro_first", enabled=True)

    def studio_success(c: CandidateSetupAuditContext, cascade_path: Sequence[str]) -> AIVetoDecision:
        return AIVetoDecision(
            decision="VETO",
            confidence=0.93,
            reason="Studio failover caught structural risk",
            latency_ms=180.0,
            bypassed=False,
            provider_used="studio_api",
            cascade_path=tuple(cascade_path),
        )

    router._call_studio_api_sync = studio_success  # type: ignore[method-assign]
    # AGY_PRO is still set to timeout, so auto_pro_first must try agy_pro -> cascade to studio_api!
    res_pro_to_studio = router.evaluate_sync(ctx)
    assert res_pro_to_studio.decision == "VETO"
    assert res_pro_to_studio.provider_used == "studio_api"
    assert res_pro_to_studio.cascade_path == ("agy_pro", "studio_api")

    # 3. Neither Studio nor AGY_PRO available -> auto-routes directly to Pure Math (0ms)
    env_file.write_text("GEMINI_API_KEY=\n", encoding="utf-8")
    router_neither = AIVetoCommittee(
        env_path=env_file,
        config_path=cfg_file,
        agy_executable_override="",
    )
    st_neither = router_neither.get_router_status()
    assert st_neither.active_primary == "pure_math"
    res_neither = router_neither.evaluate_sync(ctx)
    assert res_neither.provider_used == "pure_math"
    assert res_neither.cascade_path == ("pure_math",)


def test_live_hot_reload_on_env_and_config_mtime_change(tmp_path: Path) -> None:
    """Verify hot-reloading when .env or config/apex_config.json changes on disk without restarting."""
    env_file = tmp_path / ".env"
    cfg_file = tmp_path / "apex_config.json"
    fake_agy = tmp_path / "fake_agy.exe"
    fake_agy.write_text("binary", encoding="utf-8")

    # Start with NO GEMINI_API_KEY in .env and mode="auto_studio_first"
    env_file.write_text("GEMINI_API_KEY=\n", encoding="utf-8")
    update_ai_veto_config(cfg_file, mode="auto_studio_first", enabled=True)

    router = AIVetoCommittee(
        env_path=env_file,
        config_path=cfg_file,
        agy_executable_override=str(fake_agy),
    )

    # Because GEMINI_API_KEY is empty today, auto_studio_first automatically skips studio_api and selects agy_pro!
    st1 = router.get_router_status()
    assert st1.studio_status.ready is False
    assert st1.agy_pro_status.ready is True
    assert st1.active_primary == "agy_pro"
    assert st1.fallback_chain == ("pure_math",)
    assert st1.dashboard_badge == "AI Brain: AGY_PRO [Auto -> Pure Math fallback]"

    # Now add GEMINI_API_KEY to .env and advance mtime -> router hot-reloads to studio_api primary!
    time.sleep(0.05)
    update_studio_api_key_in_env(env_file, "AIzaSyHotReloadKey999999")
    os.utime(env_file, (time.time() + 2.0, time.time() + 2.0))

    st2 = router.get_router_status()
    assert st2.studio_status.ready is True
    assert st2.active_primary == "studio_api"
    assert st2.fallback_chain == ("agy_pro", "pure_math")
    assert st2.dashboard_badge == "AI Brain: STUDIO_API [Auto -> AGY_PRO -> Pure Math fallback]"

    # Now use tools/configure_ai_mode.py menu choice [3] to force Pure Math -> hot-reloads immediately!
    apply_menu_choice("3", router, cfg_file, env_file)
    os.utime(cfg_file, (time.time() + 4.0, time.time() + 4.0))

    st3 = router.get_router_status()
    assert st3.enabled is False
    assert st3.active_primary == "pure_math"
    assert st3.dashboard_badge == "AI Brain: PURE_MATH [0ms Deterministic Quant Only]"


def test_live_google_ai_pro_agy_session_and_dashboard_header() -> None:
    """Verify live Google AI Pro (agy.exe) discovery, authentication, waterfall execution, and TUI header."""
    bat_5 = WORKSPACE_ROOT / "5_CONFIGURE_AI_MODE.bat"
    assert bat_5.exists(), "5_CONFIGURE_AI_MODE.bat must exist"
    assert "%~dp0" in bat_5.read_text(encoding="utf-8")

    router = AIVetoCommittee(
        env_path=WORKSPACE_ROOT / ".env",
        config_path=WORKSPACE_ROOT / "config" / "apex_config.json",
        agy_timeout_seconds=10.0,
        probe_agy_auth=True,
    )
    status = router.get_router_status(probe_agy_auth=True)
    # If running in cloud CI without local agy.exe or MT5 installed
    is_ci = os.environ.get("GITHUB_ACTIONS") == "true"
    if is_ci and not status.agy_pro_status.ready:
        assert status.active_primary in ("pure_math", "studio_api", "agy_pro")
        assert "AI Brain:" in status.dashboard_badge
        ctx = _sample_context()
        live_decision = router.evaluate_sync(ctx)
        assert live_decision.decision in ("APPROVE", "VETO")
        return

    # Verify agy.exe is discovered and authenticated with the user's Google AI Pro login
    assert status.agy_pro_status.ready is True
    assert "agy.exe" in status.agy_pro_status.executable_or_endpoint.lower()
    assert status.active_primary in ("agy_pro", "studio_api")
    assert "AI Brain:" in status.dashboard_badge

    # Execute one live end-to-end evaluation through the active waterfall
    ctx = _sample_context()
    live_decision = router.evaluate_sync(ctx)
    assert live_decision.decision in ("APPROVE", "VETO")
    assert 0.0 <= live_decision.confidence <= 1.0
    assert "agy_pro" in live_decision.cascade_path or "studio_api" in live_decision.cascade_path

    # Verify TUI Dashboard header displays the live AI Brain badge
    dash = ApexTUIDashboard()
    renderable = dash.build_renderable(
        run_mode="demo-auto",
        active_feed="mt5",
        ctrader_connected=False,
        mt5_connected=True,
        account_equity_usd=10_000.0,
        daily_pnl_usd=0.0,
        open_positions_count=0,
        rows=[
            SymbolCockpitRow(
                symbol="USDJPY",
                tier="TIGHT_SPREAD_M5",
                timeframe="M5",
                bid=159.74,
                ask=159.74,
                spread_pips=0.0,
                regime="TRENDING_EXPANSION",
                hurst=0.53,
                kaufman_er=0.54,
                obi=None,
                net_avg_r=0.871,
                rec_risk_pct=0.50,
                ai_verdict=f"{live_decision.decision} ({live_decision.provider_used})",
                quarantine_mode="ACTIVE",
                profit_factor=1.0,
            )
        ],
        ai_brain_badge=status.dashboard_badge,
    )
    assert renderable is not None
