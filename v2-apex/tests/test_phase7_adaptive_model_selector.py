"""Phase 7 Verification Suite: Adaptive Multi-Factor Gemini Model & Quota Selector."""

from __future__ import annotations

import json
import time
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

from apex.ai.model_selector import (
    AdaptiveModelSelector,
    ModelTier,
    SetupStakes,
    classify_model_tier,
    compute_setup_stakes,
)
from apex.ai.veto_committee import AIVetoCommittee, CandidateSetupAuditContext
from apex.feeds.models import Side
from tools.configure_ai_mode import inspect_discovered_models


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
        "recommended_risk_pct": 0.5,
        "session_utc_iso": "2026-11-12T18:00:00Z",
        "bar_timestamp": 1700000000.0,
    }
    defaults.update(overrides)
    return CandidateSetupAuditContext(**defaults)  # type: ignore[arg-type]


def test_dynamic_model_discovery_and_tier_classification() -> None:
    """Verify dynamic /v1beta/models and agy models parsing & tier classification."""
    assert classify_model_tier("gemini-2.5-pro") == ModelTier.PRO_REASONING
    assert classify_model_tier("gemini-4-hydrogen-medium") == ModelTier.PRO_REASONING
    assert classify_model_tier("gemini-2.5-flash") == ModelTier.FLASH_FAST
    assert classify_model_tier("gemini-2.5-flash-lite") == ModelTier.FLASH_LITE
    assert classify_model_tier("gemini-3.8-flash-low") == ModelTier.FLASH_LITE

    selector = AdaptiveModelSelector()

    fake_studio_payload = {
        "models": [
            {
                "name": "models/gemini-2.5-pro",
                "displayName": "Gemini 2.5 Pro",
                "supportedGenerationMethods": ["generateContent"],
            },
            {
                "name": "models/gemini-2.5-flash",
                "displayName": "Gemini 2.5 Flash",
                "supportedGenerationMethods": ["generateContent"],
            },
            {
                "name": "models/gemini-2.5-flash-lite",
                "displayName": "Gemini 2.5 Flash-Lite",
                "supportedGenerationMethods": ["generateContent"],
            },
            {
                "name": "models/text-embedding-004",
                "displayName": "Embedding",
                "supportedGenerationMethods": ["embedContent"],
            },
        ]
    }
    fake_resp = MagicMock()
    fake_resp.read.return_value = json.dumps(fake_studio_payload).encode("utf-8")
    fake_resp.__enter__.return_value = fake_resp

    with patch("urllib.request.urlopen", return_value=fake_resp):
        studio_models = selector.discover_studio_models("AIzaFakeKeyForTest", force=True)

    studio_ids = {m.model_id for m in studio_models}
    assert studio_ids == {"gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"}

    # Mock `agy models` stdout
    fake_proc = MagicMock(
        returncode=0,
        stdout=(
            "gemini-4-hydrogen-medium\tGemini 4 Hydrogen (Medium)\n"
            "gemini-4-hydrogen-xhigh\tGemini 4 Hydrogen (XHigh)\n"
            "gemini-3.8-flash-low\tGemini 3.8 Flash (Low)\n"
            "claude-sonnet-5-5-medium\tClaude Sonnet 5.5 (Medium)\n"
        ),
    )
    with patch("subprocess.run", return_value=fake_proc):
        agy_models = selector.discover_agy_models("agy.exe", probe_cli=False, force=True)

    agy_ids = {m.model_id for m in agy_models}
    assert "gemini-4-hydrogen-medium" in agy_ids
    assert "gemini-3.8-flash-low" in agy_ids
    assert "gemini-4-hydrogen-xhigh" not in agy_ids
    assert "claude-sonnet-5-5-medium" not in agy_ids


def test_stakes_and_latency_scoring_preferences() -> None:
    """Verify high-stakes setups prefer Pro (when fast enough) while urgent M5 penalizes >6s models."""
    selector = AdaptiveModelSelector()
    now_utc = time.time()
    selector.register_model("studio_api", "gemini-2.5-pro", ModelTier.PRO_REASONING, initial_latency_ms=3200.0)
    selector.register_model("studio_api", "gemini-2.5-flash", ModelTier.FLASH_FAST, initial_latency_ms=900.0)

    # High-stakes non-urgent M15 setup (high vol + borderline EV + high risk) -> PRO_REASONING wins
    high_stakes = compute_setup_stakes(
        volatility_percentile=88.0,
        net_avg_R=0.16,
        recommended_risk_pct=0.80,
        timeframe="M15",
    )
    assert high_stakes.is_high_stakes is True
    assert high_stakes.is_urgent is False
    ranked_high = selector.rank_models("studio_api", high_stakes, now_utc=now_utc)
    assert ranked_high[0].model.model_id == "gemini-2.5-pro"

    # Now simulate PRO latency degradation (>6s) on an urgent M5 breakout -> FLASH_FAST wins!
    selector.record_success("studio_api", "gemini-2.5-pro", latency_ms=14000.0, now_utc=now_utc)
    selector.record_success("studio_api", "gemini-2.5-pro", latency_ms=14000.0, now_utc=now_utc)
    urgent_m5 = compute_setup_stakes(
        volatility_percentile=50.0,
        net_avg_R=0.40,
        recommended_risk_pct=0.50,
        timeframe="M5",
    )
    assert urgent_m5.is_urgent is True
    ranked_m5 = selector.rank_models("studio_api", urgent_m5, now_utc=now_utc)
    assert ranked_m5[0].model.model_id == "gemini-2.5-flash"


def test_quota_depletion_shifting_429_cooldown_and_pure_math_cascade(tmp_path: Path) -> None:
    """Verify Pro conservation, 429 exponential cooldown, and clean cascade to pure_math."""
    selector = AdaptiveModelSelector(soft_pro_threshold_pct=0.70)
    now_utc = time.time()
    pro = selector.register_model("studio_api", "gemini-2.5-pro", ModelTier.PRO_REASONING, rpd_limit=100, initial_latency_ms=2500.0)
    flash = selector.register_model("studio_api", "gemini-2.5-flash", ModelTier.FLASH_FAST, rpd_limit=1500, initial_latency_ms=1000.0)

    # Consume 75% of Pro's daily budget -> conservation penalty shifts even high-stakes to Flash
    pro.daily_calls = 75
    high_stakes = SetupStakes(stakes_score=0.85, is_high_stakes=True, is_urgent=False)
    ranked = selector.rank_models("studio_api", high_stakes, now_utc=now_utc)
    assert ranked[0].model.model_id == "gemini-2.5-flash"

    # Simulate HTTP 429 on Flash -> quarantined with exponential cooldown
    selector.record_failure("studio_api", "gemini-2.5-flash", "HTTP 429 RESOURCE_EXHAUSTED", is_quota_or_429=True, now_utc=now_utc)
    assert flash.cooldown_until_utc > now_utc
    ranked_after_429 = selector.rank_models("studio_api", high_stakes, now_utc=now_utc)
    assert all(r.model.model_id != "gemini-2.5-flash" for r in ranked_after_429)

    # Now quarantine Pro too -> summary reports COOLDOWN and committee cascades to pure_math
    selector.record_failure("studio_api", "gemini-2.5-pro", "HTTP 429 quota exceeded", is_quota_or_429=True, now_utc=now_utc)
    qh = selector.get_quota_health_summary("studio_api", now_utc=now_utc)
    assert qh["health"] == "COOLDOWN"

    cfg_file = tmp_path / "apex_config.json"
    cfg_file.write_text(
        json.dumps(
            {
                "filters": {
                    "ai_veto": {
                        "enabled": True,
                        "mode": "auto_studio_first",
                        "priority_order": ["studio_api", "pure_math"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    committee = AIVetoCommittee(
        api_key="AIzaFakeKey",
        config_path=cfg_file,
        env_path=tmp_path / ".env",
        agy_executable_override="__nonexistent_agy__",
    )
    with patch("urllib.request.urlopen", side_effect=RuntimeError("HTTP 429 Too Many Requests")):
        dec = committee.evaluate_sync(_make_ctx())
    assert dec.provider_used == "pure_math"
    assert dec.decision == "APPROVE"
    assert dec.bypassed is True


def test_per_candle_cache_prevents_duplicate_quota_burn(tmp_path: Path) -> None:
    """Verify auditing the same (symbol, side, tf, bar_ts) twice hits cache and burns 0 extra quota."""
    cfg_file = tmp_path / "apex_config.json"
    cfg_file.write_text(
        json.dumps({"filters": {"ai_veto": {"enabled": True, "mode": "force_pure_math"}}}),
        encoding="utf-8",
    )
    committee = AIVetoCommittee(
        config_path=cfg_file,
        env_path=tmp_path / ".env",
        agy_executable_override="__nonexistent_agy__",
    )
    committee.model_selector.register_model("agy_pro", "gemini-3.8-flash-low", ModelTier.FLASH_LITE)
    committee.model_selector.record_success("agy_pro", "gemini-3.8-flash-low", 1800.0)

    status = committee.get_router_status()
    assert "AI Brain:" in status.model_quota_badge

    ctx = _make_ctx(bar_timestamp=1700000600.0)
    d1 = committee.evaluate_sync(ctx)
    d2 = committee.evaluate_sync(ctx)
    assert d1.decision == d2.decision == "APPROVE"
    assert len(committee._verdict_cache) == 1

    # Option [6] diagnostic function runs cleanly without error
    inspect_discovered_models(committee)
