"""Phase 8.1 / v2.3.1 hotfix verification: cross-family cascade, contagion Z, Stage A ranking, in-place TUI & veto logs."""

from __future__ import annotations

import datetime
import json
import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from apex.ai.model_selector import AdaptiveModelSelector, ModelTier, SetupStakes, classify_model_family
from apex.ai.veto_committee import AIVetoCommittee, CandidateSetupAuditContext
from apex.cockpit.tui_dashboard import ApexTUIDashboard, BrokerExecutionEvent
from apex.feeds.models import AssetClass, Candle, DiscoveredSymbol, Side, Tick
from apex.quant.contagion_oracle import ContagionOracle


def test_cross_family_quarantine_skips_argon_low_and_hops_to_hydrogen(tmp_path: Path) -> None:
    """When argon hits credit exhaustion, entire argon family is quarantined 60m and cascade hops to hydrogen."""
    selector = AdaptiveModelSelector()
    selector.register_model("agy_pro", "gemini-4-argon-medium", ModelTier.FLASH_FAST)
    selector.register_model("agy_pro", "gemini-4-argon-low", ModelTier.FLASH_LITE)
    selector.register_model("agy_pro", "gemini-4-hydrogen-medium", ModelTier.FLASH_FAST)
    selector.register_model("agy_pro", "gemini-3.8-flash-medium", ModelTier.FLASH_FAST)

    assert classify_model_family("gemini-4-argon-medium") == "argon"
    assert classify_model_family("gemini-4-argon-low") == "argon"
    assert classify_model_family("gemini-pro-orchestrator-g1-pro") == "argon"
    assert classify_model_family("gemini-4-hydrogen-medium") == "hydrogen"
    assert classify_model_family("gemini-3.8-flash-medium") == "flash"

    # Simulate credit exhaustion on gemini-4-argon-medium
    selector.record_failure(
        "agy_pro",
        "gemini-4-argon-medium",
        "Your AI credits balance is too low to continue.",
        is_quota_or_429=True,
    )

    q_flag, rem_s, _ = selector.is_family_quarantined("agy_pro", "argon")
    assert q_flag is True
    assert rem_s > 3500

    # Ranked cross-family candidates must NEVER include gemini-4-argon-low or bare 'default'
    stakes = SetupStakes(stakes_score=0.3, is_high_stakes=False, is_urgent=True)
    candidates = selector.rank_cross_family_candidates("agy_pro", stakes)
    ids = [c.model.model_id for c in candidates]
    assert "gemini-4-argon-medium" not in ids
    assert "gemini-4-argon-low" not in ids
    assert "default" not in ids
    assert "gemini-4-hydrogen-medium" in ids
    assert "gemini-3.8-flash-medium" in ids


def test_veto_committee_cross_family_cascade_live_mock(tmp_path: Path) -> None:
    """AIVetoCommittee catches credit error on argon, hops straight to hydrogen, and updates verified badge."""
    with patch.object(AIVetoCommittee, "check_agy_pro_authenticated", return_value=(True, "Mock OK")):
        committee = AIVetoCommittee(
            enabled=True,
            api_key="",
            mode="auto_pro_first",
            priority_order=("agy_pro", "pure_math"),
            env_path=tmp_path / "no.env",
            config_path=tmp_path / "no_cfg.json",
            agy_executable_override="agy.exe",
        )
    committee._cached_agy_path = "agy.exe"
    committee._cached_agy_auth_ok = True
    committee.model_selector._models.clear()
    committee.model_selector.register_model(
        "agy_pro", "gemini-4-argon-medium", ModelTier.FLASH_FAST, initial_latency_ms=400.0
    )
    committee.model_selector.register_model("agy_pro", "gemini-4-argon-low", ModelTier.FLASH_LITE)
    committee.model_selector.register_model(
        "agy_pro", "gemini-4-hydrogen-medium", ModelTier.FLASH_FAST, initial_latency_ms=800.0
    )
    committee.model_selector.register_model(
        "agy_pro", "gemini-3.8-flash-medium", ModelTier.FLASH_FAST, initial_latency_ms=1200.0
    )

    ctx = CandidateSetupAuditContext(
        symbol="EURUSD",
        side=Side.BUY,
        timeframe="M5",
        entry_price=1.1000,
        stop_loss=1.0980,
        take_profit=1.1040,
        regime="TRENDING_EXPANSION",
        hurst_exponent=0.62,
        kaufman_er=0.45,
        volatility_percentile=65.0,
        dom_obi=0.42,
        spread_pct_of_sl=0.05,
        net_avg_R=0.32,
        prob_profit=0.60,
        cvar_95_R=-0.65,
        recommended_risk_pct=0.50,
        session_utc_iso="2026-11-12T14:00:00Z",
        bar_timestamp=1794492000.0,
    )

    valid_payload = json.dumps(
        {
            "bull_case_claim": {
                "field": "net_avg_R",
                "op": ">=",
                "cited_value": 0.32,
                "limit": 0.10,
            },
            "bear_trap_claim": {
                "field": "spread_pct_of_sl",
                "op": "<=",
                "cited_value": 0.05,
                "limit": 0.15,
            },
            "evidence_claims": [
                {"field": "net_avg_R", "op": ">=", "cited_value": 0.32, "limit": 0.10},
                {"field": "hurst_exponent", "op": ">=", "cited_value": 0.62, "limit": 0.50},
            ],
            "decision": "APPROVE",
            "confidence": 0.85,
            "risk_multiplier": 1.0,
            "reason": "Positive expectancy and trending expansion regime.",
        }
    )

    called_models: list[str] = []

    def _fake_run(cmd, **kwargs):
        m_idx = cmd.index("--model") + 1 if "--model" in cmd else -1
        m_name = cmd[m_idx] if m_idx >= 0 else "default"
        called_models.append(m_name)
        if "argon" in m_name:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=1,
                stdout='{"status":"ERROR","error":{"message":"Your AI credits balance is too low to continue."}}',
                stderr="",
            )
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout=json.dumps({"status": "SUCCESS", "response": valid_payload}),
            stderr="",
        )

    with patch("apex.ai.veto_committee.subprocess.run", side_effect=_fake_run), patch.object(
        committee.model_selector, "discover_agy_models", return_value=[]
    ), patch.object(AIVetoCommittee, "check_agy_pro_authenticated", return_value=(True, "Mock OK")):
        dec = committee.evaluate_sync(ctx)
        status = committee.get_router_status()

    # Must NEVER retry argon-low or bare default after argon-medium failed!
    assert "gemini-4-argon-low" not in called_models
    assert "default" not in called_models
    assert len(called_models) == 2
    assert dec.decision == "APPROVE"
    assert dec.model_used != "gemini-4-argon-medium"
    assert dec.aegis_verified is True

    # Dashboard badge must reflect the actual verified working model
    assert status.model_quota_summary["selected_model"] == dec.model_used


def test_contagion_oracle_ignores_stale_maintenance_bars_and_quiet_rollover() -> None:
    """Stale bars (>900s) and quiet rollover bars must not freeze Contagion Shock Z at 3.76."""
    oracle = ContagionOracle()
    now_dt = datetime.datetime(2026, 11, 12, 23, 30, tzinfo=datetime.timezone.utc)
    stale_ts = now_dt.timestamp() - 3600.0  # 1 hour old (maintenance gap)

    stale_candles = [
        Candle(
            symbol="US500",
            timeframe="M5",
            timestamp_utc=stale_ts - (39 - i) * 300,
            open=5000.0,
            high=5001.0,
            low=4999.0,
            close=5000.0 + (20.0 if i == 39 else 0.01 * i),
            volume=100.0,
            source="mt5",
        )
        for i in range(40)
    ]

    mock_router = MagicMock()
    mock_router.get_candles.return_value = stale_candles

    tel = oracle.evaluate(feed_router=mock_router, utc_dt=now_dt)
    assert tel.cross_asset_shock_z == 0.0


def test_inplace_rich_live_dashboard_and_veto_event_rendering() -> None:
    """Verify ApexTUIDashboard updates rich.live.Live in-place and renders VETO events."""
    dashboard = ApexTUIDashboard()
    mock_live = MagicMock()
    veto_ev = BrokerExecutionEvent(
        timestamp_str="20:55:00",
        symbol="EURUSD",
        side="BUY",
        lots=0.0,
        price=1.1000,
        stop_loss=1.0980,
        take_profit=1.1040,
        venue="AI/AGY_PRO",
        status="VETO",
        details="model=gemini-4-hydrogen-medium | Z3=PASS | Spread too high",
    )
    dashboard.print_snapshot(
        run_mode="demo-auto",
        active_feed="MT5",
        ctrader_connected=False,
        mt5_connected=True,
        account_equity_usd=10000.0,
        daily_pnl_usd=0.0,
        open_positions_count=0,
        rows=[],
        broker_events=[veto_ev],
        live=mock_live,
    )
    mock_live.update.assert_called_once()
