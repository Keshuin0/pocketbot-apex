"""Phase 9/10 Tests: WebSocket Chrome App Cockpit, 100x Historical Replay, and Live Drill Guards."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import aiohttp
import numpy as np
import pytest

from apex.cockpit.replay_engine import HistoricalReplayEngine
from apex.cockpit.web_cockpit import WebCockpitServer, launch_chrome_app
from apex.feeds.models import Candle
from apex.ledger.walkforward_db import SymbolExecutionMode, WalkForwardLedgerDB
from tools.live_lifecycle_drill import assert_demo_account


def _http_get_json(url: str) -> dict:
    with urlopen(url, timeout=5.0) as resp:
        assert resp.status == 200
        return json.loads(resp.read().decode("utf-8"))


def _http_post_json(url: str, headers: dict[str, str] | None = None) -> tuple[int, dict]:
    req = Request(url, data=b"{}", headers=headers or {}, method="POST")
    try:
        with urlopen(req, timeout=5.0) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def test_web_cockpit_http_ws_csrf_and_port_fallback() -> None:
    srv1 = WebCockpitServer(port=8791, max_port=8798)
    srv2 = WebCockpitServer(port=8791, max_port=8798)
    try:
        url1 = srv1.start()
        url2 = srv2.start()
        assert srv1.port == 8791
        assert srv2.port == 8792
        assert url1.startswith("http://127.0.0.1:")
        assert url2.startswith("http://127.0.0.1:")

        with urlopen(f"{url1}/", timeout=5.0) as resp:
            assert resp.status == 200
            html = resp.read().decode("utf-8")
            assert "PocketBot Apex" in html

        state = _http_get_json(f"{url1}/api/state")
        csrf = state["csrf_token"]
        assert isinstance(csrf, str) and len(csrf) >= 16

        # Missing or bad CSRF must be rejected with 403
        code_no_csrf, _ = _http_post_json(f"{url1}/api/control/pause")
        assert code_no_csrf == 403
        code_bad_csrf, _ = _http_post_json(
            f"{url1}/api/control/pause",
            headers={"X-Apex-CSRF": "wrong-token"},
        )
        assert code_bad_csrf == 403

        # Valid CSRF toggles pause state
        code_ok, body_ok = _http_post_json(
            f"{url1}/api/control/pause",
            headers={"X-Apex-CSRF": csrf},
        )
        assert code_ok == 200
        assert body_ok["ok"] is True
        assert body_ok["paused"] is True

        # WebSocket receives live push
        async def _check_ws() -> None:
            ws_url = url1.replace("http://", "ws://") + "/ws"
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(ws_url, timeout=5.0) as ws:
                    first = await ws.receive_json(timeout=5.0)
                    assert first["paused"] is True
                    srv1.push_state({"account_equity": 10250.55, "verified_ai_model": "gemini-4-hydrogen-medium"})
                    pushed = await ws.receive_json(timeout=5.0)
                    assert pushed["account_equity"] == 10250.55
                    assert pushed["verified_ai_model"] == "gemini-4-hydrogen-medium"

        asyncio.run(_check_ws())
    finally:
        srv2.stop()
        srv1.stop()


def test_launch_chrome_app_skips_when_headless(monkeypatch: pytest.MonkeyPatch) -> None:
    assert launch_chrome_app("http://127.0.0.1:8765", no_gui=True) is None
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    assert launch_chrome_app("http://127.0.0.1:8765", no_gui=False) is None


def test_replay_engine_100x_isolated_from_live_ledger(tmp_path: Path) -> None:
    db_path = tmp_path / "apex_ledger.db"
    ledger = WalkForwardLedgerDB(db_path=db_path)

    # Generate 120 realistic trending candles for EURUSD
    candles = []
    px = 1.0800
    for i in range(120):
        px += 0.00025 * (1.0 if (i % 7) != 0 else -0.4)
        candles.append(
            Candle(
                symbol="EURUSD",
                timeframe="M5",
                timestamp_utc=1_700_000_000.0 + i * 300.0,
                open=round(px - 0.0001, 5),
                high=round(px + 0.0003, 5),
                low=round(px - 0.0003, 5),
                close=round(px, 5),
                volume=100.0 + i,
                source="synthetic",
            )
        )

    updates: list[dict] = []
    engine = HistoricalReplayEngine(starting_equity=10000.0, step_delay_seconds=0.0)
    summary = engine.run_replay(
        candles_by_symbol={"EURUSD": candles},
        symbols=("EURUSD",),
        on_update=updates.append,
    )
    assert summary.bars_processed > 0
    assert len(updates) >= 2
    assert updates[-1]["status"] == "COMPLETED"
    assert isinstance(summary.equity_curve, list)

    # Confirm zero pollution of live trade_ledger
    with sqlite3.connect(str(db_path)) as conn:
        count = conn.execute("SELECT COUNT(*) FROM trade_ledger").fetchone()[0]
    assert count == 0


def test_ledger_is_drill_isolation(tmp_path: Path) -> None:
    db_path = tmp_path / "drill_isolation.db"
    ledger = WalkForwardLedgerDB(db_path=db_path)

    for i in range(20):
        ledger.log_trade_outcome(
            order_id=f"DRILL-{i}",
            symbol="EURUSD",
            side="BUY",
            execution_mode=SymbolExecutionMode.ACTIVE,
            regime="DRILL",
            dom_obi=0.1,
            predicted_net_r=0.4,
            ai_decision="APPROVE",
            ai_confidence=1.0,
            ai_reason="drill",
            entry_price=1.08,
            executed_price=1.0802,
            stop_loss=1.0780,
            take_profit=1.0840,
            realized_slippage_r=0.25,
            realized_net_r=-1.0,
            is_drill=True,
        )

    # Drill losses must NOT quarantine EURUSD or pollute slippage/session priors
    q_state = ledger.evaluate_quarantine("EURUSD")
    assert q_state.mode == SymbolExecutionMode.ACTIVE
    assert q_state.window_trades == 0
    assert ledger.compute_slippage_calibration_multipliers() == {}
    assert ledger.compute_session_hour_priors() == {}


def test_live_drill_demo_only_guard() -> None:
    with pytest.raises(RuntimeError, match="DEMO-only"):
        assert_demo_account(SimpleNamespace(trade_mode=2, login=999999, currency="CAD"))
    login, currency = assert_demo_account(SimpleNamespace(trade_mode=0, login=568159, currency="CAD"))
    assert login == 568159
    assert currency == "CAD"


def test_tui_no_wrap_v241_and_web_banner_no_scroll_spam() -> None:
    from io import StringIO
    from rich.console import Console
    from apex.cockpit.tui_dashboard import ApexTUIDashboard
    from apex.cockpit.web_cockpit import find_browser_executable
    from apex_runner import format_web_cockpit_banner, format_single_line_heartbeat

    buf = StringIO()
    dash = ApexTUIDashboard(console=Console(file=buf, width=145, force_terminal=False))
    group = dash.build_renderable(
        run_mode="demo-auto",
        active_feed="MT5",
        ctrader_connected=False,
        mt5_connected=True,
        account_equity_usd=10000.0,
        daily_pnl_usd=0.0,
        open_positions_count=0,
        rows=[],
    )
    dash.console.print(group)
    rendered = buf.getvalue()
    assert "POCKETBOT APEX v2.4.1" in rendered
    for renderable in group.renderables:
        if hasattr(renderable, "columns"):
            assert all(col.no_wrap for col in renderable.columns)

    banner = format_web_cockpit_banner("http://127.0.0.1:8765", "demo-auto")
    assert "POCKETBOT APEX v2.6.0" in banner
    assert "http://127.0.0.1:8765" in banner
    assert len(banner.strip().splitlines()) <= 6

    fake_orch = SimpleNamespace(
        execution_router=SimpleNamespace(account_equity_usd=9999.64),
        last_scan_summary=SimpleNamespace(open_scanned=18, total_scanned=251),
        paused=False,
    )
    hb = format_single_line_heartbeat(fake_orch)  # type: ignore[arg-type]
    assert "\n" not in hb
    assert len(hb) < 100
    assert "APEX v2.6.0" in hb

    # Chrome or Edge should be discoverable on Windows
    exe = find_browser_executable()
    assert exe is None or exe.lower().endswith(".exe")
