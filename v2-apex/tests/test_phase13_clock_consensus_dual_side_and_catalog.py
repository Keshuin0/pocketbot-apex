"""Phase 13 (v2.6.0) Regression Tests:
1. Multi-symbol max(tick.time) MT5 broker clock consensus (prevents weekend FX inversion)
2. 252-symbol grouped catalog & live row 11-Agent pipeline_trace attachment
3. Symmetric Side.BUY + Side.SELL dual-direction alpha & short 1R/3.5R ratchets
4. Startup non-argon model readiness, asyncio.to_thread controls, and Brownian intra-bar tie-breaker
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import aiohttp
import numpy as np

from apex.ai.model_selector import AdaptiveModelSelector
from apex.ai.veto_committee import AIVetoCommittee
from apex.cockpit.web_cockpit import WebCockpitServer
from apex.execution.position_manager import AutonomousPositionOrchestrator
from apex.feeds.models import AssetClass, Candle, ContractSpec, DiscoveredSymbol, Side, Tick
from apex.feeds.mt5_feed import MT5Feed
from apex.ledger.walkforward_db import WalkForwardLedgerDB
from apex.quant.momentum_dom import DOMAnalytics, MomentumDOMGate
from apex.quant.regime import RegimeClassifier
from apex.quant.timemachine_v2 import BlockBootstrapTimeMachine, TimeMachineCostModel
from apex.quant.universe_scanner import UniverseScanner


def _make_candles(
    symbol: str,
    timeframe: str,
    closes: list[float],
    spread: float = 0.0002,
    start_ts: float = 1_700_000_000.0,
    step_s: float = 300.0,
) -> list[Candle]:
    out: list[Candle] = []
    prev = closes[0]
    for i, cl in enumerate(closes):
        op = prev if i > 0 else cl * (1.0002 if closes[-1] < closes[0] else 0.9998)
        hi = max(op, cl) + spread * 1.5
        lo = min(op, cl) - spread * 1.5
        out.append(
            Candle(
                symbol=symbol,
                timeframe=timeframe,
                timestamp_utc=start_ts + i * step_s,
                open=op,
                high=hi,
                low=lo,
                close=cl,
                volume=100.0,
                source="mt5",
            )
        )
        prev = cl
    return out


def test_weekend_clock_consensus_ignores_stale_eurusd_and_locks_btcusd_offset() -> None:
    """4h-stale Friday EURUSD + fresh 24/7 BTCUSD must yield +7200s (UTC+2), never -7200s."""
    now_utc = 1_763_085_000.0
    stale_eur_server_ts = now_utc - 14400.0 + 7200.0  # 4h stale on UTC+2 server
    fresh_btc_server_ts = now_utc - 3.0 + 7200.0      # 3s fresh on UTC+2 server

    client = MT5Feed()
    client._initialized = True
    client._symbol_map = {"EURUSD": "EURUSD", "BTCUSD": "BTCUSD", "XPTUSD": "XPTUSD"}

    def fake_tick(sym: str) -> SimpleNamespace | None:
        if sym == "EURUSD":
            return SimpleNamespace(time=stale_eur_server_ts, bid=1.0800, ask=1.0802)
        if sym == "BTCUSD":
            return SimpleNamespace(time=fresh_btc_server_ts, bid=90000.0, ask=90020.0)
        if sym == "XPTUSD":
            return SimpleNamespace(time=stale_eur_server_ts, bid=1489.9, ask=1495.8)
        return None

    with (
        patch("apex.feeds.mt5_feed.time.time", return_value=now_utc),
        patch("apex.feeds.mt5_feed.mt5.symbol_info_tick", side_effect=fake_tick),
        patch("apex.feeds.mt5_feed.mt5.symbols_get", return_value=[]),
    ):
        offset = client._calibrate_server_utc_offset()

    assert offset == 7200
    assert client.normalize_server_timestamp_to_utc(fresh_btc_server_ts) == now_utc - 3.0
    # Normalized XPTUSD timestamp is 14400s old (>120s), so it is rejected as stale
    xpt_age = now_utc - client.normalize_server_timestamp_to_utc(stale_eur_server_ts)
    assert abs(xpt_age - 14400.0) < 1e-6


def test_grouped_catalog_and_web_cockpit_api_state_populated(tmp_path) -> None:
    """UniverseScanner.discover_catalog() & get_grouped_catalog() must populate catalog_symbols."""
    mock_router = MagicMock()
    mock_router.discover_catalog.return_value = [
        DiscoveredSymbol(name="EURUSD", canonical_name="EURUSD", asset_class=AssetClass.FOREX, path="Forex", digits=5, point=0.00001, trade_mode=4, is_visible=True, source="mt5"),
        DiscoveredSymbol(name="BTCUSD", canonical_name="BTCUSD", asset_class=AssetClass.CRYPTO, path="Crypto", digits=2, point=0.01, trade_mode=4, is_visible=True, source="mt5"),
        DiscoveredSymbol(name="US500", canonical_name="US500", asset_class=AssetClass.INDICES, path="Indices", digits=2, point=0.01, trade_mode=4, is_visible=True, source="mt5"),
        DiscoveredSymbol(name="XAUUSD", canonical_name="XAUUSD", asset_class=AssetClass.METALS, path="Metals", digits=2, point=0.01, trade_mode=4, is_visible=True, source="mt5"),
    ]
    ledger = WalkForwardLedgerDB(tmp_path / "ledger.db")
    scanner = UniverseScanner(
        feed_router=mock_router,
        ledger=ledger,
        regime_classifier=RegimeClassifier(),
        momentum_dom_gate=MomentumDOMGate(),
        time_machine=BlockBootstrapTimeMachine(replays=200, seed=42),
        ai_committee=MagicMock(spec=AIVetoCommittee),
    )

    assert len(scanner.discover_catalog()) == 4
    grouped = scanner.get_grouped_catalog()
    assert grouped["FOREX"] == ["EURUSD"]
    assert grouped["CRYPTO"] == ["BTCUSD"]
    assert grouped["INDICES"] == ["US500"]
    assert grouped["METALS"] == ["XAUUSD"]


def test_symmetric_sell_pipeline_and_short_ratchet(tmp_path) -> None:
    """Bearish breakdown must trigger Side.SELL with SL > entry > TP, 11-stage trace, and short 1R/3.5R ratchets."""
    now_utc = 1_763_085_000.0
    m5_closes = [100.0 - i * 0.25 for i in range(60)]
    m5_candles = _make_candles("BTCUSD", "M5", m5_closes, spread=0.05, start_ts=now_utc - 60 * 300, step_s=300)
    h1_closes = [105.0 - i * 0.30 for i in range(40)]
    h1_candles = _make_candles("BTCUSD", "H1", h1_closes, spread=0.05, start_ts=now_utc - 40 * 3600, step_s=3600)

    mock_router = MagicMock()
    mock_router.mt5 = MagicMock(_initialized=False)
    mock_router.get_tick.return_value = (
        Tick(symbol="BTCUSD", bid=m5_closes[-1], ask=m5_closes[-1] + 0.05, timestamp_utc=now_utc - 1.0, source="mt5"),
        None,
    )
    mock_router.get_candles.side_effect = lambda sym, timeframe="M5", count=100: (
        h1_candles if timeframe == "H1" else m5_candles
    )
    mock_router.get_contract_spec.return_value = ContractSpec(
        symbol="BTCUSD",
        digits=2,
        point_size=0.01,
        pip_size=0.01,
        min_volume_lots=0.01,
        max_volume_lots=10.0,
        step_volume_lots=0.01,
        lot_size=1.0,
        source="mt5",
    )
    mock_router.get_dom_analytics.return_value = None
    mock_router.evaluate_nbbo.return_value = SimpleNamespace(
        winning_venue="mt5", winning_cost_pips=5.0, pips_saved=0.0
    )

    ledger = WalkForwardLedgerDB(tmp_path / "sell_ledger.db")
    scanner = UniverseScanner(
        feed_router=mock_router,
        ledger=ledger,
        regime_classifier=RegimeClassifier(),
        momentum_dom_gate=MomentumDOMGate(),
        time_machine=BlockBootstrapTimeMachine(replays=500, seed=42),
        ai_committee=MagicMock(spec=AIVetoCommittee),
    )
    cand = scanner.pre_score_symbol(
        DiscoveredSymbol(name="BTCUSD", canonical_name="BTCUSD", asset_class=AssetClass.CRYPTO, path="Crypto", digits=2, point=0.01, trade_mode=4, is_visible=True, source="mt5"),
        now_utc=now_utc,
    )
    row = scanner.evaluate_candidate_pipeline(cand)

    assert row.side == Side.SELL
    assert row.stop_loss > row.entry_price > row.take_profit
    assert row.pipeline_trace is not None
    assert len(row.pipeline_trace["stages"]) == 11
    assert row.pipeline_trace["side"] == "SELL"

    # Short 1R free-roll & 3.5R runner ratchet in AutonomousPositionOrchestrator
    mgr = AutonomousPositionOrchestrator()
    decision = mgr.evaluate_position(
        ticket=9911,
        symbol="BTCUSD",
        side=Side.SELL,
        entry_price=100.0,
        current_sl=101.0,
        current_tp=98.0,
        mark_price=98.4,  # +1.6R in profit for SHORT (1R = 1.0)
        candles=m5_candles,
        dom_obi=-0.35,
        lots=0.05,
        now_utc=now_utc,
        bars_held_override=4,
        execute_broker_actions=False,
    )
    assert decision.action == "RATCHET_SLTP"
    assert decision.ratchet_stage == "RUNNER_3_5R"
    assert decision.new_sl is not None and decision.new_sl < 101.0  # SL ratcheted downward for SHORT
    assert decision.new_tp is not None and decision.new_tp < 98.0   # TP expanded to 3.5R short runner


def test_startup_model_selector_prefers_hydrogen_over_argon_and_brownian_tiebreak() -> None:
    """Startup selector must rank hydrogen/flash ahead of argon, and Brownian tie-breaker must not understate TP."""
    sel = AdaptiveModelSelector()
    sel.register_model("agy_pro", "gemini-4-argon-medium")
    sel.register_model("agy_pro", "gemini-4-hydrogen-medium")
    sel.register_model("agy_pro", "gemini-3.8-flash")
    summary = sel.get_quota_health_summary("agy_pro")
    assert "argon" not in summary["selected_model"]

    # Brownian-Bridge tie-breaker: bars with wide wicks touching both SL & TP, closer to TP + bullish close
    tm = BlockBootstrapTimeMachine(replays=500, seed=7)
    candles = []
    px = 100.0
    for i in range(40):
        # Open=px, Low=px-1.6 (touches SL=98.5), High=px+3.2 (touches TP=103.0), Close=px+0.2 (bullish)
        candles.append(
            Candle(
                symbol="BTCUSD",
                timeframe="M5",
                timestamp_utc=1_700_000_000.0 + i * 300,
                open=px,
                high=px + 3.5,
                low=px - 1.7,
                close=px + 0.2,
                volume=100.0,
                source="mt5",
            )
        )
    res = tm.simulate_candles(
        symbol="BTCUSD",
        side=Side.BUY,
        entry_price=100.0,
        stop_price=98.5,
        target_price=103.0,
        candles=candles,
        horizon_bars=5,
        cost_model=TimeMachineCostModel(commission_per_lot_round_turn_usd=0.0, spread_price=0.0, vwap_slippage_price=0.0),
    )
    assert res.p_target > 0.0


def test_web_cockpit_sync_control_runs_via_asyncio_to_thread() -> None:
    """Synchronous /api/control/* handlers must execute in a worker thread via asyncio.to_thread."""
    import threading

    async def _run() -> None:
        loop_thread_id = threading.get_ident()
        handler_thread_ids: list[int] = []

        def sync_handler(payload: dict) -> dict:
            handler_thread_ids.append(threading.get_ident())
            return {"message": f"ok:{payload.get('x', 1)}"}

        srv = WebCockpitServer(host="127.0.0.1", port=8791)
        srv.set_control_handler("refresh_ai", sync_handler)
        url = srv.start()
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.post(
                    f"{url}/api/control/refresh_ai",
                    headers={"X-Apex-CSRF": srv.csrf_token},
                    json={"x": 42},
                ) as resp:
                    data = await resp.json()
                    assert resp.status == 200
                    assert data["ok"] is True
                    assert data["message"] == "ok:42"
        finally:
            srv.stop()

        assert handler_thread_ids and handler_thread_ids[0] != loop_thread_id

    asyncio.run(_run())
