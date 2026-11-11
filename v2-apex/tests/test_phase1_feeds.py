"""Phase 1 Verification Suite for PocketBot Apex Dual-Broker Ingestion & Microstructure Layer."""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from pathlib import Path

import pytest
from ctrader_open_api.messages import (
    OpenApiMessages_pb2 as oa_msg,
    OpenApiModelMessages_pb2 as oa_model,
)

from apex.feeds.ctrader_feed import CTraderOpenApiFeed, OrderBookEngine
from apex.feeds.dual_router import CrossFeedTickGuard, MarketFeedRouter
from apex.feeds.models import (
    LiveLockViolationError,
    Side,
    Tick,
    verify_live_unlock,
)
from apex.feeds.mt5_feed import MT5Feed
from tools.setup_ctrader_oauth import (
    REDIRECT_URI,
    build_authorization_url,
    save_credentials_to_env,
    wait_for_oauth_callback,
)

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent


def test_architecture_and_mcp_configs() -> None:
    """Verify AGENTS.md, .agents/mcp_config.json ('serverUrl' invariant), and config/apex_config.json."""
    agents_md = WORKSPACE_ROOT / "AGENTS.md"
    assert agents_md.exists(), "AGENTS.md must exist in workspace root"
    content = agents_md.read_text(encoding="utf-8")
    assert ".venv\\Scripts\\python.exe" in content
    assert "--unlock-live" in content
    assert "APEX_UNLOCK_LIVE=I_ACCEPT_REAL_RISK" in content

    mcp_cfg_path = WORKSPACE_ROOT / ".agents" / "mcp_config.json"
    assert mcp_cfg_path.exists(), ".agents/mcp_config.json must exist"
    mcp_cfg = json.loads(mcp_cfg_path.read_text(encoding="utf-8"))
    assert "mcpServers" in mcp_cfg
    assert "ctrader-local" in mcp_cfg["mcpServers"]
    server_entry = mcp_cfg["mcpServers"]["ctrader-local"]
    assert "serverUrl" in server_entry, "Must use Antigravity's required 'serverUrl' key"
    assert "url" not in server_entry and "httpUrl" not in server_entry
    assert server_entry["serverUrl"] == "http://127.0.0.1:9876/mcp/"
    assert server_entry["timeoutSeconds"] == 120

    apex_cfg_path = WORKSPACE_ROOT / "config" / "apex_config.json"
    assert apex_cfg_path.exists(), "config/apex_config.json must exist"
    apex_cfg = json.loads(apex_cfg_path.read_text(encoding="utf-8"))
    required_filters = ("regime", "dom_obi", "correlation_cap", "ai_veto", "dual_feed_guard")
    for flt in required_filters:
        assert flt in apex_cfg["filters"], f"Missing filter config '{flt}'"
        assert isinstance(apex_cfg["filters"][flt].get("enabled"), bool)


def test_demo_live_safety_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify live account instantiation is hard-locked unless BOTH CLI flag AND env var are set."""
    monkeypatch.delenv("APEX_UNLOCK_LIVE", raising=False)

    # DEMO mode always allowed
    verify_live_unlock(is_live=False, cli_args=[])

    # Missing both -> raises
    with pytest.raises(LiveLockViolationError):
        verify_live_unlock(is_live=True, cli_args=[])

    # Only CLI flag -> raises
    with pytest.raises(LiveLockViolationError):
        verify_live_unlock(is_live=True, cli_args=["--unlock-live"])

    # Only env var -> raises
    monkeypatch.setenv("APEX_UNLOCK_LIVE", "I_ACCEPT_REAL_RISK")
    with pytest.raises(LiveLockViolationError):
        verify_live_unlock(is_live=True, cli_args=[])

    # Both present -> allowed
    verify_live_unlock(is_live=True, cli_args=["--unlock-live"])


def test_level2_dom_obi_and_vwap_slippage_math() -> None:
    """Deterministic unit test for Level 2 DOM Order-Book Imbalance, liquidity walls, and VWAP slippage."""
    engine = OrderBookEngine(
        symbol="EURUSD",
        pip_size=0.0001,
        digits=5,
        top_n_levels=5,
        wall_multiplier=3.0,
    )

    # Construct ProtoOADepthQuote messages (Spotware scale: price * 100,000; size * 100)
    # 5 Bid levels (descending: 1.10000 to 1.09960) and 5 Ask levels (ascending: 1.10010 to 1.10050)
    # Bid volumes (units): 200k, 150k, 100k, 100k, 650k (wall at 1.09960) -> sum = 1,200,000
    # Ask volumes (units): 100k, 100k, 100k, 100k, 200k -> sum = 600,000
    raw_bids = [
        (1, 1.10000, 200_000.0),
        (2, 1.09990, 150_000.0),
        (3, 1.09980, 100_000.0),
        (4, 1.09970, 100_000.0),
        (5, 1.09960, 650_000.0),
    ]
    raw_asks = [
        (6, 1.10010, 100_000.0),
        (7, 1.10020, 100_000.0),
        (8, 1.10030, 100_000.0),
        (9, 1.10040, 100_000.0),
        (10, 1.10050, 200_000.0),
    ]

    proto_quotes: list[oa_model.ProtoOADepthQuote] = []
    for qid, price, vol_units in raw_bids:
        q = oa_model.ProtoOADepthQuote()
        q.id = qid
        q.bid = int(round(price * 100_000))
        q.size = int(round(vol_units * 100))
        proto_quotes.append(q)

    for qid, price, vol_units in raw_asks:
        q = oa_model.ProtoOADepthQuote()
        q.id = qid
        q.ask = int(round(price * 100_000))
        q.size = int(round(vol_units * 100))
        proto_quotes.append(q)

    engine.apply_protobuf_depth_event(new_quotes=proto_quotes, deleted_quote_ids=[], timestamp_utc=1700000000.0)

    # 1. Verify OBI = (1,200,000 - 600,000) / (1,200,000 + 600,000) = +1/3 = 0.3333333333
    obi = engine.compute_obi(top_n=5)
    assert pytest.approx(obi, rel=1e-9) == (1_200_000.0 - 600_000.0) / 1_800_000.0

    # 2. Verify Liquidity Wall detection (median volume across 10 levels is 125,000; 650,000 is 5.2x median >= 3.0x)
    analytics = engine.compute_analytics(top_n=5)
    assert len(analytics.liquidity_walls) == 1
    wall = analytics.liquidity_walls[0]
    assert wall.side == Side.BUY
    assert pytest.approx(wall.price, abs=1e-6) == 1.09960
    assert pytest.approx(wall.volume_units, rel=1e-6) == 650_000.0
    assert wall.ratio_to_median >= 3.0

    # 3. Verify True VWAP Fill & Slippage for a 250,000 unit (2.5 lot) BUY order
    # Sweeps 100k @ 1.10010 + 100k @ 1.10020 + 50k @ 1.10030
    # Expected VWAP = (100k*1.10010 + 100k*1.10020 + 50k*1.10030) / 250k = 1.10018
    # Top of book = 1.10010 -> Slippage = 0.00008 = 0.8 pips across 3 levels
    buy_est = engine.compute_vwap_fill(side=Side.BUY, target_volume_units=250_000.0)
    assert buy_est.is_fully_fillable is True
    assert buy_est.levels_consumed == 3
    assert pytest.approx(buy_est.top_of_book_price, abs=1e-6) == 1.10010
    assert pytest.approx(buy_est.vwap_fill_price, abs=1e-7) == 1.10018
    assert pytest.approx(buy_est.slippage_pips, abs=1e-5) == 0.8

    # 4. Verify SELL VWAP Fill & Slippage for 300,000 units
    # Sweeps 200k @ 1.10000 + 100k @ 1.09990 -> VWAP = 1.09996667 -> Slippage = 0.3333 pips
    sell_est = engine.compute_vwap_fill(side=Side.SELL, target_volume_units=300_000.0)
    assert sell_est.is_fully_fillable is True
    assert sell_est.levels_consumed == 2
    assert pytest.approx(sell_est.vwap_fill_price, abs=1e-7) == (200_000 * 1.10000 + 100_000 * 1.09990) / 300_000.0
    assert pytest.approx(sell_est.slippage_pips, abs=1e-4) == (1.0 / 3.0)

    # 5. Verify incremental quote deletion removes quote 1 (top bid 1.10000)
    engine.apply_protobuf_depth_event(new_quotes=[], deleted_quote_ids=[1], timestamp_utc=1700000001.0)
    snap_after = engine.snapshot()
    assert pytest.approx(snap_after.bids[0].price, abs=1e-6) == 1.09990


def test_ctrader_trendbar_delta_decoding() -> None:
    """Verify Spotware ProtoOATrendbar relative delta decoding matches exact OHLC prices."""
    tb = oa_model.ProtoOATrendbar()
    tb.low = 109950  # 1.09950
    tb.deltaOpen = 25  # 1.09975
    tb.deltaHigh = 80  # 1.10030
    tb.deltaClose = 50  # 1.10000
    tb.volume = 1234
    tb.utcTimestampInMinutes = 28_000_000

    candles = CTraderOpenApiFeed.decode_trendbars("EURUSD", "M5", [tb], digits=5)
    assert len(candles) == 1
    c = candles[0]
    assert c.symbol == "EURUSD"
    assert c.timeframe == "M5"
    assert pytest.approx(c.low, abs=1e-6) == 1.09950
    assert pytest.approx(c.open, abs=1e-6) == 1.09975
    assert pytest.approx(c.high, abs=1e-6) == 1.10030
    assert pytest.approx(c.close, abs=1e-6) == 1.10000
    assert c.volume == 1234.0
    assert c.timestamp_utc == 28_000_000 * 60.0


def test_cross_feed_tick_discrepancy_guard() -> None:
    """Verify CrossFeedTickGuard passes normal ticks and flags >3z single-broker bad ticks."""
    guard = CrossFeedTickGuard(
        enabled=True,
        z_score_threshold=3.0,
        rolling_window_ticks=50,
        min_samples=10,
        max_tick_age_seconds=5.0,
    )

    base_ts = 1700000000.0
    # Feed 20 normal ticks with normal 0.1 - 0.3 pip inter-broker basis around 1.10000
    offsets = [0.00001, 0.00002, 0.000015, 0.000025, 0.000012, 0.000018, 0.000022, 0.000014, 0.000019, 0.000021] * 2
    for i, diff in enumerate(offsets):
        t_ct = Tick(symbol="EURUSD", bid=1.10000, ask=1.10010, timestamp_utc=base_ts + i, source="ctrader")
        t_mt5 = Tick(
            symbol="EURUSD",
            bid=1.10000 + diff,
            ask=1.10010 + diff,
            timestamp_utc=base_ts + i + 0.05,
            source="mt5",
        )
        alert = guard.evaluate(t_ct, t_mt5)
        assert alert.is_bad_tick is False

    # Now inject an abnormal 15-pip single-broker bad tick on MT5 (way > 3z)
    bad_mt5 = Tick(symbol="EURUSD", bid=1.10150, ask=1.10160, timestamp_utc=base_ts + 25.0, source="mt5")
    good_ct = Tick(symbol="EURUSD", bid=1.10000, ask=1.10010, timestamp_utc=base_ts + 25.0, source="ctrader")
    spike_alert = guard.evaluate(good_ct, bad_mt5)
    assert spike_alert.is_bad_tick is True
    assert spike_alert.z_score > 3.0
    assert "CROSS_FEED_DIVERGENCE" in spike_alert.reason


def test_oauth_wizard_callback_and_env_persistence(tmp_path: Path) -> None:
    """Verify tools/setup_ctrader_oauth.py URL builder, local HTTP callback capture, and .env persistence."""
    url = build_authorization_url("test_client_123", "test_secret_456")
    assert "client_id=test_client_123" in url
    assert urllib.parse.quote(REDIRECT_URI, safe=":/") in url or "127.0.0.1:8787/callback" in url

    def trigger_callback() -> None:
        time.sleep(0.15)
        with urllib.request.urlopen("http://127.0.0.1:8787/callback?code=AUTH_CODE_XYZ_999", timeout=3) as resp:
            assert resp.status == 200

    t = threading.Thread(target=trigger_callback, daemon=True)
    t.start()
    captured_code = wait_for_oauth_callback(timeout_seconds=5.0)
    assert captured_code == "AUTH_CODE_XYZ_999"

    env_file = tmp_path / ".env"
    env_file.write_text("EXISTING_KEY=keep_me\n", encoding="utf-8")
    save_credentials_to_env(
        env_path=env_file,
        client_id="cid_1",
        client_secret="csec_2",
        access_token="at_3",
        refresh_token="rt_4",
        account_id=9876543,
    )
    env_text = env_file.read_text(encoding="utf-8")
    assert "EXISTING_KEY=keep_me" in env_text
    assert "CTRADER_CLIENT_ID" in env_text and "cid_1" in env_text
    assert "CTRADER_ACCOUNT_ID" in env_text and "9876543" in env_text


def test_dual_router_and_live_mt5_probe() -> None:
    """Test MarketFeedRouter auto-fallback to MT5 and live MT5 ingestion (if MT5 terminal is open)."""
    router = MarketFeedRouter(config_path=WORKSPACE_ROOT / "config" / "apex_config.json")
    source = router.connect()
    try:
        assert source in ("ctrader", "mt5", "offline")
        if router.mt5.is_connected:
            spec = router.get_contract_spec("EURUSD")
            assert spec is not None
            assert spec.digits >= 4
            assert spec.lot_size > 0

            tick, alert = router.get_tick("EURUSD")
            assert tick is not None
            assert tick.bid > 0 and tick.ask >= tick.bid

            m5_candles = router.get_candles("EURUSD", timeframe="M5", count=20)
            m15_candles = router.get_candles("EURUSD", timeframe="M15", count=20)
            assert len(m5_candles) > 0
            assert len(m15_candles) > 0
            assert m5_candles[-1].high >= m5_candles[-1].low
    finally:
        router.shutdown()
