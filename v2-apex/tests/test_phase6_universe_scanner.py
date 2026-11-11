"""Phase 6 Verification Suite: Full Broker Catalog Auto-Discovery & Two-Stage Universe Funnel."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from apex.ai.veto_committee import AIVetoCommittee
from apex.cockpit.tui_dashboard import ApexTUIDashboard, SymbolCockpitRow
from apex.feeds.alias_mapper import SmartAliasMapper
from apex.feeds.dual_router import MarketFeedRouter
from apex.feeds.models import AssetClass, Candle, DiscoveredSymbol, Side, Tick
from apex.ledger.walkforward_db import WalkForwardLedgerDB
from apex.quant.momentum_dom import MomentumDOMGate
from apex.quant.regime import RegimeClassifier
from apex.quant.timemachine_v2 import BlockBootstrapTimeMachine
from apex.quant.universe_scanner import CandidatePreScore, UniverseScanner, UniverseScanSummary


def test_smart_alias_mapper_equities_and_indices() -> None:
    """Verify bidirectional translation between standard tickers and broker names."""
    mapper = SmartAliasMapper()

    # US Equities (Fusion Markets names)
    assert mapper.resolve_to_broker("NVDA", "mt5") == "NVIDIA"
    assert mapper.resolve_to_broker("TSLA", "mt5") == "Tesla"
    assert mapper.resolve_to_broker("AAPL", "mt5") == "Apple"
    assert mapper.resolve_to_broker("MSFT", "mt5") == "Microsoft"
    assert mapper.resolve_to_broker("GOOGL", "mt5") == "Alphabet-A"
    assert mapper.canonicalize("NVIDIA") == "NVDA"
    assert mapper.canonicalize("Tesla") == "TSLA"
    assert mapper.canonicalize("Alphabet-A") == "GOOGL"

    # Indices
    assert mapper.resolve_to_broker("SPX500", "mt5") == "US500"
    assert mapper.resolve_to_broker("SPX", "mt5") == "US500"
    assert mapper.resolve_to_broker("NQ", "mt5") == "NAS100"
    assert mapper.resolve_to_broker("USTEC", "mt5") == "NAS100"
    assert mapper.resolve_to_broker("DJ30", "mt5") == "US30"
    assert mapper.resolve_to_broker("DAX", "mt5") == "GER40"
    assert mapper.canonicalize("SPX500") == "US500"

    # Energy & Metals
    assert mapper.resolve_to_broker("USOUSD", "mt5") == "XTIUSD"
    assert mapper.resolve_to_broker("WTI", "mt5") == "XTIUSD"
    assert mapper.resolve_to_broker("BRENT", "mt5") == "XBRUSD"
    assert mapper.resolve_to_broker("GOLD", "mt5") == "XAUUSD"
    assert mapper.resolve_to_broker("SILVER", "mt5") == "XAGUSD"
    assert mapper.canonicalize("GOLD") == "XAUUSD"


def test_asset_class_classification() -> None:
    """Verify accurate AssetClass classification for all major asset segments."""
    mapper = SmartAliasMapper()

    # Forex
    assert mapper.classify_asset_class("EURUSD") == AssetClass.FOREX
    assert mapper.classify_asset_class("AUDHUF", path="Forex Exotics") == AssetClass.FOREX

    # Metals (both path-based and pattern-based)
    assert mapper.classify_asset_class("XAUUSD") == AssetClass.METALS
    assert mapper.classify_asset_class("XAGEUR", path="Commodities") == AssetClass.METALS
    assert mapper.classify_asset_class("XPTUSD") == AssetClass.METALS

    # Energy
    assert mapper.classify_asset_class("XTIUSD") == AssetClass.ENERGY
    assert mapper.classify_asset_class("XBRUSD", path="Energy") == AssetClass.ENERGY
    assert mapper.classify_asset_class("XNGUSD") == AssetClass.ENERGY

    # Crypto
    assert mapper.classify_asset_class("BTCUSD") == AssetClass.CRYPTO
    assert mapper.classify_asset_class("SOLUSD", path="Crypto") == AssetClass.CRYPTO

    # Indices
    assert mapper.classify_asset_class("US500") == AssetClass.INDICES
    assert mapper.classify_asset_class("NAS100", path="Indices") == AssetClass.INDICES
    assert mapper.classify_asset_class("GER40") == AssetClass.INDICES

    # Commodities / Soft Commodities
    assert mapper.classify_asset_class("COFARA", path="Soft Commodity") == AssetClass.COMMODITIES
    assert mapper.classify_asset_class("CORN") == AssetClass.COMMODITIES

    # US Equities
    assert mapper.classify_asset_class("NVIDIA", path="Equities\\US\\NASDAQ") == AssetClass.US_EQUITIES
    assert mapper.classify_asset_class("Tesla") == AssetClass.US_EQUITIES
    assert mapper.classify_asset_class("NVDA") == AssetClass.US_EQUITIES


def test_market_open_and_closed_detection() -> None:
    """Verify session open vs closed filters properly reject stale and invalid ticks."""
    feed_router = MagicMock(spec=MarketFeedRouter)
    scanner = UniverseScanner(
        feed_router=feed_router,
        regime_classifier=RegimeClassifier(),
        momentum_dom_gate=MomentumDOMGate(),
        time_machine=BlockBootstrapTimeMachine(replays=100),
        ai_committee=AIVetoCommittee(enabled=False),
        ledger=MagicMock(spec=WalkForwardLedgerDB),
        config={"scanner": {"max_tick_age_seconds": 60.0}},
    )

    now = 1700000000.0

    # 1. Valid live tick
    live_tick = Tick(symbol="EURUSD", bid=1.0850, ask=1.0851, timestamp_utc=now - 5.0, source="mt5")
    is_open, reason = scanner.is_symbol_market_open("EURUSD", live_tick, now_utc=now)
    assert is_open is True
    assert reason == "ACTIVE"

    # 2. None tick (market closed / symbol unavailable)
    is_open, reason = scanner.is_symbol_market_open("EURUSD", None, now_utc=now)
    assert is_open is False
    assert reason == "NO_TICK"

    # 3. Zero price tick (weekend/closed CFD)
    zero_tick = Tick(symbol="AAPL", bid=0.0, ask=0.0, timestamp_utc=now - 10.0, source="mt5")
    is_open, reason = scanner.is_symbol_market_open("AAPL", zero_tick, now_utc=now)
    assert is_open is False
    assert reason == "ZERO_PRICE"

    # 4. Stale tick (age > 60s)
    stale_tick = Tick(symbol="MSFT", bid=400.0, ask=400.1, timestamp_utc=now - 300.0, source="mt5")
    is_open, reason = scanner.is_symbol_market_open("MSFT", stale_tick, now_utc=now)
    assert is_open is False
    assert "STALE_TICK" in reason


def test_stage_a_sweep_ranking() -> None:
    """Verify Stage A ranks symbols by momentum and Kaufman ER mover scores."""
    feed_router = MagicMock(spec=MarketFeedRouter)
    now = 1700000000.0

    # Setup 3 mock symbols
    sym_a = DiscoveredSymbol(name="TREND_SYM", canonical_name="TREND_SYM", asset_class=AssetClass.FOREX, path="", digits=5, point=0.00001, trade_mode=4, is_visible=True, source="test")
    sym_b = DiscoveredSymbol(name="CHOP_SYM", canonical_name="CHOP_SYM", asset_class=AssetClass.FOREX, path="", digits=5, point=0.00001, trade_mode=4, is_visible=True, source="test")
    sym_c = DiscoveredSymbol(name="CLOSED_SYM", canonical_name="CLOSED_SYM", asset_class=AssetClass.US_EQUITIES, path="", digits=2, point=0.01, trade_mode=4, is_visible=True, source="test")

    tick_a = Tick(symbol="TREND_SYM", bid=1.1000, ask=1.1001, timestamp_utc=now - 2.0, source="test")
    tick_b = Tick(symbol="CHOP_SYM", bid=1.2000, ask=1.2001, timestamp_utc=now - 2.0, source="test")
    tick_c = Tick(symbol="CLOSED_SYM", bid=0.0, ask=0.0, timestamp_utc=now - 1000.0, source="test")

    # TREND_SYM: clean steady uptrend -> high ER
    candles_a = [
        Candle(symbol="TREND_SYM", timeframe="M5", timestamp_utc=now - (50 - i) * 300, open=1.0 + i * 0.002, high=1.0 + i * 0.002 + 0.001, low=1.0 + i * 0.002 - 0.0005, close=1.0 + i * 0.002 + 0.0008, volume=100.0, source="test")
        for i in range(50)
    ]
    # CHOP_SYM: oscillating sideways -> low ER
    candles_b = [
        Candle(symbol="CHOP_SYM", timeframe="M5", timestamp_utc=now - (50 - i) * 300, open=1.2000 + (0.001 if i % 2 == 0 else -0.001), high=1.2020, low=1.1980, close=1.2000 + (-0.001 if i % 2 == 0 else 0.001), volume=100.0, source="test")
        for i in range(50)
    ]

    def mock_get_tick(s: str) -> tuple[Tick | None, None]:
        if s == "TREND_SYM":
            return tick_a, None
        if s == "CHOP_SYM":
            return tick_b, None
        return tick_c, None

    def mock_get_candles(s: str, timeframe: str = "M5", count: int = 50) -> list[Candle]:
        if s == "TREND_SYM":
            return candles_a
        if s == "CHOP_SYM":
            return candles_b
        return []

    feed_router.get_tick.side_effect = mock_get_tick
    feed_router.get_candles.side_effect = mock_get_candles
    feed_router.get_contract_spec.return_value = None

    scanner = UniverseScanner(
        feed_router=feed_router,
        regime_classifier=RegimeClassifier(),
        momentum_dom_gate=MomentumDOMGate(),
        time_machine=BlockBootstrapTimeMachine(replays=100),
        ai_committee=AIVetoCommittee(enabled=False),
        ledger=MagicMock(spec=WalkForwardLedgerDB),
        config={"scanner": {"max_candidates_per_cycle": 2, "max_tick_age_seconds": 60.0}},
    )

    open_candidates, summary = scanner.sweep_universe(target_symbols=[sym_a, sym_b, sym_c], now_utc=now)

    assert summary.total_scanned == 3
    assert summary.open_scanned == 2
    assert summary.closed_skipped == 1
    assert len(open_candidates) == 2

    # TREND_SYM must rank higher than CHOP_SYM due to higher Kaufman ER and momentum
    assert open_candidates[0].symbol == "TREND_SYM"
    assert open_candidates[0].kaufman_er > open_candidates[1].kaufman_er
    assert open_candidates[0].composite_mover_score > open_candidates[1].composite_mover_score


def test_tui_dashboard_universe_header_formatting() -> None:
    """Verify that ApexTUIDashboard formats universe scan progress in its header panel."""
    dash = ApexTUIDashboard()
    rows = [
        SymbolCockpitRow(
            symbol="US500",
            tier="TIGHT M5",
            timeframe="M5",
            bid=7580.0,
            ask=7580.4,
            spread_pips=4.0,
            regime="TRENDING_EXPANSION",
            hurst=0.58,
            kaufman_er=0.72,
            obi=0.25,
            net_avg_r=0.45,
            rec_risk_pct=0.50,
            ai_verdict="APPROVE",
            quarantine_mode="ACTIVE",
            profit_factor=1.85,
        )
    ]

    group = dash.build_renderable(
        run_mode="DEMO-AUTO",
        active_feed="mt5",
        ctrader_connected=False,
        mt5_connected=True,
        account_equity_usd=10000.0,
        daily_pnl_usd=0.0,
        open_positions_count=0,
        rows=rows,
        universe_scanned_count=108,
        universe_total_count=251,
    )

    header_panel = group.renderables[0]
    # Check that panel contains the formatted string
    assert "Universe: 108/251 Symbols Scanned" in str(header_panel.renderable)
    assert "(143 Closed)" in str(header_panel.renderable)
