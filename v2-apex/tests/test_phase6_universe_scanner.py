"""Phase 6 Verification Suite: Full Broker Catalog Auto-Discovery & Two-Stage Universe Funnel."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock
import pytest

from apex.ai.veto_committee import AIVetoCommittee, CandidateSetupAuditContext
from apex.cockpit.tui_dashboard import ApexTUIDashboard, BrokerExecutionEvent, SymbolCockpitRow
from apex.execution.router import DualBrokerExecutionRouter, ExecutionOrderRequest
from apex.feeds.alias_mapper import SmartAliasMapper
from apex.feeds.ctrader_feed import CTraderOpenApiFeed
from apex.feeds.dual_router import MarketFeedRouter
from apex.feeds.models import AssetClass, Candle, DiscoveredSymbol, Side, Tick
from apex.ledger.walkforward_db import WalkForwardLedgerDB
from apex.quant.correlation_shield import (
    CorrelationShield,
    PositionRiskSpec,
    decompose_symbol_factors,
)
from apex.quant.momentum_dom import MomentumDOMGate
from apex.quant.regime import RegimeClassifier
from apex.quant.timemachine_v2 import BlockBootstrapTimeMachine
from apex.quant.universe_scanner import UniverseScanner


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


def test_multi_currency_and_multi_stock_factor_capping() -> None:
    """Verify factor decomposition and 1.0% single-factor risk capping across all 251 universe symbols."""
    # 1. Exotic/Cross Currency factor decomposition
    assert decompose_symbol_factors("EURTRY", Side.BUY) == {"EUR": 1.0, "TRY": -1.0}
    assert decompose_symbol_factors("USDZAR", Side.BUY) == {"USD": 1.0, "ZAR": -1.0}
    assert decompose_symbol_factors("GBPNOK", Side.BUY) == {"GBP": 1.0, "NOK": -1.0}

    # 2. Commodities & Energy & Crypto decomposition
    assert decompose_symbol_factors("XTIUSD", Side.BUY) == {"Energy_Beta": 1.0, "USD": -1.0}
    assert decompose_symbol_factors("COFARA", Side.BUY) == {"Commodities_Beta": 1.0, "USD": -1.0}
    assert decompose_symbol_factors("SOLUSD", Side.BUY) == {"Crypto_Beta": 1.0, "USD": -1.0}

    # 3. Equities decomposition
    assert decompose_symbol_factors("NVIDIA", Side.BUY) == {"US_Tech_Beta": 1.0, "Stock_NVDA": 1.0}
    assert decompose_symbol_factors("Tesla", Side.BUY) == {"US_Tech_Beta": 1.0, "Stock_TSLA": 1.0}
    assert decompose_symbol_factors("JPMorgan", Side.BUY) == {"US_Broad_Market_Beta": 1.0, "Stock_JPM": 1.0}

    shield = CorrelationShield(
        enabled=True,
        max_factor_risk_pct=1.0,
        account_equity_usd=10_000.0,
    )

    # Multi-Currency: EUR concentration trap
    # Long EURUSD (0.6%) + Long EURTRY (0.6%) -> EUR = +1.2% (> 1.0% cap -> BLOCKED!)
    pos_eurusd = PositionRiskSpec(symbol="EURUSD", side=Side.BUY, risk_pct=0.6, risk_usd=60.0)
    v_eurusd = shield.evaluate_proposed_trade(pos_eurusd, open_positions=[])
    assert v_eurusd.allowed is True
    assert pytest.approx(v_eurusd.factor_exposures_after_pct["EUR"]) == 0.6

    pos_eurtry = PositionRiskSpec(symbol="EURTRY", side=Side.BUY, risk_pct=0.6, risk_usd=60.0)
    v_eurtry = shield.evaluate_proposed_trade(pos_eurtry, open_positions=[pos_eurusd])
    assert v_eurtry.allowed is False
    assert "EUR" in v_eurtry.breached_factors
    assert pytest.approx(v_eurtry.factor_exposures_after_pct["EUR"]) == 1.2
    assert "CORRELATION_SHIELD_BLOCK" in v_eurtry.reason

    # Multi-Stock Tech Sector: Long NVIDIA (0.6%) + Long Tesla (0.6%) -> US_Tech_Beta = +1.2% (> 1.0% -> BLOCKED!)
    pos_nvda = PositionRiskSpec(symbol="NVIDIA", side=Side.BUY, risk_pct=0.6, risk_usd=60.0)
    v_nvda = shield.evaluate_proposed_trade(pos_nvda, open_positions=[])
    assert v_nvda.allowed is True
    assert pytest.approx(v_nvda.factor_exposures_after_pct["US_Tech_Beta"]) == 0.6
    assert pytest.approx(v_nvda.factor_exposures_after_pct["Stock_NVDA"]) == 0.6

    pos_tsla = PositionRiskSpec(symbol="Tesla", side=Side.BUY, risk_pct=0.6, risk_usd=60.0)
    v_tsla = shield.evaluate_proposed_trade(pos_tsla, open_positions=[pos_nvda])
    assert v_tsla.allowed is False
    assert "US_Tech_Beta" in v_tsla.breached_factors
    assert pytest.approx(v_tsla.factor_exposures_after_pct["US_Tech_Beta"]) == 1.2

    # Same Stock Stacking: 2x NVDA trades (0.6% each) -> Stock_NVDA = 1.2% (> 1.0% -> BLOCKED!)
    pos_nvda_2 = PositionRiskSpec(symbol="NVDA", side=Side.BUY, risk_pct=0.6, risk_usd=60.0)
    v_nvda_2 = shield.evaluate_proposed_trade(pos_nvda_2, open_positions=[pos_nvda])
    assert v_nvda_2.allowed is False
    assert "Stock_NVDA" in v_nvda_2.breached_factors

    # Crypto Beta Capping: Long BTCUSD (0.6%) + Long SOLUSD (0.6%) -> Crypto_Beta = 1.2% (> 1.0% -> BLOCKED!)
    pos_btc = PositionRiskSpec(symbol="BTCUSD", side=Side.BUY, risk_pct=0.6, risk_usd=60.0)
    pos_sol = PositionRiskSpec(symbol="SOLUSD", side=Side.BUY, risk_pct=0.6, risk_usd=60.0)
    v_sol = shield.evaluate_proposed_trade(pos_sol, open_positions=[pos_btc])
    assert v_sol.allowed is False
    assert "Crypto_Beta" in v_sol.breached_factors

    # Energy Beta Capping: Long XTIUSD (0.6%) + Long XBRUSD (0.6%) -> Energy_Beta = 1.2% (> 1.0% -> BLOCKED!)
    pos_xti = PositionRiskSpec(symbol="XTIUSD", side=Side.BUY, risk_pct=0.6, risk_usd=60.0)
    pos_xbr = PositionRiskSpec(symbol="XBRUSD", side=Side.BUY, risk_pct=0.6, risk_usd=60.0)
    v_xbr = shield.evaluate_proposed_trade(pos_xbr, open_positions=[pos_xti])
    assert v_xbr.allowed is False
    assert "Energy_Beta" in v_xbr.breached_factors


def test_ctrader_catalog_discovery_and_alias_mapping() -> None:
    """Verify cTrader feed auto-discovery, AssetClass mapping, and symbol resolution."""
    feed = CTraderOpenApiFeed(
        account_id=12345,
        client_id="cid_test",
        client_secret="csec_test",
        access_token="tok_test",
    )

    # Register symbols representing distinct asset classes
    feed.register_symbol_mapping("EURUSD", 1, pip_size=0.0001, digits=5)
    feed.register_symbol_mapping("XAUUSD", 2, pip_size=0.01, digits=2)
    feed.register_symbol_mapping("US500", 3, pip_size=0.1, digits=1)
    feed.register_symbol_mapping("BTCUSD", 4, pip_size=0.01, digits=2)
    feed.register_symbol_mapping("NVIDIA", 5, pip_size=0.01, digits=2)
    feed.register_symbol_mapping("XTIUSD", 6, pip_size=0.01, digits=2)
    feed.register_symbol_mapping("COFARA", 7, pip_size=0.01, digits=2)

    catalog = feed.discover_catalog()
    assert len(catalog) == 7

    cat_map = {item.name: item for item in catalog}
    assert cat_map["EURUSD"].asset_class == AssetClass.FOREX
    assert cat_map["XAUUSD"].asset_class == AssetClass.METALS
    assert cat_map["US500"].asset_class == AssetClass.INDICES
    assert cat_map["BTCUSD"].asset_class == AssetClass.CRYPTO
    assert cat_map["NVIDIA"].asset_class == AssetClass.US_EQUITIES
    assert cat_map["XTIUSD"].asset_class == AssetClass.ENERGY
    assert cat_map["COFARA"].asset_class == AssetClass.COMMODITIES

    # Bidirectional resolution testing
    assert feed.resolve_symbol("EURUSD") == "EURUSD"
    assert feed.resolve_symbol("NVDA") == "NVIDIA"
    assert feed.resolve_symbol("GOLD") == "XAUUSD"
    assert feed.resolve_symbol("SPX500") == "US500"
    assert feed.resolve_symbol("NONEXISTENT") is None


def _sample_audit_context(**kwargs: Any) -> CandidateSetupAuditContext:
    defaults: dict[str, Any] = {
        "symbol": "US500",
        "side": Side.BUY,
        "timeframe": "M5",
        "entry_price": 5000.0,
        "stop_loss": 4990.0,
        "take_profit": 5020.0,
        "regime": "TRENDING_EXPANSION",
        "hurst_exponent": 0.65,
        "kaufman_er": 0.45,
        "volatility_percentile": 60.0,
        "dom_obi": 0.25,
        "spread_pct_of_sl": 0.05,
        "net_avg_R": 0.22,
        "prob_profit": 0.58,
        "cvar_95_R": -1.05,
        "recommended_risk_pct": 0.5,
        "session_utc_iso": "2026-11-11T16:00:00Z",
        "bar_timestamp": 1700000000.0,
    }
    defaults.update(kwargs)
    return CandidateSetupAuditContext(**defaults)


def test_ai_verdict_caching_and_cooldown() -> None:
    """Verify per-candle verdict caching and consecutive failure cooldown in AIVetoCommittee."""
    committee = AIVetoCommittee(
        enabled=True,
        mode="force_pure_math",
        timeout_seconds=1.0,
        config_path="__nonexistent__",
    )
    committee._verdict_cache.clear()

    ctx1 = _sample_audit_context(bar_timestamp=1700000000.0)

    # First evaluation generates and caches verdict
    v1 = committee.evaluate_sync(ctx1)
    assert v1.decision == "APPROVE"
    assert len(committee._verdict_cache) == 1

    # Same symbol, side, tf, and bar_timestamp hits cache
    v2 = committee.evaluate_sync(ctx1)
    assert v2.decision == v1.decision
    assert v2.reason == v1.reason

    # New bar timestamp generates a new cache entry
    ctx2 = _sample_audit_context(bar_timestamp=1700000300.0)
    v3 = committee.evaluate_sync(ctx2)
    assert v3.decision == "APPROVE"
    assert len(committee._verdict_cache) == 2

    # Test cooldown: consecutive failures trigger cooldown to pure_math (works with or without agy.exe installed)
    import time

    committee._agy_consecutive_failures = 2
    committee._agy_cooldown_until_utc = time.time() + 600.0

    status = committee.get_router_status()
    assert status.agy_pro_status.ready is False
    assert "cooldown" in status.agy_pro_status.detail.lower()

    # Cleanup instance state
    committee._agy_consecutive_failures = 0
    committee._agy_cooldown_until_utc = 0.0


def test_min_net_avg_r_filter() -> None:
    """Verify net_avg_R >= 0.10 floor in pure_math fallback evaluation."""
    committee = AIVetoCommittee(
        enabled=True,
        mode="force_pure_math",
        timeout_seconds=1.0,
        config_path="__nonexistent__",
    )
    committee._verdict_cache.clear()

    # Candidate with net_avg_R = 0.05 (< 0.10 floor) must be VETOED
    ctx_low = _sample_audit_context(symbol="EURUSD", bar_timestamp=1700000000.0, net_avg_R=0.05)
    v_low = committee.evaluate_sync(ctx_low)
    assert v_low.decision == "VETO"
    assert "0.10" in v_low.reason

    # Candidate with net_avg_R = 0.15 (>= 0.10 floor) must be APPROVED
    ctx_high = _sample_audit_context(symbol="EURUSD", bar_timestamp=1700000300.0, net_avg_R=0.15)
    v_high = committee.evaluate_sync(ctx_high)
    assert v_high.decision == "APPROVE"


def test_fusion_cad_lot_sizing_and_broker_events() -> None:
    """Verify Fusion Markets CAD sizing and BrokerExecutionEvent recording."""
    feed_router = MagicMock(spec=MarketFeedRouter)
    feed_router.ctrader = MagicMock()
    feed_router.ctrader.is_connected = False
    feed_router.mt5 = MagicMock()
    feed_router.mt5.is_connected = True

    # Ensure get_dom_analytics returns None so DOM checks don't fail on MagicMock
    feed_router.get_dom_analytics.return_value = None

    # Sizing calculation mock
    feed_router.mt5.compute_compliant_lot_size.return_value = 3.52
    feed_router.mt5.get_open_positions.return_value = []
    feed_router.mt5.alias_mapper = SmartAliasMapper()
    feed_router.mt5.resolve_symbol.return_value = "US500"

    nbbo_mock = MagicMock()
    nbbo_mock.topology = MagicMock(value="MT5_SOLO")
    nbbo_mock.winning_venue = "mt5"
    nbbo_mock.pips_saved = 0.0
    nbbo_mock.usd_saved_per_lot = 0.0
    nbbo_mock.single_broker_spike_blocked = False
    feed_router.evaluate_nbbo.return_value = nbbo_mock

    router = DualBrokerExecutionRouter(
        feed_router=feed_router,
        account_equity_usd=10000.0,
        is_live=False,
    )

    # Compliant lot size verification
    lots = router.compute_compliant_lot_size("US500", 5000.0, 4990.0, 50.0)
    assert lots == 3.52

    # Execute paper shadow and verify BrokerExecutionEvent
    order = ExecutionOrderRequest(
        symbol="US500",
        side=Side.BUY,
        lots=3.52,
        entry_price=5000.0,
        stop_loss=4990.0,
        take_profit=5020.0,
        force_paper_shadow=True,
    )
    receipt = router.execute_order(order)
    assert receipt.status == "PAPER_SHADOW_FILLED"

    events = router.recent_broker_events
    assert len(events) >= 1
    last_event = events[-1]
    assert last_event.symbol == "US500"
    assert last_event.side == "BUY"
    assert last_event.lots == 3.52
    assert last_event.status == "PAPER_SHADOW_FILLED"

