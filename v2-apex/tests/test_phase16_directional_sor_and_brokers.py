"""Phase 16 (Apex Omega v3.0 — Phase 2) Tests: Directional SOR, Broker Aliasing, Conservative Floor Lots & Dual-Venue Position Management."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from ctrader_open_api.messages import (
    OpenApiMessages_pb2 as oa_msg,
    OpenApiModelMessages_pb2 as oa_model,
)

from apex.execution.position_manager import AutonomousPositionOrchestrator
from apex.execution.router import DualBrokerExecutionRouter, ExecutionOrderRequest
from apex.feeds.alias_mapper import SmartAliasMapper
from apex.feeds.ctrader_feed import CTRADER_VOLUME_CENTS_PER_UNIT
from apex.feeds.dual_router import MarketFeedRouter
from apex.feeds.hydra_nbbo import HydraNBBOEngine, VenueFeeProfile
from apex.feeds.models import AssetClass, Candle, ContractSpec, Side, Tick


def _make_candles(symbol: str, timeframe: str, closes: list[float], spread_bar: float = 0.0004) -> list[Candle]:
    out: list[Candle] = []
    for i, c in enumerate(closes):
        prev = closes[i - 1] if i > 0 else c - 0.0001
        out.append(
            Candle(
                symbol=symbol,
                timeframe=timeframe,
                timestamp_utc=1_700_000_000.0 + i * 300.0,
                open=round(prev, 5),
                high=round(max(prev, c) + spread_bar * 0.5, 5),
                low=round(min(prev, c) - spread_bar * 0.5, 5),
                close=round(c, 5),
                volume=100.0,
                source="synthetic",
            )
        )
    return out


def test_directional_buy_vs_sell_sor_venue_selection() -> None:
    """BUY must favor tighter/lower Ask; SELL must favor higher Bid even when spreads are equal."""
    engine = HydraNBBOEngine(
        ctrader_profile=VenueFeeProfile(venue="ctrader", commission_per_lot_round_turn_usd=1.0, ping_ms=10.0),
        mt5_profile=VenueFeeProfile(venue="mt5", commission_per_lot_round_turn_usd=0.0, ping_ms=10.0),
    )

    # Equal 0.2 pip spreads, skewed quotes: cTrader has lower Ask (better BUY), MT5 has higher Bid (better SELL)
    ct_tick = Tick(symbol="EURUSD", bid=1.10000, ask=1.10002, timestamp_utc=1_700_000_100.0, source="ctrader")
    mt5_tick = Tick(symbol="EURUSD", bid=1.10004, ask=1.10006, timestamp_utc=1_700_000_100.0, source="mt5")

    buy_res = engine.evaluate_symbol(
        symbol="EURUSD",
        configured_mode="AUTO_HYDRA_MESH",
        ctrader_tick=ct_tick,
        mt5_tick=mt5_tick,
        pip_size=0.0001,
        ctrader_vwap_slippage_pips=0.05,
        mt5_vwap_slippage_pips=0.05,
        side=Side.BUY,
    )
    assert buy_res.winning_venue == "ctrader"

    sell_res = engine.evaluate_symbol(
        symbol="EURUSD",
        configured_mode="AUTO_HYDRA_MESH",
        ctrader_tick=ct_tick,
        mt5_tick=mt5_tick,
        pip_size=0.0001,
        ctrader_vwap_slippage_pips=0.05,
        mt5_vwap_slippage_pips=0.05,
        side=Side.SELL,
    )
    assert sell_res.winning_venue == "mt5"


def test_conservative_floor_lot_sizing_and_alias_normalization(tmp_path) -> None:
    """Verify math.floor prevents exceeding risk budget on MT5 & cTrader fallback, plus uppercase suffix stripping."""
    # Case-insensitive micro '_i' / 'm' suffix normalization
    assert SmartAliasMapper.strip_broker_suffixes("EURUSDm") == "EURUSD"
    assert SmartAliasMapper.strip_broker_suffixes("EURUSDM") == "EURUSD"
    assert SmartAliasMapper.strip_broker_suffixes("EURUSD_i") == "EURUSD"
    assert SmartAliasMapper.strip_broker_suffixes("EURUSD_I") == "EURUSD"
    assert SmartAliasMapper.classify_asset_class("XAUUSDm") == AssetClass.METALS
    assert SmartAliasMapper.classify_asset_class("BTCUSDC") == AssetClass.CRYPTO

    # MT5 feed conservative floor: lots_raw = 0.199 -> must snap DOWN to 0.19, NEVER round up to 0.20
    from apex.feeds.mt5_feed import MT5Feed

    feed = MT5Feed()
    feed._initialized = True
    feed._symbol_map["EURUSD"] = "EURUSD"
    fake_info = SimpleNamespace(
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
        trade_tick_size=0.00001,
        trade_tick_value=1.0,
        trade_tick_value_loss=1.0,
        trade_contract_size=100_000.0,
    )
    with patch("apex.feeds.mt5_feed.mt5.symbol_info", return_value=fake_info):
        # loss_per_lot = (0.0010 / 0.00001) * 1.0 = 100.0; risk=19.90 => raw=0.199 -> floor 0.19
        lots = feed.compute_compliant_lot_size("EURUSD", entry_price=1.1000, stop_loss=1.0990, risk_amount=19.90)
        assert lots == pytest.approx(0.19, abs=1e-6)

    # cTrader fallback 0.001 step floor (e.g. BTCUSD)
    cfg = tmp_path / "apex_config.json"
    cfg.write_text('{"account_mode": "DEMO"}', encoding="utf-8")
    router = MarketFeedRouter(config_path=cfg)
    router.mt5._connected = False
    router.ctrader._connected = True
    router.ctrader._contract_specs["BTCUSD"] = ContractSpec(
        symbol="BTCUSD",
        digits=2,
        pip_size=0.01,
        point_size=0.01,
        lot_size=1.0,
        min_volume_lots=0.001,
        max_volume_lots=50.0,
        step_volume_lots=0.001,
        source="ctrader",
    )
    exec_router = DualBrokerExecutionRouter(feed_router=router)
    # sl_dist = 1000 -> loss_per_lot = 1000; risk=19.99 -> raw = 0.01999 -> floor 0.019
    ct_lots = exec_router.compute_compliant_lot_size("BTCUSD", entry_price=90000.0, stop_loss=89000.0, risk_usd=19.99)
    assert ct_lots == pytest.approx(0.019, abs=1e-6)


def test_ctrader_trendbars_res_cache_reconcile_and_router_candle_fallback(tmp_path) -> None:
    """ProtoOAGetTrendbarsRes & ProtoOAReconcileRes populate cTrader buffers; MarketFeedRouter falls back cleanly."""
    cfg = tmp_path / "apex_config.json"
    cfg.write_text('{"account_mode": "DEMO"}', encoding="utf-8")
    router = MarketFeedRouter(config_path=cfg)
    router.mt5._connected = False
    router.ctrader._connected = True
    router.ctrader.register_symbol_mapping("EURUSD", symbol_id=101, pip_size=0.0001, digits=5)

    # Inject ProtoOAGetTrendbarsRes directly via _handle_trendbars_res / envelope fallback
    res = oa_msg.ProtoOAGetTrendbarsRes()
    res.ctidTraderAccountId = 999
    res.symbolId = 101
    res.period = oa_model.ProtoOATrendbarPeriod.M5
    for i in range(5):
        tb = oa_model.ProtoOATrendbar()
        tb.low = 109900 + i * 10
        tb.deltaOpen = 5
        tb.deltaHigh = 25
        tb.deltaClose = 15
        tb.volume = 500 + i
        tb.utcTimestampInMinutes = 28_000_000 + i * 5
        res.trendbar.append(tb)

    decoded = router.ctrader._handle_trendbars_res(res)
    assert len(decoded) == 5
    bars = router.get_candles("EURUSD", timeframe="M5", count=5)
    assert len(bars) == 5
    assert bars[-1].source == "ctrader"
    assert pytest.approx(bars[0].low, abs=1e-6) == 1.09900

    # Inject ProtoOAReconcileRes and verify execution router tracks open cTrader position + panic_flatten_all closes it
    rec = oa_msg.ProtoOAReconcileRes()
    rec.ctidTraderAccountId = 999
    pos = oa_model.ProtoOAPosition()
    pos.positionId = 778899
    pos.tradeData.symbolId = 101
    pos.tradeData.volume = int(0.20 * 100_000 * CTRADER_VOLUME_CENTS_PER_UNIT)
    pos.tradeData.tradeSide = oa_model.ProtoOATradeSide.BUY
    pos.price = 1.10000
    pos.stopLoss = 1.09900
    pos.takeProfit = 1.10200
    rec.position.append(pos)

    router.ctrader._handle_reconcile_res(rec)
    ct_open = router.ctrader.get_open_positions()
    assert len(ct_open) == 1
    assert ct_open[0]["position_id"] == 778899
    assert ct_open[0]["lots"] == pytest.approx(0.20, abs=1e-4)

    def _ack_close(req, *_args, **_kwargs):
        if isinstance(req, oa_msg.ProtoOAClosePositionReq):
            ev = oa_msg.ProtoOAExecutionEvent()
            ev.executionType = oa_model.ProtoOAExecutionType.ORDER_FILLED
            ev.position.positionId = int(req.positionId)
            ev.position.positionStatus = 2
            router.ctrader._handle_execution_event(ev)

    router.ctrader._client = MagicMock()
    router.ctrader._client.send.side_effect = _ack_close
    exec_router = DualBrokerExecutionRouter(feed_router=router)
    pos_specs = exec_router.get_open_positions()
    assert any(p.symbol == "EURUSD" for p in pos_specs)
    assert "EURUSD" in exec_router.get_open_symbols()

    flat = exec_router.panic_flatten_all()
    assert flat["ctrader_closed"] == 1
    assert router.ctrader.get_open_positions() == []


def test_ctrader_execution_alias_resolution_and_dual_venue_position_manager(tmp_path) -> None:
    """Verify aliased symbol execution on cTrader and cTrader SL/TP amend + close dispatch in PositionOrchestrator."""
    cfg = tmp_path / "apex_config.json"
    cfg.write_text('{"account_mode": "DEMO"}', encoding="utf-8")
    router = MarketFeedRouter(config_path=cfg)
    router.mt5._connected = False
    router.ctrader._connected = True

    def _ack_all(req, *_args, **_kwargs):
        ev = oa_msg.ProtoOAExecutionEvent()
        ev.executionType = oa_model.ProtoOAExecutionType.ORDER_FILLED
        if isinstance(req, oa_msg.ProtoOANewOrderReq):
            ev.order.clientOrderId = str(req.clientOrderId)
            ev.order.orderId = 9001
            ev.position.positionId = 554433
            ev.position.positionStatus = 1
            ev.position.tradeData.symbolId = int(req.symbolId)
            ev.position.tradeData.volume = int(req.volume)
            ev.position.tradeData.tradeSide = int(req.tradeSide)
            ev.position.price = 140.05
            router.ctrader._handle_execution_event(ev)
        elif isinstance(req, oa_msg.ProtoOAAmendPositionSLTPReq):
            ev.position.positionId = int(req.positionId)
            ev.position.positionStatus = 1
            router.ctrader._handle_execution_event(ev)
        elif isinstance(req, oa_msg.ProtoOAClosePositionReq):
            ev.position.positionId = int(req.positionId)
            ev.position.positionStatus = 2
            router.ctrader._handle_execution_event(ev)

    router.ctrader._client = MagicMock()
    router.ctrader._client.send.side_effect = _ack_all
    router.ctrader.register_symbol_mapping("NVIDIA", symbol_id=505, pip_size=0.01, digits=2)
    router.ctrader._latest_ticks["NVIDIA"] = Tick("NVIDIA", bid=140.00, ask=140.05, timestamp_utc=1_700_000_500.0, source="ctrader")

    # Resolving NVDA -> NVIDIA succeeds through _try_execute_ctrader
    exec_router = DualBrokerExecutionRouter(feed_router=router, dom_obi_enabled=False)
    order = ExecutionOrderRequest(
        symbol="NVDA",
        side=Side.BUY,
        lots=1.0,
        entry_price=140.05,
        stop_loss=138.05,
        take_profit=144.05,
    )
    receipt = exec_router.execute_order(order=order, current_spread_price=0.05, vwap_slippage_price=0.01)
    assert receipt.venue == "ctrader"
    assert receipt.status == "FILLED"
    assert router.ctrader._client.send.call_count >= 1
    sent_new_order = router.ctrader._client.send.call_args[0][0]
    assert sent_new_order.symbolId == 505

    # Position manager routes SLTP amend + EARLY_EXIT close to cTrader when ticket is absent from MT5
    router.ctrader._client.reset_mock()
    candles = _make_candles("EURUSD", "M5", [1.0800 + i * 0.0001 for i in range(45)])
    mgr = AutonomousPositionOrchestrator(feed_router=router)
    with patch("apex.execution.position_manager.mt5.positions_get", return_value=[]):
        dec = mgr.evaluate_position(
            ticket=554433,
            symbol="EURUSD",
            side=Side.BUY,
            entry_price=1.0800,
            current_sl=1.0780,
            current_tp=1.0840,
            mark_price=1.0822,
            candles=candles,
            spread_price=0.0001,
            now_utc=1_700_049_600.0,
            bars_held_override=4,
            force_evaluate=True,
            execute_broker_actions=True,
        )
        assert dec.action == "RATCHET_SLTP"
        assert router.ctrader._client.send.call_count == 1
        amend_msg = router.ctrader._client.send.call_args[0][0]
        assert isinstance(amend_msg, oa_msg.ProtoOAAmendPositionSLTPReq)
        assert amend_msg.positionId == 554433

        # Now test Thesis Exit -> ProtoOAClosePositionReq
        router.ctrader._client.reset_mock()
        down_candles = _make_candles("EURUSD", "M5", [1.1000 - i * 0.0005 for i in range(59)] + [1.0680], spread_bar=0.0003)
        dec_exit = mgr.evaluate_position(
            ticket=554433,
            symbol="EURUSD",
            side=Side.BUY,
            entry_price=1.0950,
            current_sl=1.0650,
            current_tp=1.1250,
            mark_price=1.0680,
            candles=down_candles,
            dom_obi=-0.60,
            cross_asset_shock_z=3.5,
            now_utc=1_700_049_900.0,
            bars_held_override=6,
            force_evaluate=True,
            execute_broker_actions=True,
        )
        assert dec_exit.action == "EARLY_EXIT"
        assert router.ctrader._client.send.call_count == 1
        close_msg = router.ctrader._client.send.call_args[0][0]
        assert isinstance(close_msg, oa_msg.ProtoOAClosePositionReq)
        assert close_msg.positionId == 554433
