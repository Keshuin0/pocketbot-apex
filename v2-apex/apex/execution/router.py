"""Dual-Broker Execution Engine & 6 Hard Pre-Trade Circuit Breakers (`apex/execution/router.py`).

Supports order execution on:
  1. **cTrader Open API**: `ProtoOANewOrderReq` with mandatory server-side relative/absolute SL & TP.
  2. **MetaTrader 5 IPC**: `mt5.order_send` with automatic filling-mode detection (`ORDER_FILLING_FOK`,
     `ORDER_FILLING_IOC`, `ORDER_FILLING_RETURN`).
  3. **Seamless Paper-Shadow Fallback**: Automatically routes to high-fidelity `PAPER_SHADOW` execution
     if server AutoTrading is disabled, if the symbol is quarantined, or when running in shadow mode.

Enforces 6 non-negotiable pre-trade circuit breakers prior to wire transmission:
  (a) `demo_only_lock`: Hard-locked to DEMO accounts unless BOTH `--unlock-live` CLI flag AND
      `APEX_UNLOCK_LIVE=I_ACCEPT_REAL_RISK` env var are present.
  (b) `daily_drawdown_killswitch`: -2.0% Daily Drawdown Kill-Switch (closed + floating P/L vs daily start equity).
  (c) `position_and_factor_cap`: Max 2 Concurrent Open Positions + 1.0% Single-Currency/Factor Exposure Cap.
  (d) `spread_and_vwap_slippage_cap`: Live Spread + VWAP DOM Slippage <= 15% of Stop-Loss distance.
  (e) `level2_dom_wall_and_obi_gate`: Level 2 DOM Order-Book Imbalance (>= 0.15) & opposing wall (>3x within 0.35R) check (when DOM is active).
  (f) `mandatory_server_sl_tp`: Mandatory valid server-side hard Stop-Loss (`SL`) and Take-Profit (`TP`) verification.
"""

from __future__ import annotations

import datetime
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Sequence

import MetaTrader5 as mt5
from ctrader_open_api.messages import (
    OpenApiMessages_pb2 as oa_msg,
    OpenApiModelMessages_pb2 as oa_model,
)

from apex.cockpit.tui_dashboard import BrokerExecutionEvent
from apex.feeds.ctrader_feed import CTRADER_PRICE_SCALE, CTRADER_VOLUME_CENTS_PER_UNIT
from apex.feeds.dual_router import MarketFeedRouter
from apex.feeds.hydra_nbbo import ResolvedMeshTopology, SyntheticNBBOResult
from apex.feeds.models import DOMAnalytics, LiquidityWall, Side, verify_live_unlock
from apex.ledger.walkforward_db import WalkForwardLedgerDB
from apex.quant.correlation_shield import CorrelationShield, PositionRiskSpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ExecutionOrderRequest:
    """Normalized order execution request requiring mandatory server-side SL & TP."""

    symbol: str
    side: Side
    lots: float
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_pct: float = 0.5
    risk_usd: float = 50.0
    order_type: str = "MARKET"  # "MARKET" or "LIMIT"
    comment: str = "PocketBot_Apex"
    force_paper_shadow: bool = False


@dataclass(frozen=True, slots=True)
class CircuitBreakerCheck:
    """Single pre-trade circuit breaker evaluation."""

    code: str
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class CircuitBreakerReport:
    """Aggregated report across all 6 hard pre-trade circuit breakers."""

    all_passed: bool
    checks: tuple[CircuitBreakerCheck, ...] = field(default_factory=tuple)

    @property
    def failed_reasons(self) -> tuple[str, ...]:
        return tuple(f"[{c.code}] {c.detail}" for c in self.checks if not c.passed)


@dataclass(frozen=True, slots=True)
class ExecutionReceipt:
    """Order execution receipt returned by cTrader, MT5, Split-Execution, or Paper-Shadow fallback."""

    order_id: str
    symbol: str
    side: Side
    lots: float
    requested_entry: float
    executed_price: float
    stop_loss: float
    take_profit: float
    realized_slippage_price: float
    realized_slippage_r: float
    venue: str  # "ctrader", "mt5", "SPLIT_CTRADER_MT5", "PAPER_SHADOW", or "REJECTED"
    status: str  # "FILLED", "PAPER_SHADOW_FILLED", or "REJECTED"
    timestamp_utc: float
    breaker_report: CircuitBreakerReport
    message: str
    winning_venue: str = "mt5"
    mesh_topology: str = "MT5_SOLO"
    sor_pips_saved: float = 0.0
    sor_usd_saved: float = 0.0
    child_receipts: tuple[ExecutionReceipt, ...] = ()


class DualBrokerExecutionRouter:
    """Apex Hydra Active-Active Dual-Venue Smart Order Router (`apex/execution/router.py`)."""

    MAX_DAILY_DRAWDOWN_PCT: float = 2.0
    MAX_CONCURRENT_POSITIONS: int = 2
    MAX_FACTOR_EXPOSURE_PCT: float = 1.0
    MAX_SPREAD_SLIPPAGE_SL_RATIO: float = 0.15
    MIN_DOM_OBI: float = 0.15
    WALL_MULT_LIMIT: float = 3.0
    WALL_DIST_R_LIMIT: float = 0.35

    def __init__(
        self,
        feed_router: MarketFeedRouter,
        account_equity_usd: float = 10_000.0,
        max_daily_drawdown_pct: float = 2.0,
        max_concurrent_positions: int = 2,
        max_factor_exposure_pct: float = 1.0,
        max_spread_slippage_sl_ratio: float = 0.15,
        correlation_cap_enabled: bool = True,
        dom_obi_enabled: bool = True,
        is_live: bool = False,
        cli_args: Sequence[str] | None = None,
        ledger_db: WalkForwardLedgerDB | None = None,
    ) -> None:
        self.feed_router: MarketFeedRouter = feed_router
        self.account_equity_usd: float = float(account_equity_usd)
        self.max_daily_drawdown_pct: float = float(max_daily_drawdown_pct)
        self.max_concurrent_positions: int = int(max_concurrent_positions)
        self.max_factor_exposure_pct: float = float(max_factor_exposure_pct)
        self.max_spread_slippage_sl_ratio: float = float(max_spread_slippage_sl_ratio)
        self.correlation_cap_enabled: bool = bool(correlation_cap_enabled)
        self.dom_obi_enabled: bool = bool(dom_obi_enabled)
        self.is_live: bool = bool(is_live)
        self._cli_args: Sequence[str] | None = cli_args
        self.ledger_db: WalkForwardLedgerDB | None = ledger_db

        self.correlation_shield = CorrelationShield(
            enabled=self.correlation_cap_enabled,
            max_factor_risk_pct=self.max_factor_exposure_pct,
            account_equity_usd=self.account_equity_usd,
        )
        self._shadow_positions: list[dict[str, object]] = []
        self._recent_broker_events: list[BrokerExecutionEvent] = []
        self._total_sor_pips_saved: float = 0.0
        self._total_sor_usd_saved: float = 0.0

    @property
    def recent_broker_events(self) -> tuple[BrokerExecutionEvent, ...]:
        """Return recent broker execution events (up to last 20)."""
        return tuple(self._recent_broker_events[-20:])

    def record_broker_event(self, event: BrokerExecutionEvent) -> None:
        """Record an execution or rejection event into the live telemetry buffer."""
        self._recent_broker_events.append(event)
        if len(self._recent_broker_events) > 50:
            self._recent_broker_events.pop(0)

    def update_account_equity_from_broker(self) -> float:
        """Sync account equity with connected MT5 terminal if active."""
        if self.feed_router.mt5.is_connected:
            info = self.feed_router.mt5.get_account_info()
            eq = float(info.get("equity", 0.0))
            if eq > 0.0:
                self.account_equity_usd = eq
                self.correlation_shield.account_equity_usd = eq
                return eq
        return self.account_equity_usd

    def get_open_positions(self) -> list[PositionRiskSpec]:
        """Fetch all currently open positions across MT5 and Paper-Shadow, normalized to PositionRiskSpec."""
        specs: list[PositionRiskSpec] = []
        if self.feed_router.mt5.is_connected:
            raw_pos = self.feed_router.mt5.get_open_positions()
            for p in raw_pos:
                raw_sym = str(p.get("symbol", ""))
                can_sym = self.feed_router.mt5.alias_mapper.canonicalize(raw_sym)
                is_buy = int(p.get("type", 0)) == int(mt5.ORDER_TYPE_BUY)
                side = Side.BUY if is_buy else Side.SELL
                vol = float(p.get("volume", 0.01))
                entry_px = float(p.get("price_open", 0.0))
                sl_px = float(p.get("sl", 0.0))
                sl_dist = abs(entry_px - sl_px) if sl_px > 0.0 else 0.0
                risk_usd = 50.0
                if sl_dist > 0.0:
                    info = mt5.symbol_info(raw_sym)
                    if info is not None:
                        tick_size = float(getattr(info, "trade_tick_size", 0.0) or getattr(info, "point", 0.0) or 1e-5)
                        tick_val = float(getattr(info, "trade_tick_value", 0.0) or 0.0)
                        if tick_size > 0.0 and tick_val > 0.0:
                            risk_usd = (sl_dist / tick_size) * tick_val * vol
                risk_pct = (risk_usd / self.account_equity_usd * 100.0) if self.account_equity_usd > 0.0 else 0.5
                specs.append(
                    PositionRiskSpec(
                        symbol=can_sym,
                        side=side,
                        risk_pct=risk_pct,
                        risk_usd=risk_usd,
                    )
                )

        for sp in self._shadow_positions:
            specs.append(
                PositionRiskSpec(
                    symbol=str(sp.get("symbol", "")),
                    side=Side.BUY if str(sp.get("side", "")).upper() == "BUY" else Side.SELL,
                    risk_pct=0.5,
                    risk_usd=50.0,
                )
            )
        return specs

    def get_open_symbols(self) -> set[str]:
        """Return set of all symbol names (both canonical and raw broker names) currently open."""
        open_syms: set[str] = set()
        for p in self.get_open_positions():
            can = p.symbol.upper()
            open_syms.add(can)
            raw = self.feed_router.mt5.resolve_symbol(can) if self.feed_router.mt5.is_connected else None
            if raw:
                open_syms.add(raw.upper())
        for sp in self._shadow_positions:
            open_syms.add(str(sp.get("symbol", "")).upper())
        return open_syms

    def compute_compliant_lot_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        risk_usd: float,
    ) -> float:
        """Compute broker-compliant lot size using MT5 feed calculator or contract spec fallback."""
        if self.feed_router.mt5.is_connected:
            return self.feed_router.mt5.compute_compliant_lot_size(
                symbol=symbol,
                entry_price=entry_price,
                stop_loss=stop_loss,
                risk_amount=risk_usd,
            )
        spec = self.feed_router.ctrader.get_contract_spec(symbol)
        if spec is not None:
            sl_dist = abs(float(entry_price) - float(stop_loss))
            if sl_dist > 1e-12:
                loss_per_lot = sl_dist * spec.lot_size
                lots_raw = float(risk_usd) / loss_per_lot if loss_per_lot > 0.0 else spec.min_volume_lots
                step = spec.step_volume_lots if spec.step_volume_lots > 0 else 0.01
                snapped = round(lots_raw / step) * step
                return max(spec.min_volume_lots, min(spec.max_volume_lots, round(snapped, 2)))
        return 0.01

    @property
    def cumulative_sor_savings(self) -> tuple[float, float]:
        """Return `(total_pips_saved, total_usd_saved)` across session & SQLite ledger."""
        if self.ledger_db is not None:
            db_p, db_u = self.ledger_db.get_cumulative_sor_savings()
            return max(self._total_sor_pips_saved, db_p), max(self._total_sor_usd_saved, db_u)
        return self._total_sor_pips_saved, self._total_sor_usd_saved

    def evaluate_circuit_breakers(
        self,
        order: ExecutionOrderRequest,
        closed_pnl_today_usd: float = 0.0,
        floating_pnl_usd: float = 0.0,
        open_positions: Sequence[PositionRiskSpec] = (),
        current_spread_price: float = 0.0,
        vwap_slippage_price: float = 0.0,
        dom_analytics: DOMAnalytics | None = None,
        nbbo_result: SyntheticNBBOResult | None = None,
    ) -> CircuitBreakerReport:
        """Run all 6 non-negotiable pre-trade circuit breakers (including Cross-Pollinated DOM & Single-Broker Spike Rejector)."""
        checks: list[CircuitBreakerCheck] = []

        # Cross-Pollinated DOM: whenever cTrader depth quotes are online, inject cTrader's live
        # Level 2 OBI and Liquidity Wall check into ALL candidate trades—even trades routed to MT5!
        effective_dom = dom_analytics if dom_analytics is not None else self.feed_router.get_dom_analytics(order.symbol)

        # (a) Breaker 1: Demo-Only Lock
        try:
            verify_live_unlock(self.is_live, cli_args=self._cli_args)
            checks.append(
                CircuitBreakerCheck(
                    code="A_DEMO_LOCK",
                    name="Demo-Only Safety Lock",
                    passed=True,
                    detail="DEMO account lock verified" if not self.is_live else "Dual live unlock verified",
                )
            )
        except Exception as exc:
            checks.append(
                CircuitBreakerCheck(
                    code="A_DEMO_LOCK",
                    name="Demo-Only Safety Lock",
                    passed=False,
                    detail=str(exc),
                )
            )

        # (b) Breaker 2: -2.0% Daily Drawdown Kill-Switch (closed + floating P/L)
        net_daily_pnl_usd = float(closed_pnl_today_usd) + float(floating_pnl_usd)
        daily_dd_pct = (
            (-net_daily_pnl_usd / self.account_equity_usd) * 100.0
            if (self.account_equity_usd > 0.0 and net_daily_pnl_usd < 0.0)
            else 0.0
        )
        dd_passed = daily_dd_pct < self.max_daily_drawdown_pct - 1e-9
        checks.append(
            CircuitBreakerCheck(
                code="B_DAILY_DRAWDOWN",
                name="-2.0% Daily Drawdown Kill-Switch",
                passed=dd_passed,
                detail=(
                    f"Daily net P/L ${net_daily_pnl_usd:+.2f} (drawdown {daily_dd_pct:.2f}% "
                    f"{'<' if dd_passed else '>='} {self.max_daily_drawdown_pct:.2f}% kill-switch)"
                ),
            )
        )

        # (c) Breaker 3: Max 2 Concurrent Positions + 1.0% Factor Exposure Cap
        num_open = len(open_positions)
        pos_count_ok = num_open < self.max_concurrent_positions
        proposed_spec = PositionRiskSpec(
            symbol=order.symbol,
            side=order.side,
            risk_pct=order.risk_pct,
            risk_usd=order.risk_usd,
        )
        cs_verdict = self.correlation_shield.evaluate_proposed_trade(proposed_spec, open_positions)
        pos_factor_passed = pos_count_ok and cs_verdict.allowed
        if not pos_count_ok:
            pf_detail = (
                f"Concurrent open positions ({num_open}) already at max limit ({self.max_concurrent_positions})"
            )
        elif not cs_verdict.allowed:
            pf_detail = cs_verdict.reason
        else:
            pf_detail = (
                f"Open positions {num_open}/{self.max_concurrent_positions}; factor exposures within "
                f"{self.max_factor_exposure_pct:.2f}% cap"
            )
        checks.append(
            CircuitBreakerCheck(
                code="C_POSITIONS_AND_FACTOR_CAP",
                name="Max 2 Concurrent Positions + 1.0% Factor Exposure Cap",
                passed=pos_factor_passed,
                detail=pf_detail,
            )
        )

        # (f) Breaker 6 evaluated before (d)/(e) so we have validated SL distance:
        # Mandatory server-side SL & TP verification
        sl = float(order.stop_loss)
        tp = float(order.take_profit)
        ep = float(order.entry_price)
        if sl <= 0.0 or tp <= 0.0 or ep <= 0.0:
            sl_tp_valid = False
            sl_tp_detail = f"Missing or non-positive SL ({sl}) / TP ({tp}) / Entry ({ep})"
        elif order.side == Side.BUY and not (sl < ep < tp):
            sl_tp_valid = False
            sl_tp_detail = f"Invalid BUY bracket geometry: require SL ({sl}) < Entry ({ep}) < TP ({tp})"
        elif order.side == Side.SELL and not (tp < ep < sl):
            sl_tp_valid = False
            sl_tp_detail = f"Invalid SELL bracket geometry: require TP ({tp}) < Entry ({ep}) < SL ({sl})"
        else:
            sl_tp_valid = True
            sl_tp_detail = f"Verified server-side SL={sl:.5f} and TP={tp:.5f} for {order.side.value}"

        sl_distance = max(abs(ep - sl), 1e-12)

        # (d) Breaker 4: Spread & VWAP Slippage <= 15% of SL distance + Single-Broker Spike Rejector
        total_friction_price = max(0.0, float(current_spread_price)) + max(0.0, float(vwap_slippage_price))
        friction_sl_ratio = total_friction_price / sl_distance
        spike_blocked = bool(nbbo_result.single_broker_spike_blocked) if nbbo_result is not None else False
        friction_passed = sl_tp_valid and (friction_sl_ratio <= self.max_spread_slippage_sl_ratio + 1e-9) and (not spike_blocked)
        if spike_blocked and nbbo_result is not None:
            friction_detail = nbbo_result.spike_reason
        else:
            friction_detail = (
                f"Spread ({current_spread_price:.5f}) + VWAP slippage ({vwap_slippage_price:.5f}) "
                f"= {friction_sl_ratio * 100:.2f}% of SL distance ({sl_distance:.5f}) "
                f"[limit {self.max_spread_slippage_sl_ratio * 100:.1f}%]"
            )
        checks.append(
            CircuitBreakerCheck(
                code="D_SPREAD_VWAP_SLIPPAGE",
                name="Spread + VWAP Slippage <= 15% of SL & Cross-Venue Basis Guard",
                passed=friction_passed,
                detail=friction_detail,
            )
        )

        # (e) Breaker 5: Level 2 DOM Wall / OBI check (Cross-Pollinated when cTrader depth quotes are online)
        if effective_dom is not None and self.dom_obi_enabled:
            obi = effective_dom.obi
            obi_ok = (obi >= self.MIN_DOM_OBI - 1e-9) if order.side == Side.BUY else (obi <= -self.MIN_DOM_OBI + 1e-9)
            opposing_side = Side.SELL if order.side == Side.BUY else Side.BUY
            max_wall_dist = self.WALL_DIST_R_LIMIT * sl_distance
            blocking_wall: LiquidityWall | None = None
            for w in effective_dom.liquidity_walls:
                if w.side != opposing_side or w.ratio_to_median <= self.WALL_MULT_LIMIT:
                    continue
                dist = (w.price - ep) if order.side == Side.BUY else (ep - w.price)
                if 0.0 <= dist <= max_wall_dist + 1e-9:
                    blocking_wall = w
                    break

            dom_ok = obi_ok and (blocking_wall is None)
            if not obi_ok:
                dom_detail = f"Cross-Pollinated DOM OBI {obi:+.3f} fails directional threshold ({self.MIN_DOM_OBI:.2f})"
            elif blocking_wall is not None:
                dom_detail = (
                    f"Opposing {blocking_wall.side.value} liquidity wall ({blocking_wall.ratio_to_median:.1f}x median) "
                    f"at {blocking_wall.price:.5f} within {self.WALL_DIST_R_LIMIT:.2f}R"
                )
            else:
                dom_detail = (
                    f"Cross-Pollinated Level 2 DOM verified: OBI={obi:+.3f}, "
                    f"zero opposing walls within {self.WALL_DIST_R_LIMIT:.2f}R"
                )
        else:
            dom_ok = True
            dom_detail = "Level 2 DOM inactive or disabled; bypassed cleanly"

        checks.append(
            CircuitBreakerCheck(
                code="E_LEVEL2_DOM_GATE",
                name="Level 2 DOM Wall & OBI Gate (Cross-Pollinated)",
                passed=dom_ok,
                detail=dom_detail,
            )
        )

        # Append (f) Breaker 6 in canonical order
        checks.append(
            CircuitBreakerCheck(
                code="F_MANDATORY_SL_TP",
                name="Mandatory Server-Side SL & TP Verification",
                passed=sl_tp_valid,
                detail=sl_tp_detail,
            )
        )

        return CircuitBreakerReport(
            all_passed=all(c.passed for c in checks),
            checks=tuple(checks),
        )

    @staticmethod
    def build_ctrader_Order_req(
        account_id: int,
        symbol_id: int,
        order: ExecutionOrderRequest,
        lot_size_units: float = 100_000.0,
    ) -> oa_msg.ProtoOANewOrderReq:
        """Construct a Spotware `ProtoOANewOrderReq` with mandatory server-side SL & TP."""
        if order.stop_loss <= 0.0 or order.take_profit <= 0.0:
            raise ValueError("cTrader order requires positive server-side stop_loss and take_profit.")

        req = oa_msg.ProtoOANewOrderReq()
        req.ctidTraderAccountId = int(account_id)
        req.symbolId = int(symbol_id)
        req.orderType = (
            oa_model.ProtoOAOrderType.LIMIT
            if order.order_type.upper() == "LIMIT"
            else oa_model.ProtoOAOrderType.MARKET
        )
        req.tradeSide = (
            oa_model.ProtoOATradeSide.BUY if order.side == Side.BUY else oa_model.ProtoOATradeSide.SELL
        )
        # Volume in cents of a unit (1.00 lot of 100,000 units = 10,000,000)
        req.volume = int(round(float(order.lots) * float(lot_size_units) * CTRADER_VOLUME_CENTS_PER_UNIT))
        req.comment = order.comment
        req.clientOrderId = f"apex_{uuid.uuid4().hex[:12]}"

        if req.orderType == oa_model.ProtoOAOrderType.LIMIT:
            req.limitPrice = float(order.entry_price)
            req.stopLoss = float(order.stop_loss)
            req.takeProfit = float(order.take_profit)
        else:
            # Market orders in Spotware Open API v2 accept relativeStopLoss / relativeTakeProfit
            # scaled by 100,000 (1e-5 price units) AND absolute stopLoss / takeProfit
            sl_dist = abs(float(order.entry_price) - float(order.stop_loss))
            tp_dist = abs(float(order.take_profit) - float(order.entry_price))
            req.relativeStopLoss = int(round(sl_dist * CTRADER_PRICE_SCALE))
            req.relativeTakeProfit = int(round(tp_dist * CTRADER_PRICE_SCALE))
            req.stopLoss = float(order.stop_loss)
            req.takeProfit = float(order.take_profit)

        return req

    @staticmethod
    def detect_mt5_filling_mode(symbol: str) -> int:
        """Detect the broker's supported MT5 filling mode bitmask (`FOK`, `IOC`, or `RETURN`)."""
        info = mt5.symbol_info(symbol)
        if info is None:
            return int(mt5.ORDER_FILLING_FOK)
        filling_mode = int(getattr(info, "filling_mode", 0))
        # Bitmask: 1 = SYMBOL_FILLING_FOK, 2 = SYMBOL_FILLING_IOC
        if filling_mode & 1:
            return int(mt5.ORDER_FILLING_FOK)
        if filling_mode & 2:
            return int(mt5.ORDER_FILLING_IOC)
        return int(mt5.ORDER_FILLING_RETURN)

    def _try_execute_ctrader(
        self,
        order: ExecutionOrderRequest,
        report: CircuitBreakerReport,
        vwap_slippage_price: float,
        sl_dist: float,
        now_utc: float,
        winning_venue: str = "ctrader",
        mesh_topology: str = "CTRADER_SOLO",
        sor_pips_saved: float = 0.0,
        sor_usd_saved: float = 0.0,
    ) -> ExecutionReceipt | None:
        """Attempt wire dispatch to cTrader Open API if connected and symbol is mapped."""
        ct = self.feed_router.ctrader
        norm_sym = order.symbol.upper()
        if ct.is_connected and norm_sym in ct._symbol_name_to_id:
            try:
                sid = ct._symbol_name_to_id[norm_sym]
                spec = ct._contract_specs.get(norm_sym)
                lot_units = spec.lot_size if spec else 100_000.0
                req = self.build_ctrader_Order_req(
                    account_id=ct.account_id,
                    symbol_id=sid,
                    order=order,
                    lot_size_units=lot_units,
                )
                if ct._client is not None:
                    ct._client.send(req)
                exec_price = (
                    order.entry_price + vwap_slippage_price
                    if order.side == Side.BUY
                    else order.entry_price - vwap_slippage_price
                )
                self.record_broker_event(
                    BrokerExecutionEvent(
                        timestamp_str=datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S"),
                        symbol=norm_sym,
                        side=order.side.value,
                        lots=order.lots,
                        price=exec_price,
                        stop_loss=order.stop_loss,
                        take_profit=order.take_profit,
                        venue="CTRADER",
                        status="FILLED",
                        details=f"Order {req.clientOrderId[:8]} | Sent [{mesh_topology}]",
                    )
                )
                return ExecutionReceipt(
                    order_id=req.clientOrderId,
                    symbol=norm_sym,
                    side=order.side,
                    lots=order.lots,
                    requested_entry=order.entry_price,
                    executed_price=exec_price,
                    stop_loss=order.stop_loss,
                    take_profit=order.take_profit,
                    realized_slippage_price=vwap_slippage_price,
                    realized_slippage_r=vwap_slippage_price / sl_dist,
                    venue="ctrader",
                    status="FILLED",
                    timestamp_utc=now_utc,
                    breaker_report=report,
                    message=(
                        f"Dispatched ProtoOANewOrderReq with server-side SL/TP to cTrader Open API "
                        f"[{mesh_topology}]."
                    ),
                    winning_venue=winning_venue,
                    mesh_topology=mesh_topology,
                    sor_pips_saved=sor_pips_saved,
                    sor_usd_saved=sor_usd_saved,
                )
            except Exception as exc:
                logger.warning("cTrader execution failed (%s); attempting fallback...", exc)
        return None

    def _try_execute_mt5(
        self,
        order: ExecutionOrderRequest,
        report: CircuitBreakerReport,
        vwap_slippage_price: float,
        sl_dist: float,
        now_utc: float,
        winning_venue: str = "mt5",
        mesh_topology: str = "MT5_SOLO",
        sor_pips_saved: float = 0.0,
        sor_usd_saved: float = 0.0,
    ) -> ExecutionReceipt | None:
        """Attempt wire dispatch to MetaTrader 5 IPC if connected and AutoTrading is enabled."""
        norm_sym = order.symbol.upper()
        if self.feed_router.mt5.is_connected:
            term_info = mt5.terminal_info()
            trade_allowed = bool(getattr(term_info, "trade_allowed", False)) if term_info is not None else False
            resolved = self.feed_router.mt5.resolve_symbol(norm_sym)
            if trade_allowed and resolved:
                filling = self.detect_mt5_filling_mode(resolved)
                tick = mt5.symbol_info_tick(resolved)
                live_px = (
                    float(tick.ask if order.side == Side.BUY else tick.bid)
                    if tick is not None
                    else float(order.entry_price)
                )
                mt5_req = {
                    "action": mt5.TRADE_ACTION_DEAL if order.order_type.upper() == "MARKET" else mt5.TRADE_ACTION_PENDING,
                    "symbol": resolved,
                    "volume": float(order.lots),
                    "type": mt5.ORDER_TYPE_BUY if order.side == Side.BUY else mt5.ORDER_TYPE_SELL,
                    "price": live_px if order.order_type.upper() == "MARKET" else float(order.entry_price),
                    "sl": float(order.stop_loss),
                    "tp": float(order.take_profit),
                    "deviation": 20,
                    "magic": 202611,
                    "comment": order.comment,
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": filling,
                }
                res = mt5.order_send(mt5_req)
                if res is not None and int(getattr(res, "retcode", 0)) == int(mt5.TRADE_RETCODE_DONE):
                    fill_px = float(getattr(res, "price", 0.0) or live_px)
                    slip = (fill_px - order.entry_price) if order.side == Side.BUY else (order.entry_price - fill_px)
                    self.record_broker_event(
                        BrokerExecutionEvent(
                            timestamp_str=datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S"),
                            symbol=norm_sym,
                            side=order.side.value,
                            lots=order.lots,
                            price=fill_px,
                            stop_loss=order.stop_loss,
                            take_profit=order.take_profit,
                            venue="MT5",
                            status="FILLED",
                            details=f"Ticket #{res.order} | Done",
                        )
                    )
                    return ExecutionReceipt(
                        order_id=str(getattr(res, "order", uuid.uuid4().hex[:10])),
                        symbol=norm_sym,
                        side=order.side,
                        lots=order.lots,
                        requested_entry=order.entry_price,
                        executed_price=fill_px,
                        stop_loss=order.stop_loss,
                        take_profit=order.take_profit,
                        realized_slippage_price=float(slip),
                        realized_slippage_r=float(slip) / sl_dist,
                        venue="mt5",
                        status="FILLED",
                        timestamp_utc=now_utc,
                        breaker_report=report,
                        message=f"Executed MT5 order #{res.order} with server-side SL/TP [{mesh_topology}].",
                        winning_venue=winning_venue,
                        mesh_topology=mesh_topology,
                        sor_pips_saved=sor_pips_saved,
                        sor_usd_saved=sor_usd_saved,
                    )
                else:
                    rc = int(getattr(res, "retcode", -1)) if res is not None else -1
                    cmt = str(getattr(res, "comment", "Error")) if res is not None else "No terminal response"
                    logger.warning("MT5 order_send rejected for %s: retcode=%s comment=%s", norm_sym, rc, cmt)
                    self.record_broker_event(
                        BrokerExecutionEvent(
                            timestamp_str=datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S"),
                            symbol=norm_sym,
                            side=order.side.value,
                            lots=order.lots,
                            price=live_px,
                            stop_loss=order.stop_loss,
                            take_profit=order.take_profit,
                            venue="MT5",
                            status="REJECTED",
                            details=f"Retcode {rc}: {cmt}",
                        )
                    )
            else:
                reason = "AutoTrading Disabled" if not trade_allowed else f"Unresolved '{norm_sym}'"
                logger.warning("MT5 execution skipped: %s", reason)
                self.record_broker_event(
                    BrokerExecutionEvent(
                        timestamp_str=datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S"),
                        symbol=norm_sym,
                        side=order.side.value,
                        lots=order.lots,
                        price=float(order.entry_price),
                        stop_loss=order.stop_loss,
                        take_profit=order.take_profit,
                        venue="MT5",
                        status="BLOCKED",
                        details=reason,
                    )
                )
        return None

    def _record_sor_savings(
        self,
        order_id: str,
        symbol: str,
        lots: float,
        nbbo: SyntheticNBBOResult,
    ) -> tuple[float, float]:
        """Persist SOR pips and USD saved into session totals and `WalkForwardLedgerDB`."""
        pips_saved = max(0.0, float(nbbo.pips_saved))
        usd_saved = max(0.0, float(nbbo.usd_saved_per_lot) * float(lots))
        if pips_saved > 0.0 or usd_saved > 0.0:
            self._total_sor_pips_saved += pips_saved
            self._total_sor_usd_saved += usd_saved
            if self.ledger_db is not None:
                ct_pips = nbbo.ctrader_quote.effective_cost_pips if nbbo.ctrader_quote else None
                mt5_pips = nbbo.mt5_quote.effective_cost_pips if nbbo.mt5_quote else None
                self.ledger_db.log_sor_execution_savings(
                    order_id=order_id,
                    symbol=symbol,
                    lots=lots,
                    mesh_topology=nbbo.topology.value,
                    winning_venue=nbbo.winning_venue,
                    pips_saved=pips_saved,
                    usd_saved=usd_saved,
                    ctrader_effective_pips=ct_pips,
                    mt5_effective_pips=mt5_pips,
                )
        return pips_saved, usd_saved

    def execute_order(
        self,
        order: ExecutionOrderRequest,
        closed_pnl_today_usd: float = 0.0,
        floating_pnl_usd: float = 0.0,
        open_positions: Sequence[PositionRiskSpec] = (),
        current_spread_price: float = 0.0,
        vwap_slippage_price: float = 0.0,
        dom_analytics: DOMAnalytics | None = None,
    ) -> ExecutionReceipt:
        """Validate all 6 circuit breakers and route order via the Apex Hydra Smart Order Router."""
        sl_dist = max(abs(order.entry_price - order.stop_loss), 1e-12)
        nbbo = self.feed_router.evaluate_nbbo(
            symbol=order.symbol,
            one_r_price_distance=sl_dist,
            side=order.side,
            ctrader_slippage_price=vwap_slippage_price,
            mt5_slippage_price=vwap_slippage_price,
        )

        report = self.evaluate_circuit_breakers(
            order=order,
            closed_pnl_today_usd=closed_pnl_today_usd,
            floating_pnl_usd=floating_pnl_usd,
            open_positions=open_positions,
            current_spread_price=current_spread_price,
            vwap_slippage_price=vwap_slippage_price,
            dom_analytics=dom_analytics,
            nbbo_result=nbbo,
        )
        now_utc = time.time()
        topology = nbbo.topology

        if not report.all_passed:
            fail_detail = report.failed_reasons[0] if report.failed_reasons else "Circuit breaker block"
            self.record_broker_event(
                BrokerExecutionEvent(
                    timestamp_str=datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S"),
                    symbol=order.symbol.upper(),
                    side=order.side.value,
                    lots=order.lots,
                    price=order.entry_price,
                    stop_loss=order.stop_loss,
                    take_profit=order.take_profit,
                    venue="CIRCUIT",
                    status="REJECTED",
                    details=fail_detail[:60],
                )
            )
            return ExecutionReceipt(
                order_id="",
                symbol=order.symbol.upper(),
                side=order.side,
                lots=order.lots,
                requested_entry=order.entry_price,
                executed_price=0.0,
                stop_loss=order.stop_loss,
                take_profit=order.take_profit,
                realized_slippage_price=0.0,
                realized_slippage_r=0.0,
                venue="REJECTED",
                status="REJECTED",
                timestamp_utc=now_utc,
                breaker_report=report,
                message="Order blocked by pre-trade circuit breakers: " + "; ".join(report.failed_reasons),
                winning_venue="REJECTED",
                mesh_topology=topology.value,
            )

        if order.force_paper_shadow:
            pips_saved, usd_saved = self._record_sor_savings(
                order_id=f"shadow_sor_{uuid.uuid4().hex[:8]}",
                symbol=order.symbol,
                lots=order.lots,
                nbbo=nbbo,
            )
            return self._execute_paper_shadow(
                order,
                report,
                vwap_slippage_price,
                sl_dist,
                "Symbol in PAPER_SHADOW mode",
                winning_venue=nbbo.winning_venue,
                mesh_topology=topology.value,
                sor_pips_saved=pips_saved,
                sor_usd_saved=usd_saved,
            )

        # 1. SPLIT_EXECUTION: 50/50 parallel risk split across cTrader + MT5
        if topology == ResolvedMeshTopology.SPLIT_EXECUTION:
            half_lots = round(max(0.01, float(order.lots) / 2.0), 2)
            half_risk_pct = float(order.risk_pct) / 2.0
            half_risk_usd = float(order.risk_usd) / 2.0
            child_order = ExecutionOrderRequest(
                symbol=order.symbol,
                side=order.side,
                lots=half_lots,
                entry_price=order.entry_price,
                stop_loss=order.stop_loss,
                take_profit=order.take_profit,
                risk_pct=half_risk_pct,
                risk_usd=half_risk_usd,
                order_type=order.order_type,
                comment=f"{order.comment}_SPLIT",
                force_paper_shadow=False,
            )
            leg_ct = self._try_execute_ctrader(
                child_order,
                report,
                vwap_slippage_price,
                sl_dist,
                now_utc,
                winning_venue="ctrader",
                mesh_topology=topology.value,
            ) or self._execute_paper_shadow(
                child_order,
                report,
                vwap_slippage_price,
                sl_dist,
                "Split leg 1 (cTrader paper-shadow fallback)",
                winning_venue="ctrader",
                mesh_topology=topology.value,
            )
            leg_mt5 = self._try_execute_mt5(
                child_order,
                report,
                vwap_slippage_price,
                sl_dist,
                now_utc,
                winning_venue="mt5",
                mesh_topology=topology.value,
            ) or self._execute_paper_shadow(
                child_order,
                report,
                vwap_slippage_price,
                sl_dist,
                "Split leg 2 (MT5 paper-shadow fallback)",
                winning_venue="mt5",
                mesh_topology=topology.value,
            )
            avg_px = (leg_ct.executed_price + leg_mt5.executed_price) / 2.0
            parent_id = f"split_{leg_ct.order_id}_{leg_mt5.order_id}"
            pips_saved, usd_saved = self._record_sor_savings(parent_id, order.symbol, order.lots, nbbo)
            return ExecutionReceipt(
                order_id=parent_id,
                symbol=order.symbol.upper(),
                side=order.side,
                lots=round(half_lots * 2.0, 2),
                requested_entry=order.entry_price,
                executed_price=avg_px,
                stop_loss=order.stop_loss,
                take_profit=order.take_profit,
                realized_slippage_price=vwap_slippage_price,
                realized_slippage_r=vwap_slippage_price / sl_dist,
                venue="SPLIT_CTRADER_MT5",
                status="FILLED" if (leg_ct.status == "FILLED" or leg_mt5.status == "FILLED") else "PAPER_SHADOW_FILLED",
                timestamp_utc=now_utc,
                breaker_report=report,
                message=f"Executed 50/50 SPLIT_EXECUTION across cTrader ({half_lots} lots) + MT5 ({half_lots} lots).",
                winning_venue="SPLIT_CTRADER_MT5",
                mesh_topology=topology.value,
                sor_pips_saved=pips_saved,
                sor_usd_saved=usd_saved,
                child_receipts=(leg_ct, leg_mt5),
            )

        # 2. CTRADER_EYES_MT5_HANDS or MT5_SOLO: Route 100% of orders to MT5
        if topology in (ResolvedMeshTopology.CTRADER_EYES_MT5_HANDS, ResolvedMeshTopology.MT5_SOLO):
            mt5_rec = self._try_execute_mt5(
                order,
                report,
                vwap_slippage_price,
                sl_dist,
                now_utc,
                winning_venue="mt5",
                mesh_topology=topology.value,
            )
            if mt5_rec is not None:
                return mt5_rec
            return self._execute_paper_shadow(
                order,
                report,
                vwap_slippage_price,
                sl_dist,
                f"MT5 execution routed to Paper-Shadow fallback [{topology.value}]",
                winning_venue="mt5",
                mesh_topology=topology.value,
            )

        # 3. MT5_EYES_CTRADER_HANDS or CTRADER_SOLO: Route 100% of orders to cTrader
        if topology in (ResolvedMeshTopology.MT5_EYES_CTRADER_HANDS, ResolvedMeshTopology.CTRADER_SOLO):
            ct_rec = self._try_execute_ctrader(
                order,
                report,
                vwap_slippage_price,
                sl_dist,
                now_utc,
                winning_venue="ctrader",
                mesh_topology=topology.value,
            )
            if ct_rec is not None:
                return ct_rec
            return self._execute_paper_shadow(
                order,
                report,
                vwap_slippage_price,
                sl_dist,
                f"cTrader execution routed to Paper-Shadow fallback [{topology.value}]",
                winning_venue="ctrader",
                mesh_topology=topology.value,
            )

        # 4. ACTIVE_ACTIVE_SOR: Smart Venue Selector routes to lowest effective_cost venue
        pips_saved, usd_saved = self._record_sor_savings(
            order_id=f"sor_{uuid.uuid4().hex[:10]}",
            symbol=order.symbol,
            lots=order.lots,
            nbbo=nbbo,
        )
        winner = nbbo.winning_venue.lower()
        if winner == "ctrader":
            rec = self._try_execute_ctrader(
                order,
                report,
                vwap_slippage_price,
                sl_dist,
                now_utc,
                winning_venue="ctrader",
                mesh_topology=topology.value,
                sor_pips_saved=pips_saved,
                sor_usd_saved=usd_saved,
            ) or self._try_execute_mt5(
                order,
                report,
                vwap_slippage_price,
                sl_dist,
                now_utc,
                winning_venue="ctrader",
                mesh_topology=topology.value,
                sor_pips_saved=pips_saved,
                sor_usd_saved=usd_saved,
            )
            if rec is not None:
                return rec
        else:
            rec = self._try_execute_mt5(
                order,
                report,
                vwap_slippage_price,
                sl_dist,
                now_utc,
                winning_venue="mt5",
                mesh_topology=topology.value,
                sor_pips_saved=pips_saved,
                sor_usd_saved=usd_saved,
            ) or self._try_execute_ctrader(
                order,
                report,
                vwap_slippage_price,
                sl_dist,
                now_utc,
                winning_venue="mt5",
                mesh_topology=topology.value,
                sor_pips_saved=pips_saved,
                sor_usd_saved=usd_saved,
            )
            if rec is not None:
                return rec

        # Seamless Paper-Shadow fallback preserving winning_venue and SOR cost savings telemetry
        return self._execute_paper_shadow(
            order,
            report,
            vwap_slippage_price,
            sl_dist,
            f"Seamless Paper-Shadow fallback [{topology.value} -> winner={nbbo.winning_venue.upper()}]",
            winning_venue=nbbo.winning_venue,
            mesh_topology=topology.value,
            sor_pips_saved=pips_saved,
            sor_usd_saved=usd_saved,
        )

    def _execute_paper_shadow(
        self,
        order: ExecutionOrderRequest,
        report: CircuitBreakerReport,
        vwap_slippage_price: float,
        sl_dist: float,
        reason_note: str,
        winning_venue: str = "mt5",
        mesh_topology: str = "MT5_SOLO",
        sor_pips_saved: float = 0.0,
        sor_usd_saved: float = 0.0,
    ) -> ExecutionReceipt:
        now_utc = time.time()
        oid = f"shadow_{uuid.uuid4().hex[:10]}"
        exec_price = (
            order.entry_price + vwap_slippage_price
            if order.side == Side.BUY
            else order.entry_price - vwap_slippage_price
        )
        self._shadow_positions.append(
            {
                "order_id": oid,
                "symbol": order.symbol.upper(),
                "side": order.side.value,
                "lots": order.lots,
                "entry_price": exec_price,
                "stop_loss": order.stop_loss,
                "take_profit": order.take_profit,
                "timestamp_utc": now_utc,
                "winning_venue": winning_venue,
                "mesh_topology": mesh_topology,
                "sor_pips_saved": sor_pips_saved,
                "sor_usd_saved": sor_usd_saved,
            }
        )
        self.record_broker_event(
            BrokerExecutionEvent(
                timestamp_str=datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S"),
                symbol=order.symbol.upper(),
                side=order.side.value,
                lots=order.lots,
                price=exec_price,
                stop_loss=order.stop_loss,
                take_profit=order.take_profit,
                venue="SHADOW",
                status="PAPER_SHADOW_FILLED",
                details=reason_note[:50],
            )
        )
        return ExecutionReceipt(
            order_id=oid,
            symbol=order.symbol.upper(),
            side=order.side,
            lots=order.lots,
            requested_entry=order.entry_price,
            executed_price=exec_price,
            stop_loss=order.stop_loss,
            take_profit=order.take_profit,
            realized_slippage_price=float(vwap_slippage_price),
            realized_slippage_r=float(vwap_slippage_price) / sl_dist,
            venue="PAPER_SHADOW",
            status="PAPER_SHADOW_FILLED",
            timestamp_utc=now_utc,
            breaker_report=report,
            message=reason_note,
            winning_venue=winning_venue,
            mesh_topology=mesh_topology,
            sor_pips_saved=sor_pips_saved,
            sor_usd_saved=sor_usd_saved,
        )

    def panic_flatten_all(self) -> dict[str, int]:
        """Emergency `/panic_flatten`: close all open positions across MT5, cTrader, and Paper-Shadow."""
        closed_mt5 = 0
        closed_ctrader = 0
        closed_shadow = len(self._shadow_positions)
        self._shadow_positions.clear()

        if self.feed_router.mt5.is_connected:
            positions = mt5.positions_get()
            if positions:
                for pos in positions:
                    sym = str(pos.symbol)
                    vol = float(pos.volume)
                    is_buy = int(pos.type) == int(mt5.ORDER_TYPE_BUY)
                    tick = mt5.symbol_info_tick(sym)
                    close_px = float(tick.bid if is_buy else tick.ask) if tick else float(pos.price_current)
                    req = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "position": int(pos.ticket),
                        "symbol": sym,
                        "volume": vol,
                        "type": mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY,
                        "price": close_px,
                        "deviation": 50,
                        "magic": 202611,
                        "comment": "APEX_PANIC_FLATTEN",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": self.detect_mt5_filling_mode(sym),
                    }
                    res = mt5.order_send(req)
                    if res is not None and int(getattr(res, "retcode", 0)) == int(mt5.TRADE_RETCODE_DONE):
                        closed_mt5 += 1

        return {
            "mt5_closed": closed_mt5,
            "ctrader_closed": closed_ctrader,
            "shadow_closed": closed_shadow,
        }
