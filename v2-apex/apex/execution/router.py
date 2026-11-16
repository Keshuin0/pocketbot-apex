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
from decimal import Decimal
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Sequence

try:
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover - headless/cTrader-only nodes
    mt5 = None  # type: ignore[assignment]
from ctrader_open_api.messages import (
    OpenApiMessages_pb2 as oa_msg,
    OpenApiModelMessages_pb2 as oa_model,
)

from apex.cockpit.tui_dashboard import BrokerExecutionEvent
from apex.feeds.ctrader_feed import CTRADER_PRICE_SCALE, CTRADER_VOLUME_CENTS_PER_UNIT
from apex.feeds.dual_router import MarketFeedRouter
from apex.feeds.hydra_nbbo import ResolvedMeshTopology, SyntheticNBBOResult
from apex.feeds.models import AssetClass, ContractSpec, DOMAnalytics, LiquidityWall, Side, verify_live_unlock
from apex.ledger.walkforward_db import SymbolExecutionMode, WalkForwardLedgerDB
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
    predicted_net_r: float = 0.0
    regime: str = "UNKNOWN"
    ai_decision: str = "APPROVE"
    ai_confidence: float = 1.0
    ai_reason: str = ""
    dom_obi: float | None = None


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
        self.prune_closed_shadow_positions()
        specs: list[PositionRiskSpec] = []
        if mt5 is not None and self.feed_router.mt5.is_connected:
            raw_pos = self.feed_router.mt5.get_open_positions()
            for p in raw_pos:
                raw_sym = str(p.get("symbol", ""))
                can_sym = self.feed_router.mt5.alias_mapper.canonicalize(raw_sym)
                is_buy = int(p.get("type", 0)) == int(mt5.ORDER_TYPE_BUY)
                side = Side.BUY if is_buy else Side.SELL
                vol = float(p.get("volume", 0.01))
                entry_px = float(p.get("price_open", 0.0))
                curr_px = float(p.get("price_current", entry_px) or entry_px)
                sl_px = float(p.get("sl", 0.0))
                tp_px = float(p.get("tp", 0.0))
                profit_usd = float(p.get("profit", 0.0) or 0.0)
                ticket = int(p.get("ticket", 0) or 0)
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
                        lots=vol,
                        entry_price=entry_px,
                        current_price=curr_px,
                        stop_loss=sl_px,
                        take_profit=tp_px,
                        profit_usd=profit_usd,
                        ticket=ticket,
                        venue="mt5",
                    )
                )

        if self.feed_router.ctrader.is_connected and hasattr(self.feed_router.ctrader, "get_open_positions"):
            for cp in self.feed_router.ctrader.get_open_positions():
                can_sym = str(cp.get("canonical_symbol") or cp.get("symbol") or "").upper()
                if not can_sym:
                    continue
                side = Side.BUY if str(cp.get("side", "BUY")).upper() == "BUY" else Side.SELL
                entry_px = float(cp.get("entry_price", 0.0) or 0.0)
                curr_px = float(cp.get("current_price", entry_px) or entry_px)
                sl_px = float(cp.get("stop_loss", 0.0) or 0.0)
                tp_px = float(cp.get("take_profit", 0.0) or 0.0)
                lots = float(cp.get("lots", 0.01) or 0.01)
                profit_usd = float(cp.get("profit_usd", 0.0) or 0.0)
                ticket = int(cp.get("position_id", cp.get("ticket", 0)) or 0)
                spec = self.feed_router.ctrader.get_contract_spec(can_sym)
                lot_units = spec.lot_size if spec else 100_000.0
                sl_dist = abs(entry_px - sl_px) if (entry_px > 0.0 and sl_px > 0.0) else 0.0
                risk_usd = (sl_dist * lot_units * lots) if sl_dist > 0.0 else 50.0
                risk_pct = (risk_usd / self.account_equity_usd * 100.0) if self.account_equity_usd > 0.0 else 0.5
                specs.append(
                    PositionRiskSpec(
                        symbol=can_sym,
                        side=side,
                        risk_pct=risk_pct,
                        risk_usd=risk_usd,
                        lots=lots,
                        entry_price=entry_px,
                        current_price=curr_px,
                        stop_loss=sl_px,
                        take_profit=tp_px,
                        profit_usd=profit_usd,
                        ticket=ticket,
                        venue="ctrader",
                    )
                )

        for sp in self._shadow_positions:
            entry_px = float(sp.get("entry_price", 0.0) or 0.0)
            specs.append(
                PositionRiskSpec(
                    symbol=str(sp.get("symbol", "")),
                    side=Side.BUY if str(sp.get("side", "")).upper() == "BUY" else Side.SELL,
                    risk_pct=float(sp.get("risk_pct", 0.5) or 0.5),
                    risk_usd=float(sp.get("risk_usd", 50.0) or 50.0),
                    lots=float(sp.get("lots", 0.01) or 0.01),
                    entry_price=entry_px,
                    current_price=float(sp.get("current_price", entry_px) or entry_px),
                    stop_loss=float(sp.get("stop_loss", 0.0) or 0.0),
                    take_profit=float(sp.get("take_profit", 0.0) or 0.0),
                    profit_usd=float(sp.get("profit_usd", 0.0) or 0.0),
                    ticket=int(sp.get("ticket", 0) or 0),
                    venue="paper_shadow",
                )
            )
        return specs

    def get_open_symbols(self) -> set[str]:
        """Return set of all symbol names (both canonical and raw broker names) currently open."""
        open_syms: set[str] = set()
        for p in self.get_open_positions():
            can = p.symbol.upper()
            open_syms.add(can)
            if self.feed_router.mt5.is_connected:
                raw = self.feed_router.mt5.resolve_symbol(can)
                if raw:
                    open_syms.add(raw.upper())
            if self.feed_router.ctrader.is_connected:
                ct_raw = self.feed_router.ctrader.resolve_symbol(can)
                if ct_raw:
                    open_syms.add(ct_raw.upper())
        for sp in self._shadow_positions:
            open_syms.add(str(sp.get("symbol", "")).upper())
        return open_syms

    def compute_compliant_lot_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        risk_usd: float,
        *,
        commission_per_lot_rt: float | None = None,
        account_currency: str | None = None,
    ) -> float:
        """Compute broker-compliant lot size using MT5 feed calculator or FX/commission-adjusted cTrader spec."""
        if self.feed_router.mt5.is_connected:
            return self.feed_router.mt5.compute_compliant_lot_size(
                symbol=symbol,
                entry_price=entry_price,
                stop_loss=stop_loss,
                risk_amount=risk_usd,
            )
        norm_sym = symbol.upper().strip()
        spec = self.feed_router.ctrader.get_contract_spec(symbol)
        if spec is None:
            clean_fb = norm_sym.split(".", 1)[0].split("_", 1)[0]
            mapper = getattr(getattr(self.feed_router, "mt5", None), "alias_mapper", None)
            ac_fb = mapper.classify_asset_class(clean_fb) if mapper is not None else AssetClass.FOREX
            if ac_fb == AssetClass.FOREX:
                fb_lot_sz = 100_000.0
            elif clean_fb.startswith("XAU"):
                fb_lot_sz = 100.0
            elif clean_fb.startswith("XAG"):
                fb_lot_sz = 5_000.0
            else:
                fb_lot_sz = 1.0
            spec = ContractSpec(
                symbol=norm_sym,
                digits=3 if "JPY" in clean_fb else (2 if fb_lot_sz <= 100.0 else 5),
                pip_size=0.01 if "JPY" in clean_fb else (0.1 if clean_fb.startswith("XAU") else 0.0001),
                point_size=0.001 if "JPY" in clean_fb else 0.00001,
                lot_size=fb_lot_sz,
                min_volume_lots=0.01,
                max_volume_lots=100.0,
                step_volume_lots=0.01,
                source="synthetic_fallback",
            )
        if spec is not None:
            sl_dist = abs(float(entry_price) - float(stop_loss))
            eff_risk = float(risk_usd)
            if sl_dist <= 1e-12 or eff_risk <= 0.0:
                return 0.0

            clean_sym = norm_sym.split(".", 1)[0].split("_", 1)[0]
            if clean_sym.startswith(("BTC", "ETH", "SOL", "XRP", "LTC", "ADA")):
                quote_ccy = "USD"
            elif clean_sym.endswith("USD"):
                quote_ccy = "USD"
            elif clean_sym.endswith(("JPY", "CHF", "CAD", "GBP", "EUR", "AUD", "NZD", "SGD", "HKD", "CNH")):
                quote_ccy = clean_sym[-3:]
            else:
                quote_ccy = "USD"

            acct_ccy = (account_currency or "USD").upper().strip()
            fx_fn = getattr(self.feed_router, "get_fx_conversion_rate", None)
            fx_q_to_acct = float(fx_fn(quote_ccy, acct_ccy)) if callable(fx_fn) else 1.0
            fx_usd_to_acct = float(fx_fn("USD", acct_ccy)) if callable(fx_fn) else 1.0

            sl_loss_per_lot = sl_dist * float(spec.lot_size) * max(1e-9, fx_q_to_acct)
            is_forex_or_metal = clean_sym.startswith(("XAU", "XAG")) or (len(clean_sym) == 6 and clean_sym.isalpha())
            if commission_per_lot_rt is not None:
                comm_usd = max(0.0, float(commission_per_lot_rt))
            elif getattr(spec, "commission_per_lot_rt", None) is not None:
                comm_usd = max(0.0, float(spec.commission_per_lot_rt))  # type: ignore[arg-type]
            else:
                comm_usd = 4.50 if is_forex_or_metal else 0.0

            all_in_risk = sl_loss_per_lot + (comm_usd * max(1e-9, fx_usd_to_acct))
            if all_in_risk <= 1e-9:
                return 0.0

            lots_raw = eff_risk / all_in_risk
            if lots_raw < float(spec.min_volume_lots):
                return 0.0

            step = float(spec.step_volume_lots) if float(spec.step_volume_lots) > 0.0 else 0.01
            steps = math.floor((lots_raw / step) + 1e-12)
            snapped = min(float(spec.max_volume_lots), steps * step)
            try:
                exp = Decimal(str(step)).normalize().as_tuple().exponent
                step_decimals = max(0, -int(exp)) if isinstance(exp, int) else 2
            except Exception:
                step_decimals = 2
            rounded = round(snapped, step_decimals)
            if rounded < float(spec.min_volume_lots):
                return 0.0
            return float(rounded)
        return 0.0

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
            req.relativeStopLoss = max(1, int(round(sl_dist * CTRADER_PRICE_SCALE)))
            req.relativeTakeProfit = max(1, int(round(tp_dist * CTRADER_PRICE_SCALE)))
            req.stopLoss = float(order.stop_loss)
            req.takeProfit = float(order.take_profit)

        return req

    @staticmethod
    def detect_mt5_filling_mode(symbol: str) -> int:
        """Detect the broker's supported MT5 filling mode bitmask (`FOK`, `IOC`, or `RETURN`)."""
        if mt5 is None:
            return 1
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
        resolved_sym = ct.resolve_symbol(norm_sym) or norm_sym
        sid = ct._symbol_name_to_id.get(resolved_sym) or ct._symbol_name_to_id.get(norm_sym)
        if ct.is_connected and sid is not None:
            try:
                spec = ct._contract_specs.get(resolved_sym) or ct._contract_specs.get(norm_sym)
                lot_units = spec.lot_size if spec else 100_000.0
                digits = int(spec.digits) if spec and getattr(spec, "digits", 0) else 5

                # Re-anchor MARKET SL/TP to live broker quote right before wire transmission
                send_order = order
                if order.order_type.upper() == "MARKET":
                    live_tick = ct.get_tick(norm_sym) or self.feed_router.get_tick(norm_sym)[0]
                    if live_tick is not None:
                        live_px = float(live_tick.ask if order.side == Side.BUY else live_tick.bid)
                        if live_px > 0.0:
                            planned_sl_dist = abs(float(order.entry_price) - float(order.stop_loss))
                            planned_tp_dist = abs(float(order.take_profit) - float(order.entry_price))
                            if planned_sl_dist > 0.0 and planned_tp_dist > 0.0:
                                new_sl = round(
                                    live_px - planned_sl_dist if order.side == Side.BUY else live_px + planned_sl_dist,
                                    digits,
                                )
                                new_tp = round(
                                    live_px + planned_tp_dist if order.side == Side.BUY else live_px - planned_tp_dist,
                                    digits,
                                )
                                send_order = ExecutionOrderRequest(
                                    symbol=order.symbol,
                                    side=order.side,
                                    lots=order.lots,
                                    entry_price=live_px,
                                    stop_loss=new_sl,
                                    take_profit=new_tp,
                                    risk_usd=order.risk_usd,
                                    order_type=order.order_type,
                                    comment=order.comment,
                                    force_paper_shadow=order.force_paper_shadow,
                                    predicted_net_r=order.predicted_net_r,
                                    regime=order.regime,
                                    ai_decision=order.ai_decision,
                                    ai_confidence=order.ai_confidence,
                                    ai_reason=order.ai_reason,
                                    dom_obi=order.dom_obi,
                                )

                req = self.build_ctrader_Order_req(
                    account_id=ct.account_id,
                    symbol_id=sid,
                    order=send_order,
                    lot_size_units=lot_units,
                )
                send_res: dict[str, object] = {
                    "ok": False,
                    "status": "REJECTED",
                    "error_code": "CTRADER_NO_EXECUTION_ACK",
                    "order_id": req.clientOrderId,
                }
                if hasattr(ct, "send_new_order"):
                    maybe_res = ct.send_new_order(req)
                    if isinstance(maybe_res, dict):
                        send_res = maybe_res
                elif ct._client is not None:
                    ct._client.send(req)
                    send_res = {"ok": True, "order_id": req.clientOrderId}

                if not bool(send_res.get("ok", False)):
                    err_detail = str(send_res.get("error_code") or "cTrader order rejected")
                    self.record_broker_event(
                        BrokerExecutionEvent(
                            timestamp_str=datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S"),
                            symbol=norm_sym,
                            side=send_order.side.value,
                            lots=send_order.lots,
                            price=send_order.entry_price,
                            stop_loss=send_order.stop_loss,
                            take_profit=send_order.take_profit,
                            venue="CTRADER",
                            status="REJECTED",
                            details=err_detail[:60],
                        )
                    )
                    return None

                exec_price = float(send_res.get("executed_price") or 0.0)
                if exec_price <= 0.0:
                    exec_price = (
                        send_order.entry_price + vwap_slippage_price
                        if send_order.side == Side.BUY
                        else send_order.entry_price - vwap_slippage_price
                    )
                if send_order.order_type.upper() == "MARKET":
                    p_sl = abs(float(send_order.entry_price) - float(send_order.stop_loss))
                    p_tp = abs(float(send_order.take_profit) - float(send_order.entry_price))
                    final_sl = round(exec_price - p_sl if send_order.side == Side.BUY else exec_price + p_sl, digits)
                    final_tp = round(exec_price + p_tp if send_order.side == Side.BUY else exec_price - p_tp, digits)
                else:
                    final_sl = send_order.stop_loss
                    final_tp = send_order.take_profit
                receipt_oid = str(send_res.get("order_id") or req.clientOrderId)
                self.record_broker_event(
                    BrokerExecutionEvent(
                        timestamp_str=datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S"),
                        symbol=norm_sym,
                        side=send_order.side.value,
                        lots=send_order.lots,
                        price=exec_price,
                        stop_loss=final_sl,
                        take_profit=final_tp,
                        venue="CTRADER",
                        status="FILLED",
                        details=f"Order {receipt_oid[:10]} | Sent [{mesh_topology}]",
                    )
                )
                return ExecutionReceipt(
                    order_id=receipt_oid,
                    symbol=norm_sym,
                    side=send_order.side,
                    lots=send_order.lots,
                    requested_entry=send_order.entry_price,
                    executed_price=exec_price,
                    stop_loss=final_sl,
                    take_profit=final_tp,
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
        if mt5 is not None and self.feed_router.mt5.is_connected:
            term_info = mt5.terminal_info()
            trade_allowed = bool(getattr(term_info, "trade_allowed", False)) if term_info is not None else False
            resolved = self.feed_router.mt5.resolve_symbol(norm_sym)
            if trade_allowed and resolved:
                info = mt5.symbol_info(resolved)
                digits = int(getattr(info, "digits", 5) or 5) if info is not None else 5
                detected_filling = self.detect_mt5_filling_mode(resolved)
                candidate_fillings = list(
                    dict.fromkeys(
                        [
                            detected_filling,
                            int(mt5.ORDER_FILLING_IOC),
                            int(mt5.ORDER_FILLING_FOK),
                            int(mt5.ORDER_FILLING_RETURN),
                        ]
                    )
                )
                tick = mt5.symbol_info_tick(resolved)
                live_px = (
                    float(tick.ask if order.side == Side.BUY else tick.bid)
                    if tick is not None
                    else float(order.entry_price)
                )
                is_market = order.order_type.upper() == "MARKET"
                req_price = live_px if is_market else float(order.entry_price)
                planned_sl_dist = abs(float(order.entry_price) - float(order.stop_loss))
                planned_tp_dist = abs(float(order.take_profit) - float(order.entry_price))
                if is_market and tick is not None and planned_sl_dist > 0.0 and planned_tp_dist > 0.0:
                    if order.side == Side.BUY:
                        send_sl = round(live_px - planned_sl_dist, digits)
                        send_tp = round(live_px + planned_tp_dist, digits)
                    else:
                        send_sl = round(live_px + planned_sl_dist, digits)
                        send_tp = round(live_px - planned_tp_dist, digits)
                else:
                    send_sl = round(float(order.stop_loss), digits)
                    send_tp = round(float(order.take_profit), digits)

                mt5_action = mt5.TRADE_ACTION_DEAL if is_market else mt5.TRADE_ACTION_PENDING
                if is_market:
                    mt5_order_type = mt5.ORDER_TYPE_BUY if order.side == Side.BUY else mt5.ORDER_TYPE_SELL
                else:
                    mt5_order_type = (
                        getattr(mt5, "ORDER_TYPE_BUY_LIMIT", 2)
                        if order.side == Side.BUY
                        else getattr(mt5, "ORDER_TYPE_SELL_LIMIT", 3)
                    )

                # Probe mt5.order_check across candidate filling modes so IOC/FOK/RETURN mismatches never drop silently
                chosen_filling = detected_filling
                order_check_fn = getattr(mt5, "order_check", None)
                if callable(order_check_fn):
                    for cand_fill in candidate_fillings:
                        probe_req = {
                            "action": mt5_action,
                            "symbol": resolved,
                            "volume": float(order.lots),
                            "type": mt5_order_type,
                            "price": round(req_price, digits),
                            "sl": send_sl,
                            "tp": send_tp,
                            "deviation": 20,
                            "magic": 202611,
                            "comment": order.comment,
                            "type_time": mt5.ORDER_TIME_GTC,
                            "type_filling": cand_fill,
                        }
                        chk = order_check_fn(probe_req)
                        if chk is not None and int(getattr(chk, "retcode", -1)) == 0:
                            chosen_filling = cand_fill
                            break

                mt5_req = {
                    "action": mt5_action,
                    "symbol": resolved,
                    "volume": float(order.lots),
                    "type": mt5_order_type,
                    "price": round(req_price, digits),
                    "sl": send_sl,
                    "tp": send_tp,
                    "deviation": 20,
                    "magic": 202611,
                    "comment": order.comment,
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": chosen_filling,
                }
                res = mt5.order_send(mt5_req)
                # Retry alternate filling mode once if broker returns INVALID_FILL (10030)
                if res is not None and int(getattr(res, "retcode", 0)) == 10030:
                    for alt_fill in candidate_fillings:
                        if alt_fill == chosen_filling:
                            continue
                        mt5_req["type_filling"] = alt_fill
                        res = mt5.order_send(mt5_req)
                        if res is not None and int(getattr(res, "retcode", 0)) == int(mt5.TRADE_RETCODE_DONE):
                            break

                if res is not None and int(getattr(res, "retcode", 0)) == int(mt5.TRADE_RETCODE_DONE):
                    fill_px = float(getattr(res, "price", 0.0) or live_px)
                    if is_market and planned_sl_dist > 0.0 and planned_tp_dist > 0.0:
                        final_sl = round(fill_px - planned_sl_dist, digits) if order.side == Side.BUY else round(fill_px + planned_sl_dist, digits)
                        final_tp = round(fill_px + planned_tp_dist, digits) if order.side == Side.BUY else round(fill_px - planned_tp_dist, digits)
                    else:
                        final_sl = send_sl
                        final_tp = send_tp
                    slip = (fill_px - order.entry_price) if order.side == Side.BUY else (order.entry_price - fill_px)
                    self.record_broker_event(
                        BrokerExecutionEvent(
                            timestamp_str=datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S"),
                            symbol=norm_sym,
                            side=order.side.value,
                            lots=order.lots,
                            price=fill_px,
                            stop_loss=final_sl,
                            take_profit=final_tp,
                            venue="MT5",
                            status="FILLED",
                            details=f"Ticket #{res.order} | Retcode {res.retcode} (Done)",
                        )
                    )
                    return ExecutionReceipt(
                        order_id=str(getattr(res, "order", uuid.uuid4().hex[:10])),
                        symbol=norm_sym,
                        side=order.side,
                        lots=order.lots,
                        requested_entry=order.entry_price,
                        executed_price=fill_px,
                        stop_loss=final_sl,
                        take_profit=final_tp,
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

    def prune_closed_shadow_positions(self) -> int:
        """Remove Paper-Shadow positions already settled in `WalkForwardLedgerDB` so Circuit Breaker C never deadlocks."""
        if self.ledger_db is None or not self._shadow_positions:
            return 0
        oids = [str(sp.get("order_id", "")) for sp in self._shadow_positions if sp.get("order_id")]
        closed_ids = self.ledger_db.get_closed_order_ids(oids)
        if not closed_ids:
            return 0
        before = len(self._shadow_positions)
        self._shadow_positions = [
            sp for sp in self._shadow_positions if str(sp.get("order_id", "")) not in closed_ids
        ]
        return before - len(self._shadow_positions)

    def _persist_trade_open(self, order: ExecutionOrderRequest, receipt: ExecutionReceipt) -> None:
        """Persist every confirmed FILLED or PAPER_SHADOW_FILLED execution into `trade_ledger` (`is_closed = 0`)."""
        if self.ledger_db is None:
            return
        if receipt.status not in ("FILLED", "PAPER_SHADOW_FILLED"):
            return
        sl_dist = abs(float(receipt.executed_price or order.entry_price) - float(receipt.stop_loss))
        targets = receipt.child_receipts if receipt.child_receipts else (receipt,)
        for rec in targets:
            if rec.status not in ("FILLED", "PAPER_SHADOW_FILLED"):
                continue
            exec_mode = (
                SymbolExecutionMode.PAPER_SHADOW
                if (rec.status == "PAPER_SHADOW_FILLED" or rec.venue == "PAPER_SHADOW" or order.force_paper_shadow)
                else SymbolExecutionMode.ACTIVE
            )
            try:
                self.ledger_db.record_trade_open(
                    order_id=rec.order_id,
                    symbol=rec.symbol,
                    side=rec.side.value,
                    lots=rec.lots,
                    entry_price=rec.executed_price or order.entry_price,
                    stop_loss=rec.stop_loss,
                    take_profit=rec.take_profit,
                    initial_one_r=sl_dist if sl_dist > 0.0 else abs(float(order.entry_price) - float(order.stop_loss)),
                    predicted_net_r=order.predicted_net_r,
                    regime=order.regime,
                    venue=rec.winning_venue or rec.venue,
                    execution_mode=exec_mode,
                    dom_obi=order.dom_obi,
                    ai_decision=order.ai_decision,
                    ai_confidence=order.ai_confidence,
                    ai_reason=order.ai_reason,
                    realized_slippage_r=rec.realized_slippage_r,
                    mesh_topology=rec.mesh_topology,
                    sor_pips_saved=rec.sor_pips_saved,
                    sor_usd_saved=rec.sor_usd_saved,
                    timestamp_utc=rec.timestamp_utc,
                )
            except Exception as exc:
                logger.debug("record_trade_open skipped for %s: %s", rec.order_id, exc)

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
        """Validate all 6 circuit breakers, route order via Apex Hydra SOR, and persist open trade."""
        receipt = self._execute_order_routed(
            order=order,
            closed_pnl_today_usd=closed_pnl_today_usd,
            floating_pnl_usd=floating_pnl_usd,
            open_positions=open_positions,
            current_spread_price=current_spread_price,
            vwap_slippage_price=vwap_slippage_price,
            dom_analytics=dom_analytics,
        )
        self._persist_trade_open(order, receipt)
        return receipt

    def _get_min_lot(self, symbol: str) -> float:
        """Resolve minimum tradable volume in lots across connected venues."""
        norm_sym = symbol.upper().strip()
        vals: list[float] = []
        if self.feed_router.mt5.is_connected and mt5 is not None:
            try:
                res = self.feed_router.mt5.resolve_symbol(norm_sym) or norm_sym
                info = mt5.symbol_info(res)
                vmin = getattr(info, "volume_min", None) if info is not None else None
                if isinstance(vmin, (int, float)) and not isinstance(vmin, bool) and float(vmin) > 0.0:
                    vals.append(float(vmin))
            except Exception as exc:
                logger.debug("MT5 min_lot probe skipped for %s: %s", norm_sym, exc)
        if hasattr(self.feed_router.ctrader, "_contract_specs"):
            ct_spec = self.feed_router.ctrader._contract_specs.get(norm_sym)
            if ct_spec is not None and float(ct_spec.min_volume_lots) > 0.0:
                vals.append(float(ct_spec.min_volume_lots))
        return max(vals) if vals else 0.01

    def _compensate_close_mt5_leg(self, order_id: str, symbol: str, side: Side, lots: float) -> bool:
        """Close an exposed MT5 split leg if the opposite split leg failed to fill."""
        if mt5 is None or not self.feed_router.mt5.is_connected:
            return False
        resolved = self.feed_router.mt5.resolve_symbol(symbol) or symbol
        tick = mt5.symbol_info_tick(resolved)
        close_px = float(tick.bid if side == Side.BUY else tick.ask) if tick is not None else 0.0
        pos_id = int(order_id) if order_id.isdigit() else 0
        for fill_mode in dict.fromkeys(
            [
                self.detect_mt5_filling_mode(resolved),
                int(mt5.ORDER_FILLING_IOC),
                int(mt5.ORDER_FILLING_FOK),
                int(mt5.ORDER_FILLING_RETURN),
            ]
        ):
            req = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": pos_id,
                "symbol": resolved,
                "volume": float(lots),
                "type": mt5.ORDER_TYPE_SELL if side == Side.BUY else mt5.ORDER_TYPE_BUY,
                "price": close_px,
                "deviation": 50,
                "magic": 202611,
                "comment": "APEX_SPLIT_ROLLBACK",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": fill_mode,
            }
            res = mt5.order_send(req)
            if res is not None and int(getattr(res, "retcode", 0)) == int(mt5.TRADE_RETCODE_DONE):
                return True
        return False

    def _compensate_close_ctrader_leg(self, symbol: str, order_id: str, lots: float) -> bool:
        """Close an exposed cTrader split leg if the opposite split leg failed to fill."""
        ct = self.feed_router.ctrader
        if not ct.is_connected or not hasattr(ct, "close_position"):
            return False
        pos_id = 0
        for pos in ct.get_open_positions() if hasattr(ct, "get_open_positions") else ():
            if str(pos.get("symbol", "")).upper() == symbol.upper():
                pos_id = int(pos.get("position_id", 0) or 0)
                if pos_id > 0:
                    break
        if pos_id <= 0 and str(order_id).isdigit():
            pos_id = int(order_id)
        if pos_id <= 0:
            return False
        res = ct.close_position(position_id=pos_id, volume_lots=float(lots), symbol=symbol)
        return bool(isinstance(res, dict) and res.get("ok"))

    def _execute_order_routed(
        self,
        order: ExecutionOrderRequest,
        closed_pnl_today_usd: float = 0.0,
        floating_pnl_usd: float = 0.0,
        open_positions: Sequence[PositionRiskSpec] = (),
        current_spread_price: float = 0.0,
        vwap_slippage_price: float = 0.0,
        dom_analytics: DOMAnalytics | None = None,
    ) -> ExecutionReceipt:
        """Internal routing implementation called by `execute_order`."""
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

        # Reject immediately when online venues fail margin / AutoTrading checks (never bypass live safety)
        if topology != ResolvedMeshTopology.OFFLINE and nbbo.winning_venue.upper() == "NONE" and not order.force_paper_shadow:
            return ExecutionReceipt(
                order_id=f"blocked_{uuid.uuid4().hex[:10]}",
                symbol=order.symbol.upper(),
                side=order.side,
                lots=order.lots,
                requested_entry=order.entry_price,
                executed_price=0.0,
                stop_loss=order.stop_loss,
                take_profit=order.take_profit,
                realized_slippage_price=0.0,
                realized_slippage_r=0.0,
                venue="NONE",
                status="REJECTED",
                timestamp_utc=now_utc,
                breaker_report=report,
                message="Rejected: no broker venue passed AutoTrading & margin checks (winning_venue=NONE).",
                winning_venue="NONE",
                mesh_topology=topology.value,
            )

        # 1. SPLIT_EXECUTION: 50/50 parallel risk split only when order.lots >= 2.0 * min_lot
        min_lot = self._get_min_lot(order.symbol)
        if topology == ResolvedMeshTopology.SPLIT_EXECUTION and float(order.lots) >= 2.0 * min_lot:
            half_lots = round(float(order.lots) / 2.0, 2)
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
            live_ct = self._try_execute_ctrader(
                child_order,
                report,
                vwap_slippage_price,
                sl_dist,
                now_utc,
                winning_venue="ctrader",
                mesh_topology=topology.value,
            )
            live_mt5 = self._try_execute_mt5(
                child_order,
                report,
                vwap_slippage_price,
                sl_dist,
                now_utc,
                winning_venue="mt5",
                mesh_topology=topology.value,
            )

            # Never synthesize a shadow leg 2 against a live leg 1: compensate-close immediately
            if live_ct is not None and live_mt5 is None:
                rolled_back = self._compensate_close_ctrader_leg(order.symbol, live_ct.order_id, live_ct.lots)
                return ExecutionReceipt(
                    order_id=f"split_abort_{live_ct.order_id}",
                    symbol=order.symbol.upper(),
                    side=order.side,
                    lots=order.lots,
                    requested_entry=order.entry_price,
                    executed_price=0.0,
                    stop_loss=order.stop_loss,
                    take_profit=order.take_profit,
                    realized_slippage_price=0.0,
                    realized_slippage_r=0.0,
                    venue="SPLIT_CTRADER_MT5",
                    status="REJECTED",
                    timestamp_utc=now_utc,
                    breaker_report=report,
                    message=f"SPLIT_EXECUTION aborted: MT5 leg 2 failed; cTrader leg 1 close={rolled_back}.",
                    winning_venue="NONE",
                    mesh_topology=topology.value,
                    child_receipts=(live_ct,),
                )
            if live_mt5 is not None and live_ct is None:
                rolled_back = self._compensate_close_mt5_leg(live_mt5.order_id, order.symbol, live_mt5.side, live_mt5.lots)
                return ExecutionReceipt(
                    order_id=f"split_abort_{live_mt5.order_id}",
                    symbol=order.symbol.upper(),
                    side=order.side,
                    lots=order.lots,
                    requested_entry=order.entry_price,
                    executed_price=0.0,
                    stop_loss=order.stop_loss,
                    take_profit=order.take_profit,
                    realized_slippage_price=0.0,
                    realized_slippage_r=0.0,
                    venue="SPLIT_CTRADER_MT5",
                    status="REJECTED",
                    timestamp_utc=now_utc,
                    breaker_report=report,
                    message=f"SPLIT_EXECUTION aborted: cTrader leg 1 failed; MT5 leg 2 close={rolled_back}.",
                    winning_venue="NONE",
                    mesh_topology=topology.value,
                    child_receipts=(live_mt5,),
                )

            if live_ct is not None and live_mt5 is not None:
                avg_px = (live_ct.executed_price + live_mt5.executed_price) / 2.0
                parent_id = f"split_{live_ct.order_id}_{live_mt5.order_id}"
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
                    status="FILLED",
                    timestamp_utc=now_utc,
                    breaker_report=report,
                    message=f"Executed 50/50 SPLIT_EXECUTION across cTrader ({half_lots} lots) + MT5 ({half_lots} lots).",
                    winning_venue="SPLIT_CTRADER_MT5",
                    mesh_topology=topology.value,
                    sor_pips_saved=pips_saved,
                    sor_usd_saved=usd_saved,
                    child_receipts=(live_ct, live_mt5),
                )

            # Both brokers offline -> coordinated Paper-Shadow simulation (without recording live SOR savings)
            leg_ct = self._execute_paper_shadow(
                child_order,
                report,
                vwap_slippage_price,
                sl_dist,
                "Split leg 1 (cTrader paper-shadow fallback)",
                winning_venue="ctrader",
                mesh_topology=topology.value,
            )
            leg_mt5 = self._execute_paper_shadow(
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
                status="PAPER_SHADOW_FILLED",
                timestamp_utc=now_utc,
                breaker_report=report,
                message=f"Simulated 50/50 SPLIT_EXECUTION Paper-Shadow ({half_lots} lots + {half_lots} lots).",
                winning_venue="SPLIT_CTRADER_MT5",
                mesh_topology=topology.value,
                sor_pips_saved=0.0,
                sor_usd_saved=0.0,
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

        # 4. ACTIVE_ACTIVE_SOR (or single-venue fallback when SPLIT_EXECUTION lot size < 2*min_lot)
        winner = nbbo.winning_venue.lower()
        if winner not in ("ctrader", "mt5", "none"):
            if nbbo.ctrader_quote and nbbo.mt5_quote:
                winner = "ctrader" if nbbo.ctrader_quote.effective_cost_pips <= nbbo.mt5_quote.effective_cost_pips else "mt5"
            elif nbbo.mt5_quote:
                winner = "mt5"
            else:
                winner = "ctrader"

        cand_pips = max(0.0, float(nbbo.pips_saved))
        cand_usd = max(0.0, float(nbbo.usd_saved_per_lot) * float(order.lots))

        if winner == "ctrader":
            rec = self._try_execute_ctrader(
                order,
                report,
                vwap_slippage_price,
                sl_dist,
                now_utc,
                winning_venue="ctrader",
                mesh_topology=topology.value,
                sor_pips_saved=cand_pips,
                sor_usd_saved=cand_usd,
            ) or self._try_execute_mt5(
                order,
                report,
                vwap_slippage_price,
                sl_dist,
                now_utc,
                winning_venue="ctrader",
                mesh_topology=topology.value,
                sor_pips_saved=cand_pips,
                sor_usd_saved=cand_usd,
            )
            if rec is not None:
                if rec.status == "FILLED":
                    self._record_sor_savings(order_id=rec.order_id, symbol=order.symbol, lots=order.lots, nbbo=nbbo)
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
                sor_pips_saved=cand_pips,
                sor_usd_saved=cand_usd,
            ) or self._try_execute_ctrader(
                order,
                report,
                vwap_slippage_price,
                sl_dist,
                now_utc,
                winning_venue="mt5",
                mesh_topology=topology.value,
                sor_pips_saved=cand_pips,
                sor_usd_saved=cand_usd,
            )
            if rec is not None:
                if rec.status == "FILLED":
                    self._record_sor_savings(order_id=rec.order_id, symbol=order.symbol, lots=order.lots, nbbo=nbbo)
                return rec

        # Seamless Paper-Shadow fallback WITHOUT claiming live SOR execution savings
        return self._execute_paper_shadow(
            order,
            report,
            vwap_slippage_price,
            sl_dist,
            f"Seamless Paper-Shadow fallback [{topology.value} -> winner={winner.upper()}]",
            winning_venue=winner,
            mesh_topology=topology.value,
            sor_pips_saved=0.0,
            sor_usd_saved=0.0,
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

        if mt5 is not None and self.feed_router.mt5.is_connected:
            positions = mt5.positions_get()
            if positions:
                for pos in positions:
                    sym = str(pos.symbol)
                    vol = float(pos.volume)
                    is_buy = int(pos.type) == int(mt5.ORDER_TYPE_BUY)
                    tick = mt5.symbol_info_tick(sym)
                    close_px = float(tick.bid if is_buy else tick.ask) if tick else float(pos.price_current)
                    for fill_mode in dict.fromkeys(
                        [
                            self.detect_mt5_filling_mode(sym),
                            int(mt5.ORDER_FILLING_IOC),
                            int(mt5.ORDER_FILLING_FOK),
                            int(mt5.ORDER_FILLING_RETURN),
                        ]
                    ):
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
                            "type_filling": fill_mode,
                        }
                        res = mt5.order_send(req)
                        if res is not None and int(getattr(res, "retcode", 0)) == int(mt5.TRADE_RETCODE_DONE):
                            closed_mt5 += 1
                            break

        if self.feed_router.ctrader.is_connected and hasattr(self.feed_router.ctrader, "get_open_positions"):
            for cpos in list(self.feed_router.ctrader.get_open_positions()):
                pid = int(cpos.get("position_id", 0) or 0)
                if pid <= 0:
                    continue
                close_res = self.feed_router.ctrader.close_position(
                    position_id=pid,
                    volume_lots=float(cpos.get("lots", 0.01) or 0.01),
                    symbol=str(cpos.get("symbol", "")),
                )
                if isinstance(close_res, dict) and close_res.get("ok"):
                    closed_ctrader += 1

        return {
            "mt5_closed": closed_mt5,
            "ctrader_closed": closed_ctrader,
            "shadow_closed": closed_shadow,
        }
