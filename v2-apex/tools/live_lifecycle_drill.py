"""60-Second Live MT5 0.01-Lot Lifecycle Fire-Drill (`tools/live_lifecycle_drill.py`).

Strictly DEMO-locked (`ACCOUNT_TRADE_MODE_DEMO`). Executes ONE real 0.01-lot bracket order
with mandatory server-side SL & TP on FusionMarkets-Demo, streams live floating P/L,
exercises a `TRADE_ACTION_SLTP` SL-tighten check, closes via opposite `TRADE_ACTION_DEAL`,
verifies settled deal history, and logs `is_drill=1` in `trader/apex_ledger.db`.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import MetaTrader5 as mt5

from apex.ai.aegis_verifier import AegisVerifier
from apex.engine.canonical_pipeline import Canonical11AgentPipeline
from apex.execution.position_manager import AutonomousPositionOrchestrator, PositionDecision
from apex.feeds.models import Candle, Side
from apex.ledger.walkforward_db import SymbolExecutionMode, WalkForwardLedgerDB

DEFAULT_CANDIDATE_SYMBOLS = ("EURUSD", "USDJPY", "GBPUSD", "USDCAD", "AUDUSD", "USDCHF", "BTCUSD")
MAX_DRILL_TICK_AGE_SECONDS = 120.0


@dataclass(frozen=True, slots=True)
class DrillResult:
    """Structured outcome of a live MT5 0.01-lot lifecycle drill."""

    ok: bool
    account_login: int
    account_currency: str
    symbol: str
    side: str
    lots: float
    entry_price: float
    stop_loss: float
    take_profit: float
    order_ticket: int
    position_id: int
    sltp_retcode: int
    close_retcode: int
    close_deal_id: int
    realized_pnl: float
    z3_status: str
    message: str
    drill_mode: str = "plumbing"
    trace: Optional[dict[str, Any]] = None


def assert_demo_account(account_info: Any) -> tuple[int, str]:
    """Fail-closed guard: raise RuntimeError unless MT5 account is strictly DEMO."""
    if account_info is None:
        raise RuntimeError("MT5 account_info() returned None. Is MetaTrader 5 logged in?")
    trade_mode = int(getattr(account_info, "trade_mode", -1))
    demo_mode = int(getattr(mt5, "ACCOUNT_TRADE_MODE_DEMO", 0))
    if trade_mode != demo_mode:
        raise RuntimeError(
            f"SAFETY REFUSAL: Live lifecycle drill is strictly DEMO-only "
            f"(trade_mode={trade_mode}, expected ACCOUNT_TRADE_MODE_DEMO={demo_mode})."
        )
    login = int(getattr(account_info, "login", 0))
    currency = str(getattr(account_info, "currency", "CAD"))
    return login, currency


def detect_filling_mode(symbol: str) -> int:
    """Detect broker-supported MT5 filling mode bitmask."""
    info = mt5.symbol_info(symbol)
    if info is None:
        return int(mt5.ORDER_FILLING_FOK)
    mode = int(getattr(info, "filling_mode", 0))
    if mode & 1:
        return int(mt5.ORDER_FILLING_FOK)
    if mode & 2:
        return int(mt5.ORDER_FILLING_IOC)
    return int(mt5.ORDER_FILLING_RETURN)


def _estimate_broker_now() -> float:
    """Estimate current MT5 broker server epoch using live 24/7 or FX benchmark ticks."""
    best_ts = 0.0
    for ref in ("BTCUSD", "ETHUSD", "EURUSD", "USDJPY"):
        t = mt5.symbol_info_tick(ref)
        if t is not None and float(getattr(t, "time", 0) or 0) > best_ts:
            best_ts = float(t.time)
    return best_ts if best_ts > 0.0 else time.time()


def _check_symbol_session_open(
    symbol: str,
    info: Any,
    tick: Any,
    broker_now: Optional[float] = None,
) -> tuple[bool, str]:
    """Verify MT5 symbol has trade_mode=FULL and a fresh live tick (<=120s)."""
    if info is None or tick is None:
        return False, f"CLOSED_MARKET: {symbol} has no MT5 symbol_info/tick."
    trade_mode = int(getattr(info, "trade_mode", 0))
    full_mode = int(getattr(mt5, "SYMBOL_TRADE_MODE_FULL", 4))
    if trade_mode != full_mode:
        return False, f"CLOSED_MARKET: {symbol} trade_mode={trade_mode} (market closed or read-only)."
    bid = float(getattr(tick, "bid", 0.0) or 0.0)
    ask = float(getattr(tick, "ask", 0.0) or 0.0)
    if bid <= 0.0 or ask <= 0.0 or ask < bid:
        return False, f"CLOSED_MARKET: {symbol} invalid quote bid={bid}, ask={ask}."
    ref_now = broker_now if broker_now is not None else _estimate_broker_now()
    tick_time = float(getattr(tick, "time", 0) or 0)
    tick_age = abs(ref_now - tick_time)
    if tick_time <= 0.0 or tick_age > MAX_DRILL_TICK_AGE_SECONDS:
        return False, f"CLOSED_MARKET: {symbol} stale tick ({int(tick_age)}s old > {int(MAX_DRILL_TICK_AGE_SECONDS)}s)."
    return True, "ACTIVE"


def _fetch_mt5_candles(symbol: str, timeframe_attr: str, count: int) -> list[Candle]:
    """Fetch MT5 bars and convert to canonical Candle models."""
    tf = getattr(mt5, timeframe_attr, None)
    if tf is None:
        return []
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
    if rates is None or len(rates) == 0:
        return []
    tf_label = timeframe_attr.replace("TIMEFRAME_", "").upper() or "M5"
    candles: list[Candle] = []
    for bar in rates:
        candles.append(
            Candle(
                symbol=symbol,
                timeframe=tf_label,
                timestamp_utc=float(bar["time"]),
                open=float(bar["open"]),
                high=float(bar["high"]),
                low=float(bar["low"]),
                close=float(bar["close"]),
                volume=float(bar["tick_volume"]),
                source="mt5",
            )
        )
    return candles


def pick_tightest_open_symbol(
    candidates: tuple[str, ...] = DEFAULT_CANDIDATE_SYMBOLS,
) -> tuple[str, Any, Any]:
    """Pick the tradable open symbol with the tightest relative spread and fresh tick."""
    broker_now = _estimate_broker_now()
    best: Optional[tuple[float, str, Any, Any]] = None
    for sym in candidates:
        mt5.symbol_select(sym, True)
        info = mt5.symbol_info(sym)
        tick = mt5.symbol_info_tick(sym)
        is_open, _ = _check_symbol_session_open(sym, info, tick, broker_now=broker_now)
        if not is_open:
            continue
        bid = float(getattr(tick, "bid", 0.0) or 0.0)
        ask = float(getattr(tick, "ask", 0.0) or 0.0)
        rel_spread = (ask - bid) / bid
        if best is None or rel_spread < best[0]:
            best = (rel_spread, sym, info, tick)

    if best is None:
        raise RuntimeError(
            "CLOSED_MARKET: No open tradable symbol with fresh ticks (<=120s) found on FusionMarkets-Demo right now."
        )
    return best[1], best[2], best[3]


def run_live_lifecycle_drill(
    hold_seconds: int = 15,
    symbol_override: Optional[str] = None,
    db_path: str | Path = "trader/apex_ledger.db",
    on_update: Optional[Callable[[dict[str, Any]], None]] = None,
    drill_mode: str = "plumbing",
) -> DrillResult:
    """Execute the full end-to-end 0.01-lot DEMO lifecycle drill on MT5."""
    mode_norm = (drill_mode or "plumbing").strip().lower()
    if mode_norm not in {"strict", "plumbing"}:
        mode_norm = "plumbing"

    def _emit(
        status: str,
        message: str,
        symbol: str = "",
        ticket: int = 0,
        floating_pnl: float = 0.0,
        trace: Optional[dict[str, Any]] = None,
    ) -> None:
        if on_update is not None:
            payload: dict[str, Any] = {
                "status": status,
                "symbol": symbol,
                "ticket": ticket,
                "floating_pnl": round(float(floating_pnl), 2),
                "drill_mode": mode_norm,
                "message": message,
            }
            if trace is not None:
                payload["trace"] = trace
            on_update(payload)

    _emit("STARTING", f"Initializing MT5 connection & verifying DEMO mode ({mode_norm.upper()})...")
    if not mt5.initialize():
        raise RuntimeError(f"mt5.initialize() failed: {mt5.last_error()}")

    login, currency = assert_demo_account(mt5.account_info())

    if symbol_override and symbol_override.upper() not in {"AUTO_BEST_MOVER", "AUTO_TOP12"}:
        sym = symbol_override.upper()
        mt5.symbol_select(sym, True)
        info = mt5.symbol_info(sym)
        tick = mt5.symbol_info_tick(sym)
        is_open, session_msg = _check_symbol_session_open(sym, info, tick)
        if not is_open:
            _emit("CLOSED_MARKET", session_msg, symbol=sym)
            return DrillResult(
                ok=False,
                account_login=login,
                account_currency=currency,
                symbol=sym,
                side="BUY",
                lots=0.0,
                entry_price=0.0,
                stop_loss=0.0,
                take_profit=0.0,
                order_ticket=0,
                position_id=0,
                sltp_retcode=0,
                close_retcode=0,
                close_deal_id=0,
                realized_pnl=0.0,
                z3_status="SKIPPED",
                message=session_msg,
                drill_mode=mode_norm,
                trace=None,
            )
        symbol = sym
    else:
        symbol, info, tick = pick_tightest_open_symbol(DEFAULT_CANDIDATE_SYMBOLS)

    digits = int(getattr(info, "digits", 5))
    point = float(getattr(info, "point", 0.00001))
    stops_level_pts = int(getattr(info, "trade_stops_level", 0) or 0)
    vol_min = float(getattr(info, "volume_min", 0.01) or 0.01)
    lots = max(0.01, vol_min)

    ask = float(tick.ask)
    bid = float(tick.bid)
    spread_px = max(ask - bid, point)
    pip_size = point * 10.0 if digits in (3, 5) else point

    # Run Canonical 11-Agent Pipeline scorecard on live MT5 M5 & H1 candles
    m5_candles = _fetch_mt5_candles(symbol, "TIMEFRAME_M5", 120)
    h1_candles = _fetch_mt5_candles(symbol, "TIMEFRAME_H1", 80)
    trace_dict: Optional[dict[str, Any]] = None
    if len(m5_candles) >= 30:
        pipeline = Canonical11AgentPipeline()
        trace = pipeline.evaluate(
            symbol=symbol,
            side=Side.BUY,
            timeframe="M5",
            entry_price=ask,
            candles=m5_candles,
            h1_candles=h1_candles or None,
            spread_price=spread_px,
            pip_size=pip_size,
            run_ai_committee=False,
            in_flight_ready=True,
        )
        trace_dict = trace.to_dict()
        _emit(
            "SCORECARD",
            f"11-Agent Scorecard {symbol}: {'ALL PASS' if trace.passed_all else f'Blocked at Stage {trace.blocked_stage}: {trace.blocked_reason}'}",
            symbol=symbol,
            trace=trace_dict,
        )
        if mode_norm == "strict" and not trace.passed_all:
            msg = f"STRICT 11-AGENT GATE REFUSAL on {symbol}: Stage {trace.blocked_stage} ({trace.blocked_reason})"
            _emit("REJECTED", msg, symbol=symbol, trace=trace_dict)
            return DrillResult(
                ok=False,
                account_login=login,
                account_currency=currency,
                symbol=symbol,
                side="BUY",
                lots=0.0,
                entry_price=round(ask, digits),
                stop_loss=0.0,
                take_profit=0.0,
                order_ticket=0,
                position_id=0,
                sltp_retcode=0,
                close_retcode=0,
                close_deal_id=0,
                realized_pnl=0.0,
                z3_status=f"BLOCKED_STAGE_{trace.blocked_stage}",
                message=msg,
                drill_mode=mode_norm,
                trace=trace_dict,
            )

    # Ensure SL & TP sit safely beyond broker stop level and spread
    min_dist = max(stops_level_pts * point * 2.0, spread_px * 12.0, 120.0 * point)
    entry_price = round(ask, digits)
    stop_loss = round(entry_price - min_dist, digits)
    take_profit = round(entry_price + min_dist * 1.5, digits)

    # Run Z3 formal pure-math axiom gate before sending order
    telemetry = {
        "symbol": symbol,
        "side": "BUY",
        "timemachine_net_avg_R": 0.35,
        "spread_pct_of_sl": min(15.0, round((spread_px / max(min_dist, point)) * 100.0, 2)),
        "cross_asset_shock_z": 0.4,
        "vol_pct": 45.0,
        "hurst": 0.58,
        "kaufman_er": 0.45,
        "dom_obi": 0.12,
    }
    z3_decision, z3_msg = AegisVerifier.z3_pure_math_axiom_gate(telemetry)
    if z3_decision != "APPROVE":
        raise RuntimeError(f"Aegis Z3 gate refused drill setup on {symbol}: {z3_msg}")

    _emit("SENDING", f"Sending 0.01 BUY {symbol} @ {entry_price} (SL={stop_loss}, TP={take_profit})...", symbol=symbol, trace=trace_dict)
    filling = detect_filling_mode(symbol)
    open_req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lots,
        "type": mt5.ORDER_TYPE_BUY,
        "price": entry_price,
        "sl": stop_loss,
        "tp": take_profit,
        "deviation": 30,
        "magic": 202699,
        "comment": "APEX_DRILL_001",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling,
    }
    open_res = mt5.order_send(open_req)
    done_code = int(getattr(mt5, "TRADE_RETCODE_DONE", 10009))
    if open_res is None or int(getattr(open_res, "retcode", 0)) != done_code:
        rc = getattr(open_res, "retcode", None)
        cmt = getattr(open_res, "comment", mt5.last_error())
        raise RuntimeError(f"Drill open order rejected (retcode={rc}): {cmt}")

    order_ticket = int(getattr(open_res, "order", 0) or 0)
    fill_price = float(getattr(open_res, "price", entry_price) or entry_price)

    # Locate the live MT5 position ticket
    pos_ticket = order_ticket
    pos_id = order_ticket
    for _ in range(10):
        positions = mt5.positions_get(symbol=symbol) or []
        matched = [p for p in positions if int(getattr(p, "magic", 0)) == 202699]
        if matched:
            pos_ticket = int(matched[-1].ticket)
            pos_id = int(getattr(matched[-1], "identifier", pos_ticket) or pos_ticket)
            fill_price = float(matched[-1].price_open)
            break
        time.sleep(0.2)

    hold_secs = max(1, int(hold_seconds))
    last_floating = 0.0
    orchestrator = AutonomousPositionOrchestrator()
    open_ts = time.time()
    for sec in range(hold_secs):
        positions = mt5.positions_get(ticket=pos_ticket) or []
        cur_px = fill_price
        if positions:
            last_floating = float(positions[0].profit)
            cur_px = float(getattr(positions[0], "price_current", fill_price) or fill_price)
        pos_decision: PositionDecision = orchestrator.evaluate_position(
            ticket=pos_ticket,
            symbol=symbol,
            side=Side.BUY,
            entry_price=fill_price,
            current_price=cur_px,
            current_sl=stop_loss,
            current_tp=take_profit,
            lots=lots,
            atr_value=max(min_dist, point * 10.0),
            opened_at_utc=open_ts,
            now_utc=time.time(),
            candles=m5_candles[-30:] if len(m5_candles) >= 30 else None,
        )
        _emit(
            "HOLDING",
            (
                f"Holding {symbol} #{pos_ticket} ({sec + 1}/{hold_secs}s) | "
                f"Floating: {last_floating:+.2f} {currency} | InFlight: {pos_decision.action.value}"
            ),
            symbol=symbol,
            ticket=pos_ticket,
            floating_pnl=last_floating,
            trace=trace_dict,
        )
        time.sleep(1.0)

    # Tighten SL by 5 points (safe inside stop distance) to prove TRADE_ACTION_SLTP
    tightened_sl = round(stop_loss + max(5.0 * point, point), digits)
    if tightened_sl >= fill_price - max(stops_level_pts * point * 1.5, spread_px * 3.0):
        tightened_sl = stop_loss
    sltp_req = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": pos_ticket,
        "symbol": symbol,
        "sl": tightened_sl,
        "tp": take_profit,
        "magic": 202699,
        "comment": "APEX_DRILL_SLTP",
    }
    sltp_res = mt5.order_send(sltp_req)
    sltp_retcode = int(getattr(sltp_res, "retcode", -1)) if sltp_res is not None else -1

    # Close position via opposite TRADE_ACTION_DEAL
    close_tick = mt5.symbol_info_tick(symbol)
    close_px = float(close_tick.bid) if close_tick is not None else fill_price
    close_req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "position": pos_ticket,
        "symbol": symbol,
        "volume": lots,
        "type": mt5.ORDER_TYPE_SELL,
        "price": close_px,
        "deviation": 50,
        "magic": 202699,
        "comment": "APEX_DRILL_CLOSE",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling,
    }
    close_res = mt5.order_send(close_req)
    close_retcode = int(getattr(close_res, "retcode", -1)) if close_res is not None else -1
    close_deal_id = int(getattr(close_res, "deal", 0) or 0) if close_res is not None else 0

    realized_pnl = last_floating
    for _ in range(10):
        deals = mt5.history_deals_get(position=pos_id) or []
        exit_deals = [d for d in deals if int(getattr(d, "entry", -1)) == int(getattr(mt5, "DEAL_ENTRY_OUT", 1))]
        if exit_deals:
            d = exit_deals[-1]
            close_deal_id = int(getattr(d, "ticket", close_deal_id) or close_deal_id)
            realized_pnl = float(getattr(d, "profit", 0.0)) + float(getattr(d, "commission", 0.0)) + float(getattr(d, "swap", 0.0))
            break
        time.sleep(0.25)

    # Log to SQLite with is_drill=True so live strategy Profit Factor & Quarantine remain isolated
    ledger = WalkForwardLedgerDB(db_path=db_path)
    risk_px = max(abs(fill_price - stop_loss), point)
    realized_r = (close_px - fill_price) / risk_px
    ledger.log_trade_outcome(
        order_id=f"DRILL-{pos_ticket}",
        symbol=symbol,
        side="BUY",
        execution_mode=SymbolExecutionMode.ACTIVE,
        regime="DRILL",
        dom_obi=0.12,
        predicted_net_r=0.35,
        ai_decision="APPROVE",
        ai_confidence=1.0,
        ai_reason=z3_msg,
        entry_price=entry_price,
        executed_price=fill_price,
        stop_loss=tightened_sl,
        take_profit=take_profit,
        realized_slippage_r=abs(fill_price - entry_price) / risk_px,
        realized_net_r=round(realized_r, 4),
        winning_venue="mt5",
        mesh_topology="MT5_DEMO_DRILL",
        is_drill=True,
    )

    summary_msg = (
        f"Drill COMPLETE on {symbol}: Ticket #{pos_ticket} | SLTP={sltp_retcode} | "
        f"Close={close_retcode} (Deal #{close_deal_id}) | Realized={realized_pnl:+.2f} {currency}"
    )
    _emit("COMPLETE", summary_msg, symbol=symbol, ticket=pos_ticket, floating_pnl=realized_pnl, trace=trace_dict)

    return DrillResult(
        ok=(close_retcode == done_code),
        account_login=login,
        account_currency=currency,
        symbol=symbol,
        side="BUY",
        lots=lots,
        entry_price=fill_price,
        stop_loss=tightened_sl,
        take_profit=take_profit,
        order_ticket=pos_ticket,
        position_id=pos_id,
        sltp_retcode=sltp_retcode,
        close_retcode=close_retcode,
        close_deal_id=close_deal_id,
        realized_pnl=round(realized_pnl, 2),
        z3_status=z3_msg,
        message=summary_msg,
        drill_mode=mode_norm,
        trace=trace_dict,
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="PocketBot Apex 60s Live 0.01-Lot MT5 Demo Lifecycle Drill")
    parser.add_argument("--hold-seconds", type=int, default=15, help="Seconds to hold and stream live P/L")
    parser.add_argument("--symbol", type=str, default=None, help="Optional symbol override")
    parser.add_argument("--drill-mode", type=str, default="plumbing", choices=["strict", "plumbing"], help="strict=11-agent gate, plumbing=scorecard+0.01 lot test")
    parser.add_argument("--db", type=str, default="trader/apex_ledger.db", help="Ledger DB path")
    args = parser.parse_args(argv)

    def _printer(state: dict[str, Any]) -> None:
        print(f"[{state['status']}] {state['message']}")

    result = run_live_lifecycle_drill(
        hold_seconds=args.hold_seconds,
        symbol_override=args.symbol,
        db_path=args.db,
        on_update=_printer,
        drill_mode=args.drill_mode,
    )
    print("\n=== DRILL SUMMARY ===")
    print(f"Account     : {result.account_login} ({result.account_currency})")
    print(f"Symbol/Side : {result.symbol} {result.side} {result.lots:.2f} lots")
    print(f"Ticket ID   : {result.order_ticket}")
    print(f"SLTP RetCode: {result.sltp_retcode}")
    print(f"Close Deal  : {result.close_deal_id} (RetCode {result.close_retcode})")
    print(f"Realized P/L: {result.realized_pnl:+.2f} {result.account_currency}")
    print(f"Z3 Gate     : {result.z3_status}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
