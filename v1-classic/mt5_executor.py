#!/usr/bin/env python3
"""MetaTrader 5 Execution, Strong Momentum & Risk Circuit-Breaker Engine for PocketBot.

Provides:
1. Tiered Symbol Classification & Strong Momentum Mode (`evaluate_strong_momentum`):
   - Tight-spread symbols (`EURUSD`, `GBPUSD`, `USDJPY`, `XAUUSD`, `SPX`, `NQ`) -> scanned on M5 (`5m`).
   - Wide-spread symbols (`BTCUSD`, `ETHUSD`, `#TSLA`, `#NVDA`) -> scanned on M15 (`15m`) with ATR-based
     Stop-Loss so target distances outweigh the spread.
   - Allows high-spread symbols whenever ALL 3 Strong Momentum checks pass:
     (a) Live spread <= 15% of the ATR-based Stop-Loss distance AND <= 1.5x the symbol's 14-bar rolling
         average spread (blocks news-spike blowouts).
     (b) Breakout candle body >= 1.5x the 14-bar ATR (fast directional surge).
     (c) Breakout candle closes in the top 25% of its high-low range for longs (or bottom 25% for shorts),
         confirming strong follow-through without wick rejection.
2. Live MT5 position sizing (`calculate_mt5_lots`) using `symbol_info`:
   - `trade_tick_value`, `trade_tick_size`, `volume_min`, `volume_step`, `volume_max`
   - Default max risk per trade = 0.5% of live MT5 equity ($50.00 on $10,000.00), scaling lot size down automatically.
3. Hard pre-trade safety circuit breakers (`run_safety_checks`) before any `mt5.order_send()`:
   - Demo-only lock: verifies `account_info().trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO`
     unless explicitly unlocked via `--unlock-live` (`allow_live=True`).
   - Daily loss kill-switch: blocks new trades if today's realized + floating P/L <= -2.0% (-$200.00).
   - Max open positions cap: blocks new trades if open positions >= 2.
   - Spread-to-SL & rolling spread filter: blocks any trade where live bid-ask spread > 15.0% of planned SL distance
     (or > 1.5x rolling average spread in Strong Momentum mode).
   - Strong Momentum 3-check gate (for wide-spread symbols or when `--strong-momentum` is active).
   - Mandatory hard server-side `sl` and `tp` + broker `trade_stops_level` check.
   - Automatic filling-mode detection (`ORDER_FILLING_FOK` / `ORDER_FILLING_IOC` / `ORDER_FILLING_RETURN`).
4. Dry-run verification (`mt5.order_check`) and live demo execution (`mt5.order_send`).

CLI Usage:
    .venv\\Scripts\\python.exe mt5_executor.py dry-run --symbol EURUSD
    .venv\\Scripts\\python.exe mt5_executor.py dry-run --symbol BTCUSD --simulate-momentum
    .venv\\Scripts\\python.exe mt5_executor.py dry-run --symbol #NVDA --simulate-momentum
    .venv\\Scripts\\python.exe mt5_executor.py check-breakers
    .venv\\Scripts\\python.exe mt5_executor.py execute --from-orders trader/orders.json --dry-run
"""

import argparse
import datetime
import json
import math
import os
from pathlib import Path
import sys

import MetaTrader5 as mt5

import mt5_bridge

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_MAX_RISK_PCT = 0.5               # 0.5% of equity per trade ($50 on $10,000)
DEFAULT_DAILY_LOSS_LIMIT_PCT = 2.0       # -2.0% daily loss kill-switch (-$200 on $10,000)
DEFAULT_MAX_OPEN_POSITIONS = 2           # Max 2 open positions simultaneously
DEFAULT_MAX_SPREAD_SL_PCT = 15.0         # Spread must not exceed 15% of SL distance
DEFAULT_MAX_SPREAD_ROLLING_MULT = 1.5    # Spread must not exceed 1.5x rolling average spread
DEFAULT_MIN_BODY_ATR_MULT = 1.5          # Breakout candle body must be >= 1.5x 14-bar ATR
DEFAULT_MIN_CLOSE_EXTREME_PCT = 75.0     # Close in top 25% (>=75%) for longs or bottom 25% for shorts
DEFAULT_ATR_PERIOD = 14                  # 14-bar ATR
DEFAULT_ATR_SL_MULT = 4.0                # Default M15 ATR stop multiplier for wide-spread symbols
MAGIC_NUMBER = 20261110

TIGHT_SPREAD_SYMBOLS = ("EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "SPX", "NQ")
WIDE_SPREAD_SYMBOLS = ("BTCUSD", "ETHUSD", "#TSLA", "#NVDA")
WIDE_SPREAD_SET = {"BTCUSD", "ETHUSD", "TSLA", "#TSLA", "NVDA", "#NVDA"}


class MT5ExecutionError(Exception):
    pass


def is_wide_spread_symbol(symbol):
    """Return True if `symbol` belongs to the high-spread M15 Strong Momentum tier."""
    s = str(symbol or "").strip().upper()
    return s in WIDE_SPREAD_SET or s.lstrip("#") in WIDE_SPREAD_SET


def default_timeframe_for_symbol(symbol):
    """Tight-spread symbols use M5 ('5m'); wide-spread symbols use M15 ('15m')."""
    return "15m" if is_wide_spread_symbol(symbol) else "5m"


# ---------------------------------------------------------------- filling mode detection

FILLING_NAMES = {
    mt5.ORDER_FILLING_FOK: "ORDER_FILLING_FOK",
    mt5.ORDER_FILLING_IOC: "ORDER_FILLING_IOC",
    mt5.ORDER_FILLING_RETURN: "ORDER_FILLING_RETURN",
}


def detect_filling_mode(symbol_info):
    """Detect the broker's supported order filling mode from symbol_info.filling_mode bitmask.
    Bit 1 (1) = SYMBOL_FILLING_FOK    -> mt5.ORDER_FILLING_FOK (0)
    Bit 2 (2) = SYMBOL_FILLING_IOC    -> mt5.ORDER_FILLING_IOC (1)
    Otherwise                         -> mt5.ORDER_FILLING_RETURN (2)
    """
    fm = getattr(symbol_info, "filling_mode", 0) or 0
    if fm & 1:
        return mt5.ORDER_FILLING_FOK
    if fm & 2:
        return mt5.ORDER_FILLING_IOC
    return mt5.ORDER_FILLING_RETURN


# ---------------------------------------------------------------- Strong Momentum & Rolling Spread

def evaluate_strong_momentum(
    symbol,
    side="buy",
    timeframe="15m",
    atr_period=DEFAULT_ATR_PERIOD,
    atr_sl_mult=DEFAULT_ATR_SL_MULT,
    sl_distance=None,
    candle_override=None,
    max_spread_sl_pct=DEFAULT_MAX_SPREAD_SL_PCT,
    max_spread_rolling_mult=DEFAULT_MAX_SPREAD_ROLLING_MULT,
    min_body_atr_mult=DEFAULT_MIN_BODY_ATR_MULT,
    min_close_extreme_pct=DEFAULT_MIN_CLOSE_EXTREME_PCT,
):
    """Evaluate the 3 Strong Momentum checks + 14-bar ATR & rolling spread on MT5 candles:
    1. Live spread <= 15% of ATR-based Stop-Loss distance AND <= 1.5x rolling 14-bar average spread.
    2. Breakout candle body (|close - open|) >= 1.5x 14-bar ATR.
    3. Breakout candle closes in top 25% of high-low range for longs (or bottom 25% for shorts).
    """
    mt5_bridge.ensure_mt5()
    mt5_sym, info = mt5_bridge.resolve_mt5_symbol(symbol)
    tick = mt5.symbol_info_tick(mt5_sym)
    if tick is None:
        raise MT5ExecutionError(f"Cannot read live tick for {mt5_sym}")

    tf_str = str(timeframe or default_timeframe_for_symbol(symbol)).lower()
    tf_map = {
        "1m": mt5.TIMEFRAME_M1,
        "5m": mt5.TIMEFRAME_M5,
        "15m": mt5.TIMEFRAME_M15,
    }
    mt5_tf = tf_map.get(tf_str, mt5.TIMEFRAME_M15)

    rates = mt5.copy_rates_from_pos(mt5_sym, mt5_tf, 0, int(atr_period) + 6)
    if rates is None or len(rates) < 4:
        raise MT5ExecutionError(f"Insufficient {tf_str} bars on {mt5_sym} to compute {atr_period}-bar ATR")

    # Drop the currently forming bar (rates[-1]) so ATR and rolling spread use completed bars
    completed = list(rates[:-1])
    use_bars = completed[-int(atr_period):] if len(completed) >= int(atr_period) else completed

    trs = []
    prev_c = float(completed[-len(use_bars) - 1]["close"]) if len(completed) > len(use_bars) else None
    for r in use_bars:
        h, l, c = float(r["high"]), float(r["low"]), float(r["close"])
        tr = (h - l) if prev_c is None else max(h - l, abs(h - prev_c), abs(l - prev_c))
        trs.append(tr)
        prev_c = c

    digits = info.digits
    point = float(info.point)
    atr_14 = sum(trs) / len(trs) if trs else point * 100.0
    atr_sl_distance = round(atr_14 * float(atr_sl_mult), digits)
    effective_sl_distance = round(float(sl_distance) if (sl_distance is not None and sl_distance > 0) else atr_sl_distance, digits)

    # Rolling average spread across completed bars (MT5 stores bar spread in points)
    bar_spreads_pts = [float(r["spread"]) for r in use_bars if float(r["spread"]) > 0]
    live_spread_price = round(float(tick.ask) - float(tick.bid), digits)
    if live_spread_price <= 0 and info.spread > 0:
        live_spread_price = round(info.spread * point, digits)
    live_spread_pts = int(round(live_spread_price / point)) if point > 0 else int(info.spread)

    rolling_avg_spread_pts = (
        sum(bar_spreads_pts) / len(bar_spreads_pts)
        if bar_spreads_pts
        else float(max(live_spread_pts, 1))
    )
    rolling_avg_spread_price = round(rolling_avg_spread_pts * point, digits)
    spread_vs_rolling_ratio = round(live_spread_pts / rolling_avg_spread_pts, 2) if rolling_avg_spread_pts > 0 else 1.0
    spread_pct_of_sl = round((live_spread_price / effective_sl_distance) * 100.0, 2) if effective_sl_distance > 0 else 100.0

    # Breakout candle inspection (use candle_override if provided, else last completed bar)
    if candle_override is not None:
        c_open = float(candle_override.get("open", candle_override.get("o")))
        c_high = float(candle_override.get("high", candle_override.get("h")))
        c_low = float(candle_override.get("low", candle_override.get("l")))
        c_close = float(candle_override.get("close", candle_override.get("c")))
        c_time = str(candle_override.get("time") or candle_override.get("bar_start") or "override")
        c_simulated = bool(candle_override.get("simulated", False))
    else:
        last_bar = use_bars[-1]
        c_open = float(last_bar["open"])
        c_high = float(last_bar["high"])
        c_low = float(last_bar["low"])
        c_close = float(last_bar["close"])
        offset_sec = mt5_bridge.server_offset_seconds()
        utc_dt = datetime.datetime.fromtimestamp(int(last_bar["time"]) - offset_sec, tz=datetime.timezone.utc).replace(tzinfo=None)
        c_time = mt5_bridge.to_eastern(utc_dt).strftime("%Y-%m-%d %H:%M ET")
        c_simulated = False

    side = str(side or "buy").lower()
    body = abs(c_close - c_open)
    bar_range = max(c_high - c_low, point)
    directional_ok = (c_close > c_open) if side == "buy" else (c_close < c_open)
    body_atr_ratio = round(body / atr_14, 2) if atr_14 > 0 else 0.0

    if side == "buy":
        close_extreme_pct = round(((c_close - c_low) / bar_range) * 100.0, 1)
        extreme_label = f"top {100.0 - close_extreme_pct:.1f}% of range ({close_extreme_pct:.1f}% from low, need >= {min_close_extreme_pct:.0f}%)"
    else:
        close_extreme_pct = round(((c_high - c_close) / bar_range) * 100.0, 1)
        extreme_label = f"bottom {100.0 - close_extreme_pct:.1f}% of range ({close_extreme_pct:.1f}% from high, need >= {min_close_extreme_pct:.0f}%)"

    # Evaluate the 3 Strong Momentum checks
    chk1_spread_sl_ok = spread_pct_of_sl <= float(max_spread_sl_pct) + 1e-9
    chk1_rolling_ok = spread_vs_rolling_ratio <= float(max_spread_rolling_mult) + 1e-9
    chk1_pass = chk1_spread_sl_ok and chk1_rolling_ok

    chk2_pass = directional_ok and (body_atr_ratio >= float(min_body_atr_mult) - 1e-9)
    chk3_pass = directional_ok and (close_extreme_pct >= float(min_close_extreme_pct) - 1e-9)

    checks = [
        {
            "check": "1_spread_vs_atr_and_rolling",
            "passed": chk1_pass,
            "detail": (
                f"Spread {live_spread_price:.{digits}f} ({live_spread_pts} pts) is {spread_pct_of_sl:.2f}% of "
                f"ATR SL {effective_sl_distance:.{digits}f} (max {max_spread_sl_pct:.1f}%) AND "
                f"{spread_vs_rolling_ratio:.2f}x rolling { len(use_bars) }-bar avg spread "
                f"{rolling_avg_spread_pts:.1f} pts (max {max_spread_rolling_mult:.1f}x)"
            ),
        },
        {
            "check": "2_breakout_body_vs_atr",
            "passed": chk2_pass,
            "detail": (
                f"Breakout candle body {body:.{digits}f} ({'bullish' if c_close >= c_open else 'bearish'}) is "
                f"{body_atr_ratio:.2f}x {tf_str.upper()} ATR({len(use_bars)})={atr_14:.{digits}f} "
                f"(need >= {min_body_atr_mult:.1f}x directional surge)"
            ),
        },
        {
            "check": "3_close_in_extreme_25pct",
            "passed": chk3_pass,
            "detail": (
                f"Candle closed at {c_close:.{digits}f} in [L={c_low:.{digits}f}, H={c_high:.{digits}f}] -> "
                f"{extreme_label}"
            ),
        },
    ]

    return {
        "enabled": True,
        "all_passed": chk1_pass and chk2_pass and chk3_pass,
        "timeframe": tf_str,
        "atr_period": len(use_bars),
        "atr_14": round(atr_14, digits),
        "atr_sl_mult": round(float(atr_sl_mult), 2),
        "atr_sl_distance": atr_sl_distance,
        "effective_sl_distance": effective_sl_distance,
        "live_spread_points": live_spread_pts,
        "live_spread_price": live_spread_price,
        "rolling_avg_spread_points": round(rolling_avg_spread_pts, 1),
        "rolling_avg_spread_price": rolling_avg_spread_price,
        "spread_vs_rolling_ratio": spread_vs_rolling_ratio,
        "spread_pct_of_sl": spread_pct_of_sl,
        "candle": {
            "time": c_time,
            "simulated": c_simulated,
            "open": round(c_open, digits),
            "high": round(c_high, digits),
            "low": round(c_low, digits),
            "close": round(c_close, digits),
            "body": round(body, digits),
            "range": round(bar_range, digits),
            "directional_ok": directional_ok,
            "body_atr_ratio": body_atr_ratio,
            "close_extreme_pct": close_extreme_pct,
        },
        "checks": checks,
    }


def build_simulated_momentum_candle(entry_price, atr_14, digits, side="buy", body_mult=1.65, close_pct=0.88):
    """Build a synthetic qualifying breakout candle (body = 1.65x ATR, close in top 12% of range)
    for dry-run verification of Strong Momentum mode on wide-spread symbols."""
    body = round(atr_14 * float(body_mult), digits)
    total_range = round(body / 0.80, digits)
    if str(side).lower() == "buy":
        c_close = round(float(entry_price), digits)
        c_open = round(c_close - body, digits)
        top_wick = round(total_range * (1.0 - float(close_pct)), digits)
        c_high = round(c_close + top_wick, digits)
        c_low = round(c_high - total_range, digits)
    else:
        c_close = round(float(entry_price), digits)
        c_open = round(c_close + body, digits)
        bot_wick = round(total_range * (1.0 - float(close_pct)), digits)
        c_low = round(c_close - bot_wick, digits)
        c_high = round(c_low + total_range, digits)
    return {
        "time": "simulated_m15_breakout",
        "simulated": True,
        "open": c_open,
        "high": c_high,
        "low": c_low,
        "close": c_close,
    }


# ---------------------------------------------------------------- daily P/L & open positions

def get_daily_pnl_summary():
    """Calculate today's realized P/L from closed deals + current floating P/L in MT5."""
    mt5_bridge.ensure_mt5()
    acct = mt5.account_info()
    if acct is None:
        raise MT5ExecutionError(f"Could not read MT5 account_info(): {mt5.last_error()}")

    offset_sec = mt5_bridge.server_offset_seconds()
    now_server = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=offset_sec)
    start_of_day_server = now_server.replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_day_server = now_server + datetime.timedelta(days=1)

    deals = mt5.history_deals_get(start_of_day_server, end_of_day_server)
    realized_pnl = 0.0
    closed_deals_count = 0
    if deals:
        for d in deals:
            if d.type in (mt5.DEAL_TYPE_BUY, mt5.DEAL_TYPE_SELL):
                net = float(d.profit) + float(d.commission) + float(d.swap) + float(getattr(d, "fee", 0.0))
                realized_pnl += net
                if d.entry in (mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_INOUT, mt5.DEAL_ENTRY_OUT_BY):
                    closed_deals_count += 1

    floating_pnl = float(acct.profit)
    total_daily_pnl = round(realized_pnl + floating_pnl, 2)
    starting_equity = float(acct.balance) - realized_pnl
    if starting_equity <= 0:
        starting_equity = float(acct.equity) or 10000.0
    daily_pnl_pct = round((total_daily_pnl / starting_equity) * 100.0, 4)

    positions = mt5.positions_get()
    open_positions = list(positions) if positions else []

    return {
        "login": acct.login,
        "server": acct.server,
        "trade_mode": acct.trade_mode,
        "is_demo": acct.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO,
        "balance": float(acct.balance),
        "equity": float(acct.equity),
        "margin_free": float(acct.margin_free),
        "starting_equity_today": round(starting_equity, 2),
        "realized_pnl_today": round(realized_pnl, 2),
        "floating_pnl": round(floating_pnl, 2),
        "total_daily_pnl": total_daily_pnl,
        "daily_pnl_pct": daily_pnl_pct,
        "closed_deals_today": closed_deals_count,
        "open_positions_count": len(open_positions),
        "open_positions": [
            {
                "ticket": p.ticket,
                "symbol": p.symbol,
                "type": "buy" if p.type == mt5.POSITION_TYPE_BUY else "sell",
                "volume": p.volume,
                "price_open": p.price_open,
                "sl": p.sl,
                "tp": p.tp,
                "profit": p.profit,
            }
            for p in open_positions
        ],
    }


# ---------------------------------------------------------------- MT5 lot sizing

def _step_decimals(step):
    s = f"{float(step):.8f}".rstrip("0")
    return len(s.split(".")[1]) if "." in s else 0


def floor_to_step(value, step):
    if step <= 0:
        return 0.0
    steps = math.floor((float(value) + 1e-9) / float(step))
    return round(steps * float(step), _step_decimals(step))


def calculate_mt5_lots(
    symbol,
    entry_price,
    sl_price,
    tp_price=None,
    side="buy",
    max_risk_pct=DEFAULT_MAX_RISK_PCT,
    max_risk_usd=None,
    equity_override=None,
    strong_momentum=None,
    timeframe=None,
    atr_sl_mult=DEFAULT_ATR_SL_MULT,
    candle_override=None,
):
    """Size a position in MT5 lots using live symbol_info (trade_tick_value, trade_tick_size,
    volume_min, volume_step, volume_max) so risk at Stop-Loss <= max_risk_usd (default 0.5% of equity).
    Automatically scales lot size down so dollar risk never exceeds $50.00 (0.5% on $10,000)."""
    mt5_bridge.ensure_mt5()
    mt5_sym, info = mt5_bridge.resolve_mt5_symbol(symbol)
    tick = mt5.symbol_info_tick(mt5_sym)
    acct = mt5.account_info()
    if acct is None:
        raise MT5ExecutionError("Cannot read MT5 account_info()")
    if tick is None:
        raise MT5ExecutionError(f"Cannot read live tick for {mt5_sym}")

    equity = float(equity_override if equity_override is not None else acct.equity)
    if max_risk_usd is None:
        max_risk_usd = round(equity * (float(max_risk_pct) / 100.0), 2)
    else:
        max_risk_usd = float(max_risk_usd)
        max_risk_pct = round((max_risk_usd / equity) * 100.0, 4) if equity > 0 else DEFAULT_MAX_RISK_PCT

    digits = info.digits
    point = float(info.point)
    tick_size = float(info.trade_tick_size) or point
    tick_value = float(info.trade_tick_value)
    if tick_value <= 0:
        tick_value = float(info.trade_contract_size) * tick_size

    vol_min = float(info.volume_min)
    vol_step = float(info.volume_step)
    vol_max = float(info.volume_max)

    side = str(side or "buy").lower()
    entry = round(float(entry_price), digits)
    sl = round(float(sl_price), digits)
    tp = round(float(tp_price), digits) if tp_price is not None else None

    if side == "buy" and sl >= entry:
        raise MT5ExecutionError(f"BUY stop-loss ({sl:.{digits}f}) must be below entry ({entry:.{digits}f})")
    if side == "sell" and sl <= entry:
        raise MT5ExecutionError(f"SELL stop-loss ({sl:.{digits}f}) must be above entry ({entry:.{digits}f})")
    if tp is not None:
        if side == "buy" and tp <= entry:
            raise MT5ExecutionError(f"BUY take-profit ({tp:.{digits}f}) must be above entry ({entry:.{digits}f})")
        if side == "sell" and tp >= entry:
            raise MT5ExecutionError(f"SELL take-profit ({tp:.{digits}f}) must be below entry ({entry:.{digits}f})")

    sl_distance = abs(entry - sl)
    tp_distance = abs(tp - entry) if tp is not None else 0.0
    reward_risk = round(tp_distance / sl_distance, 2) if (tp is not None and sl_distance > 0) else None

    ticks_at_risk = sl_distance / tick_size
    loss_per_lot = ticks_at_risk * tick_value
    if loss_per_lot <= 0:
        raise MT5ExecutionError(f"Invalid loss_per_lot ({loss_per_lot}) for {mt5_sym}")

    raw_lots = max_risk_usd / loss_per_lot
    risk_lots = floor_to_step(raw_lots, vol_step)

    # Check margin limit (keep 10% free margin buffer)
    order_type = mt5.ORDER_TYPE_BUY if side == "buy" else mt5.ORDER_TYPE_SELL
    margin_1lot = mt5.order_calc_margin(order_type, mt5_sym, 1.0, entry)
    if not margin_1lot or margin_1lot <= 0:
        margin_1lot = (float(info.trade_contract_size) * entry) / max(float(acct.leverage), 1.0)
    usable_margin = max(0.0, float(acct.margin_free) * 0.90)
    margin_lots = floor_to_step(usable_margin / margin_1lot, vol_step) if margin_1lot > 0 else vol_max

    lots = min(risk_lots, margin_lots, vol_max)
    bound_by = "max loss"
    if lots == margin_lots and margin_lots < risk_lots:
        bound_by = "free margin"
    elif lots == vol_max and vol_max < risk_lots:
        bound_by = "volume_max"

    min_lot_loss = round(vol_min * loss_per_lot, 2)
    refusal = None
    if lots < vol_min - 1e-9:
        lots = 0.0
        if bound_by == "max loss":
            refusal = (
                f"Minimum lot {vol_min} on {mt5_sym} risks ${min_lot_loss:.2f} "
                f"(SL distance {sl_distance:.{digits}f} = {ticks_at_risk:.1f} ticks x ${tick_value:.4f}/tick), "
                f"which exceeds your ${max_risk_usd:.2f} ({max_risk_pct}% equity) max risk per trade."
            )
        else:
            refusal = f"Insufficient free margin (${acct.margin_free:.2f}) for minimum lot {vol_min} on {mt5_sym}."

    actual_risk_usd = round(lots * loss_per_lot, 2)
    actual_reward_usd = round(lots * (tp_distance / tick_size) * tick_value, 2) if tp is not None else None
    margin_required = round(lots * margin_1lot, 2) if lots > 0 else 0.0
    notional_usd = round(lots * float(info.trade_contract_size) * entry, 2)

    # Live spread metrics
    spread_price = round(float(tick.ask) - float(tick.bid), digits)
    if spread_price <= 0 and info.spread > 0:
        spread_price = round(info.spread * point, digits)
    spread_points = int(round(spread_price / point)) if point > 0 else int(info.spread)
    spread_pct_of_sl = round((spread_price / sl_distance) * 100.0, 2) if sl_distance > 0 else 100.0
    spread_cost_usd = round(lots * (spread_price / tick_size) * tick_value, 2) if lots > 0 else 0.0

    filling_mode = detect_filling_mode(info)

    use_strong_momentum = is_wide_spread_symbol(mt5_sym) if strong_momentum is None else bool(strong_momentum)
    tf_str = timeframe or default_timeframe_for_symbol(mt5_sym)
    sm_report = None
    if use_strong_momentum:
        sm_report = evaluate_strong_momentum(
            symbol=mt5_sym,
            side=side,
            timeframe=tf_str,
            atr_sl_mult=atr_sl_mult,
            sl_distance=sl_distance,
            candle_override=candle_override,
        )

    return {
        "symbol": mt5_bridge.canonical_symbol(symbol, mt5_sym),
        "mt5_symbol": mt5_sym,
        "description": info.description,
        "tier": "wide_spread_m15_momentum" if use_strong_momentum else "tight_spread_m5",
        "timeframe": tf_str,
        "side": side,
        "digits": digits,
        "point": point,
        "trade_tick_size": tick_size,
        "trade_tick_value": round(tick_value, 6),
        "contract_size": float(info.trade_contract_size),
        "volume_min": vol_min,
        "volume_step": vol_step,
        "volume_max": vol_max,
        "stops_level_points": int(info.trade_stops_level),
        "filling_mode_flag": int(info.filling_mode),
        "type_filling": int(filling_mode),
        "type_filling_name": FILLING_NAMES.get(filling_mode, str(filling_mode)),
        "bid": float(tick.bid),
        "ask": float(tick.ask),
        "spread_points": spread_points,
        "spread_price": spread_price,
        "spread_pct_of_sl": spread_pct_of_sl,
        "spread_cost_usd": spread_cost_usd,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "sl_distance": round(sl_distance, digits),
        "tp_distance": round(tp_distance, digits) if tp is not None else None,
        "ticks_at_risk": round(ticks_at_risk, 2),
        "loss_per_lot_usd": round(loss_per_lot, 2),
        "equity": round(equity, 2),
        "max_risk_pct": max_risk_pct,
        "max_risk_usd": round(max_risk_usd, 2),
        "raw_lots": round(raw_lots, 4),
        "lots": lots,
        "bound_by": bound_by,
        "actual_risk_usd": actual_risk_usd,
        "actual_reward_usd": actual_reward_usd,
        "reward_risk": reward_risk,
        "margin_required_usd": margin_required,
        "notional_usd": notional_usd,
        "refusal": refusal,
        "strong_momentum": sm_report,
    }


# ---------------------------------------------------------------- Hard Safety Circuit Breakers

def run_safety_checks(
    sizing,
    allow_live=False,
    daily_loss_limit_pct=DEFAULT_DAILY_LOSS_LIMIT_PCT,
    max_open_positions=DEFAULT_MAX_OPEN_POSITIONS,
    max_spread_sl_pct=DEFAULT_MAX_SPREAD_SL_PCT,
):
    """Run all hard safety circuit breakers (including Strong Momentum checks on wide-spread symbols)
    before any `mt5.order_send()`."""
    pnl = get_daily_pnl_summary()
    breakers = []
    passed = True

    # 1. Demo Account Lock
    unlock_env = os.environ.get("POCKETBOT_UNLOCK_LIVE", "").strip().upper() == "YES"
    demo_ok = pnl["is_demo"] or (allow_live and unlock_env)
    breakers.append({
        "rule": "demo_account_only",
        "passed": demo_ok,
        "detail": (
            f"Account {pnl['login']} on {pnl['server']} is DEMO (trade_mode={pnl['trade_mode']} == ACCOUNT_TRADE_MODE_DEMO)"
            if pnl["is_demo"]
            else "LIVE REAL-MONEY ACCOUNT BLOCKED (trade_mode != ACCOUNT_TRADE_MODE_DEMO)"
        ),
    })
    if not demo_ok:
        passed = False

    # 2. Daily Loss Kill-Switch (-2.0% / -$200.00)
    max_daily_loss_usd = round(pnl["starting_equity_today"] * (float(daily_loss_limit_pct) / 100.0), 2)
    daily_ok = pnl["total_daily_pnl"] > -max_daily_loss_usd + 1e-9
    breakers.append({
        "rule": "daily_loss_kill_switch",
        "passed": daily_ok,
        "detail": (
            f"Daily P/L is ${pnl['total_daily_pnl']:+,.2f} ({pnl['daily_pnl_pct']:+.2f}%), "
            f"limit is -${max_daily_loss_usd:,.2f} (-{daily_loss_limit_pct:.1f}%)"
        ),
    })
    if not daily_ok:
        passed = False

    # 3. Max 2 Open Positions
    pos_count = pnl["open_positions_count"]
    pos_ok = pos_count < int(max_open_positions)
    breakers.append({
        "rule": "max_open_positions",
        "passed": pos_ok,
        "detail": f"Open positions: {pos_count} of {int(max_open_positions)} allowed",
    })
    if not pos_ok:
        passed = False

    # 4. Spread Filter (<= 15% of Stop-Loss distance, plus <= 1.5x rolling avg spread in Strong Momentum mode)
    spread_pct = sizing.get("spread_pct_of_sl", 100.0)
    sm = sizing.get("strong_momentum")
    d = sizing.get("digits", 5)
    if sm and sm.get("enabled"):
        rolling_ratio = sm.get("spread_vs_rolling_ratio", 1.0)
        spread_ok = (spread_pct <= float(max_spread_sl_pct) + 1e-9) and (
            rolling_ratio <= DEFAULT_MAX_SPREAD_ROLLING_MULT + 1e-9
        )
        spread_detail = (
            f"Live spread {sizing['spread_price']:.{d}f} ({sizing['spread_points']} pts) is "
            f"{spread_pct:.2f}% of ATR SL {sizing['sl_distance']:.{d}f} (max {max_spread_sl_pct:.1f}%) "
            f"& {rolling_ratio:.2f}x rolling {sm['atr_period']}-bar avg spread "
            f"({sm['rolling_avg_spread_points']:.1f} pts, max {DEFAULT_MAX_SPREAD_ROLLING_MULT:.1f}x)"
        )
    else:
        spread_ok = spread_pct <= float(max_spread_sl_pct) + 1e-9
        spread_detail = (
            f"Live spread {sizing['spread_price']:.{d}f} ({sizing['spread_points']} pts) is "
            f"{spread_pct:.2f}% of SL distance {sizing['sl_distance']:.{d}f} (max {max_spread_sl_pct:.1f}%)"
        )
    breakers.append({
        "rule": "spread_filter",
        "passed": spread_ok,
        "detail": spread_detail,
    })
    if not spread_ok:
        passed = False

    # 5. Strong Momentum 3-Check Gate (when symbol is in Wide-Spread M15 tier or --strong-momentum enabled)
    if sm and sm.get("enabled"):
        sm_ok = bool(sm.get("all_passed"))
        c_info = sm["candle"]
        breakers.append({
            "rule": "strong_momentum_3_checks",
            "passed": sm_ok,
            "detail": (
                f"[{sm['timeframe'].upper()}] Spread={sm['spread_pct_of_sl']:.1f}% SL ({sm['spread_vs_rolling_ratio']:.2f}x avg) | "
                f"Body={c_info['body']:.{d}f} ({c_info['body_atr_ratio']:.2f}x ATR(14)={sm['atr_14']:.{d}f}, need >=1.50x) | "
                f"CloseLoc={c_info['close_extreme_pct']:.1f}% (need >=75.0%)"
            ),
        })
        if not sm_ok:
            passed = False

    # 6. Hard Server-Side SL & TP + Minimum Stops Level + Strict 0.5% ($50) Risk Cap
    sl = sizing.get("sl")
    tp = sizing.get("tp")
    lots = sizing.get("lots", 0.0)
    min_stop_dist = sizing.get("stops_level_points", 0) * sizing.get("point", 0.0)
    sl_dist = sizing.get("sl_distance") or 0.0
    tp_dist = sizing.get("tp_distance") or 0.0
    risk_cap_ok = sizing.get("actual_risk_usd", 0.0) <= sizing.get("max_risk_usd", 50.0) + 1e-6
    sl_tp_ok = (
        lots >= sizing.get("volume_min", 0.01) - 1e-9
        and sizing.get("refusal") is None
        and risk_cap_ok
        and sl is not None and sl > 0
        and tp is not None and tp > 0
        and sl_dist >= min_stop_dist - 1e-9
        and tp_dist >= min_stop_dist - 1e-9
    )
    breakers.append({
        "rule": "hard_sl_tp_and_volume",
        "passed": sl_tp_ok,
        "detail": (
            f"Lots={lots} (Risk=${sizing['actual_risk_usd']:.2f} <= ${sizing['max_risk_usd']:.2f} cap), "
            f"SL={sl:.{d}f}, TP={tp:.{d}f} (min stop={min_stop_dist:.{d}f}, filling={sizing.get('type_filling_name')})"
            if (sl and tp and sizing.get("refusal") is None)
            else (sizing.get("refusal") or "Missing hard SL or TP")
        ),
    })
    if not sl_tp_ok:
        passed = False

    return {
        "passed": passed,
        "breakers": breakers,
        "account_summary": pnl,
    }


def build_mt5_order_request(sizing, comment="PocketBot ORB"):
    """Build the exact dictionary for `mt5.order_check()` and `mt5.order_send()`."""
    side = sizing["side"]
    digits = sizing["digits"]
    exec_price = round(sizing["ask"] if side == "buy" else sizing["bid"], digits)
    sl = round(float(sizing["sl"]), digits)
    tp = round(float(sizing["tp"]), digits)

    return {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": sizing["mt5_symbol"],
        "volume": float(sizing["lots"]),
        "type": mt5.ORDER_TYPE_BUY if side == "buy" else mt5.ORDER_TYPE_SELL,
        "price": exec_price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": MAGIC_NUMBER,
        "comment": str(comment)[:31],
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": int(sizing["type_filling"]),
    }


def verify_or_send_order(
    sizing,
    dry_run=True,
    allow_live=False,
    daily_loss_limit_pct=DEFAULT_DAILY_LOSS_LIMIT_PCT,
    max_open_positions=DEFAULT_MAX_OPEN_POSITIONS,
    max_spread_sl_pct=DEFAULT_MAX_SPREAD_SL_PCT,
    comment="PocketBot ORB",
):
    """Run all circuit breakers, validate with `mt5.order_check()`, and if `dry_run=False`,
    send the order via `mt5.order_send()` with automatic filling-mode fallback."""
    mt5_bridge.ensure_mt5()
    safety = run_safety_checks(
        sizing,
        allow_live=allow_live,
        daily_loss_limit_pct=daily_loss_limit_pct,
        max_open_positions=max_open_positions,
        max_spread_sl_pct=max_spread_sl_pct,
    )

    if not safety["passed"]:
        failed_rules = [b["rule"] + ": " + b["detail"] for b in safety["breakers"] if not b["passed"]]
        return {
            "status": "blocked_by_circuit_breaker",
            "dry_run": dry_run,
            "safety": safety,
            "sizing": sizing,
            "failed_rules": failed_rules,
            "order_check": None,
            "order_send": None,
        }

    req = build_mt5_order_request(sizing, comment=comment)

    fill_candidates = [req["type_filling"]] + [
        f for f in (mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN)
        if f != req["type_filling"]
    ]

    chk_dict = None
    for fill_mode in fill_candidates:
        req["type_filling"] = int(fill_mode)
        chk = mt5.order_check(req)
        if chk is None:
            raise MT5ExecutionError(f"mt5.order_check() returned None: {mt5.last_error()}")
        chk_dict = chk._asdict()
        if "request" in chk_dict and hasattr(chk_dict["request"], "_asdict"):
            chk_dict["request"] = chk_dict["request"]._asdict()
        if chk.retcode != 10030:
            sizing["type_filling"] = int(fill_mode)
            sizing["type_filling_name"] = FILLING_NAMES.get(int(fill_mode), str(fill_mode))
            break

    if dry_run or chk_dict["retcode"] != 0:
        return {
            "status": "dry_run_verified" if chk_dict["retcode"] == 0 else f"order_check_failed_{chk_dict['retcode']}",
            "dry_run": True,
            "safety": safety,
            "sizing": sizing,
            "request": req,
            "order_check": chk_dict,
            "order_send": None,
        }

    term = mt5.terminal_info()
    if term and not term.trade_allowed:
        return {
            "status": "mt5_autotrading_disabled_in_terminal",
            "dry_run": False,
            "safety": safety,
            "sizing": sizing,
            "request": req,
            "order_check": chk_dict,
            "order_send": {
                "retcode": 10027,
                "comment": "Algo Trading button is disabled in MT5 toolbar (click 'Algo Trading' in MetaTrader 5 to enable live demo execution)",
            },
        }

    res = mt5.order_send(req)
    if res is None:
        raise MT5ExecutionError(f"mt5.order_send() returned None: {mt5.last_error()}")
    res_dict = res._asdict()
    if "request" in res_dict and hasattr(res_dict["request"], "_asdict"):
        res_dict["request"] = res_dict["request"]._asdict()

    placed_ok = res.retcode in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_DONE_PARTIAL, mt5.TRADE_RETCODE_PLACED)
    paper_ticket = None
    if res.retcode == 10026:
        acct_info = mt5.account_info()
        paper_path = str(BASE_DIR / "trader" / "paper_positions.json")
        os.makedirs(os.path.dirname(paper_path), exist_ok=True)
        existing = []
        if os.path.exists(paper_path):
            try:
                with open(paper_path, encoding="utf-8") as fh:
                    existing = json.load(fh)
            except Exception:
                existing = []
        paper_ticket = 90001000 + len(existing) + 1
        existing.append({
            "ticket": paper_ticket,
            "symbol": sizing["mt5_symbol"],
            "type": sizing["side"],
            "volume": float(sizing["lots"]),
            "price_open": float(req["price"]),
            "sl": float(req["sl"]),
            "tp": float(req["tp"]),
            "risk_usd": float(sizing["actual_risk_usd"]),
            "reward_usd": float(sizing["actual_reward_usd"] or 0.0),
            "opened_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
            "status": "open",
            "mode": "paper_demo_fallback (PoTrade-MT5 retcode 10026: trade_allowed=False, trade_expert=False)",
        })
        with open(paper_path, "w", encoding="utf-8") as fh:
            json.dump(existing, fh, indent=2)
        res_dict["paper_ticket"] = paper_ticket
        res_dict["server_note"] = (
            f"PoTrade-MT5 server returned 10026 (AutoTrading disabled by server) because account {acct_info.login if acct_info else ''} "
            f"has trade_allowed={getattr(acct_info, 'trade_allowed', False)} and trade_expert={getattr(acct_info, 'trade_expert', False)} "
            f"(either logged in with Investor read-only password, or PoTrade-MT5 restricts EA/API orders on this demo login). "
            f"Order was recorded in local Paper-Demo ledger (trader/paper_positions.json) as Ticket #{paper_ticket}."
        )

    return {
        "status": "placed" if placed_ok else ("paper_demo_recorded_10026" if paper_ticket else f"send_failed_{res.retcode}"),
        "dry_run": False,
        "safety": safety,
        "sizing": sizing,
        "request": req,
        "order_check": chk_dict,
        "order_send": res_dict,
    }


# ---------------------------------------------------------------- CLI formatting & entrypoints

def format_execution_report(result):
    s = result["sizing"]
    safety = result["safety"]
    pnl = safety["account_summary"]
    d = s["digits"]
    sm = s.get("strong_momentum")
    lines = [
        "=" * 84,
        f"  MT5 EXECUTOR — {'DRY-RUN VERIFICATION' if result['dry_run'] else 'LIVE DEMO EXECUTION'} ({s['mt5_symbol']} | Tier: {s['tier']})",
        "=" * 84,
        f"  Account      : {pnl['login']} ({pnl['server']}) | Mode: {'DEMO' if pnl['is_demo'] else 'LIVE'}",
        f"  Equity / Bal : ${pnl['equity']:,.2f} / ${pnl['balance']:,.2f} | Free Margin: ${pnl['margin_free']:,.2f}",
        f"  Daily P/L    : ${pnl['total_daily_pnl']:+,.2f} ({pnl['daily_pnl_pct']:+.2f}%) | Open Positions: {pnl['open_positions_count']}/2",
        "-" * 84,
        "  SYMBOL & CONTRACT SPECS (from mt5.symbol_info):",
        f"  Symbol       : {s['mt5_symbol']} ({s['description']}) | Timeframe: {s['timeframe'].upper()} | Digits: {d} | Point: {s['point']}",
        f"  Tick Math    : trade_tick_size = {s['trade_tick_size']} | trade_tick_value = ${s['trade_tick_value']:.4f} | Contract = {s['contract_size']:g}",
        f"  Volume Specs : volume_min = {s['volume_min']} | volume_step = {s['volume_step']} | volume_max = {s['volume_max']}",
        f"  Filling Mode : symbol.filling_mode = {s['filling_mode_flag']} -> {s['type_filling_name']} ({s['type_filling']})",
        f"  Live Quote   : Bid = {s['bid']:.{d}f} | Ask = {s['ask']:.{d}f} | Spread = {s['spread_price']:.{d}f} ({s['spread_points']} pts)",
        "-" * 84,
        "  POSITION SIZING & SL/TP MATH (Strict 0.5% Equity Max Risk Cap):",
        f"  Direction    : {s['side'].upper()}",
        f"  Entry Price  : {s['entry']:.{d}f}",
        f"  Stop-Loss    : {s['sl']:.{d}f} (SL Distance = {s['sl_distance']:.{d}f} = {s['ticks_at_risk']:.1f} ticks)",
        f"  Take-Profit  : {s['tp']:.{d}f} (TP Distance = {s['tp_distance']:.{d}f} | Reward:Risk = {s['reward_risk']:.2f}R)",
        f"  Risk Formula : {s['ticks_at_risk']:.1f} ticks x ${s['trade_tick_value']:.4f}/tick = ${s['loss_per_lot_usd']:,.2f} risk per 1.00 lot",
        f"  Max Risk Cap : {s['max_risk_pct']}% of ${s['equity']:,.2f} = ${s['max_risk_usd']:,.2f}",
        f"  Lot Scaling  : ${s['max_risk_usd']:,.2f} / ${s['loss_per_lot_usd']:,.2f} = {s['raw_lots']:.4f} raw lots -> floored to step {s['volume_step']} = {s['lots']} lots (bound by {s['bound_by']})",
        f"  Actual Risk  : ${s['actual_risk_usd']:,.2f} at SL | Target Reward: ${s['actual_reward_usd']:,.2f} at TP | Margin Req: ${s['margin_required_usd']:,.2f}",
        f"  Spread Drag  : {s['spread_pct_of_sl']:.2f}% of SL distance (${s['spread_cost_usd']:,.2f} spread cost)",
    ]

    if sm and sm.get("enabled"):
        c = sm["candle"]
        lines.extend([
            "-" * 84,
            f"  STRONG MOMENTUM QUALIFICATION ({sm['timeframe'].upper()} Candles | ATR({sm['atr_period']}) = {sm['atr_14']:.{d}f}):",
            f"  Breakout Bar : {c['time']} {'(SIMULATED BREAKOUT)' if c['simulated'] else '(LIVE COMPLETED BAR)'} | "
            f"O={c['open']:.{d}f} H={c['high']:.{d}f} L={c['low']:.{d}f} C={c['close']:.{d}f}",
        ])
        for chk in sm["checks"]:
            tag = "PASS" if chk["passed"] else "FAIL"
            lines.append(f"    [{tag}] {chk['check']:<28} : {chk['detail']}")

    lines.extend([
        "-" * 84,
        "  SAFETY CIRCUIT BREAKERS:",
    ])
    for b in safety["breakers"]:
        tag = "PASS" if b["passed"] else "FAIL"
        lines.append(f"    [{tag}] {b['rule']:<26} : {b['detail']}")

    chk = result.get("order_check")
    if chk:
        lines.extend([
            "-" * 84,
            "  MT5 SERVER PRE-CHECK (mt5.order_check):",
            f"    retcode      : {chk['retcode']} ({chk['comment']})",
            f"    margin_req   : ${chk['margin']:,.2f} | margin_free_after: ${chk['margin_free']:,.2f} | margin_level: {chk['margin_level']:.2f}%",
            f"    request      : action=TRADE_ACTION_DEAL, symbol={result['request']['symbol']}, volume={result['request']['volume']}, "
            f"type={'BUY' if result['request']['type'] == 0 else 'SELL'}, price={result['request']['price']:.{d}f}, "
            f"sl={result['request']['sl']:.{d}f}, tp={result['request']['tp']:.{d}f}, type_filling={s['type_filling_name']}",
        ])
    snd = result.get("order_send")
    if snd:
        lines.extend([
            "-" * 84,
            "  MT5 ORDER SEND RESULT (mt5.order_send):",
            f"    retcode      : {snd.get('retcode')} ({snd.get('comment')})",
            f"    order / deal : order={snd.get('order')} | deal={snd.get('deal')}"
            + (f" | paper_ticket=#{snd.get('paper_ticket')}" if snd.get("paper_ticket") else ""),
        ])
        if snd.get("server_note"):
            lines.append(f"    note         : {snd['server_note']}")
    lines.append("=" * 84)
    return "\n".join(lines)


def cmd_dry_run(args):
    mt5_bridge.ensure_mt5()
    mt5_sym, info = mt5_bridge.resolve_mt5_symbol(args.symbol)
    tick = mt5.symbol_info_tick(mt5_sym)
    if not tick:
        raise MT5ExecutionError(f"No live tick for {mt5_sym}")

    side = args.side.lower()
    entry = float(args.entry) if args.entry else (float(tick.ask) if side == "buy" else float(tick.bid))
    digits = info.digits
    point = info.point

    use_sm = is_wide_spread_symbol(mt5_sym) or bool(args.strong_momentum)
    tf_str = args.timeframe or default_timeframe_for_symbol(mt5_sym)

    candle_override = None
    sm_probe = None
    if use_sm:
        sm_probe = evaluate_strong_momentum(
            symbol=mt5_sym,
            side=side,
            timeframe=tf_str,
            atr_sl_mult=args.atr_sl_mult,
        )
        if args.simulate_momentum:
            candle_override = build_simulated_momentum_candle(
                entry_price=entry,
                atr_14=sm_probe["atr_14"],
                digits=digits,
                side=side,
            )

    if args.sl is not None:
        sl = float(args.sl)
    elif use_sm and sm_probe is not None and args.sl_pips is None:
        sl_dist = sm_probe["atr_sl_distance"]
        sl = entry - sl_dist if side == "buy" else entry + sl_dist
    else:
        sl_pips = float(args.sl_pips if args.sl_pips is not None else 20.0)
        pip_size = point * 10.0
        sl_dist = sl_pips * pip_size
        sl = entry - sl_dist if side == "buy" else entry + sl_dist

    sl_dist = abs(entry - sl)
    if args.tp is not None:
        tp = float(args.tp)
    else:
        tp_dist = sl_dist * float(args.rr)
        tp = entry + tp_dist if side == "buy" else entry - tp_dist

    sizing = calculate_mt5_lots(
        symbol=args.symbol,
        entry_price=entry,
        sl_price=sl,
        tp_price=tp,
        side=side,
        max_risk_pct=args.risk_pct,
        max_risk_usd=args.risk_usd,
        strong_momentum=use_sm,
        timeframe=tf_str,
        atr_sl_mult=args.atr_sl_mult,
        candle_override=candle_override,
    )
    res = verify_or_send_order(
        sizing,
        dry_run=not args.send,
        allow_live=args.unlock_live,
    )
    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print(format_execution_report(res))
    return 0 if res["status"] in ("dry_run_verified", "placed", "paper_demo_recorded_10026") else 1


def cmd_check_breakers(args):
    pnl = get_daily_pnl_summary()
    print(json.dumps(pnl, indent=2))
    return 0


def cmd_execute_orders(args):
    if not os.path.exists(args.from_orders):
        raise MT5ExecutionError(f"No orders file at {args.from_orders}")
    with open(args.from_orders, encoding="utf-8") as fh:
        doc = json.load(fh)
    orders = doc.get("orders", doc) if isinstance(doc, dict) else doc
    if not orders:
        raise MT5ExecutionError(f"No orders inside {args.from_orders}")

    order = orders[args.index - 1]
    plan = order.get("plan") or {}
    sym = order.get("symbol")
    side = order.get("side", "buy")
    entry = float(order.get("limit_price") or plan.get("entry") or 0)
    sl = float(order.get("stop_loss_stop_price") or plan.get("stop") or 0)
    tp = float(order.get("take_profit_limit_price") or plan.get("target") or 0)

    mt5_bridge.ensure_mt5()
    mt5_sym, _ = mt5_bridge.resolve_mt5_symbol(sym)
    tick = mt5.symbol_info_tick(mt5_sym)
    if entry <= 0 and tick:
        entry = float(tick.ask if side == "buy" else tick.bid)

    sizing = calculate_mt5_lots(
        symbol=sym,
        entry_price=entry,
        sl_price=sl,
        tp_price=tp,
        side=side,
        max_risk_pct=args.risk_pct,
    )
    res = verify_or_send_order(
        sizing,
        dry_run=args.dry_run,
        allow_live=args.unlock_live,
    )
    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print(format_execution_report(res))
    return 0 if res["status"] in ("dry_run_verified", "placed", "paper_demo_recorded_10026") else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description="MetaTrader 5 Execution, Strong Momentum & Safety Circuit-Breaker Engine.")
    sub = ap.add_subparsers(dest="cmd")

    p_dr = sub.add_parser("dry-run", help="Verify MT5 lot sizing, SL/TP math, Strong Momentum checks, circuit breakers, and mt5.order_check")
    p_dr.add_argument("--symbol", default="EURUSD", help="Symbol to test (e.g. EURUSD, BTCUSD, ETHUSD, #TSLA, #NVDA)")
    p_dr.add_argument("--side", default="buy", choices=["buy", "sell"])
    p_dr.add_argument("--timeframe", choices=["1m", "5m", "15m"], help="Override candle timeframe (default: 5m for tight-spread, 15m for wide-spread)")
    p_dr.add_argument("--entry", type=float, help="Override entry price (default live Ask/Bid)")
    p_dr.add_argument("--sl", type=float, help="Explicit Stop-Loss price")
    p_dr.add_argument("--tp", type=float, help="Explicit Take-Profit price")
    p_dr.add_argument("--sl-pips", type=float, default=None, help="SL distance in pips (default 20 pips for tight-spread FX; ATR-based for wide-spread)")
    p_dr.add_argument("--atr-sl-mult", type=float, default=DEFAULT_ATR_SL_MULT, help="ATR Stop-Loss multiplier for Strong Momentum symbols (default 3.5x M15 ATR(14))")
    p_dr.add_argument("--rr", type=float, default=2.0, help="Reward:Risk ratio if --tp not given (default 2.0R)")
    p_dr.add_argument("--risk-pct", type=float, default=DEFAULT_MAX_RISK_PCT, help="Max risk %% of equity (default 0.5%%)")
    p_dr.add_argument("--risk-usd", type=float, help="Explicit max risk USD override")
    p_dr.add_argument("--strong-momentum", action="store_true", help="Force Strong Momentum 3-check qualification on any symbol")
    p_dr.add_argument("--simulate-momentum", action="store_true", help="Synthesize a qualifying >=1.5x ATR M15 breakout candle for dry-run verification")
    p_dr.add_argument("--send", action="store_true", help="Actually call mt5.order_send (default is dry-run order_check only)")
    p_dr.add_argument("--unlock-live", action="store_true", help="Unlock live real-money account (also requires POCKETBOT_UNLOCK_LIVE=YES)")
    p_dr.add_argument("--json", action="store_true")

    p_cb = sub.add_parser("check-breakers", help="Inspect daily P/L, open positions, and account status")
    p_cb.add_argument("--json", action="store_true")

    p_ex = sub.add_parser("execute", help="Execute or dry-run an order from trader/orders.json")
    p_ex.add_argument("--from-orders", default="trader/orders.json")
    p_ex.add_argument("--index", type=int, default=1)
    p_ex.add_argument("--risk-pct", type=float, default=DEFAULT_MAX_RISK_PCT)
    p_ex.add_argument("--dry-run", action="store_true", default=False)
    p_ex.add_argument("--unlock-live", action="store_true")
    p_ex.add_argument("--json", action="store_true")

    args = ap.parse_args(argv)
    try:
        if not args.cmd or args.cmd == "dry-run":
            if not args.cmd:
                args = ap.parse_args(["dry-run"] + (argv or []))
            return cmd_dry_run(args)
        return {
            "check-breakers": cmd_check_breakers,
            "execute": cmd_execute_orders,
        }[args.cmd](args)
    except (MT5ExecutionError, mt5_bridge.MT5BridgeError) as e:
        print(f"mt5_executor.py: {e}", file=sys.stderr)
        return 1
    finally:
        if mt5_bridge._INITIALIZED:
            mt5.shutdown()
            mt5_bridge._INITIALIZED = False


if __name__ == "__main__":
    sys.exit(main())
