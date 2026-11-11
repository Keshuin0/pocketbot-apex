#!/usr/bin/env python3
"""PocketBot MT5 Account, Position & Performance Summary (stats.py).

Displays:
  1. Current MT5 Demo Balance, Equity, Free Margin, and Server/Terminal AutoTrading status
  2. Today's Realized + Floating P/L and Circuit-Breaker status
  3. Open Positions (live MT5 positions + Paper-Demo ledger marked-to-market at live MT5 ticks)
  4. Closed Trade Performance: Total Trades, Win Rate, Gross Profit/Loss, and Profit Factor
  5. Last 10 actions from `trader/bot_log.txt`

Usage:
    .venv\\Scripts\\python.exe stats.py
"""

import datetime
import json
import os
from pathlib import Path
import sys

import MetaTrader5 as mt5

import mt5_bridge
import mt5_executor

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = str(BASE_DIR)
LOG_PATH = str(BASE_DIR / "trader" / "bot_log.txt")
PAPER_PATH = str(BASE_DIR / "trader" / "paper_positions.json")


def compute_all_time_deal_stats():
    """Compute win rate, gross profit, gross loss, and profit factor from MT5 closed deals."""
    mt5_bridge.ensure_mt5()
    start = datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc)
    end = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=2)
    deals = mt5.history_deals_get(start, end)

    closed_pnls = []
    if deals:
        for d in deals:
            if d.type in (mt5.DEAL_TYPE_BUY, mt5.DEAL_TYPE_SELL) and d.entry in (
                mt5.DEAL_ENTRY_OUT,
                mt5.DEAL_ENTRY_INOUT,
                mt5.DEAL_ENTRY_OUT_BY,
            ):
                net = float(d.profit) + float(d.commission) + float(d.swap) + float(getattr(d, "fee", 0.0))
                closed_pnls.append(net)

    total = len(closed_pnls)
    wins = [p for p in closed_pnls if p > 0]
    losses = [p for p in closed_pnls if p < 0]
    gross_profit = round(sum(wins), 2)
    gross_loss = round(abs(sum(losses)), 2)
    net_profit = round(sum(closed_pnls), 2)
    win_rate = round((len(wins) / total) * 100.0, 1) if total > 0 else 0.0
    if gross_loss > 0:
        profit_factor = f"{gross_profit / gross_loss:.2f}"
    elif gross_profit > 0:
        profit_factor = "INF"
    else:
        profit_factor = "N/A (0 closed trades)"

    return {
        "total_trades": total,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": win_rate,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "net_profit": net_profit,
        "profit_factor": profit_factor,
    }


def evaluate_paper_positions():
    """Mark-to-market any paper-demo positions against live MT5 Bid/Ask ticks."""
    if not os.path.exists(PAPER_PATH):
        return []
    try:
        with open(PAPER_PATH, encoding="utf-8") as fh:
            items = json.load(fh)
    except Exception:
        return []

    out = []
    for p in items:
        sym = p.get("symbol", "EURUSD")
        side = str(p.get("type", "buy")).lower()
        vol = float(p.get("volume", 0.01))
        entry = float(p.get("price_open", 0.0))
        sl = float(p.get("sl", 0.0))
        tp = float(p.get("tp", 0.0))
        try:
            mt5_sym, info = mt5_bridge.resolve_mt5_symbol(sym)
            tick = mt5.symbol_info_tick(mt5_sym)
            digits = info.digits
            tick_size = float(info.trade_tick_size) or float(info.point)
            tick_val = float(info.trade_tick_value) or (float(info.trade_contract_size) * tick_size)
            mark = float(tick.bid if side == "buy" else tick.ask) if tick else entry
            diff = (mark - entry) if side == "buy" else (entry - mark)
            pnl_usd = round(vol * (diff / tick_size) * tick_val, 2)
        except Exception:
            digits = 5
            mark = entry
            pnl_usd = 0.0

        out.append({
            "ticket": p.get("ticket"),
            "symbol": sym,
            "type": side,
            "volume": vol,
            "price_open": entry,
            "mark_price": mark,
            "sl": sl,
            "tp": tp,
            "digits": digits,
            "profit": pnl_usd,
            "opened_at": p.get("opened_at", ""),
        })
    return out


def read_last_log_lines(n=10):
    if not os.path.exists(LOG_PATH):
        return ["(No trader/bot_log.txt found yet)"]
    with open(LOG_PATH, encoding="utf-8", errors="replace") as fh:
        lines = [line.rstrip("\r\n") for line in fh if line.strip()]
    return lines[-n:] if lines else ["(trader/bot_log.txt is empty)"]


def main():
    try:
        mt5_bridge.ensure_mt5()
        acct = mt5.account_info()
        term = mt5.terminal_info()
        pnl = mt5_executor.get_daily_pnl_summary()
        stats = compute_all_time_deal_stats()
        paper_pos = evaluate_paper_positions()
        paper_floating = round(sum(p["profit"] for p in paper_pos), 2)
        recent_logs = read_last_log_lines(10)

        print("=" * 88)
        print("  POCKETBOT MT5 — LIVE ACCOUNT, POSITION & PERFORMANCE DASHBOARD")
        print("=" * 88)
        print(f"  Account Login     : {pnl['login']} ({pnl['server']}) | Mode: {'DEMO (Safe)' if pnl['is_demo'] else 'LIVE'}")
        print(f"  Demo Balance      : ${pnl['balance']:,.2f}")
        print(f"  Live Equity       : ${pnl['equity']:,.2f} | Free Margin: ${pnl['margin_free']:,.2f}")
        print(
            f"  AutoTrading Flags : Terminal AlgoTrading={'ON' if (term and term.trade_allowed) else 'OFF'} | "
            f"Broker Account trade_allowed={getattr(acct, 'trade_allowed', False)} | "
            f"trade_expert={getattr(acct, 'trade_expert', False)}"
        )
        print("-" * 88)
        print("  TODAY'S P/L & CIRCUIT BREAKERS:")
        print(f"  Realized Today    : ${pnl['realized_pnl_today']:+,.2f} ({pnl['closed_deals_today']} closed deals)")
        print(f"  Floating (MT5)    : ${pnl['floating_pnl']:+,.2f}" + (f" | Paper-Demo Floating: ${paper_floating:+,.2f}" if paper_pos else ""))
        print(f"  Total Daily P/L   : ${pnl['total_daily_pnl']:+,.2f} ({pnl['daily_pnl_pct']:+.2f}%) | Kill-Switch Limit: -$200.00 (-2.0%)")
        print(f"  Open Positions    : {pnl['open_positions_count']} / {mt5_executor.DEFAULT_MAX_OPEN_POSITIONS} max allowed on MT5 server")
        print("-" * 88)
        print("  PERFORMANCE METRICS (Closed MT5 Deals):")
        print(f"  Total Closed      : {stats['total_trades']} trades (Wins: {stats['wins']} | Losses: {stats['losses']})")
        print(f"  Win Rate          : {stats['win_rate_pct']:.1f}%")
        print(f"  Profit Factor     : {stats['profit_factor']} (Gross Profit: ${stats['gross_profit']:,.2f} | Gross Loss: -${stats['gross_loss']:,.2f})")
        print(f"  Net Realized P/L  : ${stats['net_profit']:+,.2f}")
        print("-" * 88)
        print("  OPEN POSITIONS (MT5 Trade Tab + Paper-Demo Ledger):")
        if not pnl["open_positions"] and not paper_pos:
            print("    (No open positions)")
        for p in pnl["open_positions"]:
            print(
                f"    [MT5 #{p['ticket']}] {p['symbol']:<7} {p['type'].upper():<4} {p['volume']} lots | "
                f"Open={p['price_open']} | SL={p['sl']} | TP={p['tp']} | P/L=${p['profit']:+,.2f}"
            )
        for p in paper_pos:
            d = p["digits"]
            print(
                f"    [PAPER #{p['ticket']}] {p['symbol']:<7} {p['type'].upper():<4} {p['volume']} lots | "
                f"Open={p['price_open']:.{d}f} | Mark={p['mark_price']:.{d}f} | "
                f"SL={p['sl']:.{d}f} | TP={p['tp']:.{d}f} | Live P/L=${p['profit']:+,.2f} ({p['opened_at']})"
            )
        print("-" * 88)
        print("  LAST 10 ACTIONS (trader/bot_log.txt):")
        for line in recent_logs:
            print(f"    {line}")
        print("=" * 88)
        return 0
    except Exception as e:
        print(f"stats.py error: {e}", file=sys.stderr)
        return 1
    finally:
        if mt5_bridge._INITIALIZED:
            mt5.shutdown()
            mt5_bridge._INITIALIZED = False


if __name__ == "__main__":
    sys.exit(main())
