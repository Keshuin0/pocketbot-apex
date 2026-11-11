#!/usr/bin/env python3
"""PocketBot Autonomous & Interactive MT5 Trading Loop with Strong Momentum Mode (pocketbot_auto.py).

Orchestrates the trader skills with live MetaTrader 5 candles and the MT5 execution /
Strong Momentum / safety circuit-breaker engine:

  1. Two-Tier Multi-Timeframe Scanning:
     - Tight-spread symbols (`EURUSD`, `GBPUSD`, `USDJPY`, `XAUUSD`, `SPX`, `NQ`) scanned on
       live **M5** (`5m`) candles with Opening-Range midpoint stop (`or-mid`) and `2R` target.
     - Wide-spread symbols (`BTCUSD`, `ETHUSD`, `#TSLA`, `#NVDA`) scanned on live **M15** (`15m`)
       candles with ATR-based Stop-Loss (`atr:3.5` on 14-bar M15 ATR) and `2R` target so target
       distances outweigh the spread.
  2. Strong Momentum Qualification for Wide-Spread Symbols (all 3 checks must pass):
     - Check 1: Spread <= 15% of ATR-based Stop-Loss distance AND <= 1.5x rolling 14-bar average spread.
     - Check 2: Breakout candle body >= 1.5x 14-bar ATR (fast directional surge).
     - Check 3: Breakout candle closes in top 25% of its high-low range for longs (or bottom 25% for shorts).
  3. Strict 0.5% ($50.00) Max Dollar Risk Cap:
     - `mt5_executor.calculate_mt5_lots()` scales lot size down automatically using live `mt5.symbol_info`
       (`trade_tick_value`, `trade_tick_size`, `volume_min`, `volume_step`) so risk never exceeds 0.5% ($50).
  4. Two Operating Modes:
     - `--mode confirm`   : Prints full setup, Strong Momentum checks, Time Machine 10,000-replay EV,
                            MT5 lot size, SL/TP, and circuit breakers, and asks YES/NO before sending.
     - `--mode demo-auto` : Automatically places the demo bracket order ONLY when Time Machine expected
                            value (`avg_R > 0`) is positive AND all risk, Strong Momentum, and safety
                            circuit breakers pass.
  5. Logs every scan, Strong Momentum evaluation, Time Machine simulation, and order decision to `trader/bot_log.txt`.

Usage:
    .venv\\Scripts\\python.exe pocketbot_auto.py --mode confirm --once
    .venv\\Scripts\\python.exe pocketbot_auto.py --mode demo-auto --once --dry-run
    .venv\\Scripts\\python.exe pocketbot_auto.py --mode confirm
    .venv\\Scripts\\python.exe pocketbot_auto.py --mode demo-auto
"""

import argparse
import datetime
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import MetaTrader5 as mt5

import mt5_bridge
import mt5_executor

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = str(BASE_DIR)
os.chdir(ROOT_DIR)
TIGHT_SPREAD_M5_SYMBOLS = list(mt5_executor.TIGHT_SPREAD_SYMBOLS)
WIDE_SPREAD_M15_SYMBOLS = list(mt5_executor.WIDE_SPREAD_SYMBOLS)
DEFAULT_SYMBOLS = TIGHT_SPREAD_M5_SYMBOLS + WIDE_SPREAD_M15_SYMBOLS
LOG_PATH = str(BASE_DIR / "trader" / "bot_log.txt")


def _load_skill_module(skill_folder, script_name):
    path = os.path.join(ROOT_DIR, ".agents", "skills", skill_folder, script_name)
    spec = importlib.util.spec_from_file_location(f"skill_{skill_folder.replace('-', '_')}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


entry_mod = _load_skill_module("trader-entry", "entry.py")
exit_mod = _load_skill_module("trader-exit", "exit.py")
tm_mod = _load_skill_module("trader-timemachine", "timemachine.py")
size_mod = _load_skill_module("trader-budget", "size.py")
risk_mod = _load_skill_module("trader-risk", "check.py")


def log_event(message, log_path=LOG_PATH):
    """Append a timestamped line to trader/bot_log.txt and print to stdout."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    line = f"[{ts}] {message}"
    print(line)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def evaluate_symbol_pipeline(sym, now_utc, replays=10000, risk_pct=0.5, atr_sl_mult=mt5_executor.DEFAULT_ATR_SL_MULT):
    """Run entry -> exit -> Strong Momentum (for M15 wide-spread tier) -> Time Machine (10,000 replays)
    -> MT5 lot sizing (capped at 0.5% / $50) -> risk checks for `sym`."""
    wide_tier = mt5_executor.is_wide_spread_symbol(sym)
    interval = "15m" if wide_tier else "5m"
    or_minutes = 15  # 3x M5 bars or 1x M15 bar (9:30-9:45 ET)

    chart_text_5d = mt5_bridge.handle_yahoo_path(f"/v8/finance/chart/{sym}?range=5d&interval={interval}")
    chart = entry_mod.parse_chart(chart_text_5d, sym)

    cfg = entry_mod.read_config("trader/config.json")
    res_entry = entry_mod.evaluate(chart, now_utc, setup="orb", or_minutes=or_minutes, interval=interval, cfg=cfg)

    # Pre-compute Strong Momentum metrics on M15 wide-spread symbols even if ORB is still waiting
    sm_preview = None
    if wide_tier:
        trig_bar = res_entry.get("trigger") if res_entry["status"] == "triggered" else None
        sm_preview = mt5_executor.evaluate_strong_momentum(
            symbol=sym,
            side="buy",
            timeframe=interval,
            atr_sl_mult=atr_sl_mult,
            candle_override=trig_bar,
        )

    if res_entry["status"] != "triggered":
        return {
            "symbol": sym,
            "tier": "M15-MOMENTUM" if wide_tier else "M5-TIGHT",
            "interval": interval,
            "triggered": False,
            "entry_status": res_entry["status"],
            "reason": res_entry.get("reason", ""),
            "opening_range": res_entry.get("opening_range"),
            "last": res_entry.get("last"),
            "strong_momentum": sm_preview,
        }

    # Save trader/entry.json
    edoc = entry_mod.entry_doc(res_entry, now_utc)
    edoc["source"] = "mt5"
    entry_mod.write_json("trader/entry.json", edoc)

    # Build Exit Plan:
    # - Tight-spread M5 symbols use Opening-Range midpoint stop ('or-mid') and '2R' target
    # - Wide-spread M15 symbols use ATR-based stop ('atr:<atr_sl_mult>') and '2R' target so target/stop outweigh spread
    stop_rule_arg = f"atr:{atr_sl_mult:g}" if wide_tier else "or-mid"
    exit_args = argparse.Namespace(
        from_path="trader/entry.json",
        symbol=None,
        entry=None,
        or_high=None,
        or_low=None,
        or_minutes=or_minutes,
        setup=None,
        date=None,
        time=None,
        entry_bar=None,
        interval=interval,
        side="buy",
        stop=stop_rule_arg,
        target="2R",
        flat_by=None,
        config="trader/config.json",
        offline=None,
        now=entry_mod.to_eastern(now_utc).strftime("%Y-%m-%d %H:%M"),
        write="trader/plan.json",
        json=False,
    )
    plan, plan_warnings = exit_mod.build_plan(exit_args, now_utc)
    entry_mod.write_json("trader/plan.json", plan)

    # Run Time Machine (10,000 replays on 60 days of MT5 5-minute bars)
    chart_text_60d = mt5_bridge.handle_yahoo_path(f"/v8/finance/chart/{sym}?range=60d&interval=5m")
    norm_plan = tm_mod.normalize_plan(plan, "trader/plan.json")
    tm_doc = tm_mod.time_machine(
        chart_text_60d,
        norm_plan,
        now_utc,
        replays=replays,
        seed=19,
        window=2,
        until=plan.get("flat_by"),
        source="mt5",
    )
    entry_mod.write_json("trader/timemachine.json", tm_doc)

    # Size position in MT5 lots using live symbol_info (strict 0.5% / $50 max risk cap + Strong Momentum check)
    trig_candle = res_entry.get("trigger")
    mt5_sizing = mt5_executor.calculate_mt5_lots(
        symbol=sym,
        entry_price=plan["entry"],
        sl_price=plan["stop"],
        tp_price=plan["target"],
        side="buy",
        max_risk_pct=risk_pct,
        strong_momentum=wide_tier,
        timeframe=interval,
        atr_sl_mult=atr_sl_mult,
        candle_override=trig_candle,
    )

    # Also write trader/orders.json and trader/state/risk.json via trader-budget & trader-risk
    s_cfg = size_mod.load_config("trader/config.json")
    s_cfg["max_risk_pct"] = risk_pct
    s_res = size_mod.size_trade(plan, s_cfg, today=plan["date"])
    order = size_mod.bracket_order(s_res)
    if order:
        size_mod.write_orders("trader/orders.json", [order], s_cfg, f"pocketbot_auto ({sym})", now_utc)
        r_res = risk_mod.check([order], s_cfg, today=plan["date"], now=now_utc)
        entry_mod.write_json("trader/state/risk.json", r_res)
    else:
        r_res = {"pass": 0, "block": 1, "results": []}

    safety = mt5_executor.run_safety_checks(mt5_sizing)

    return {
        "symbol": sym,
        "tier": "M15-MOMENTUM" if wide_tier else "M5-TIGHT",
        "interval": interval,
        "triggered": True,
        "entry": res_entry,
        "plan": plan,
        "plan_warnings": plan_warnings,
        "timemachine": tm_doc,
        "mt5_sizing": mt5_sizing,
        "strong_momentum": mt5_sizing.get("strong_momentum"),
        "safety": safety,
        "order": order,
        "risk_check": r_res,
    }


def run_scan_cycle(
    symbols,
    mode,
    now_utc=None,
    replays=10000,
    risk_pct=0.5,
    atr_sl_mult=mt5_executor.DEFAULT_ATR_SL_MULT,
    dry_run=False,
    allow_live=False,
):
    """Run one complete scan cycle across M5 tight-spread and M15 wide-spread symbols."""
    mt5_bridge.ensure_mt5()
    now_utc = now_utc or datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0, tzinfo=None)
    now_et = mt5_bridge.to_eastern(now_utc)

    m5_list = [s for s in symbols if not mt5_executor.is_wide_spread_symbol(s)]
    m15_list = [s for s in symbols if mt5_executor.is_wide_spread_symbol(s)]

    pnl = mt5_executor.get_daily_pnl_summary()
    log_event(
        f"SCAN START | mode={mode} | NY={now_et.strftime('%Y-%m-%d %H:%M:%S')} ET | "
        f"Equity=${pnl['equity']:,.2f} | Daily P/L=${pnl['total_daily_pnl']:+,.2f} ({pnl['daily_pnl_pct']:+.2f}%) | "
        f"OpenPos={pnl['open_positions_count']}/2 | M5={','.join(m5_list) or 'none'} | M15-Momentum={','.join(m15_list) or 'none'}"
    )

    # Global pre-scan circuit breakers
    if not pnl["is_demo"] and not allow_live:
        log_event("CIRCUIT BREAKER TRIPPED: Account is not a DEMO account (trade_mode != ACCOUNT_TRADE_MODE_DEMO). Aborting scan.")
        return 1
    if pnl["daily_pnl_pct"] <= -mt5_executor.DEFAULT_DAILY_LOSS_LIMIT_PCT:
        log_event(f"CIRCUIT BREAKER TRIPPED: Daily loss kill-switch active ({pnl['daily_pnl_pct']:+.2f}% <= -2.0%). Stopping trades for today.")
        return 1
    if pnl["open_positions_count"] >= mt5_executor.DEFAULT_MAX_OPEN_POSITIONS:
        log_event(f"CIRCUIT BREAKER TRIPPED: Max open positions reached ({pnl['open_positions_count']}/2). Skipping new entries.")
        return 0

    for sym in symbols:
        try:
            res = evaluate_symbol_pipeline(
                sym,
                now_utc,
                replays=replays,
                risk_pct=risk_pct,
                atr_sl_mult=atr_sl_mult,
            )
        except Exception as e:
            log_event(f"  [{sym:<6}] ERROR during evaluation: {e}")
            continue

        tier_tag = f"{res['tier']}:{res['interval'].upper()}"
        if not res["triggered"]:
            sm = res.get("strong_momentum")
            if sm:
                c = sm["candle"]
                log_event(
                    f"  [{sym:<6} | {tier_tag:<16}] status={res['entry_status']:<9} | {res['reason']} | "
                    f"MomentumCheck={'PASS' if sm['all_passed'] else 'WAIT'} "
                    f"(Spread={sm['spread_pct_of_sl']:.1f}% of {sm['atr_sl_mult']}xATR SL, {sm['spread_vs_rolling_ratio']:.2f}x avg; "
                    f"LastM15Body={c['body_atr_ratio']:.2f}x ATR; CloseLoc={c['close_extreme_pct']:.1f}%)"
                )
            else:
                log_event(f"  [{sym:<6} | {tier_tag:<16}] status={res['entry_status']:<9} | {res['reason']}")
            continue

        plan = res["plan"]
        tm = res["timemachine"]
        mc = tm["mc"]
        s = res["mt5_sizing"]
        sm = res.get("strong_momentum")
        safety = res["safety"]
        d = s["digits"]
        ev_r = mc["avg_R"]
        ev_positive = ev_r > 0.0
        all_checks_ok = safety["passed"] and (res["risk_check"].get("block", 0) == 0) and (s["lots"] > 0)

        sm_summary = ""
        if sm and sm.get("enabled"):
            c = sm["candle"]
            sm_summary = (
                f" | StrongMomentum={'PASS' if sm['all_passed'] else 'FAIL'} "
                f"[Spread={sm['spread_pct_of_sl']:.1f}% SL & {sm['spread_vs_rolling_ratio']:.2f}x avg, "
                f"Body={c['body_atr_ratio']:.2f}x ATR, CloseLoc={c['close_extreme_pct']:.1f}%]"
            )

        log_event(
            f"  [{sym:<6} | {tier_tag:<16}] TRIGGERED ORB BUY @ {s['entry']:.{d}f} | SL={s['sl']:.{d}f} | TP={s['tp']:.{d}f} ({s['reward_risk']}R) | "
            f"Lots={s['lots']} (Risk=${s['actual_risk_usd']:.2f} <= ${s['max_risk_usd']:.2f}, Spread={s['spread_pct_of_sl']:.1f}% of SL)"
            f"{sm_summary} | "
            f"TimeMachine({mc['replays']} runs): target={mc['p_target']*100:.1f}% stop={mc['p_stop']*100:.1f}% avg_R={ev_r:+.3f} ({tm['verdict']}) | "
            f"Safety={'PASS' if all_checks_ok else 'BLOCK'}"
        )

        if not all_checks_ok:
            failed = [b["rule"] + ": " + b["detail"] for b in safety["breakers"] if not b["passed"]]
            if s.get("refusal"):
                failed.append(s["refusal"])
            log_event(f"  [{sym:<6} | {tier_tag:<16}] BLOCKED BY SAFETY / MOMENTUM CHECKS: {'; '.join(failed)}")
            continue

        if mode == "confirm":
            preview = mt5_executor.verify_or_send_order(s, dry_run=True, allow_live=allow_live, comment=f"PocketBot {sym} {res['interval']}")
            print("\n" + mt5_executor.format_execution_report(preview))
            print(f"  Time Machine ({mc['replays']:,} replays): avg_R = {ev_r:+.3f} | Target {mc['p_target']*100:.1f}% vs Stop {mc['p_stop']*100:.1f}%")
            print(f"  Verdict: {tm['verdict']}")
            try:
                ans = input(f"\n>>> Place this {s['lots']} lot {s['mt5_symbol']} BUY bracket order on MT5 Demo? [YES/NO]: ").strip()
            except EOFError:
                ans = "NO"

            if ans.upper() == "YES":
                exec_res = mt5_executor.verify_or_send_order(s, dry_run=dry_run, allow_live=allow_live, comment=f"PocketBot {sym} {res['interval']}")
                log_event(
                    f"  [{sym:<6} | {tier_tag:<16}] USER CONFIRMED YES -> status={exec_res['status']} | "
                    f"lots={s['lots']} entry={s['entry']:.{d}f} sl={s['sl']:.{d}f} tp={s['tp']:.{d}f}"
                )
                print(mt5_executor.format_execution_report(exec_res))
            else:
                log_event(f"  [{sym:<6} | {tier_tag:<16}] USER DECLINED ({ans or 'NO'}) — order not sent.")

        elif mode == "demo-auto":
            if not ev_positive:
                log_event(
                    f"  [{sym:<6} | {tier_tag:<16}] DEMO-AUTO SKIPPED: Time Machine Expected Value is non-positive "
                    f"(avg_R = {ev_r:+.3f} <= 0.000; target={mc['p_target']*100:.1f}%, stop={mc['p_stop']*100:.1f}%)."
                )
                continue

            exec_res = mt5_executor.verify_or_send_order(s, dry_run=dry_run, allow_live=allow_live, comment=f"PocketBot {sym} Auto")
            log_event(
                f"  [{sym:<6} | {tier_tag:<16}] DEMO-AUTO EXECUTED (EV avg_R={ev_r:+.3f} > 0 & all checks PASS) -> "
                f"status={exec_res['status']} | lots={s['lots']} entry={s['entry']:.{d}f} sl={s['sl']:.{d}f} tp={s['tp']:.{d}f}"
            )
            print(mt5_executor.format_execution_report(exec_res))

    log_event("SCAN COMPLETE.")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="PocketBot MT5 Autonomous & Confirm-Mode Trading Runner (M5 Tight + M15 Strong Momentum).")
    ap.add_argument("--mode", required=True, choices=["confirm", "demo-auto"],
                    help="'confirm' asks YES/NO before sending; 'demo-auto' auto-places when Time Machine EV > 0 and checks pass")
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS),
                    help=f"Comma-separated symbols to scan (default: {','.join(DEFAULT_SYMBOLS)})")
    ap.add_argument("--replays", type=int, default=10000, help="Time Machine Monte Carlo replays (default 10,000)")
    ap.add_argument("--risk-pct", type=float, default=0.5, help="Max risk %% of equity per trade (default 0.5%% = $50 on $10,000)")
    ap.add_argument("--atr-sl-mult", type=float, default=mt5_executor.DEFAULT_ATR_SL_MULT,
                    help="M15 ATR Stop-Loss multiplier for wide-spread Strong Momentum symbols (default 3.5x ATR(14))")
    ap.add_argument("--interval-sec", type=int, default=300, help="Seconds between scan cycles (default 300 = 5 minutes)")
    ap.add_argument("--once", action="store_true", help="Run a single scan cycle and exit")
    ap.add_argument("--now", help='Pin New York clock for replay/testing, e.g. "2026-11-10 10:15"')
    ap.add_argument("--dry-run", action="store_true", help="Verify orders with mt5.order_check() without calling mt5.order_send()")
    ap.add_argument("--unlock-live", action="store_true", help="Unlock live account (requires POCKETBOT_UNLOCK_LIVE=YES)")
    args = ap.parse_args(argv)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    now_override = entry_mod.parse_now(args.now) if args.now else None

    try:
        while True:
            run_scan_cycle(
                symbols=symbols,
                mode=args.mode,
                now_utc=now_override,
                replays=args.replays,
                risk_pct=args.risk_pct,
                atr_sl_mult=args.atr_sl_mult,
                dry_run=args.dry_run,
                allow_live=args.unlock_live,
            )
            if args.once or now_override is not None:
                break
            log_event(f"Sleeping {args.interval_sec}s until next candle close...")
            time.sleep(args.interval_sec)
        return 0
    except KeyboardInterrupt:
        log_event("PocketBot stopped by user (KeyboardInterrupt).")
        return 0
    finally:
        if mt5_bridge._INITIALIZED:
            mt5.shutdown()
            mt5_bridge._INITIALIZED = False


if __name__ == "__main__":
    sys.exit(main())
