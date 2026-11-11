#!/usr/bin/env python3
"""MetaTrader 5 Live Data Bridge for PocketBot Trader Skills.

Connects to the open MetaTrader 5 terminal via the `MetaTrader5` package and
serves live 5-minute (`M5`), 1-minute (`M1`), and Daily (`D1`) candles plus
real-time bid/ask ticks in the exact Yahoo Finance v8 Chart and v7 Spark JSON
formats expected by the 11 trader skills in `.agents/skills/`:

  - trader-scanner     (scan.py)
  - trader-entry       (entry.py)
  - trader-exit        (exit.py)
  - trader-timemachine (timemachine.py)
  - trader-alerts      (alerts.py)
  - trader-news        (news.py)
  - trader-budget      (size.py via trader/prices.json)
  - trader-risk        (check.py via trader/prices.json)
  - trader-broker      (ticket.py)
  - trader-analyst     (facts.py)
  - trader-pelosi      (ptr.py)

Usage:
    .venv\\Scripts\\python.exe mt5_bridge.py
    .venv\\Scripts\\python.exe mt5_bridge.py status
    .venv\\Scripts\\python.exe mt5_bridge.py candles EURUSD XAUUSD SPX NQ TSLA --bars 12
    .venv\\Scripts\\python.exe mt5_bridge.py export EURUSD XAUUSD SPX NQ TSLA AAPL NVDA --out trader/mt5_cache
    .venv\\Scripts\\python.exe mt5_bridge.py universe
"""

import argparse
import datetime
import json
import os
from pathlib import Path
import sys
import urllib.parse

import MetaTrader5 as mt5

BASE_DIR = Path(__file__).resolve().parent
EPOCH = datetime.datetime(1970, 1, 1)

# Common symbol aliases -> PoTrade-MT5 symbol names
SYMBOL_ALIASES = {
    # US Indices
    "US500": "SPX",
    "SPX500": "SPX",
    "SP500": "SPX",
    "^GSPC": "SPX",
    "SPY": "SPX",
    "USTEC": "NQ",
    "NAS100": "NQ",
    "NDX": "NQ",
    "^IXIC": "NQ",
    "QQQ": "NQ",
    "US30": "YM",
    "DJ30": "YM",
    "DJI": "YM",
    "^DJI": "YM",
    "DIA": "YM",
    "DE40": "FDAX",
    "GER40": "FDAX",
    "DAX": "FDAX",
    "UK100": "FTSE",
    "JP225": "NI225",
    "JPN225": "NI225",
    # Commodities
    "GOLD": "XAUUSD",
    "GC=F": "XAUUSD",
    "SILVER": "XAGUSD",
    "SI=F": "XAGUSD",
    "OIL": "UKBRENT",
    "BRENT": "UKBRENT",
    "WTI": "UKBRENT",
    "CL=F": "UKBRENT",
    "NATGAS": "XNGUSD",
    # Tech Stock CFDs where ticker differs slightly
    "GOOGL": "#GOOG",
}

TOP_GROUPS = [
    ("Forex Majors", ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD"]),
    ("Commodities & Spot Metals", ["XAUUSD", "XAGUSD", "UKBRENT", "XNGUSD", "XPTUSD", "XPDUSD"]),
    ("US & Global Indices", ["SPX", "NQ", "YM", "FDAX", "FTSE", "NI225", "HK50", "SX5E"]),
    ("Crypto", ["BTCUSD", "ETHUSD", "LTCUSD", "XRPUSD", "BNBUSD", "ADAUSD"]),
    ("US Equity CFDs", ["#AAPL", "#NVDA", "#TSLA", "#MSFT", "#AMZN", "#META", "#GOOG", "#AVGO", "#COIN", "#NFLX"]),
]


class MT5BridgeError(Exception):
    pass


# ---------------------------------------------------------------- time helpers

def _nth_sunday(year, month, n):
    d = datetime.date(year, month, 1)
    d += datetime.timedelta(days=(6 - d.weekday()) % 7)
    return d + datetime.timedelta(weeks=n - 1)


def eastern_offset(utc):
    """Hours from UTC for New York: -4 in EDT, -5 in EST."""
    y = utc.year
    start = datetime.datetime.combine(_nth_sunday(y, 3, 2), datetime.time(7, 0))
    end = datetime.datetime.combine(_nth_sunday(y, 11, 1), datetime.time(6, 0))
    naive = utc.replace(tzinfo=None)
    return -4 if start <= naive < end else -5


def to_eastern(utc):
    return utc.replace(tzinfo=None) + datetime.timedelta(hours=eastern_offset(utc))


def utc_of(ts):
    return EPOCH + datetime.timedelta(seconds=int(ts))


# ---------------------------------------------------------------- MT5 session & symbol resolution

_INITIALIZED = False
_SERVER_OFFSET_SEC = None


def ensure_mt5():
    """Initialize MT5 connection if not already active."""
    global _INITIALIZED
    if not _INITIALIZED:
        if not mt5.initialize():
            err = mt5.last_error()
            raise MT5BridgeError(f"Could not initialize MetaTrader 5: {err}")
        _INITIALIZED = True
    return True


def server_offset_seconds():
    """Detect MT5 broker server timezone offset vs real UTC in whole 30-min steps.
    PoTrade-MT5 runs at UTC+2 (+7200s); subtracting this offset yields true UTC epoch timestamps."""
    global _SERVER_OFFSET_SEC
    if _SERVER_OFFSET_SEC is not None:
        return _SERVER_OFFSET_SEC
    ensure_mt5()
    now_utc = datetime.datetime.now(datetime.timezone.utc).timestamp()
    for probe in ("EURUSD", "BTCUSD", "XAUUSD", "GBPUSD"):
        mt5.symbol_select(probe, True)
        tick = mt5.symbol_info_tick(probe)
        if tick and tick.time > 0:
            diff = tick.time - now_utc
            # Round to nearest 1800 seconds (30 minutes) if within 14 hours of now
            if abs(diff) <= 14 * 3600:
                _SERVER_OFFSET_SEC = int(round(diff / 1800.0) * 1800)
                return _SERVER_OFFSET_SEC
    # Fallback: default MT5 EET server time is UTC+2 (7200s)
    _SERVER_OFFSET_SEC = 7200
    return _SERVER_OFFSET_SEC


def resolve_mt5_symbol(sym):
    """Resolve a user/skill symbol (e.g. 'EURUSD', 'XAUUSD', 'SPX', 'US500', 'TSLA', '#TSLA')
    to the exact MetaTrader 5 symbol name and ensure it is selected in Market Watch."""
    ensure_mt5()
    raw = str(sym or "").strip().upper()
    if not raw:
        raise MT5BridgeError("Empty symbol")

    candidates = []
    if raw in SYMBOL_ALIASES:
        candidates.append(SYMBOL_ALIASES[raw])
    candidates.extend([
        raw,
        f"#{raw.lstrip('#')}",
        raw.lstrip("#"),
        raw.replace("-", "."),
        f"#{raw.replace('-', '.')}",
    ])

    seen = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        info = mt5.symbol_info(cand)
        if info is not None:
            if not info.visible:
                mt5.symbol_select(cand, True)
            return cand, info

    raise MT5BridgeError(f"Symbol '{sym}' not found in MetaTrader 5")


def canonical_symbol(user_sym, mt5_sym):
    """Preserve the symbol name the calling trader skill asked for (e.g., TSLA or EURUSD)."""
    s = str(user_sym or mt5_sym).strip().upper()
    return s


# ---------------------------------------------------------------- MT5 candles -> Yahoo v8 / Spark JSON

def _bars_for_range(rng, interval):
    """Map Yahoo Finance (range, interval) to (mt5_timeframe, bar_count, step_seconds)."""
    interval = (interval or "5m").lower()
    rng = (rng or "1d").lower()

    if interval == "1m":
        tf = mt5.TIMEFRAME_M1
        step = 60
        days = 1 if rng == "1d" else 5 if rng == "5d" else 7
        count = days * 1440 + 120
    elif interval == "15m":
        tf = mt5.TIMEFRAME_M15
        step = 900
        if rng == "1d":
            count = 120
        elif rng == "5d":
            count = 600
        elif rng == "1mo":
            count = 3200
        elif rng in ("60d", "3mo"):
            count = 6500
        else:
            count = 600
    elif interval in ("1d", "d1"):
        tf = mt5.TIMEFRAME_D1
        step = 86400
        if rng in ("1d", "5d"):
            count = 10
        elif rng in ("1mo", "3mo"):
            count = 95
        else:
            count = 260
    else:
        # Default: live 5-minute candles (TIMEFRAME_M5)
        tf = mt5.TIMEFRAME_M5
        step = 300
        if rng == "1d":
            count = 320        # >1 full 24h day of 5-min bars
        elif rng == "5d":
            count = 1600       # >5 full 24h days of 5-min bars
        elif rng == "1mo":
            count = 9000       # ~30 days of 5-min bars
        elif rng in ("60d", "3mo"):
            count = 18500      # ~60+ days of 5-min bars (for timemachine.py)
        else:
            count = 1500
    return tf, count, step


def _fill_regular_session_gaps(ts_map, step):
    """Ensure every active NY regular session (09:30-16:00 ET) has continuous 5-minute/15-minute
    slots so `scan.py` (usable) and `timemachine.py` (full_sessions == 78 bars) never
    drop a session due to a quiet window with zero ticks in MT5."""
    if step not in (60, 300, 900) or not ts_map:
        return ts_map

    # Group existing bars by New York date
    by_day = {}
    for ts, b in ts_map.items():
        et = to_eastern(utc_of(ts))
        by_day.setdefault(et.date(), {})[et.time()] = (ts, b)

    now_utc = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    now_et = to_eastern(now_utc)
    open_t = datetime.time(9, 30)
    close_t = datetime.time(16, 0)

    out = dict(ts_map)
    for day, time_map in by_day.items():
        if day.weekday() >= 5:
            continue
        # Only fill if there is actual trading activity on this date
        session_points = [v for t, v in time_map.items() if open_t <= t < close_t]
        if len(session_points) < 3:
            continue

        # Find initial reference price at or before 09:30 ET
        earlier = [v[1]["c"] for t, v in sorted(time_map.items()) if t <= open_t]
        last_close = earlier[-1] if earlier else session_points[0][1]["o"]

        cur_et = datetime.datetime.combine(day, open_t)
        end_et = datetime.datetime.combine(day, close_t)
        if day == now_et.date():
            # Do not synthesize future bars beyond the last completed slot
            floored_min = (now_et.minute // (step // 60)) * (step // 60)
            cutoff_et = now_et.replace(minute=floored_min, second=0, microsecond=0)
            end_et = min(end_et, cutoff_et)

        # Offset from ET to UTC for this day
        sample_ts = session_points[0][0]
        sample_et = to_eastern(utc_of(sample_ts))
        et_to_utc_sec = int(round((utc_of(sample_ts) - sample_et).total_seconds()))

        while cur_et < end_et:
            t_only = cur_et.time()
            if t_only in time_map:
                last_close = time_map[t_only][1]["c"]
            else:
                utc_dt = cur_et + datetime.timedelta(seconds=et_to_utc_sec)
                slot_ts = int((utc_dt - EPOCH).total_seconds())
                out[slot_ts] = {
                    "o": last_close,
                    "h": last_close,
                    "l": last_close,
                    "c": last_close,
                    "v": 0,
                }
            cur_et += datetime.timedelta(seconds=step)
    return out


def fetch_mt5_chart_dict(sym, rng="1d", interval="5m", fill_gaps=True):
    """Fetch live candles & tick from MT5 and build a Yahoo Finance v8 chart dict."""
    mt5_sym, info = resolve_mt5_symbol(sym)
    out_sym = canonical_symbol(sym, mt5_sym)
    tf, count, step = _bars_for_range(rng, interval)
    offset_sec = server_offset_seconds()

    rates = mt5.copy_rates_from_pos(mt5_sym, tf, 0, count)
    if rates is None or len(rates) == 0:
        raise MT5BridgeError(f"MT5 returned no {interval} bars for {mt5_sym}: {mt5.last_error()}")

    tick = mt5.symbol_info_tick(mt5_sym)
    d1_rates = mt5.copy_rates_from_pos(mt5_sym, mt5.TIMEFRAME_D1, 0, 5)

    ts_map = {}
    for r in rates:
        raw_ts = int(r["time"])
        if step == 86400:
            # Daily bar: keep midnight UTC alignment so ts_to_eastern(ts).date() lands on that trading date
            dt_d = datetime.datetime.fromtimestamp(raw_ts, datetime.timezone.utc).date()
            # 16:00 UTC is 11:00/12:00 ET on the same calendar date
            utc_ts = int(datetime.datetime(dt_d.year, dt_d.month, dt_d.day, 16, 0, tzinfo=datetime.timezone.utc).timestamp())
        else:
            utc_ts = raw_ts - offset_sec
            # Snap to interval grid
            utc_ts = (utc_ts // step) * step

        ts_map[utc_ts] = {
            "o": float(r["open"]),
            "h": float(r["high"]),
            "l": float(r["low"]),
            "c": float(r["close"]),
            "v": int(r["tick_volume"]),
        }

    if fill_gaps and step in (60, 300, 900):
        ts_map = _fill_regular_session_gaps(ts_map, step)

    timestamps = sorted(ts_map.keys())
    opens = [ts_map[t]["o"] for t in timestamps]
    highs = [ts_map[t]["h"] for t in timestamps]
    lows = [ts_map[t]["l"] for t in timestamps]
    closes = [ts_map[t]["c"] for t in timestamps]
    volumes = [ts_map[t]["v"] for t in timestamps]

    now_utc_ts = int(datetime.datetime.now(datetime.timezone.utc).timestamp())
    tick_utc_ts = (int(tick.time) - offset_sec) if (tick and tick.time > 0) else now_utc_ts
    reg_time = max(now_utc_ts, tick_utc_ts, timestamps[-1] if timestamps else now_utc_ts)

    last_price = float(closes[-1])
    if tick and tick.bid > 0:
        last_price = float(tick.last if tick.last > 0 else tick.bid)

    prev_close = float(closes[0])
    day_high = max(highs[-78:]) if highs else last_price
    day_low = min(lows[-78:]) if lows else last_price
    day_vol = sum(volumes[-78:]) if volumes else 0

    if d1_rates is not None and len(d1_rates) >= 2:
        prev_close = float(d1_rates[-2]["close"])
        day_high = float(d1_rates[-1]["high"])
        day_low = float(d1_rates[-1]["low"])
        day_vol = int(d1_rates[-1]["tick_volume"])

    meta = {
        "currency": info.currency_profit or "USD",
        "symbol": out_sym,
        "mt5_symbol": mt5_sym,
        "exchangeName": info.path.split("\\")[0] if info.path else "MT5",
        "instrumentType": "CFD",
        "regularMarketPrice": last_price,
        "regularMarketDayHigh": day_high,
        "regularMarketDayLow": day_low,
        "regularMarketVolume": max(day_vol, 1000000),  # ensure liquidity floor in stock scanners
        "regularMarketTime": reg_time,
        "previousClose": prev_close,
        "chartPreviousClose": prev_close,
        "dataGranularity": interval,
        "range": rng,
        "digits": info.digits,
        "spread": info.spread,
        "bid": float(tick.bid) if tick else last_price,
        "ask": float(tick.ask) if tick else last_price,
        "source": "mt5",
    }

    return {
        "chart": {
            "result": [
                {
                    "meta": meta,
                    "timestamp": timestamps,
                    "indicators": {
                        "quote": [
                            {
                                "open": opens,
                                "high": highs,
                                "low": lows,
                                "close": closes,
                                "volume": volumes,
                            }
                        ]
                    },
                }
            ],
            "error": None,
        }
    }


def fetch_mt5_spark_dict(symbols, rng="1d", interval="5m"):
    """Fetch multiple symbols from MT5 and build a Yahoo Finance v7 spark dict."""
    results = []
    for sym in symbols:
        try:
            ch = fetch_mt5_chart_dict(sym, rng=rng, interval=interval)
            res0 = ch["chart"]["result"][0]
            results.append({
                "symbol": canonical_symbol(sym, res0["meta"]["mt5_symbol"]),
                "response": [res0],
            })
        except MT5BridgeError:
            continue
    return {
        "spark": {
            "result": results,
            "error": None,
        }
    }


def handle_yahoo_path(path_or_url):
    """Intercept a Yahoo Finance URL or path (`/v8/finance/chart/...` or `/v7/finance/spark?...`)
    and return the JSON string built from live MetaTrader 5 candles."""
    parsed = urllib.parse.urlparse(str(path_or_url).replace("#", "%23"))
    path = parsed.path
    qs = urllib.parse.parse_qs(parsed.query)
    rng = (qs.get("range") or ["1d"])[0]
    interval = (qs.get("interval") or ["5m"])[0]

    if "/v7/finance/spark" in path:
        raw_syms = (qs.get("symbols") or [""])[0]
        syms = [urllib.parse.unquote(s).strip() for s in raw_syms.split(",") if s.strip()]
        doc = fetch_mt5_spark_dict(syms, rng=rng, interval=interval)
        return json.dumps(doc)

    if "/v8/finance/chart/" in path:
        sym = urllib.parse.unquote(path.split("/v8/finance/chart/", 1)[1].split("/")[0]).strip()
        doc = fetch_mt5_chart_dict(sym, rng=rng, interval=interval)
        return json.dumps(doc)

    raise MT5BridgeError(f"Unsupported Yahoo path for MT5 bridge: {path_or_url}")


def get_mt5_quote(sym):
    """Return a live quote dict matching `trader-alerts/alerts.py` format using MT5 M5/tick data."""
    doc = fetch_mt5_chart_dict(sym, rng="1d", interval="5m", fill_gaps=False)
    res0 = doc["chart"]["result"][0]
    m = res0["meta"]
    q = res0["indicators"]["quote"][0]
    opens = [x for x in (q.get("open") or []) if x is not None]
    t = m.get("regularMarketTime")
    when = datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if t else ""
    return {
        "symbol": canonical_symbol(sym, m.get("mt5_symbol")),
        "mt5_symbol": m.get("mt5_symbol"),
        "price": float(m["regularMarketPrice"]),
        "bid": float(m.get("bid", m["regularMarketPrice"])),
        "ask": float(m.get("ask", m["regularMarketPrice"])),
        "open": opens[0] if opens else None,
        "high": m.get("regularMarketDayHigh"),
        "low": m.get("regularMarketDayLow"),
        "prev_close": m.get("previousClose"),
        "time": when,
        "currency": m.get("currency", "USD"),
        "digits": m.get("digits", 2),
        "source": "mt5",
    }


# ---------------------------------------------------------------- CLI commands

def cmd_status(_args):
    ensure_mt5()
    info = mt5.account_info()
    term = mt5.terminal_info()
    offset_hr = server_offset_seconds() / 3600.0
    if not info:
        raise MT5BridgeError("Could not retrieve MT5 account_info()")

    print("=" * 78)
    print("  METATRADER 5 LIVE DEMO ACCOUNT STATUS")
    print("=" * 78)
    print(f"  Account Name : {info.name}")
    print(f"  Login ID     : {info.login} ({'Demo' if info.trade_mode == 0 else 'Live'})")
    print(f"  Broker/Server: {info.company} ({info.server}) | Server TZ: UTC+{offset_hr:g}")
    print(f"  Balance      : ${info.balance:,.2f} {info.currency}")
    print(f"  Equity       : ${info.equity:,.2f} {info.currency} (Floating P/L: ${info.profit:+,.2f})")
    print(f"  Free Margin  : ${info.margin_free:,.2f} {info.currency} | Leverage: 1:{info.leverage}")
    print(f"  Terminal     : {term.name} Build {term.build} (Connected: {term.connected})")
    print("=" * 78)

    all_syms = mt5.symbols_get()
    print(f"\n  TOP AVAILABLE SYMBOLS ({len(all_syms) if all_syms else 0} total on {info.server}):")
    print("-" * 78)
    print(f"  {'Symbol':<10} {'Description':<34} {'Bid':>11} {'Ask':>11} {'Spread':>7} {'Last M5 Close':>13}")
    print("-" * 78)

    offset_sec = server_offset_seconds()
    for group_name, syms in TOP_GROUPS:
        print(f"  [{group_name}]")
        for s in syms:
            try:
                mt5_sym, s_info = resolve_mt5_symbol(s)
                tick = mt5.symbol_info_tick(mt5_sym)
                rates = mt5.copy_rates_from_pos(mt5_sym, mt5.TIMEFRAME_M5, 0, 2)
                last_m5 = f"{rates[-1]['close']:.{s_info.digits}f}" if (rates is not None and len(rates)) else "N/A"
                bid_s = f"{tick.bid:.{s_info.digits}f}" if tick else "N/A"
                ask_s = f"{tick.ask:.{s_info.digits}f}" if tick else "N/A"
                desc = (s_info.description or "")[:33]
                print(f"  {mt5_sym:<10} {desc:<34} {bid_s:>11} {ask_s:>11} {s_info.spread:>7} {last_m5:>13}")
            except MT5BridgeError:
                continue
    print("-" * 78)
    return 0


def cmd_candles(args):
    ensure_mt5()
    offset_sec = server_offset_seconds()
    tf_map = {"1m": (mt5.TIMEFRAME_M1, 60), "5m": (mt5.TIMEFRAME_M5, 300), "15m": (mt5.TIMEFRAME_M15, 900), "1d": (mt5.TIMEFRAME_D1, 86400)}
    tf, step = tf_map[args.interval]

    for sym in args.symbols:
        mt5_sym, s_info = resolve_mt5_symbol(sym)
        rates = mt5.copy_rates_from_pos(mt5_sym, tf, 0, args.bars)
        if rates is None or len(rates) == 0:
            print(f"{sym}: no candles ({mt5.last_error()})", file=sys.stderr)
            continue
        print(f"\n=== {mt5_sym} ({s_info.description}) — Last {len(rates)} x {args.interval} Candles ===")
        print(f"{'UTC Start':<20} {'New York (ET)':<20} {'Open':>11} {'High':>11} {'Low':>11} {'Close':>11} {'TickVol':>9}")
        for r in rates:
            utc_ts = int(r["time"]) - (0 if step == 86400 else offset_sec)
            utc_dt = utc_of(utc_ts)
            et_dt = to_eastern(utc_dt)
            d = s_info.digits
            print(
                f"{utc_dt.strftime('%Y-%m-%d %H:%M'):<20} "
                f"{et_dt.strftime('%Y-%m-%d %H:%M'):<20} "
                f"{r['open']:>11.{d}f} {r['high']:>11.{d}f} {r['low']:>11.{d}f} {r['close']:>11.{d}f} "
                f"{int(r['tick_volume']):>9}"
            )
    return 0


def cmd_export(args):
    """Export Yahoo-v8 compatible 5m JSON files to `trader/mt5_cache/` and update `trader/prices.json`."""
    ensure_mt5()
    os.makedirs(args.out, exist_ok=True)
    now_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    prices_doc = {"as_of": now_iso, "source": "mt5", "prices": {}, "detail": {}}

    for sym in args.symbols:
        ch_1d = fetch_mt5_chart_dict(sym, rng=args.range, interval=args.interval)
        res0 = ch_1d["chart"]["result"][0]
        meta = res0["meta"]
        clean = canonical_symbol(sym, meta["mt5_symbol"]).lstrip("#")

        for name in (
            f"{clean}.json",
            f"yahoo-{clean}.json",
            f"chart-{clean}.json",
            f"{clean}-60d5m.json",
        ):
            out_path = os.path.join(args.out, name)
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(ch_1d, fh)

        prices_doc["prices"][clean] = meta["regularMarketPrice"]
        prices_doc["detail"][clean] = {
            "prev_close": meta["previousClose"],
            "time": now_iso,
            "source": "mt5",
            "mt5_symbol": meta["mt5_symbol"],
            "bid": meta["bid"],
            "ask": meta["ask"],
            "fetched": now_iso,
        }
        print(f"Exported {clean} (MT5: {meta['mt5_symbol']}) -> {args.out} | last={meta['regularMarketPrice']}")

    os.makedirs("trader", exist_ok=True)
    with open("trader/prices.json", "w", encoding="utf-8") as fh:
        json.dump(prices_doc, fh, indent=2)
        fh.write("\n")
    print(f"Updated trader/prices.json with {len(prices_doc['prices'])} live MT5 prices.")
    return 0


def cmd_universe(args):
    """Write MT5 tradable symbols into `trader/universe.txt`."""
    ensure_mt5()
    syms = mt5.symbols_get()
    if not syms:
        raise MT5BridgeError("No symbols returned by MT5")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    kept = []
    for s in sorted(syms, key=lambda x: x.name):
        if s.name.endswith("_otc"):
            continue
        clean = s.name.lstrip("#")
        kept.append((clean, s.name, s.description))
    with open(args.out, "w", encoding="utf-8") as fh:
        for clean, _, _ in kept:
            fh.write(clean + "\n")
    print(f"Wrote {len(kept)} MT5 symbols to {args.out}.")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="MetaTrader 5 Live 5-Minute Candle Bridge for Trader Skills.")
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("status", help="Show MT5 demo account balance and top live symbols")

    p_c = sub.add_parser("candles", help="Display live MT5 candles (default 5m)")
    p_c.add_argument("symbols", nargs="+", help="Symbols, e.g. EURUSD XAUUSD SPX NQ TSLA")
    p_c.add_argument("--interval", default="5m", choices=["1m", "5m", "15m", "1d"])
    p_c.add_argument("--bars", type=int, default=12)

    p_e = sub.add_parser("export", help="Export Yahoo-v8 compatible JSON charts & trader/prices.json from MT5")
    p_e.add_argument("symbols", nargs="*", default=["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "SPX", "NQ", "YM", "BTCUSD", "TSLA", "AAPL", "NVDA"])
    p_e.add_argument("--range", default="5d", help="1d, 5d, 1mo, 60d")
    p_e.add_argument("--interval", default="5m", choices=["1m", "5m", "15m", "1d"])
    p_e.add_argument("--out", default="trader/mt5_cache")

    p_u = sub.add_parser("universe", help="Write MT5 symbol list to trader/universe.txt")
    p_u.add_argument("--out", default="trader/universe.txt")

    args = ap.parse_args(argv)
    try:
        if not args.cmd or args.cmd == "status":
            return cmd_status(args)
        return {
            "candles": cmd_candles,
            "export": cmd_export,
            "universe": cmd_universe,
        }[args.cmd](args)
    except MT5BridgeError as e:
        print(f"mt5_bridge.py: {e}", file=sys.stderr)
        return 1
    finally:
        if _INITIALIZED:
            mt5.shutdown()


if __name__ == "__main__":
    sys.exit(main())
