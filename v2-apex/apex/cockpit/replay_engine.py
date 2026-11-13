"""100x Speed Historical Market Replay Engine (`apex/cockpit/replay_engine.py`).

Replays up to the last 5 trading days of M5/M15 candles across FX, Gold, Indices, and US Stocks
at 50x-100x speed, streaming simulated entries, Z3 Aegis vetoes, SL/TP closes, Win Rate, Profit
Factor, and an equity curve to the WebSocket Cockpit. Strictly isolated from live `trade_ledger`.
"""

from __future__ import annotations

import datetime
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from apex.ai.aegis_verifier import AegisVerifier
from apex.ai.veto_committee import CandidateSetupAuditContext
from apex.feeds.models import Candle, Side
from apex.quant.regime import MarketRegime, RegimeClassifier

DEFAULT_REPLAY_SYMBOLS: tuple[str, ...] = (
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "XAUUSD",
    "US500",
    "NAS100",
    "NVDA",
    "TSLA",
)


@dataclass
class ReplaySummary:
    """Final isolated summary of a 100x historical market replay."""

    symbols_replayed: list[str] = field(default_factory=list)
    bars_processed: int = 0
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    vetoes: int = 0
    win_rate_pct: float = 0.0
    profit_factor: float = 0.0
    starting_equity: float = 10000.0
    ending_equity: float = 10000.0
    equity_curve: list[float] = field(default_factory=lambda: [10000.0])


class HistoricalReplayEngine:
    """Isolated 100x historical replay engine (zero writes to live `trade_ledger`)."""

    def __init__(
        self,
        starting_equity: float = 10000.0,
        risk_per_trade_pct: float = 0.50,
        step_delay_seconds: float = 0.02,
    ) -> None:
        self.starting_equity = float(starting_equity)
        self.risk_per_trade_pct = float(risk_per_trade_pct)
        self.step_delay_seconds = float(step_delay_seconds)
        self.regime_classifier = RegimeClassifier()

    def run_replay(
        self,
        feed_router: Any = None,
        candles_by_symbol: dict[str, Sequence[Candle]] | None = None,
        symbols: Sequence[str] = DEFAULT_REPLAY_SYMBOLS,
        on_update: Callable[[dict[str, Any]], None] | None = None,
        stop_check: Callable[[], bool] | None = None,
    ) -> ReplaySummary:
        """Execute a 100x replay over up to 5 trading days of M5/M15 candles (no live DB writes)."""
        sym_candles: dict[str, list[Candle]] = {}
        if candles_by_symbol is not None:
            for s, cs in candles_by_symbol.items():
                if len(cs) >= 35:
                    sym_candles[s] = list(cs)
        elif feed_router is not None:
            for sym in symbols:
                try:
                    cs = feed_router.get_candles(sym, timeframe="M15", count=480)
                    if len(cs) < 35:
                        cs = feed_router.get_candles(sym, timeframe="M5", count=480)
                    if len(cs) >= 35:
                        sym_candles[sym] = list(cs)
                except Exception:
                    continue

        if not sym_candles:
            summary = ReplaySummary(starting_equity=self.starting_equity, ending_equity=self.starting_equity)
            if on_update:
                on_update(
                    {
                        "status": "NO_DATA",
                        "message": "No historical candles available for replay.",
                        "equity_curve": summary.equity_curve,
                    }
                )
            return summary

        equity = self.starting_equity
        equity_curve: list[float] = [round(equity, 2)]
        gross_profit = 0.0
        gross_loss = 0.0
        wins = 0
        losses = 0
        vetoes = 0
        bars_processed = 0
        risk_amt = self.starting_equity * (self.risk_per_trade_pct / 100.0)

        min_len = min(len(v) for v in sym_candles.values())
        window = 30
        stride = max(1, (min_len - window - 6) // 45)  # ~45 animated frames (~1-2s at 0.02s delay, or ~60s at higher counts)

        if on_update:
            on_update(
                {
                    "status": "RUNNING",
                    "message": f"Replaying {len(sym_candles)} symbols ({min_len} bars) at 100x speed...",
                    "equity_curve": list(equity_curve),
                }
            )

        for idx in range(window, min_len - 5, stride):
            if stop_check and stop_check():
                break
            for sym, cs in sym_candles.items():
                bars_processed += 1
                hist = cs[idx - window : idx]
                future = cs[idx : idx + 5]
                closes = np.asarray([c.close for c in hist], dtype=float)
                diag = self.regime_classifier.classify_candles(sym, hist)
                if diag.regime != MarketRegime.TRENDING_EXPANSION:
                    continue

                direction = Side.BUY if closes[-1] > closes[-5] else Side.SELL
                entry = float(hist[-1].close)
                atr = float(np.mean(np.abs(np.diff(closes[-15:])))) or (entry * 0.002)
                sl_dist = max(atr * 1.5, entry * 0.001)
                tp_dist = sl_dist * 2.0
                sl = entry - sl_dist if direction == Side.BUY else entry + sl_dist
                tp = entry + tp_dist if direction == Side.BUY else entry - tp_dist

                ctx = CandidateSetupAuditContext(
                    symbol=sym,
                    side=direction,
                    timeframe=hist[-1].timeframe,
                    entry_price=entry,
                    stop_loss=sl,
                    take_profit=tp,
                    regime=diag.regime.value,
                    hurst_exponent=diag.hurst_exponent,
                    kaufman_er=diag.kaufman_er,
                    volatility_percentile=diag.volatility_percentile,
                    dom_obi=0.35 if direction == Side.BUY else -0.35,
                    spread_pct_of_sl=0.05,
                    net_avg_R=0.28,
                    prob_profit=0.58,
                    cvar_95_R=-0.65,
                    recommended_risk_pct=self.risk_per_trade_pct,
                    session_utc_iso=datetime.datetime.fromtimestamp(
                        float(hist[-1].timestamp_utc or time.time()), tz=datetime.timezone.utc
                    ).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    bar_timestamp=float(hist[-1].timestamp_utc or 0.0),
                )
                blinded = AegisVerifier.normalize_ticker_blinded_telemetry(ctx)
                z3_dec, _ = AegisVerifier.z3_pure_math_axiom_gate(blinded.telemetry)
                if z3_dec == "VETO":
                    vetoes += 1
                    continue

                # Simulate forward 5 bars for SL / TP hit
                pnl_r = 0.0
                for bar in future:
                    if direction == Side.BUY:
                        if bar.low <= sl:
                            pnl_r = -1.0
                            break
                        if bar.high >= tp:
                            pnl_r = 2.0
                            break
                    else:
                        if bar.high >= sl:
                            pnl_r = -1.0
                            break
                        if bar.low <= tp:
                            pnl_r = 2.0
                            break
                if pnl_r == 0.0:
                    last_c = float(future[-1].close)
                    pnl_r = ((last_c - entry) / sl_dist) if direction == Side.BUY else ((entry - last_c) / sl_dist)
                    pnl_r = max(-1.0, min(2.0, pnl_r))

                pnl_cash = pnl_r * risk_amt
                equity += pnl_cash
                if pnl_cash >= 0.0:
                    wins += 1
                    gross_profit += pnl_cash
                else:
                    losses += 1
                    gross_loss += abs(pnl_cash)
                equity_curve.append(round(equity, 2))

            if on_update:
                total_tr = wins + losses
                wr = (wins / total_tr * 100.0) if total_tr > 0 else 0.0
                pf = (gross_profit / gross_loss) if gross_loss > 0 else (2.0 if gross_profit > 0 else 0.0)
                on_update(
                    {
                        "status": "RUNNING",
                        "message": (
                            f"100x Replay: {bars_processed} bars | Trades={total_tr} | "
                            f"WinRate={wr:.1f}% | PF={pf:.2f} | Vetoes={vetoes} | Eq={equity:.2f}"
                        ),
                        "equity_curve": equity_curve[-80:],
                    }
                )
            if self.step_delay_seconds > 0:
                time.sleep(self.step_delay_seconds)

        total_trades = wins + losses
        win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0
        pf = (gross_profit / gross_loss) if gross_loss > 0 else (2.0 if gross_profit > 0 else 0.0)
        summary = ReplaySummary(
            symbols_replayed=list(sym_candles.keys()),
            bars_processed=bars_processed,
            total_trades=total_trades,
            wins=wins,
            losses=losses,
            vetoes=vetoes,
            win_rate_pct=round(win_rate, 2),
            profit_factor=round(pf, 2),
            starting_equity=self.starting_equity,
            ending_equity=round(equity, 2),
            equity_curve=equity_curve,
        )
        if on_update:
            on_update(
                {
                    "status": "COMPLETED",
                    "message": (
                        f"Replay Complete: {total_trades} trades | Win Rate={summary.win_rate_pct:.1f}% | "
                        f"PF={summary.profit_factor:.2f} | Z3 Vetoes={vetoes} | "
                        f"Final Eq={summary.ending_equity:.2f}"
                    ),
                    "equity_curve": equity_curve[-80:],
                }
            )
        return summary
