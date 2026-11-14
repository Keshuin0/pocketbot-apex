"""Unified 11-Agent Historical Market Replay Engine (`apex/cockpit/replay_engine.py`).

Replays 2d/5d/10d of M5/M15 candles at 20x/50x/100x speed through the shared
`Canonical11AgentPipeline` + `AutonomousPositionOrchestrator` (Grace Lock, +1.0R Free-Roll,
+1.5R -> 3.5R Runner Mode), streaming trades, 11-stage X-Ray traces, Win Rate, Profit
Factor, and equity curve over `/ws`. Strictly isolated from live `trade_ledger`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from apex.engine.canonical_pipeline import AgentPipelineTrace, Canonical11AgentPipeline
from apex.execution.position_manager import AutonomousPositionOrchestrator
from apex.feeds.models import Candle, Side
from apex.quant.regime import RegimeClassifier
from apex.quant.timemachine_v2 import BlockBootstrapTimeMachine

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
class ReplayTradeRecord:
    """Single simulated trade produced by 11-Agent + In-Flight Replay."""

    symbol: str
    side: str
    entry_price: float
    exit_price: float
    ratchet_stage: str
    exit_reason: str
    net_r: float
    pnl_cad: float
    trace: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReplaySummary:
    """Final isolated summary of a historical market replay."""

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
    trades: list[dict[str, Any]] = field(default_factory=list)
    last_trace: Optional[dict[str, Any]] = None


class HistoricalReplayEngine:
    """Isolated 11-Agent + In-Flight historical replay engine (zero writes to live `trade_ledger`)."""

    def __init__(
        self,
        starting_equity: float = 10000.0,
        risk_per_trade_pct: float = 0.50,
        step_delay_seconds: float = 0.02,
        min_net_avg_r: float = 0.10,
    ) -> None:
        self.starting_equity = float(starting_equity)
        self.risk_per_trade_pct = float(risk_per_trade_pct)
        self.step_delay_seconds = float(step_delay_seconds)
        self.regime_classifier = RegimeClassifier()
        self.time_machine = BlockBootstrapTimeMachine(replays=1500, seed=19)
        self.pipeline = Canonical11AgentPipeline(
            regime_classifier=self.regime_classifier,
            time_machine=self.time_machine,
            min_net_avg_r=min_net_avg_r,
        )

    def run_replay(
        self,
        feed_router: Any = None,
        candles_by_symbol: dict[str, Sequence[Candle]] | None = None,
        h1_by_symbol: dict[str, Sequence[Candle]] | None = None,
        symbols: Sequence[str] = DEFAULT_REPLAY_SYMBOLS,
        window_days: int = 5,
        speed: int = 100,
        on_update: Callable[[dict[str, Any]], None] | None = None,
        stop_check: Callable[[], bool] | None = None,
    ) -> ReplaySummary:
        """Execute a historical replay through the shared 11-agent pipeline & InFlightPositionManager."""
        bar_count = max(120, int(window_days) * 96)  # ~96 M15 bars/day
        sym_candles: dict[str, list[Candle]] = {}
        sym_h1: dict[str, list[Candle]] = {}

        if candles_by_symbol is not None:
            for s, cs in candles_by_symbol.items():
                if len(cs) >= 35:
                    sym_candles[s] = list(cs)
            if h1_by_symbol:
                for s, cs in h1_by_symbol.items():
                    sym_h1[s] = list(cs)
        elif feed_router is not None:
            for sym in symbols:
                try:
                    cs = feed_router.get_candles(sym, timeframe="M15", count=bar_count)
                    if len(cs) < 35:
                        cs = feed_router.get_candles(sym, timeframe="M5", count=bar_count)
                    if len(cs) >= 35:
                        sym_candles[sym] = list(cs)
                        sym_h1[sym] = list(feed_router.get_candles(sym, timeframe="H1", count=max(40, int(window_days) * 24)))
                except Exception:
                    continue

        if not sym_candles:
            summary = ReplaySummary(starting_equity=self.starting_equity, ending_equity=self.starting_equity)
            if on_update:
                on_update(
                    {
                        "status": "NO_DATA",
                        "symbol": ", ".join(symbols[:2]),
                        "message": "No historical candles available for replay.",
                        "equity_curve": summary.equity_curve,
                        "trades": [],
                    }
                )
            return summary

        equity = self.starting_equity
        equity_curve: list[float] = [round(equity, 2)]
        trades_log: list[dict[str, Any]] = []
        last_trace_dict: Optional[dict[str, Any]] = None
        gross_profit = 0.0
        gross_loss = 0.0
        wins = 0
        losses = 0
        vetoes = 0
        bars_processed = 0
        risk_amt = self.starting_equity * (self.risk_per_trade_pct / 100.0)
        delay = self.step_delay_seconds * (100.0 / max(float(speed), 10.0)) if self.step_delay_seconds > 0 else 0.0

        min_len = min(len(v) for v in sym_candles.values())
        window = 30
        stride = max(1, (min_len - window - 10) // 35)
        sym_label = ", ".join(list(sym_candles.keys())[:3]) + ("..." if len(sym_candles) > 3 else "")

        if on_update:
            on_update(
                {
                    "status": "RUNNING",
                    "symbol": sym_label,
                    "message": f"Replaying {sym_label} ({window_days}d @ {speed}x) via 11-Agent + In-Flight Pipeline...",
                    "equity_curve": list(equity_curve),
                    "trades": list(trades_log),
                }
            )

        inflight_mgr = AutonomousPositionOrchestrator(
            regime_classifier=self.regime_classifier,
            time_machine=self.time_machine,
            min_hold_bars=2,
        )
        next_ticket = 90001

        for idx in range(window, min_len - 8, stride):
            if stop_check and stop_check():
                break
            for sym, cs in sym_candles.items():
                bars_processed += 1
                hist = cs[idx - window : idx]
                direction = Side.BUY if hist[-1].close >= hist[-5].close else Side.SELL
                entry = float(hist[-1].close)
                h1_cs = sym_h1.get(sym)

                trace: AgentPipelineTrace = self.pipeline.evaluate(
                    symbol=sym,
                    side=direction,
                    timeframe=hist[-1].timeframe,
                    entry_price=entry,
                    candles=hist,
                    h1_candles=h1_cs,
                    spread_price=max(entry * 0.0001, 1e-5),
                    pip_size=0.01 if "JPY" in sym or "XAU" in sym else 0.0001,
                    run_ai_committee=False,
                )
                last_trace_dict = trace.to_dict()

                if trace.final_decision == "VETO":
                    vetoes += 1
                    continue
                if not trace.passed_all:
                    continue

                # Step simulated position bar-by-bar through InFlightPositionManager
                ticket = next_ticket
                next_ticket += 1
                cur_sl = trace.stop_loss
                cur_tp = trace.take_profit
                one_r = max(abs(entry - cur_sl), 1e-6)
                exit_px = float(cs[min(idx + 7, len(cs) - 1)].close)
                exit_reason = "HORIZON_CLOSE"
                ratchet_stage = "GRACE_LOCK"

                for step_i in range(0, min(8, len(cs) - idx)):
                    bar = cs[idx + step_i]
                    bar_slice = cs[idx - window + step_i : idx + step_i + 1]
                    # Check SL/TP hit inside bar
                    if direction == Side.BUY:
                        if bar.low <= cur_sl:
                            exit_px = cur_sl
                            exit_reason = "SL_HIT"
                            break
                        if bar.high >= cur_tp:
                            exit_px = cur_tp
                            exit_reason = "TP_HIT"
                            break
                    else:
                        if bar.high >= cur_sl:
                            exit_px = cur_sl
                            exit_reason = "SL_HIT"
                            break
                        if bar.low <= cur_tp:
                            exit_px = cur_tp
                            exit_reason = "TP_HIT"
                            break

                    dec = inflight_mgr.evaluate_position(
                        ticket=ticket,
                        symbol=sym,
                        side=direction,
                        entry_price=entry,
                        current_sl=cur_sl,
                        current_tp=cur_tp,
                        mark_price=float(bar.close),
                        candles=bar_slice,
                        timeframe=bar.timeframe,
                        now_utc=float(bar.timestamp_utc),
                        bars_held_override=step_i,
                        force_evaluate=True,
                    )
                    ratchet_stage = dec.ratchet_stage
                    if dec.action == "RATCHET_SLTP":
                        if dec.new_sl is not None:
                            cur_sl = dec.new_sl
                        if dec.new_tp is not None:
                            cur_tp = dec.new_tp
                    elif dec.action == "EARLY_EXIT":
                        exit_px = float(bar.close)
                        exit_reason = dec.ratchet_stage
                        break

                inflight_mgr.tracked.pop(ticket, None)
                signed_r = ((exit_px - entry) if direction == Side.BUY else (entry - exit_px)) / one_r
                pnl_cad = signed_r * risk_amt
                equity += pnl_cad
                equity_curve.append(round(equity, 2))
                if signed_r >= 0:
                    wins += 1
                    gross_profit += pnl_cad
                else:
                    losses += 1
                    gross_loss += abs(pnl_cad)

                trades_log.append(
                    {
                        "symbol": sym,
                        "side": direction.value,
                        "entry_price": round(entry, 5),
                        "exit_price": round(exit_px, 5),
                        "ratchet_stage": ratchet_stage,
                        "exit_reason": exit_reason,
                        "net_r": round(signed_r, 2),
                        "pnl_cad": round(pnl_cad, 2),
                        "trace": last_trace_dict,
                    }
                )

            if on_update:
                on_update(
                    {
                        "status": "RUNNING",
                        "symbol": sym_label,
                        "message": f"Replay {sym_label}: {bars_processed} bars | {wins}W/{losses}L | Z3 Vetoes: {vetoes} | Eq: {equity:,.2f} CAD",
                        "equity_curve": list(equity_curve),
                        "trades": trades_log[-25:],
                        "last_trace": last_trace_dict,
                    }
                )
            if delay > 0:
                time.sleep(delay)

        total = wins + losses
        wr = (wins / total * 100.0) if total > 0 else 0.0
        pf = (gross_profit / gross_loss) if gross_loss > 1e-9 else (gross_profit if gross_profit > 0 else 0.0)
        summary = ReplaySummary(
            symbols_replayed=list(sym_candles.keys()),
            bars_processed=bars_processed,
            total_trades=total,
            wins=wins,
            losses=losses,
            vetoes=vetoes,
            win_rate_pct=round(wr, 1),
            profit_factor=round(pf, 2),
            starting_equity=self.starting_equity,
            ending_equity=round(equity, 2),
            equity_curve=equity_curve,
            trades=trades_log,
            last_trace=last_trace_dict,
        )
        if on_update:
            on_update(
                {
                    "status": "COMPLETED",
                    "symbol": sym_label,
                    "message": (
                        f"Replay Completed ({sym_label}): {total} trades ({wr:.1f}% WR, PF {pf:.2f}) | "
                        f"Z3 Vetoes: {vetoes} | End Eq: {equity:,.2f} CAD"
                    ),
                    "equity_curve": equity_curve,
                    "trades": trades_log[-25:],
                    "last_trace": last_trace_dict,
                }
            )
        return summary
