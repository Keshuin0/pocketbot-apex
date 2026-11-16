"""Unified 11-Agent Historical Market Replay Engine (`apex/cockpit/replay_engine.py`).

Replays 2d/5d/10d of M5/M15 candles at 20x/50x/100x speed through the shared
`Canonical11AgentPipeline` + `AutonomousPositionOrchestrator` (Grace Lock, +1.0R Free-Roll,
+1.5R -> 3.5R Runner Mode), streaming trades, 11-stage X-Ray traces, Win Rate, Profit
Factor, and equity curve over `/ws`. Strictly isolated from live `trade_ledger`.
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any, Callable, Optional, Sequence

import numpy as np

from apex.engine.canonical_pipeline import AgentPipelineTrace, Canonical11AgentPipeline
from apex.execution.position_manager import AutonomousPositionOrchestrator
from apex.feeds.models import Candle, Side
from apex.quant.momentum_dom import InstrumentTier, classify_instrument_tier
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


def compute_cpcv_metrics(
    trade_r: Sequence[float],
    n_blocks: int = 6,
    k_test: int = 2,
    purge_gap: int = 1,
) -> tuple[int, float, float]:
    """Combinatorially Purged Cross-Validation (CPCV) returning `(paths, mean_oos_sharpe, pbo_probability)`."""
    n = len(trade_r)
    if n < 6:
        return 0, 0.0, 0.0
    blocks_cnt = min(max(4, int(n_blocks)), n)
    test_cnt = max(1, min(int(k_test), blocks_cnt - 2))
    arr = np.asarray(trade_r, dtype=np.float64)
    block_slices = [idx for idx in np.array_split(np.arange(n), blocks_cnt) if len(idx) > 0]
    actual_blocks = len(block_slices)
    if actual_blocks <= test_cnt:
        return 0, 0.0, 0.0

    is_sharpes: list[float] = []
    oos_sharpes: list[float] = []
    for test_combo in itertools.combinations(range(actual_blocks), test_cnt):
        test_set = set(test_combo)
        test_idx = np.concatenate([block_slices[b] for b in test_combo])
        oos_vals = arr[test_idx]
        if len(oos_vals) < 2:
            continue

        purge_indices: set[int] = set()
        for t_idx in test_idx:
            for g in range(-purge_gap, purge_gap + 1):
                purge_indices.add(int(t_idx) + g)
        train_parts = [
            np.asarray([i for i in block_slices[b] if int(i) not in purge_indices], dtype=np.int64)
            for b in range(actual_blocks)
            if b not in test_set
        ]
        train_parts = [p for p in train_parts if len(p) > 0]
        if not train_parts:
            continue
        is_vals = arr[np.concatenate(train_parts)]
        if len(is_vals) < 2:
            continue

        is_std = float(np.std(is_vals, ddof=1))
        oos_std = float(np.std(oos_vals, ddof=1))
        is_sr = (float(np.mean(is_vals)) / is_std) if is_std > 1e-9 else (1.0 if float(np.mean(is_vals)) > 0 else -1.0)
        oos_sr = (float(np.mean(oos_vals)) / oos_std) if oos_std > 1e-9 else (1.0 if float(np.mean(oos_vals)) > 0 else -1.0)
        is_sharpes.append(is_sr)
        oos_sharpes.append(oos_sr)

    paths = len(oos_sharpes)
    if paths == 0:
        return 0, 0.0, 0.0
    med_oos = float(np.median(oos_sharpes))
    med_is = float(np.median(is_sharpes))
    # Overfitting event: path achieves above-median IS performance yet falls <= 0 (or below median OOS)
    pbo_hits = sum(
        1
        for i_sr, o_sr in zip(is_sharpes, oos_sharpes)
        if o_sr <= 0.0 or (i_sr >= med_is and o_sr < med_oos)
    )
    return paths, round(float(np.mean(oos_sharpes)), 4), round(float(pbo_hits / paths), 4)


def compute_deflated_sharpe_ratio(
    trade_r: Sequence[float],
    num_trials: int = 12,
    trial_sharpe_variance: float = 0.04,
) -> float:
    """Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio probabilistic score in `[0.0, 1.0]`."""
    n = len(trade_r)
    if n < 3:
        return 0.0
    arr = np.asarray(trade_r, dtype=np.float64)
    std = float(np.std(arr, ddof=1))
    if std <= 1e-9:
        return 1.0 if float(np.mean(arr)) > 0.0 else 0.0
    sr = float(np.mean(arr)) / std
    z = (arr - float(np.mean(arr))) / std
    skew = float(np.mean(z**3))
    kurt = float(np.mean(z**4))

    norm = NormalDist()
    m_trials = max(2, int(num_trials))
    euler_gamma = 0.5772156649015329
    sr_zero = math.sqrt(max(trial_sharpe_variance, 1e-6)) * (
        (1.0 - euler_gamma) * norm.inv_cdf(1.0 - 1.0 / m_trials)
        + euler_gamma * norm.inv_cdf(1.0 - 1.0 / (m_trials * math.e))
    )
    denom_sq = max(1e-6, 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * (sr**2))
    z_stat = (sr - sr_zero) * math.sqrt(n - 1) / math.sqrt(denom_sq)
    return round(float(norm.cdf(z_stat)), 4)


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
    cpcv_paths: int = 0
    cpcv_mean_sharpe: float = 0.0
    pbo_probability: float = 0.0
    deflated_sharpe_ratio: float = 0.0


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
        symbols: Sequence[str] | str | None = DEFAULT_REPLAY_SYMBOLS,
        window_days: int = 5,
        speed: int = 100,
        on_update: Callable[[dict[str, Any]], None] | None = None,
        stop_check: Callable[[], bool] | None = None,
        max_concurrent_positions: int = 3,
    ) -> ReplaySummary:
        """Execute a causal chronological replay with zero lookahead, CPCV, and DSR."""
        bar_count = max(120, int(window_days) * 96)  # ~96 M15 bars/day
        if isinstance(symbols, str):
            norm_sym = symbols.strip().upper()
            if norm_sym in {"", "AUTO_TOP12", "ALL", "ALL_SCAN"}:
                symbols_list = list(DEFAULT_REPLAY_SYMBOLS)
            else:
                symbols_list = [x.strip().upper() for x in norm_sym.split(",") if x.strip()]
        elif symbols is None:
            symbols_list = list(DEFAULT_REPLAY_SYMBOLS)
        else:
            symbols_list = [str(s).strip().upper() for s in symbols if str(s).strip()]
        if not symbols_list:
            symbols_list = list(DEFAULT_REPLAY_SYMBOLS)

        sym_candles: dict[str, list[Candle]] = {}
        sym_h1: dict[str, list[Candle]] = {}

        if candles_by_symbol is not None:
            for s, cs in candles_by_symbol.items():
                if cs and len(cs) >= 35:
                    sym_candles[str(s).upper()] = list(cs)
            if h1_by_symbol:
                for s, cs in h1_by_symbol.items():
                    if cs:
                        sym_h1[str(s).upper()] = list(cs)
        elif feed_router is not None:
            for sym in symbols_list:
                try:
                    tier = classify_instrument_tier(sym)
                    primary_tf = "M5" if tier == InstrumentTier.TIGHT_SPREAD_M5 else "M15"
                    fallback_tf = "M15" if primary_tf == "M5" else "M5"
                    cs = feed_router.get_candles(sym, timeframe=primary_tf, count=bar_count) or []
                    if len(cs) < 35:
                        cs = feed_router.get_candles(sym, timeframe=fallback_tf, count=bar_count) or []
                    if len(cs) >= 35:
                        sym_candles[sym] = list(cs)
                        h1_cs = feed_router.get_candles(sym, timeframe="H1", count=max(40, int(window_days) * 24)) or []
                        sym_h1[sym] = list(h1_cs)
                except Exception:
                    continue

        if not sym_candles:
            summary = ReplaySummary(starting_equity=self.starting_equity, ending_equity=self.starting_equity)
            if on_update:
                on_update(
                    {
                        "status": "NO_DATA",
                        "symbol": ", ".join(symbols_list[:2]),
                        "message": "No historical candles available for replay.",
                        "equity_curve": summary.equity_curve,
                        "trades": [],
                    }
                )
            return summary

        equity = self.starting_equity
        equity_curve: list[float] = [round(equity, 2)]
        trades_log: list[dict[str, Any]] = []
        trade_r_series: list[float] = []
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
        eval_stride = max(1, (min_len - window - 2) // 60)
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
        active_positions: dict[str, dict[str, Any]] = {}
        max_slots = max(1, int(max_concurrent_positions))

        def _settle_closed_pos(
            pos: dict[str, Any],
            exit_px: float,
            exit_reason: str,
            exit_idx: int,
        ) -> None:
            nonlocal equity, gross_profit, gross_loss, wins, losses
            direction: Side = pos["side"]
            entry = pos["entry_price"]
            one_r = pos["one_r"]
            signed_r = ((exit_px - entry) if direction == Side.BUY else (entry - exit_px)) / one_r
            pnl_cad = signed_r * risk_amt
            equity += pnl_cad
            equity_curve.append(round(equity, 2))
            trade_r_series.append(signed_r)
            if signed_r >= 0:
                wins += 1
                gross_profit += pnl_cad
            else:
                losses += 1
                gross_loss += abs(pnl_cad)

            inflight_mgr.tracked.pop(pos["ticket"], None)
            trades_log.append(
                {
                    "symbol": pos["symbol"],
                    "side": direction.value,
                    "entry_price": round(entry, 5),
                    "exit_price": round(exit_px, 5),
                    "entry_bar": pos["entry_idx"],
                    "exit_bar": exit_idx,
                    "ratchet_stage": pos["ratchet_stage"],
                    "exit_reason": exit_reason,
                    "net_r": round(signed_r, 2),
                    "pnl_cad": round(pnl_cad, 2),
                    "trace": pos["trace"],
                }
            )

        for idx in range(window, min_len):
            if stop_check and stop_check():
                break

            # Step 1: Chronologically advance & settle existing active positions on bar `idx`
            for sym in list(active_positions.keys()):
                pos = active_positions[sym]
                cs = sym_candles[sym]
                bar = cs[idx]
                direction: Side = pos["side"]
                cur_sl = pos["cur_sl"]
                cur_tp = pos["cur_tp"]
                bars_held = idx - pos["entry_idx"]
                closed_px: float | None = None
                closed_reason = ""

                if direction == Side.BUY:
                    if bar.low <= cur_sl:
                        closed_px = cur_sl
                        closed_reason = "SL_HIT"
                    elif bar.high >= cur_tp:
                        closed_px = cur_tp
                        closed_reason = "TP_HIT"
                else:
                    if bar.high >= cur_sl:
                        closed_px = cur_sl
                        closed_reason = "SL_HIT"
                    elif bar.low <= cur_tp:
                        closed_px = cur_tp
                        closed_reason = "TP_HIT"

                if closed_px is None:
                    # Causal slice ending strictly at current bar `idx` (no future peek)
                    bar_slice = cs[max(0, idx - window + 1) : idx + 1]
                    dec = inflight_mgr.evaluate_position(
                        ticket=pos["ticket"],
                        symbol=sym,
                        side=direction,
                        entry_price=pos["entry_price"],
                        current_sl=cur_sl,
                        current_tp=cur_tp,
                        mark_price=float(bar.close),
                        candles=bar_slice,
                        timeframe=bar.timeframe,
                        now_utc=float(bar.timestamp_utc),
                        bars_held_override=bars_held,
                        force_evaluate=True,
                    )
                    pos["ratchet_stage"] = dec.ratchet_stage
                    if dec.action == "RATCHET_SLTP":
                        if dec.new_sl is not None:
                            pos["cur_sl"] = dec.new_sl
                        if dec.new_tp is not None:
                            pos["cur_tp"] = dec.new_tp
                    elif dec.action == "EARLY_EXIT":
                        closed_px = float(bar.close)
                        closed_reason = dec.ratchet_stage

                if closed_px is None and (bars_held >= 7 or idx == min_len - 1):
                    closed_px = float(bar.close)
                    closed_reason = "HORIZON_CLOSE"

                if closed_px is not None:
                    active_positions.pop(sym, None)
                    _settle_closed_pos(pos, closed_px, closed_reason, idx)

            # Step 2: Evaluate causal new entries strictly from closed history up to `idx`
            is_eval_bar = ((idx - window) % eval_stride == 0) and (idx < min_len - 1)
            if is_eval_bar:
                for sym, cs in sym_candles.items():
                    bars_processed += 1
                    if sym in active_positions or len(active_positions) >= max_slots:
                        continue
                    hist = cs[idx - window : idx + 1][-window:]
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

                    ticket = next_ticket
                    next_ticket += 1
                    cur_sl = trace.stop_loss
                    cur_tp = trace.take_profit
                    one_r = max(abs(entry - cur_sl), 1e-6)
                    active_positions[sym] = {
                        "ticket": ticket,
                        "symbol": sym,
                        "side": direction,
                        "entry_price": entry,
                        "cur_sl": cur_sl,
                        "cur_tp": cur_tp,
                        "one_r": one_r,
                        "entry_idx": idx,
                        "ratchet_stage": "GRACE_LOCK",
                        "trace": last_trace_dict,
                    }

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

        # Force mark-to-market settle if stopped early with open positions
        for sym, pos in list(active_positions.items()):
            last_close = float(sym_candles[sym][min_len - 1].close)
            _settle_closed_pos(pos, last_close, "END_OF_REPLAY", min_len - 1)
        active_positions.clear()

        total = wins + losses
        if len(equity_curve) < 2:
            equity_curve.append(round(equity, 2))
        wr = (wins / total * 100.0) if total > 0 else 0.0
        if gross_loss > 1e-9:
            pf = gross_profit / gross_loss
        elif gross_profit > 1e-9:
            pf = 99.99
        else:
            pf = 0.0
        cpcv_paths, cpcv_mean_sr, pbo_prob = compute_cpcv_metrics(trade_r_series)
        dsr = compute_deflated_sharpe_ratio(trade_r_series)
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
            cpcv_paths=cpcv_paths,
            cpcv_mean_sharpe=cpcv_mean_sr,
            pbo_probability=pbo_prob,
            deflated_sharpe_ratio=dsr,
        )
        if on_update:
            on_update(
                {
                    "status": "COMPLETED",
                    "symbol": sym_label,
                    "message": (
                        f"Replay Completed ({sym_label}): {total} trades ({wr:.1f}% WR, PF {pf:.2f}) | "
                        f"CPCV PBO={pbo_prob:.2f}, DSR={dsr:.2f} | Z3 Vetoes: {vetoes} | End Eq: {equity:,.2f} CAD"
                    ),
                    "equity_curve": equity_curve,
                    "trades": trades_log[-25:],
                    "last_trace": last_trace_dict,
                    "cpcv_paths": cpcv_paths,
                    "cpcv_mean_sharpe": cpcv_mean_sr,
                    "pbo_probability": pbo_prob,
                    "deflated_sharpe_ratio": dsr,
                }
            )
        return summary
