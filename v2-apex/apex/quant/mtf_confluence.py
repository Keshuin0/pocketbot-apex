"""Higher-Timeframe (H1) Structural Alignment & Anti-Chase Filter (`apex/quant/mtf_confluence.py`).

Prevents M5/M15 breakout entries from:
  1. Buying directly into a 20-bar H1 swing-high resistance ceiling (within 0.30 * ATR_M5)
     or fighting a strongly bearish H1 trend (and symmetric for shorts).
  2. Chasing an already-overextended breakout candle (> 2.2 * ATR_M5 beyond breakout level
     without pullback).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from apex.feeds.models import Candle, Side


@dataclass(frozen=True, slots=True)
class MTFConfluenceVerdict:
    """Result of H1 structural alignment & M5/M15 anti-chase evaluation."""

    passed: bool
    h1_trend_aligned: bool
    anti_chase_ok: bool
    h1_ema_slope: float
    h1_swing_high: float
    h1_swing_low: float
    stretch_atr_multiple: float
    reason: str


def _ema(values: np.ndarray, period: int = 20) -> np.ndarray:
    alpha = 2.0 / (period + 1.0)
    out = np.empty_like(values, dtype=np.float64)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


class MTFConfluenceFilter:
    """Higher-timeframe H1 structural alignment + M5/M15 anti-chase guard."""

    def __init__(
        self,
        enabled: bool = True,
        h1_lookback: int = 20,
        ceiling_atr_buffer: float = 0.30,
        max_chase_atr_multiple: float = 2.2,
        strong_counter_slope: float = 0.35,
    ) -> None:
        self.enabled = bool(enabled)
        self.h1_lookback = max(10, int(h1_lookback))
        self.ceiling_atr_buffer = float(ceiling_atr_buffer)
        self.max_chase_atr_multiple = float(max_chase_atr_multiple)
        self.strong_counter_slope = float(strong_counter_slope)

    def evaluate(
        self,
        symbol: str,
        side: Side,
        entry_price: float,
        atr_m5: float,
        m5_candles: Sequence[Candle],
        h1_candles: Sequence[Candle] | None = None,
    ) -> MTFConfluenceVerdict:
        """Evaluate H1 structural alignment and M5/M15 anti-chase overextension."""
        if not self.enabled:
            return MTFConfluenceVerdict(
                passed=True,
                h1_trend_aligned=True,
                anti_chase_ok=True,
                h1_ema_slope=0.0,
                h1_swing_high=0.0,
                h1_swing_low=0.0,
                stretch_atr_multiple=0.0,
                reason="MTF Confluence Filter disabled",
            )

        safe_atr = max(float(atr_m5), 1e-9)

        # 1. Anti-Chase Overextension Check on M5/M15 bars
        anti_chase_ok = True
        stretch_mult = 0.0
        if len(m5_candles) >= 14:
            prior = m5_candles[-13:-1]
            if side == Side.BUY:
                breakout_lvl = max(float(c.high) for c in prior)
                stretch_mult = max(0.0, (float(entry_price) - breakout_lvl) / safe_atr)
            else:
                breakout_lvl = min(float(c.low) for c in prior)
                stretch_mult = max(0.0, (breakout_lvl - float(entry_price)) / safe_atr)
            if stretch_mult > self.max_chase_atr_multiple:
                anti_chase_ok = False

        # 2. H1 Structural Alignment & 20-bar Swing Ceiling/Floor Check
        h1_aligned = True
        h1_slope_norm = 0.0
        swing_high = 0.0
        swing_low = 0.0
        h1_detail = "H1_OK"

        if h1_candles and len(h1_candles) >= self.h1_lookback:
            recent_h1 = list(h1_candles[-max(self.h1_lookback, 25) :])
            closes_h1 = np.asarray([float(c.close) for c in recent_h1], dtype=np.float64)
            highs_h1 = np.asarray([float(c.high) for c in recent_h1], dtype=np.float64)
            lows_h1 = np.asarray([float(c.low) for c in recent_h1], dtype=np.float64)

            ema20 = _ema(closes_h1, period=20)
            atr_h1 = max(float(np.mean(highs_h1[-14:] - lows_h1[-14:])), safe_atr)
            h1_slope_norm = float((ema20[-1] - ema20[-min(5, len(ema20))]) / atr_h1)

            lookback_window = recent_h1[-self.h1_lookback :]
            swing_high = max(float(c.high) for c in lookback_window)
            swing_low = min(float(c.low) for c in lookback_window)
            last_h1_close = float(closes_h1[-1])
            last_ema = float(ema20[-1])

            if side == Side.BUY:
                if last_h1_close < last_ema and h1_slope_norm < -self.strong_counter_slope:
                    h1_aligned = False
                    h1_detail = f"H1_BEARISH_COUNTER_TREND (slope={h1_slope_norm:.2f})"
                elif 0.0 <= (swing_high - float(entry_price)) <= (self.ceiling_atr_buffer * safe_atr):
                    h1_aligned = False
                    h1_detail = f"H1_SWING_HIGH_CEILING ({entry_price:.5f} near {swing_high:.5f})"
            else:
                if last_h1_close > last_ema and h1_slope_norm > self.strong_counter_slope:
                    h1_aligned = False
                    h1_detail = f"H1_BULLISH_COUNTER_TREND (slope={h1_slope_norm:.2f})"
                elif 0.0 <= (float(entry_price) - swing_low) <= (self.ceiling_atr_buffer * safe_atr):
                    h1_aligned = False
                    h1_detail = f"H1_SWING_LOW_FLOOR ({entry_price:.5f} near {swing_low:.5f})"
        else:
            h1_detail = "H1_UNAVAILABLE_BYPASS"

        passed = h1_aligned and anti_chase_ok
        if not anti_chase_ok:
            reason = f"ANTI_CHASE_OVEREXTENDED ({stretch_mult:.2f}x ATR > {self.max_chase_atr_multiple:.2f}x)"
        elif not h1_aligned:
            reason = h1_detail
        else:
            reason = f"MTF_ALIGNED ({h1_detail}, stretch={stretch_mult:.2f}x)"

        return MTFConfluenceVerdict(
            passed=passed,
            h1_trend_aligned=h1_aligned,
            anti_chase_ok=anti_chase_ok,
            h1_ema_slope=round(h1_slope_norm, 4),
            h1_swing_high=swing_high,
            h1_swing_low=swing_low,
            stretch_atr_multiple=round(stretch_mult, 4),
            reason=reason,
        )
