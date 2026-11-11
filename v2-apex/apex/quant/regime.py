"""Real-Time Market Regime Classifier for PocketBot Apex.

Computes:
  1. Rolling Hurst Exponent (rescaled range R/S & multi-scale variance ratio).
  2. Volatility Percentile (current ATR(14) vs 60-day historical ATR distribution).
  3. Directional Efficiency Ratio (Kaufman ER).
  4. Microstructure Shock Metrics (live spread vs rolling median spread, tick arrival velocity Z-score).

Classifies symbols dynamically into:
  - ``TRENDING_EXPANSION``: enable ORB & M15 Strong Momentum breakouts.
  - ``MEAN_REVERTING_CHOP``: auto-block breakout entries to prevent chop bleed.
  - ``VOLATILITY_SHOCK``: lock out entries when spread > 1.5x rolling median or tick velocity spikes > 4 sigma.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np
from scipy.stats import percentileofscore

from apex.feeds.models import Candle


class MarketRegime(str, Enum):
    """Dynamic market regime classification."""

    TRENDING_EXPANSION = "TRENDING_EXPANSION"
    MEAN_REVERTING_CHOP = "MEAN_REVERTING_CHOP"
    VOLATILITY_SHOCK = "VOLATILITY_SHOCK"


@dataclass(frozen=True, slots=True)
class RegimeDiagnostics:
    """Complete quantitative regime telemetry for a symbol."""

    symbol: str
    regime: MarketRegime
    hurst_exponent: float
    volatility_percentile: float
    kaufman_er: float
    current_atr: float
    spread_to_median_ratio: float
    tick_velocity_zscore: float
    allow_breakout_entries: bool
    filter_enabled: bool
    reason: str


def compute_hurst_exponent(
    prices: Sequence[float] | np.ndarray,
    max_lag: int = 20,
) -> float:
    """Compute the Hurst Exponent $H \\in (0, 1)$ using multi-scale log-return variance / R/S scaling.

    For a geometric stochastic process $P_t$, the standard deviation of $k$-step log price
    increments scales as $\\sigma(\\tau) \\propto \\tau^H$:
      - $H > 0.5$: persistent / trending expansion series.
      - $H = 0.5$: geometric Brownian motion (random walk).
      - $H < 0.5$: anti-persistent / mean-reverting chop series.

    Combines multi-lag variogram regression ($\\log \\text{Std}(\\Delta_\\tau \\log P)$ vs $\\log \\tau$)
    with classical Rescaled Range ($R/S$) estimation for robust finite-sample stability.
    """
    arr = np.asarray(prices, dtype=np.float64)
    if arr.ndim != 1 or len(arr) < 20:
        raise ValueError(f"compute_hurst_exponent requires at least 20 1D prices, got {len(arr)}")

    if np.any(arr <= 0.0):
        # Shift series positively if non-positive values were passed
        arr = arr - np.min(arr) + 1.0

    log_p = np.log(arr)
    n = len(log_p)
    upper_lag = min(max(int(max_lag), 6), n // 3)
    lags = np.arange(1, upper_lag + 1, dtype=np.int64)

    # 1. Variogram / Variance-Ratio scaling estimator on log prices
    tau_stds: list[float] = []
    valid_lags: list[float] = []
    for lag in lags:
        diffs = log_p[lag:] - log_p[:-lag]
        std_val = float(np.std(diffs, ddof=1))
        if std_val > 1e-12:
            tau_stds.append(std_val)
            valid_lags.append(float(lag))

    if len(valid_lags) < 3:
        return 0.5

    slope_var, _ = np.polyfit(np.log(valid_lags), np.log(tau_stds), 1)

    # 2. Rescaled Range (R/S) estimator across sub-windows if series has sufficient length
    returns = np.diff(log_p)
    rs_lags: list[float] = []
    rs_vals: list[float] = []
    min_chunk = 8
    max_chunk = min(len(returns) // 2, max(upper_lag * 2, 16))
    if max_chunk >= min_chunk + 4:
        chunk_sizes = np.unique(
            np.linspace(min_chunk, max_chunk, num=min(8, max_chunk - min_chunk + 1), dtype=np.int64)
        )
        for w in chunk_sizes:
            num_chunks = len(returns) // int(w)
            if num_chunks < 1:
                continue
            chunk_rs: list[float] = []
            for c_idx in range(num_chunks):
                sub = returns[c_idx * int(w) : (c_idx + 1) * int(w)]
                mean_sub = float(np.mean(sub))
                cum_dev = np.cumsum(sub - mean_sub)
                r_val = float(np.max(cum_dev) - np.min(cum_dev))
                s_val = float(np.std(sub, ddof=1))
                if s_val > 1e-12 and r_val > 0.0:
                    chunk_rs.append(r_val / s_val)
            if chunk_rs:
                rs_lags.append(float(w))
                rs_vals.append(float(np.mean(chunk_rs)))

    if len(rs_lags) >= 3:
        slope_rs, _ = np.polyfit(np.log(rs_lags), np.log(rs_vals), 1)
        # Variogram estimator is asymptotically unbiased on small windows; blend 75% VR + 25% R/S
        h_est = 0.75 * float(slope_var) + 0.25 * float(slope_rs)
    else:
        h_est = float(slope_var)

    return float(np.clip(h_est, 0.01, 0.99))


def compute_kaufman_er(
    closes: Sequence[float] | np.ndarray,
    period: int = 14,
) -> float:
    """Compute Kaufman's Directional Efficiency Ratio $ER \\in [0.0, 1.0]$.

    $$ER = \\frac{|Close_t - Close_{t - N}|}{\\sum_{i=t-N+1}^{t} |Close_i - Close_{i-1}|}$$
    """
    arr = np.asarray(closes, dtype=np.float64)
    if len(arr) < 2:
        return 0.0
    eff_period = min(int(period), len(arr) - 1)
    if eff_period < 1:
        return 0.0

    window = arr[-(eff_period + 1) :]
    net_change = abs(float(window[-1] - window[0]))
    sum_abs_steps = float(np.sum(np.abs(np.diff(window))))
    if sum_abs_steps <= 1e-12:
        return 0.0
    return float(np.clip(net_change / sum_abs_steps, 0.0, 1.0))


def compute_atr_series(
    highs: Sequence[float] | np.ndarray,
    lows: Sequence[float] | np.ndarray,
    closes: Sequence[float] | np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Compute Wilder's True Range and rolling Average True Range (ATR) series."""
    h = np.asarray(highs, dtype=np.float64)
    l = np.asarray(lows, dtype=np.float64)
    c = np.asarray(closes, dtype=np.float64)
    if not (len(h) == len(l) == len(c)) or len(c) == 0:
        raise ValueError("highs, lows, closes must have matching non-zero length.")

    n = len(c)
    tr = np.empty(n, dtype=np.float64)
    tr[0] = h[0] - l[0]
    if n > 1:
        prev_c = c[:-1]
        tr[1:] = np.maximum(
            h[1:] - l[1:],
            np.maximum(np.abs(h[1:] - prev_c), np.abs(l[1:] - prev_c)),
        )

    eff_p = max(1, min(int(period), n))
    atr = np.empty(n, dtype=np.float64)
    # Initial seed is simple average up to eff_p, followed by Wilder smoothing
    cum = np.cumsum(tr)
    for i in range(eff_p):
        atr[i] = cum[i] / float(i + 1)
    alpha = 1.0 / float(eff_p)
    for i in range(eff_p, n):
        atr[i] = (1.0 - alpha) * atr[i - 1] + alpha * tr[i]
    return atr


def compute_volatility_percentile(
    atr_series: Sequence[float] | np.ndarray,
    lookback_bars: int | None = None,
) -> float:
    """Compute the percentile rank (0.0 to 100.0) of the current ATR(14) vs the 60-day distribution."""
    arr = np.asarray(atr_series, dtype=np.float64)
    if len(arr) == 0:
        return 50.0
    if lookback_bars is not None and lookback_bars > 0 and len(arr) > lookback_bars:
        arr = arr[-int(lookback_bars) :]
    current = float(arr[-1])
    pct = float(percentileofscore(arr, current, kind="rank"))
    return float(np.clip(pct, 0.0, 100.0))


class RegimeClassifier:
    """Real-Time Market Regime Classifier (`apex/quant/regime.py`).

    Respects `config["filters"]["regime"]["enabled"]` for walk-forward A/B testing:
      - When `enabled=False`, `allow_breakout_entries` is always `True` while still reporting
        the underlying Hurst, Kaufman ER, and Volatility Percentile telemetry.
    """

    def __init__(
        self,
        enabled: bool = True,
        hurst_window: int = 100,
        adx_atr_period: int = 14,
        er_period: int = 14,
        trending_hurst_threshold: float = 0.52,
        trending_er_threshold: float = 0.25,
        max_spread_median_multiple: float = 1.5,
        max_tick_velocity_sigma: float = 4.0,
        volatility_lookback_bars: int = 4680,  # ~60 sessions of M5 bars (78 * 60)
    ) -> None:
        self.enabled: bool = bool(enabled)
        self.hurst_window: int = int(hurst_window)
        self.adx_atr_period: int = int(adx_atr_period)
        self.er_period: int = int(er_period)
        self.trending_hurst_threshold: float = float(trending_hurst_threshold)
        self.trending_er_threshold: float = float(trending_er_threshold)
        self.max_spread_median_multiple: float = float(max_spread_median_multiple)
        self.max_tick_velocity_sigma: float = float(max_tick_velocity_sigma)
        self.volatility_lookback_bars: int = int(volatility_lookback_bars)

    @staticmethod
    def compute_spread_ratio(
        current_spread: float | None,
        historical_spreads: Sequence[float] | None,
    ) -> float:
        """Compute `current_spread / median(historical_spreads)`."""
        if current_spread is None or current_spread <= 0.0:
            return 1.0
        if not historical_spreads:
            return 1.0
        arr = np.asarray(historical_spreads, dtype=np.float64)
        arr = arr[arr > 0.0]
        if len(arr) == 0:
            return 1.0
        med = float(np.median(arr))
        if med <= 1e-12:
            return 1.0
        return float(current_spread / med)

    @staticmethod
    def compute_tick_velocity_zscore(
        current_tick_rate: float | None,
        historical_tick_rates: Sequence[float] | None,
    ) -> float:
        """Compute Z-score of current tick arrival velocity against rolling historical tick rates."""
        if current_tick_rate is None or not historical_tick_rates or len(historical_tick_rates) < 5:
            return 0.0
        arr = np.asarray(historical_tick_rates, dtype=np.float64)
        mean_v = float(np.mean(arr))
        std_v = max(float(np.std(arr, ddof=1)), 1e-8)
        return float((float(current_tick_rate) - mean_v) / std_v)

    def classify_arrays(
        self,
        symbol: str,
        highs: Sequence[float] | np.ndarray,
        lows: Sequence[float] | np.ndarray,
        closes: Sequence[float] | np.ndarray,
        current_spread: float | None = None,
        historical_spreads: Sequence[float] | None = None,
        current_tick_rate: float | None = None,
        historical_tick_rates: Sequence[float] | None = None,
    ) -> RegimeDiagnostics:
        """Classify the current market regime from OHLC arrays and microstructure telemetry."""
        c_arr = np.asarray(closes, dtype=np.float64)
        h_arr = np.asarray(highs, dtype=np.float64)
        l_arr = np.asarray(lows, dtype=np.float64)

        hurst_slice = c_arr[-self.hurst_window :] if len(c_arr) > self.hurst_window else c_arr
        hurst = compute_hurst_exponent(hurst_slice) if len(hurst_slice) >= 20 else 0.5
        er = compute_kaufman_er(c_arr, period=self.er_period)
        atr_series = compute_atr_series(h_arr, l_arr, c_arr, period=self.adx_atr_period)
        current_atr = float(atr_series[-1]) if len(atr_series) > 0 else 0.0
        vol_pct = compute_volatility_percentile(
            atr_series,
            lookback_bars=self.volatility_lookback_bars,
        )

        spread_ratio = self.compute_spread_ratio(current_spread, historical_spreads)
        vel_z = self.compute_tick_velocity_zscore(current_tick_rate, historical_tick_rates)

        # 1. Check VOLATILITY_SHOCK first (microstructure safety lockout)
        if spread_ratio > self.max_spread_median_multiple:
            regime = MarketRegime.VOLATILITY_SHOCK
            raw_allow = False
            reason = (
                f"VOLATILITY_SHOCK: live spread is {spread_ratio:.2f}x rolling median "
                f"(> {self.max_spread_median_multiple:.2f}x lockout)"
            )
        elif vel_z > self.max_tick_velocity_sigma:
            regime = MarketRegime.VOLATILITY_SHOCK
            raw_allow = False
            reason = (
                f"VOLATILITY_SHOCK: tick arrival velocity spiked to +{vel_z:.2f} sigma "
                f"(> +{self.max_tick_velocity_sigma:.2f} sigma lockout)"
            )
        # 2. Check TRENDING_EXPANSION vs MEAN_REVERTING_CHOP
        elif hurst >= self.trending_hurst_threshold and er >= self.trending_er_threshold:
            regime = MarketRegime.TRENDING_EXPANSION
            raw_allow = True
            reason = (
                f"TRENDING_EXPANSION: H={hurst:.3f} (>= {self.trending_hurst_threshold:.2f}), "
                f"ER={er:.3f} (>= {self.trending_er_threshold:.2f}), VolPct={vol_pct:.1f}%"
            )
        else:
            regime = MarketRegime.MEAN_REVERTING_CHOP
            raw_allow = False
            reason = (
                f"MEAN_REVERTING_CHOP: H={hurst:.3f}, ER={er:.3f} "
                f"(breakout entries blocked to prevent chop bleed)"
            )

        allow_entries = True if not self.enabled else raw_allow
        if not self.enabled:
            reason = f"[regime_filter_disabled] {reason}"

        return RegimeDiagnostics(
            symbol=symbol.upper(),
            regime=regime,
            hurst_exponent=hurst,
            volatility_percentile=vol_pct,
            kaufman_er=er,
            current_atr=current_atr,
            spread_to_median_ratio=spread_ratio,
            tick_velocity_zscore=vel_z,
            allow_breakout_entries=allow_entries,
            filter_enabled=self.enabled,
            reason=reason,
        )

    def classify_candles(
        self,
        symbol: str,
        candles: Sequence[Candle],
        current_spread: float | None = None,
        historical_spreads: Sequence[float] | None = None,
        current_tick_rate: float | None = None,
        historical_tick_rates: Sequence[float] | None = None,
    ) -> RegimeDiagnostics:
        """Classify market regime directly from a sequence of ``Candle`` objects."""
        if len(candles) < 20:
            raise ValueError(f"RegimeClassifier requires at least 20 candles, got {len(candles)}")
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        closes = [c.close for c in candles]
        return self.classify_arrays(
            symbol=symbol,
            highs=highs,
            lows=lows,
            closes=closes,
            current_spread=current_spread,
            historical_spreads=historical_spreads,
            current_tick_rate=current_tick_rate,
            historical_tick_rates=historical_tick_rates,
        )
