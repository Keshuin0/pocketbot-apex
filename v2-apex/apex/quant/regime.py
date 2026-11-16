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

import math
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
    adx: float = 0.0
    hawkes_intensity_ratio: float = 1.0


def _anis_lloyd_expected_rs(w: int) -> float:
    """Compute Anis & Lloyd (1976) exact expected white-noise E[(R/S)_w] for finite window w."""
    if w < 2:
        return 1.0
    coeff = math.exp(math.lgamma(0.5 * float(w - 1)) - math.lgamma(0.5 * float(w))) / math.sqrt(math.pi)
    r_idx = np.arange(1, w, dtype=np.float64)
    series_sum = float(np.sum(np.sqrt((float(w) - r_idx) / r_idx)))
    return max(coeff * series_sum, 1e-12)


def compute_hurst_exponent(
    prices: Sequence[float] | np.ndarray,
    max_lag: int = 20,
) -> float:
    """Compute the Hurst Exponent H in (0, 1) using Anis-Lloyd (1976) de-biased R/S + variogram scaling.

    For a geometric stochastic process P_t:
      - H > 0.5: persistent / trending expansion series.
      - H = 0.5: geometric Brownian motion (random walk, unbiased via Anis-Lloyd correction).
      - H < 0.5: anti-persistent / mean-reverting chop series.
    """
    arr = np.asarray(prices, dtype=np.float64)
    if arr.ndim != 1 or len(arr) < 20:
        raise ValueError(f"compute_hurst_exponent requires at least 20 1D prices, got {len(arr)}")

    if np.any(arr <= 0.0):
        arr = arr - np.min(arr) + 1.0

    log_p = np.log(arr)
    n = len(log_p)
    upper_lag = min(max(int(max_lag), 6), n // 3)
    lags = np.arange(1, upper_lag + 1, dtype=np.int64)

    # 1. Multi-horizon structure-function (RMS) & demeaned variogram scaling on log prices
    tau_stds: list[float] = []
    tau_rms: list[float] = []
    valid_lags: list[float] = []
    for lag in lags:
        diffs = log_p[lag:] - log_p[:-lag]
        std_val = float(np.std(diffs, ddof=1))
        rms_val = float(np.sqrt(np.mean(diffs * diffs)))
        if std_val > 1e-12 and rms_val > 1e-12:
            tau_stds.append(std_val)
            tau_rms.append(rms_val)
            valid_lags.append(float(lag))

    if len(valid_lags) < 3:
        return 0.5

    slope_var, _ = np.polyfit(np.log(valid_lags), np.log(tau_stds), 1)
    slope_rms, _ = np.polyfit(np.log(valid_lags), np.log(tau_rms), 1)

    # 2. Tail-aligned Rescaled Range (R/S) with Anis-Lloyd (1976) finite-sample white-noise debiasing
    returns = np.diff(log_p)
    rs_lags: list[float] = []
    rs_adj_logs: list[float] = []
    min_chunk = 8
    max_chunk = min(len(returns) // 2, max(upper_lag * 2, 16))
    if max_chunk >= min_chunk + 4:
        chunk_sizes = np.unique(
            np.linspace(min_chunk, max_chunk, num=min(8, max_chunk - min_chunk + 1), dtype=np.int64)
        )
        for w_raw in chunk_sizes:
            w = int(w_raw)
            num_chunks = len(returns) // w
            if num_chunks < 1:
                continue
            # Slice from the END so the most recent bars are always preserved
            tail_returns = returns[len(returns) - num_chunks * w :]
            chunk_rs: list[float] = []
            for c_idx in range(num_chunks):
                sub = tail_returns[c_idx * w : (c_idx + 1) * w]
                mean_sub = float(np.mean(sub))
                cum_dev = np.cumsum(sub - mean_sub)
                r_val = float(np.max(cum_dev) - np.min(cum_dev))
                s_val = float(np.std(sub, ddof=1))
                if s_val > 1e-12 and r_val > 0.0:
                    chunk_rs.append(r_val / s_val)
            if chunk_rs:
                e_rs = _anis_lloyd_expected_rs(w)
                rs_lags.append(float(w))
                rs_adj_logs.append(math.log(float(np.mean(chunk_rs))) - math.log(e_rs))

    if len(rs_lags) >= 3:
        slope_delta, _ = np.polyfit(np.log(rs_lags), np.asarray(rs_adj_logs, dtype=np.float64), 1)
        h_rs_adj = 0.50 + float(slope_delta)
        h_est = 0.35 * float(slope_rms) + 0.30 * float(slope_var) + 0.35 * h_rs_adj
    else:
        h_est = 0.50 * float(slope_rms) + 0.50 * float(slope_var)

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


def compute_adx_series(
    highs: Sequence[float] | np.ndarray,
    lows: Sequence[float] | np.ndarray,
    closes: Sequence[float] | np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Compute Wilder's Average Directional Index (ADX) series."""
    h = np.asarray(highs, dtype=np.float64)
    l = np.asarray(lows, dtype=np.float64)
    c = np.asarray(closes, dtype=np.float64)
    n = len(c)
    if not (len(h) == len(l) == n) or n == 0:
        raise ValueError("highs, lows, closes must have matching non-zero length.")
    if n < 3:
        return np.zeros(n, dtype=np.float64)

    eff_p = max(2, min(int(period), max(2, n // 2)))
    tr = np.empty(n, dtype=np.float64)
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)

    tr[0] = max(h[0] - l[0], 1e-12)
    up_move = h[1:] - h[:-1]
    down_move = l[:-1] - l[1:]
    prev_c = c[:-1]
    tr[1:] = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - prev_c), np.abs(l[1:] - prev_c)))
    plus_dm[1:] = np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0)
    minus_dm[1:] = np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0)

    sm_tr = np.zeros(n, dtype=np.float64)
    sm_plus = np.zeros(n, dtype=np.float64)
    sm_minus = np.zeros(n, dtype=np.float64)
    sm_tr[eff_p] = float(np.sum(tr[1 : eff_p + 1]))
    sm_plus[eff_p] = float(np.sum(plus_dm[1 : eff_p + 1]))
    sm_minus[eff_p] = float(np.sum(minus_dm[1 : eff_p + 1]))
    for i in range(eff_p + 1, n):
        sm_tr[i] = sm_tr[i - 1] - (sm_tr[i - 1] / eff_p) + tr[i]
        sm_plus[i] = sm_plus[i - 1] - (sm_plus[i - 1] / eff_p) + plus_dm[i]
        sm_minus[i] = sm_minus[i - 1] - (sm_minus[i - 1] / eff_p) + minus_dm[i]

    dx = np.zeros(n, dtype=np.float64)
    valid = sm_tr > 1e-12
    plus_di = np.zeros(n, dtype=np.float64)
    minus_di = np.zeros(n, dtype=np.float64)
    plus_di[valid] = 100.0 * sm_plus[valid] / sm_tr[valid]
    minus_di[valid] = 100.0 * sm_minus[valid] / sm_tr[valid]
    di_sum = plus_di + minus_di
    dx_mask = di_sum > 1e-12
    dx[dx_mask] = 100.0 * np.abs(plus_di[dx_mask] - minus_di[dx_mask]) / di_sum[dx_mask]

    adx = np.zeros(n, dtype=np.float64)
    seed_end = min(n, 2 * eff_p)
    if seed_end > eff_p:
        adx[seed_end - 1] = float(np.mean(dx[eff_p:seed_end]))
        for i in range(seed_end, n):
            adx[i] = ((adx[i - 1] * (eff_p - 1)) + dx[i]) / float(eff_p)
    elif n > eff_p:
        adx[-1] = float(np.mean(dx[eff_p:]))
    return adx


def compute_hawkes_intensity_ratio(
    closes: Sequence[float] | np.ndarray,
    highs: Sequence[float] | np.ndarray | None = None,
    lows: Sequence[float] | np.ndarray | None = None,
    tick_rates: Sequence[float] | None = None,
    alpha: float = 0.55,
    beta_fast: float = 0.45,
    beta_slow: float = 0.08,
) -> float:
    """Compute univariate exponential Hawkes arrival-intensity acceleration ratio (lambda_fast / lambda_slow).

    Returns > 1.0 when impulse range expansions or tick bursts exhibit self-exciting clustering.
    """
    if tick_rates is not None and len(tick_rates) >= 6:
        arr = np.asarray(tick_rates, dtype=np.float64)
        base = max(float(np.median(arr[:-1])), 1e-6)
        events = (arr > base * 1.20).astype(np.float64)
    else:
        c = np.asarray(closes, dtype=np.float64)
        if len(c) < 10:
            return 1.0
        if highs is not None and lows is not None and len(highs) == len(c) == len(lows):
            ranges = np.maximum(np.asarray(highs, dtype=np.float64) - np.asarray(lows, dtype=np.float64), 1e-12)
        else:
            ranges = np.maximum(np.abs(np.diff(c, prepend=c[0])), 1e-12)
        med_r = max(float(np.median(ranges)), 1e-12)
        events = np.clip((ranges / med_r) - 0.85, 0.0, 3.0)

    n = len(events)
    if n < 5:
        return 1.0
    mu = 0.25
    lam_fast = mu
    lam_slow = mu
    d_fast = math.exp(-beta_fast)
    d_slow = math.exp(-beta_slow)
    for ev in events:
        lam_fast = mu + (lam_fast - mu) * d_fast + alpha * float(ev)
        lam_slow = mu + (lam_slow - mu) * d_slow + alpha * 0.25 * float(ev)
    return float(np.clip(lam_fast / max(lam_slow, 1e-9), 0.5, 5.0))


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
        adx_period: int | None = None,
        er_period: int = 14,
        trending_hurst_threshold: float = 0.52,
        trending_er_threshold: float = 0.25,
        max_spread_median_multiple: float = 1.5,
        max_tick_velocity_sigma: float = 4.0,
        volatility_lookback_bars: int = 4680,  # ~60 sessions of M5 bars (78 * 60)
        min_adx_threshold: float = 12.0,
    ) -> None:
        self.enabled: bool = bool(enabled)
        self.hurst_window: int = int(hurst_window)
        eff_adx_p = int(adx_period) if adx_period is not None else int(adx_atr_period)
        self.adx_atr_period: int = eff_adx_p
        self.adx_period: int = eff_adx_p
        self.er_period: int = int(er_period)
        self.trending_hurst_threshold: float = float(trending_hurst_threshold)
        self.trending_er_threshold: float = float(trending_er_threshold)
        self.max_spread_median_multiple: float = float(max_spread_median_multiple)
        self.max_tick_velocity_sigma: float = float(max_tick_velocity_sigma)
        self.volatility_lookback_bars: int = int(volatility_lookback_bars)
        self.min_adx_threshold: float = float(min_adx_threshold)

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
        adx_series = compute_adx_series(h_arr, l_arr, c_arr, period=self.adx_period)
        current_adx = float(adx_series[-1]) if len(adx_series) > 0 else 0.0
        hawkes_ratio = compute_hawkes_intensity_ratio(
            c_arr,
            highs=h_arr,
            lows=l_arr,
            tick_rates=historical_tick_rates,
        )
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
        # 2. Check TRENDING_EXPANSION vs MEAN_REVERTING_CHOP (Wilder ADX confirms directional strength)
        elif (
            hurst >= self.trending_hurst_threshold
            and er >= self.trending_er_threshold
            and (len(c_arr) < 2 * self.adx_period or current_adx >= self.min_adx_threshold)
        ):
            regime = MarketRegime.TRENDING_EXPANSION
            raw_allow = True
            reason = (
                f"TRENDING_EXPANSION: H={hurst:.3f} (>= {self.trending_hurst_threshold:.2f}), "
                f"ER={er:.3f} (>= {self.trending_er_threshold:.2f}), ADX={current_adx:.1f}, VolPct={vol_pct:.1f}%"
            )
        else:
            regime = MarketRegime.MEAN_REVERTING_CHOP
            raw_allow = False
            reason = (
                f"MEAN_REVERTING_CHOP: H={hurst:.3f}, ER={er:.3f}, ADX={current_adx:.1f} "
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
            adx=round(current_adx, 2),
            hawkes_intensity_ratio=round(hawkes_ratio, 4),
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
