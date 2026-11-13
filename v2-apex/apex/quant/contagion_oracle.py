"""Live 251-Symbol Cross-Asset Contagion & Session Rollover Oracle (`apex/quant/contagion_oracle.py`).

Computes deterministic ground-truth macro & liquidity telemetry at $0 AI cost directly from
the live market feed and UTC clock, replacing LLM training-set macro guesswork:
  - `rollover_illiquidity_flag`: detects daily FX/Index maintenance & rollover spread-widening
    windows (21:55-23:10 UTC).
  - `cross_asset_shock_z`: measures simultaneous >2.5-sigma 5m return shocks across USD FX pairs,
    INDICES (US500/NAS100), METALS (XAUUSD), and CRYPTO (BTCUSD).
"""

from __future__ import annotations

import datetime
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    from apex.feeds.dual_router import MarketFeedRouter

DEFAULT_ANCHOR_BUCKETS: dict[str, tuple[str, ...]] = {
    "USD_FX": ("EURUSD", "GBPUSD", "USDJPY"),
    "INDICES": ("US500", "NAS100"),
    "METALS": ("XAUUSD",),
    "CRYPTO": ("BTCUSD",),
}


@dataclass(frozen=True, slots=True)
class ContagionTelemetry:
    """Ground-truth cross-asset contagion and session rollover telemetry."""

    rollover_illiquidity_flag: bool
    cross_asset_shock_z: float
    shocked_buckets: tuple[str, ...] = ()
    session_window_label: str = "NORMAL_LIQUIDITY"
    detail: str = ""


class ContagionOracle:
    """Deterministic $0 cross-asset contagion and rollover illiquidity oracle."""

    SHOCK_BUCKET_Z_THRESHOLD: float = 2.5

    @staticmethod
    def is_rollover_illiquidity_window(
        utc_dt: datetime.datetime | None = None,
        spread_ratio: float | None = None,
    ) -> tuple[bool, str]:
        """Detect daily FX/Index maintenance & rollover spread-widening windows (21:55-23:10 UTC)."""
        dt = (
            utc_dt.astimezone(datetime.timezone.utc)
            if utc_dt is not None
            else datetime.datetime.now(datetime.timezone.utc)
        )
        mins = dt.hour * 60 + dt.minute
        # 21:55 UTC (1315m) through 23:10 UTC (1390m)
        if 1315 <= mins <= 1390:
            return True, f"ROLLOVER_MAINTENANCE_WINDOW ({dt.hour:02d}:{dt.minute:02d} UTC)"

        if spread_ratio is not None and spread_ratio >= 2.5 and (1260 <= mins <= 1410):
            return True, f"HANDOFF_SPREAD_WIDENING ({spread_ratio:.1f}x @ {dt.hour:02d}:{dt.minute:02d} UTC)"

        return False, "NORMAL_LIQUIDITY"

    @staticmethod
    def _series_latest_z(returns: Sequence[float]) -> float:
        if len(returns) < 12:
            return 0.0
        arr = np.asarray(returns, dtype=float)
        hist = arr[:-1]
        mu = float(np.mean(hist))
        sigma = float(np.std(hist, ddof=1)) if len(hist) > 1 else 0.0
        if sigma <= 1e-12 or math.isnan(sigma):
            return 0.0
        # Session-aware robust baseline floor so ultra-quiet rollover bars don't explode Z
        med = float(np.median(hist))
        mad = float(np.median(np.abs(hist - med)))
        robust_sigma = max(sigma, 1.4826 * mad, 0.00035)
        z = abs(float(arr[-1]) - mu) / robust_sigma
        return round(float(z), 4)

    def compute_cross_asset_shock_z(
        self,
        anchor_returns: dict[str, Sequence[float]] | None = None,
        feed_router: MarketFeedRouter | None = None,
        now_utc: float | None = None,
    ) -> tuple[float, tuple[str, ...]]:
        """Measure simultaneous >2.5-sigma 5m return shocks strictly on 7 liquid core benchmarks."""
        bucket_max_z: dict[str, float] = {}
        cur_ts = float(now_utc) if now_utc is not None else datetime.datetime.now(datetime.timezone.utc).timestamp()

        for bucket, symbols in DEFAULT_ANCHOR_BUCKETS.items():
            max_z = 0.0
            for sym in symbols:
                rets: Sequence[float] | None = None
                if anchor_returns is not None and sym in anchor_returns:
                    rets = anchor_returns[sym]
                elif feed_router is not None:
                    try:
                        candles = feed_router.get_candles(sym, "M5", count=40)
                        if len(candles) >= 15:
                            # Ignore frozen/maintenance bars older than 15m so shock Z never freezes
                            last_ts = float(getattr(candles[-1], "timestamp_utc", 0.0) or 0.0)
                            if last_ts <= 0.0 or (cur_ts - last_ts) <= 900.0:
                                closes = np.asarray([c.close for c in candles], dtype=float)
                                rets = np.diff(closes) / np.maximum(closes[:-1], 1e-9)
                    except Exception:
                        rets = None
                if rets is not None:
                    z = self._series_latest_z(rets)
                    if z > max_z:
                        max_z = z
            bucket_max_z[bucket] = max_z

        shocked = tuple(
            b for b, z in bucket_max_z.items() if z >= self.SHOCK_BUCKET_Z_THRESHOLD
        )
        if not bucket_max_z:
            return 0.0, ()

        top_z = max(bucket_max_z.values())
        if len(shocked) >= 2:
            # Simultaneous multi-asset shock amplification
            sorted_shocked = sorted((bucket_max_z[b] for b in shocked), reverse=True)
            composite = (sorted_shocked[0] + sorted_shocked[1]) / 2.0 + 0.5 * (len(shocked) - 1)
            return round(max(top_z, composite), 3), shocked
        return round(top_z, 3), shocked

    def evaluate(
        self,
        feed_router: MarketFeedRouter | None = None,
        anchor_returns: dict[str, Sequence[float]] | None = None,
        utc_dt: datetime.datetime | None = None,
        spread_ratio: float | None = None,
    ) -> ContagionTelemetry:
        """Compute complete ground-truth contagion & rollover telemetry."""
        rollover_flag, window_label = self.is_rollover_illiquidity_window(
            utc_dt=utc_dt,
            spread_ratio=spread_ratio,
        )
        now_ts = utc_dt.timestamp() if utc_dt is not None else None
        shock_z, shocked_buckets = self.compute_cross_asset_shock_z(
            anchor_returns=anchor_returns,
            feed_router=feed_router,
            now_utc=now_ts,
        )
        detail = (
            f"Rollover={rollover_flag} ({window_label}) | "
            f"CrossAssetShockZ={shock_z:.2f} (shocked={list(shocked_buckets)})"
        )
        return ContagionTelemetry(
            rollover_illiquidity_flag=rollover_flag,
            cross_asset_shock_z=shock_z,
            shocked_buckets=shocked_buckets,
            session_window_label=window_label,
            detail=detail,
        )
