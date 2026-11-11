"""Strict data models and safety enforcement for PocketBot Apex feeds."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence


class Side(str, Enum):
    """Order or liquidity book side."""

    BUY = "BUY"
    SELL = "SELL"


class LiveLockViolationError(RuntimeError):
    """Raised when attempting to connect to a LIVE broker without dual safety unlock."""


def verify_live_unlock(is_live: bool, cli_args: Sequence[str] | None = None) -> None:
    """Enforce the non-negotiable PocketBot Apex DEMO lock invariant.

    Hard-locked to DEMO accounts unless BOTH:
      1. ``--unlock-live`` is present in CLI arguments, AND
      2. ``APEX_UNLOCK_LIVE=I_ACCEPT_REAL_RISK`` is set in environment variables.
    """
    if not is_live:
        return

    import sys

    args_to_check = list(cli_args) if cli_args is not None else sys.argv
    has_cli_flag = "--unlock-live" in args_to_check
    env_val = os.environ.get("APEX_UNLOCK_LIVE", "")
    has_env_unlock = env_val == "I_ACCEPT_REAL_RISK"

    if not (has_cli_flag and has_env_unlock):
        raise LiveLockViolationError(
            "SAFETY VIOLATION: Live account routing is hard-locked. "
            "Both '--unlock-live' CLI flag AND 'APEX_UNLOCK_LIVE=I_ACCEPT_REAL_RISK' "
            "environment variable are strictly required to unlock live accounts."
        )


@dataclass(frozen=True, slots=True)
class Tick:
    """Normalized real-time spot tick across cTrader and MT5."""

    symbol: str
    bid: float
    ask: float
    timestamp_utc: float
    source: str
    volume: float = 0.0

    @property
    def mid(self) -> float:
        """Midpoint price."""
        return 0.5 * (self.bid + self.ask)

    @property
    def spread(self) -> float:
        """Raw bid-ask spread in price terms."""
        return self.ask - self.bid


@dataclass(frozen=True, slots=True)
class Candle:
    """Normalized OHLCV trendbar / candle."""

    symbol: str
    timeframe: str
    timestamp_utc: float
    open: float
    high: float
    low: float
    close: float
    volume: float
    source: str


@dataclass(frozen=True, slots=True)
class ContractSpec:
    """Normalized instrument contract specifications."""

    symbol: str
    digits: int
    pip_size: float
    point_size: float
    lot_size: float
    min_volume_lots: float
    max_volume_lots: float
    step_volume_lots: float
    source: str


@dataclass(frozen=True, slots=True)
class DepthLevel:
    """Single price level in a Level 2 Depth of Market (DOM) book."""

    quote_id: int
    price: float
    volume_units: float
    side: Side


@dataclass(frozen=True, slots=True)
class LiquidityWall:
    """Detected abnormal liquidity concentration in the Level 2 order book."""

    side: Side
    price: float
    volume_units: float
    ratio_to_median: float
    distance_from_mid_pips: float


@dataclass(frozen=True, slots=True)
class VWAPExecutionEstimate:
    """True Level 2 DOM VWAP fill price and slippage calculation for a target volume."""

    symbol: str
    side: Side
    target_volume_units: float
    filled_volume_units: float
    unfilled_volume_units: float
    top_of_book_price: float
    vwap_fill_price: float
    slippage_price: float
    slippage_pips: float
    levels_consumed: int
    is_fully_fillable: bool


@dataclass(frozen=True, slots=True)
class DOMAnalytics:
    """Real-time microstructure analytics computed from Level 2 DOM."""

    symbol: str
    timestamp_utc: float
    top_n_levels: int
    bid_volume_top_n: float
    ask_volume_top_n: float
    obi: float
    best_bid: float
    best_ask: float
    spread_pips: float
    liquidity_walls: tuple[LiquidityWall, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class OrderBookSnapshot:
    """Complete sorted Level 2 order book snapshot (bids descending, asks ascending)."""

    symbol: str
    timestamp_utc: float
    bids: tuple[DepthLevel, ...]
    asks: tuple[DepthLevel, ...]
    pip_size: float


@dataclass(frozen=True, slots=True)
class TickDiscrepancyAlert:
    """Alert emitted when cross-feed price divergence exceeds the configured Z-score threshold."""

    symbol: str
    timestamp_utc: float
    primary_mid: float
    standby_mid: float
    abs_diff: float
    rolling_mean_diff: float
    rolling_std_diff: float
    z_score: float
    threshold_z: float
    is_bad_tick: bool
    reason: str
