"""Tiered Multi-Timeframe + Level 2 Microstructure Gate for PocketBot Apex.

Enforces:
  1. Tight-Spread Tier (M5): EURUSD, GBPUSD, USDJPY, XAUUSD, US500/SPX, NAS100/NQ.
  2. Wide-Spread Momentum Tier (M15): BTCUSD, ETHUSD, TSLA, NVDA — requires:
       - Live spread <= 15% of ATR Stop-Loss distance
       - Impulse candle body (`|close - open|`) >= 1.5x ATR(14)
       - Close in the extreme 25% of the bar range (top 25% for BUY, bottom 25% for SELL)
  3. Level 2 DOM Gate (when cTrader Depth Quotes / `DOMAnalytics` are active):
       - Order-Book Imbalance alignment (`OBI >= +0.15` for longs, `OBI <= -0.15` for shorts)
       - Opposing Resting Liquidity Wall check: rejects entries if an opposing resting wall
         (`> 3x` median book level) sits within `0.35R` of entry in the trade direction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

from apex.feeds.models import Candle, DOMAnalytics, LiquidityWall, Side
from apex.quant.regime import compute_atr_series


class InstrumentTier(str, Enum):
    """Execution timeframe & spread tier classification."""

    TIGHT_SPREAD_M5 = "TIGHT_SPREAD_M5"
    WIDE_SPREAD_M15 = "WIDE_SPREAD_M15"


TIGHT_SPREAD_M5_SYMBOLS: frozenset[str] = frozenset(
    {
        "EURUSD",
        "GBPUSD",
        "USDJPY",
        "USDCHF",
        "AUDUSD",
        "USDCAD",
        "NZDUSD",
        "XAUUSD",
        "US500",
        "SPX",
        "SPX500",
        "NAS100",
        "NQ",
        "USTEC",
    }
)

WIDE_SPREAD_M15_SYMBOLS: frozenset[str] = frozenset(
    {
        "BTCUSD",
        "ETHUSD",
        "SOLUSD",
        "TSLA",
        "NVDA",
        "AMD",
        "AAPL",
        "MSFT",
        "META",
        "AMZN",
        "GOOGL",
    }
)


def classify_instrument_tier(symbol: str) -> InstrumentTier:
    """Map an instrument symbol to its required execution tier (M5 Tight-Spread vs M15 Wide-Spread)."""
    clean = symbol.upper().strip().lstrip("#")
    for prefix in TIGHT_SPREAD_M5_SYMBOLS:
        if clean == prefix or clean.startswith(prefix):
            return InstrumentTier.TIGHT_SPREAD_M5
    for prefix in WIDE_SPREAD_M15_SYMBOLS:
        if clean == prefix or clean.startswith(prefix):
            return InstrumentTier.WIDE_SPREAD_M15
    # Default FX pairs (6 letters) to Tight-Spread M5, equities/crypto to Wide-Spread M15
    if len(clean) == 6 and clean.isalpha():
        return InstrumentTier.TIGHT_SPREAD_M5
    return InstrumentTier.WIDE_SPREAD_M15


@dataclass(frozen=True, slots=True)
class MomentumDOMVerdict:
    """Detailed pass/block verdict from the Tiered Multi-Timeframe + Level 2 DOM Gate."""

    symbol: str
    side: Side
    tier: InstrumentTier
    required_timeframe: str
    passed: bool
    spread_pct_of_sl: float
    impulse_body_atr_ratio: float
    close_range_position: float
    dom_active: bool
    obi: float | None
    blocking_wall: LiquidityWall | None
    reasons: tuple[str, ...] = field(default_factory=tuple)


class MomentumDOMGate:
    """Tiered Multi-Timeframe + Level 2 Microstructure Gate (`apex/quant/momentum_dom.py`).

    Respects `dom_obi_enabled` (`config["filters"]["dom_obi"]["enabled"]`) so walk-forward
    ablation tests can toggle the Level 2 DOM filter independently.
    """

    def __init__(
        self,
        dom_obi_enabled: bool = True,
        min_abs_obi: float = 0.15,
        max_spread_pct_of_sl: float = 0.15,
        min_impulse_atr_multiple: float = 1.5,
        extreme_close_fraction: float = 0.25,
        wall_min_median_multiple: float = 3.0,
        wall_max_distance_r: float = 0.35,
    ) -> None:
        self.dom_obi_enabled: bool = bool(dom_obi_enabled)
        self.min_abs_obi: float = float(min_abs_obi)
        self.max_spread_pct_of_sl: float = float(max_spread_pct_of_sl)
        self.min_impulse_atr_multiple: float = float(min_impulse_atr_multiple)
        self.extreme_close_fraction: float = float(extreme_close_fraction)
        self.wall_min_median_multiple: float = float(wall_min_median_multiple)
        self.wall_max_distance_r: float = float(wall_max_distance_r)

    def evaluate(
        self,
        symbol: str,
        side: Side,
        entry_price: float,
        stop_price: float,
        candles: Sequence[Candle],
        current_spread: float,
        dom_analytics: DOMAnalytics | None = None,
    ) -> MomentumDOMVerdict:
        """Evaluate whether a candidate setup passes both its Tiered Bar Gate and Level 2 DOM Gate."""
        norm_sym = symbol.upper().strip()
        tier = classify_instrument_tier(norm_sym)
        required_tf = "M5" if tier == InstrumentTier.TIGHT_SPREAD_M5 else "M15"

        sl_distance = abs(float(entry_price) - float(stop_price))
        if sl_distance <= 1e-12:
            raise ValueError(f"Stop-loss distance must be > 0 (entry={entry_price}, stop={stop_price})")

        spread_pct_of_sl = float(current_spread) / sl_distance
        reasons: list[str] = []

        # Calculate ATR(14) and impulse metrics from the trigger candle (last candle in sequence)
        if not candles:
            raise ValueError("candles sequence must contain at least 1 completed bar.")

        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        closes = [c.close for c in candles]
        atr_series = compute_atr_series(highs, lows, closes, period=14)
        current_atr = max(float(atr_series[-1]), 1e-12)

        trigger_bar = candles[-1]
        bar_tf = trigger_bar.timeframe.upper()
        if bar_tf and bar_tf != required_tf:
            reasons.append(
                f"timeframe_mismatch: {norm_sym} belongs to {tier.value} (requires {required_tf} bars, got {bar_tf})"
            )

        impulse_body = abs(trigger_bar.close - trigger_bar.open)
        impulse_atr_ratio = impulse_body / current_atr
        bar_range = max(trigger_bar.high - trigger_bar.low, 1e-12)
        # Position of close inside bar range: 1.0 = high of bar, 0.0 = low of bar
        close_pos = (trigger_bar.close - trigger_bar.low) / bar_range

        # Enforce Wide-Spread Momentum Tier (M15) rules
        if tier == InstrumentTier.WIDE_SPREAD_M15:
            if spread_pct_of_sl > self.max_spread_pct_of_sl + 1e-9:
                reasons.append(
                    f"wide_spread_sl_ratio: spread ({current_spread:.4f}) is {spread_pct_of_sl * 100:.1f}% "
                    f"of ATR Stop-Loss distance ({sl_distance:.4f}), exceeding {self.max_spread_pct_of_sl * 100:.1f}% limit"
                )
            if impulse_atr_ratio < self.min_impulse_atr_multiple - 1e-9:
                reasons.append(
                    f"weak_impulse_body: M15 body ({impulse_body:.4f}) is {impulse_atr_ratio:.2f}x ATR(14) "
                    f"({current_atr:.4f}), requiring >= {self.min_impulse_atr_multiple:.2f}x ATR(14)"
                )
            if side == Side.BUY:
                min_pos = 1.0 - self.extreme_close_fraction
                if close_pos < min_pos - 1e-9:
                    reasons.append(
                        f"close_not_in_extreme_25pct: BUY close is at {close_pos * 100:.1f}% of bar range "
                        f"(requires top {self.extreme_close_fraction * 100:.0f}% >= {min_pos * 100:.0f}%)"
                    )
                if trigger_bar.close <= trigger_bar.open:
                    reasons.append("impulse_direction_mismatch: BUY trigger bar closed bearish")
            else:
                max_pos = self.extreme_close_fraction
                if close_pos > max_pos + 1e-9:
                    reasons.append(
                        f"close_not_in_extreme_25pct: SELL close is at {close_pos * 100:.1f}% of bar range "
                        f"(requires bottom {self.extreme_close_fraction * 100:.0f}% <= {max_pos * 100:.0f}%)"
                    )
                if trigger_bar.close >= trigger_bar.open:
                    reasons.append("impulse_direction_mismatch: SELL trigger bar closed bullish")
        else:
            # Tight-spread M5 tier still enforces a sanity check that spread <= 15% of SL
            if spread_pct_of_sl > self.max_spread_pct_of_sl + 1e-9:
                reasons.append(
                    f"spread_sl_ratio: spread ({current_spread:.5f}) is {spread_pct_of_sl * 100:.1f}% "
                    f"of SL distance ({sl_distance:.5f}) > {self.max_spread_pct_of_sl * 100:.1f}%"
                )

        # Level 2 DOM Microstructure Gate (when cTrader Depth Quotes are active & filter enabled)
        dom_active = dom_analytics is not None and self.dom_obi_enabled
        obi_val: float | None = dom_analytics.obi if dom_analytics is not None else None
        blocking_wall: LiquidityWall | None = None

        if dom_active and dom_analytics is not None:
            if side == Side.BUY and dom_analytics.obi < self.min_abs_obi - 1e-9:
                reasons.append(
                    f"dom_obi_misaligned: BUY requires OBI >= +{self.min_abs_obi:.2f}, "
                    f"got {dom_analytics.obi:+.3f}"
                )
            elif side == Side.SELL and dom_analytics.obi > -self.min_abs_obi + 1e-9:
                reasons.append(
                    f"dom_obi_misaligned: SELL requires OBI <= -{self.min_abs_obi:.2f}, "
                    f"got {dom_analytics.obi:+.3f}"
                )

            # Check for opposing resting liquidity walls (> 3x median book level) within 0.35R of entry
            # For a BUY entry, an opposing wall is a resting SELL (ask) wall above entry within 0.35R.
            # For a SELL entry, an opposing wall is a resting BUY (bid) wall below entry within 0.35R.
            opposing_side = Side.SELL if side == Side.BUY else Side.BUY
            max_wall_dist_price = self.wall_max_distance_r * sl_distance

            for wall in dom_analytics.liquidity_walls:
                if wall.side != opposing_side:
                    continue
                if wall.ratio_to_median <= self.wall_min_median_multiple:
                    continue
                if side == Side.BUY:
                    dist_ahead = wall.price - float(entry_price)
                else:
                    dist_ahead = float(entry_price) - wall.price

                if 0.0 <= dist_ahead <= max_wall_dist_price + 1e-9:
                    blocking_wall = wall
                    dist_r = dist_ahead / sl_distance
                    reasons.append(
                        f"opposing_liquidity_wall: {wall.side.value} wall at {wall.price:.5f} "
                        f"({wall.ratio_to_median:.1f}x median book volume) sits at {dist_r:.2f}R "
                        f"(<= {self.wall_max_distance_r:.2f}R limit)"
                    )
                    break

        return MomentumDOMVerdict(
            symbol=norm_sym,
            side=side,
            tier=tier,
            required_timeframe=required_tf,
            passed=len(reasons) == 0,
            spread_pct_of_sl=spread_pct_of_sl,
            impulse_body_atr_ratio=impulse_atr_ratio,
            close_range_position=close_pos,
            dom_active=dom_active,
            obi=obi_val,
            blocking_wall=blocking_wall,
            reasons=tuple(reasons),
        )
