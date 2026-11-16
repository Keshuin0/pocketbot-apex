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

from apex.feeds.alias_mapper import SmartAliasMapper
from apex.feeds.models import AssetClass, Candle, DOMAnalytics, LiquidityWall, Side
from apex.quant.regime import compute_atr_series


_TIER_MAPPER = SmartAliasMapper()


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
        "XAGUSD",
        "US500",
        "SPX",
        "SPX500",
        "NAS100",
        "US100",
        "US30",
        "GER40",
        "UK100",
        "USTEC",
    }
)

WIDE_SPREAD_M15_SYMBOLS: frozenset[str] = frozenset(
    {
        "BTCUSD",
        "ETHUSD",
        "SOLUSD",
        "USOIL",
        "UKOIL",
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
    """Delegate instrument tier mapping strictly to SmartAliasMapper.classify_asset_class()."""
    canon = _TIER_MAPPER.canonicalize(symbol).upper()
    if canon in WIDE_SPREAD_M15_SYMBOLS:
        return InstrumentTier.WIDE_SPREAD_M15
    asset_class = SmartAliasMapper.classify_asset_class(symbol)
    if asset_class in (AssetClass.FOREX, AssetClass.METALS):
        return InstrumentTier.TIGHT_SPREAD_M5
    if asset_class == AssetClass.INDICES:
        return InstrumentTier.TIGHT_SPREAD_M5
    if canon in TIGHT_SPREAD_M5_SYMBOLS:
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
    ofi: float | None = None
    vpin: float | None = None
    setup_type: str = "MOMENTUM"


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
        max_vpin: float = 0.75,
        min_abs_ofi: float = 0.05,
        max_exhaustion_atr_multiple: float = 2.4,
    ) -> None:
        self.dom_obi_enabled: bool = bool(dom_obi_enabled)
        self.min_abs_obi: float = float(min_abs_obi)
        self.max_spread_pct_of_sl: float = float(max_spread_pct_of_sl)
        self.min_impulse_atr_multiple: float = float(min_impulse_atr_multiple)
        self.extreme_close_fraction: float = float(extreme_close_fraction)
        self.wall_min_median_multiple: float = float(wall_min_median_multiple)
        self.wall_max_distance_r: float = float(wall_max_distance_r)
        self.max_vpin: float = float(max_vpin)
        self.min_abs_ofi: float = float(min_abs_ofi)
        self.max_exhaustion_atr_multiple: float = float(max_exhaustion_atr_multiple)

    def evaluate(
        self,
        symbol: str,
        side: Side,
        entry_price: float,
        stop_price: float,
        candles: Sequence[Candle],
        current_spread: float,
        dom_analytics: DOMAnalytics | None = None,
        hawkes_intensity_ratio: float = 1.0,
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
        close_pos = (trigger_bar.close - trigger_bar.low) / bar_range

        # Uniform Spread-to-SL Budget Check across ALL instrument tiers
        if spread_pct_of_sl > self.max_spread_pct_of_sl + 1e-9:
            reasons.append(
                f"spread_sl_ratio: spread ({current_spread:.5f}) is {spread_pct_of_sl * 100:.1f}% "
                f"of SL distance ({sl_distance:.5f}) > {self.max_spread_pct_of_sl * 100:.1f}%"
            )

        # Structural Liquidity Sweep / Displacement Retest & Candle Polarity Gate (uniform across M5 & M15)
        prior_window = candles[-7:-1] if len(candles) >= 3 else []
        is_sweep_reclaim = False
        is_displacement_retest = False
        if prior_window:
            if side == Side.BUY:
                prior_low = min(c.low for c in prior_window)
                swept_low = (trigger_bar.low <= prior_low) or (len(candles) >= 2 and candles[-2].low <= prior_low)
                is_sweep_reclaim = swept_low and (trigger_bar.close >= trigger_bar.open or close_pos >= 0.55)
                prior_bull_disp = any((c.close - c.open) >= 0.45 * current_atr for c in prior_window[-3:])
                shallow_pullback = (trigger_bar.open - trigger_bar.close) <= 0.40 * current_atr
                is_displacement_retest = prior_bull_disp and shallow_pullback
            else:
                prior_high = max(c.high for c in prior_window)
                swept_high = (trigger_bar.high >= prior_high) or (len(candles) >= 2 and candles[-2].high >= prior_high)
                is_sweep_reclaim = swept_high and (trigger_bar.close <= trigger_bar.open or close_pos <= 0.45)
                prior_bear_disp = any((c.open - c.close) >= 0.45 * current_atr for c in prior_window[-3:])
                shallow_pullback = (trigger_bar.close - trigger_bar.open) <= 0.40 * current_atr
                is_displacement_retest = prior_bear_disp and shallow_pullback

        setup_type = "LIQUIDITY_SWEEP_RECLAIM" if is_sweep_reclaim else (
            "DISPLACEMENT_RETEST" if is_displacement_retest else "MOMENTUM"
        )

        # Candle polarity + OFI/Hawkes confirmation on ALL tiers (replaces empty M5 else-pass & 1.5x-2.2x exhaustion chase)
        dom_active = dom_analytics is not None and self.dom_obi_enabled
        ofi_val: float | None = float(getattr(dom_analytics, "ofi", 0.0)) if dom_analytics is not None else None
        vpin_val: float | None = float(getattr(dom_analytics, "vpin", 0.0)) if dom_analytics is not None else None
        ofi_aligned = (
            ofi_val is not None
            and ((side == Side.BUY and ofi_val >= self.min_abs_ofi) or (side == Side.SELL and ofi_val <= -self.min_abs_ofi))
        )
        hawkes_accel = float(hawkes_intensity_ratio) >= 1.10

        if side == Side.BUY:
            bullish_bar = trigger_bar.close > trigger_bar.open
            if not bullish_bar and not is_sweep_reclaim and not (is_displacement_retest and (close_pos >= 0.40 or ofi_aligned or hawkes_accel)):
                reasons.append("impulse_direction_mismatch: BUY trigger bar closed bearish without sweep/retest reclaim")
            elif impulse_atr_ratio > self.max_exhaustion_atr_multiple and close_pos >= 0.90 and not is_sweep_reclaim and not (ofi_aligned and hawkes_accel):
                reasons.append(
                    f"exhaustion_climax_chase: BUY bar stretched {impulse_atr_ratio:.2f}x ATR at {close_pos*100:.0f}% top without sweep/OFI"
                )
        else:
            bearish_bar = trigger_bar.close < trigger_bar.open
            if not bearish_bar and not is_sweep_reclaim and not (is_displacement_retest and (close_pos <= 0.60 or ofi_aligned or hawkes_accel)):
                reasons.append("impulse_direction_mismatch: SELL trigger bar closed bullish without sweep/retest reclaim")
            elif impulse_atr_ratio > self.max_exhaustion_atr_multiple and close_pos <= 0.10 and not is_sweep_reclaim and not (ofi_aligned and hawkes_accel):
                reasons.append(
                    f"exhaustion_climax_chase: SELL bar stretched {impulse_atr_ratio:.2f}x ATR at {close_pos*100:.0f}% bottom without sweep/OFI"
                )

        # Level 2 DOM + OFI + VPIN + Liquidity Wall Gate
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

            if ofi_val is not None and abs(ofi_val) > 1e-6:
                if side == Side.BUY and ofi_val < -self.min_abs_ofi:
                    reasons.append(f"dom_ofi_misaligned: BUY opposing OFI={ofi_val:+.3f} < -{self.min_abs_ofi:.2f}")
                elif side == Side.SELL and ofi_val > self.min_abs_ofi:
                    reasons.append(f"dom_ofi_misaligned: SELL opposing OFI={ofi_val:+.3f} > +{self.min_abs_ofi:.2f}")

            if vpin_val is not None and vpin_val > self.max_vpin:
                reasons.append(f"vpin_toxicity_extreme: VPIN={vpin_val:.3f} > {self.max_vpin:.2f} flow toxicity threshold")

            # Check for opposing resting liquidity walls (> 3x median book level) within 0.35R of entry
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
            ofi=ofi_val,
            vpin=vpin_val,
            setup_type=setup_type,
        )
