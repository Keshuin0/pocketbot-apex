"""Phase 2 Verification Suite for PocketBot Apex Institutional Quant & Portfolio Risk Core."""

from __future__ import annotations

import numpy as np
import pytest

from apex.feeds.models import Candle, DOMAnalytics, LiquidityWall, Side
from apex.quant.correlation_shield import (
    CorrelationShield,
    PositionRiskSpec,
    decompose_symbol_factors,
)
from apex.quant.momentum_dom import (
    InstrumentTier,
    MomentumDOMGate,
    classify_instrument_tier,
)
from apex.quant.regime import (
    MarketRegime,
    RegimeClassifier,
    compute_hurst_exponent,
    compute_kaufman_er,
    compute_volatility_percentile,
)
from apex.quant.timemachine_v2 import (
    BlockBootstrapTimeMachine,
    TimeMachineCostModel,
)


def _make_candles_from_closes(
    symbol: str,
    timeframe: str,
    closes: np.ndarray,
    bar_half_range: float = 0.0005,
) -> list[Candle]:
    candles: list[Candle] = []
    base_ts = 1700000000.0
    step_s = 300.0 if timeframe == "M5" else 900.0
    for i, cl in enumerate(closes):
        op = float(closes[i - 1]) if i > 0 else float(cl)
        hi = max(op, float(cl)) + bar_half_range
        lo = min(op, float(cl)) - bar_half_range
        candles.append(
            Candle(
                symbol=symbol,
                timeframe=timeframe,
                timestamp_utc=base_ts + i * step_s,
                open=op,
                high=hi,
                low=lo,
                close=float(cl),
                volume=1000.0,
                source="test",
            )
        )
    return candles


def test_hurst_and_regime_classification_trending_vs_mean_reverting_vs_shock() -> None:
    """Validate Hurst exponent & RegimeClassifier on synthetic trending, mean-reverting, and shock series."""
    rng = np.random.default_rng(42)

    # 1. Persistent Trending Series (positively autocorrelated returns + strong drift)
    n = 150
    trend_steps = np.zeros(n, dtype=np.float64)
    eps = rng.normal(0.0006, 0.00025, size=n)
    for i in range(1, n):
        trend_steps[i] = 0.65 * trend_steps[i - 1] + eps[i]
    trending_prices = 100.0 * np.exp(np.cumsum(trend_steps))

    # 2. Mean-Reverting Chop Series (Ornstein-Uhlenbeck / alternating negative autocorrelation)
    mr_prices = np.empty(n, dtype=np.float64)
    mr_prices[0] = 100.0
    noise = rng.normal(0.0, 0.15, size=n)
    for i in range(1, n):
        # Strong pull back to 100.0 every step
        mr_prices[i] = 100.0 - 0.70 * (mr_prices[i - 1] - 100.0) + noise[i]

    h_trend = compute_hurst_exponent(trending_prices)
    h_mr = compute_hurst_exponent(mr_prices)
    er_trend = compute_kaufman_er(trending_prices, period=14)
    er_mr = compute_kaufman_er(mr_prices, period=14)

    assert h_trend > 0.60, f"Expected persistent trending Hurst > 0.60, got {h_trend:.4f}"
    assert h_mr < 0.40, f"Expected anti-persistent mean-reverting Hurst < 0.40, got {h_mr:.4f}"
    assert er_trend > 0.50, f"Expected high directional Kaufman ER on trend, got {er_trend:.4f}"
    assert er_mr < 0.25, f"Expected low Kaufman ER on chop, got {er_mr:.4f}"

    classifier = RegimeClassifier(enabled=True, hurst_window=100)

    # Classify Trending Series -> TRENDING_EXPANSION (breakout entries enabled)
    trend_candles = _make_candles_from_closes("EURUSD", "M5", trending_prices, bar_half_range=0.05)
    diag_trend = classifier.classify_candles(
        "EURUSD",
        trend_candles,
        current_spread=0.00010,
        historical_spreads=[0.00010] * 30,
        current_tick_rate=5.0,
        historical_tick_rates=[4.8, 5.1, 4.9, 5.2, 5.0] * 6,
    )
    assert diag_trend.regime == MarketRegime.TRENDING_EXPANSION
    assert diag_trend.allow_breakout_entries is True
    assert 0.0 <= diag_trend.volatility_percentile <= 100.0

    # Classify Mean-Reverting Series -> MEAN_REVERTING_CHOP (breakout entries auto-blocked)
    mr_candles = _make_candles_from_closes("EURUSD", "M5", mr_prices, bar_half_range=0.05)
    diag_mr = classifier.classify_candles(
        "EURUSD",
        mr_candles,
        current_spread=0.00010,
        historical_spreads=[0.00010] * 30,
    )
    assert diag_mr.regime == MarketRegime.MEAN_REVERTING_CHOP
    assert diag_mr.allow_breakout_entries is False

    # Classify Spread Shock (> 1.5x rolling median spread) -> VOLATILITY_SHOCK (entries locked out)
    diag_spread_shock = classifier.classify_candles(
        "EURUSD",
        trend_candles,
        current_spread=0.00020,  # 2.0x median (> 1.5x limit)
        historical_spreads=[0.00010] * 30,
    )
    assert diag_spread_shock.regime == MarketRegime.VOLATILITY_SHOCK
    assert diag_spread_shock.allow_breakout_entries is False
    assert diag_spread_shock.spread_to_median_ratio == pytest.approx(2.0)

    # Classify Tick Velocity Spike (> 4 sigma) -> VOLATILITY_SHOCK
    diag_vel_shock = classifier.classify_candles(
        "EURUSD",
        trend_candles,
        current_spread=0.00010,
        historical_spreads=[0.00010] * 30,
        current_tick_rate=30.0,  # > 4 sigma above ~5.0 +/- 0.2
        historical_tick_rates=[4.8, 5.1, 4.9, 5.2, 5.0] * 6,
    )
    assert diag_vel_shock.regime == MarketRegime.VOLATILITY_SHOCK
    assert diag_vel_shock.allow_breakout_entries is False
    assert diag_vel_shock.tick_velocity_zscore > 4.0


def test_momentum_dom_tiered_and_level2_wall_gate() -> None:
    """Validate Tight-Spread M5 vs Wide-Spread M15 rules, OBI >= 0.15 alignment, and 0.35R wall rejection."""
    assert classify_instrument_tier("EURUSD") == InstrumentTier.TIGHT_SPREAD_M5
    assert classify_instrument_tier("XAUUSD") == InstrumentTier.TIGHT_SPREAD_M5
    assert classify_instrument_tier("NAS100") == InstrumentTier.TIGHT_SPREAD_M5
    assert classify_instrument_tier("BTCUSD") == InstrumentTier.WIDE_SPREAD_M15
    assert classify_instrument_tier("NVDA") == InstrumentTier.WIDE_SPREAD_M15

    gate = MomentumDOMGate(dom_obi_enabled=True)

    # Build 20 M15 candles for BTCUSD where ATR(14) is ~$200, and the final M15 impulse bar has:
    # open=90000, low=89980, high=90400, close=90380 -> body = $380 (>= 1.5 * $200), close in top 5% of range
    btc_candles: list[Candle] = []
    base_ts = 1700000000.0
    for i in range(19):
        btc_candles.append(
            Candle(
                symbol="BTCUSD",
                timeframe="M15",
                timestamp_utc=base_ts + i * 900.0,
                open=90000.0,
                high=90100.0,
                low=89900.0,
                close=90000.0,
                volume=50.0,
                source="test",
            )
        )
    # Strong M15 impulse breakout candle
    btc_candles.append(
        Candle(
            symbol="BTCUSD",
            timeframe="M15",
            timestamp_utc=base_ts + 19 * 900.0,
            open=90000.0,
            high=90400.0,
            low=89980.0,
            close=90380.0,
            volume=200.0,
            source="test",
        )
    )

    # 1. Valid M15 impulse + positive OBI (+0.28 >= +0.15) + no opposing wall -> PASS
    clean_dom = DOMAnalytics(
        symbol="BTCUSD",
        timestamp_utc=base_ts + 19 * 900.0,
        top_n_levels=5,
        bid_volume_top_n=64.0,
        ask_volume_top_n=36.0,
        obi=0.28,
        best_bid=90379.0,
        best_ask=90381.0,
        spread_pips=20.0,
        liquidity_walls=(),
    )
    v_pass = gate.evaluate(
        symbol="BTCUSD",
        side=Side.BUY,
        entry_price=90380.0,
        stop_price=90080.0,  # 1R = $300 SL distance; spread=$20 is 6.67% of SL (<= 15%)
        candles=btc_candles,
        current_spread=20.0,
        dom_analytics=clean_dom,
    )
    assert v_pass.passed is True
    assert v_pass.reasons == ()

    # 2. Reject when opposing SELL liquidity wall (>3x median) sits within 0.35R of entry
    # 1R = $300 -> 0.35R = $105 above 90380 (i.e. up to 90485). Place a 4.5x SELL wall at 90450 (+0.23R).
    wall_dom = DOMAnalytics(
        symbol="BTCUSD",
        timestamp_utc=base_ts + 19 * 900.0,
        top_n_levels=5,
        bid_volume_top_n=64.0,
        ask_volume_top_n=36.0,
        obi=0.28,
        best_bid=90379.0,
        best_ask=90381.0,
        spread_pips=20.0,
        liquidity_walls=(
            LiquidityWall(
                side=Side.SELL,
                price=90450.0,
                volume_units=45.0,
                ratio_to_median=4.5,
                distance_from_mid_pips=700.0,
            ),
        ),
    )
    v_wall_block = gate.evaluate(
        symbol="BTCUSD",
        side=Side.BUY,
        entry_price=90380.0,
        stop_price=90080.0,
        candles=btc_candles,
        current_spread=20.0,
        dom_analytics=wall_dom,
    )
    assert v_wall_block.passed is False
    assert v_wall_block.blocking_wall is not None
    assert any("opposing_liquidity_wall" in r for r in v_wall_block.reasons)

    # 3. Reject when OBI is misaligned (< +0.15 for BUY)
    weak_obi_dom = DOMAnalytics(
        symbol="BTCUSD",
        timestamp_utc=base_ts + 19 * 900.0,
        top_n_levels=5,
        bid_volume_top_n=52.0,
        ask_volume_top_n=48.0,
        obi=0.04,
        best_bid=90379.0,
        best_ask=90381.0,
        spread_pips=20.0,
        liquidity_walls=(),
    )
    v_obi_block = gate.evaluate(
        symbol="BTCUSD",
        side=Side.BUY,
        entry_price=90380.0,
        stop_price=90080.0,
        candles=btc_candles,
        current_spread=20.0,
        dom_analytics=weak_obi_dom,
    )
    assert v_obi_block.passed is False
    assert any("dom_obi_misaligned" in r for r in v_obi_block.reasons)


def test_timemachine_v2_block_bootstrap_cost_deduction_and_kelly_cap() -> None:
    """Validate 10,000-path Stationary Block Bootstrap, exact net_avg_R cost deduction, CVaR, and 0.5% Kelly cap."""
    rng = np.random.default_rng(7)
    # Create 60 days of M5 bars (60 * 78 = 4680 bars) with slight upward drift
    n_bars = 4680
    log_rets = rng.normal(0.00004, 0.00045, size=n_bars)
    closes = 1.10000 * np.exp(np.cumsum(log_rets))
    candles = _make_candles_from_closes("EURUSD", "M5", closes, bar_half_range=0.00025)

    tm = BlockBootstrapTimeMachine(
        replays=10_000,
        mean_block_length=6.0,
        seed=19,
        account_equity_usd=10_000.0,
        max_risk_pct=0.5,
    )

    # 1. Run with zero friction vs Fusion Zero ($4.50/lot + 1.0 pip spread + 0.4 pip VWAP slippage)
    # 1R distance = 10 pips (0.00100), risk_per_lot_usd = $100.00
    # Commission R = $4.50 / $100.00 = 0.045 R
    # Spread + Slippage R = (0.00010 + 0.00004) / 0.00100 = 0.140 R
    # Total round-trip friction = 0.185 R per path!
    zero_cost = TimeMachineCostModel(
        commission_per_lot_round_turn_usd=0.0,
        lots=1.0,
        risk_per_lot_usd=100.0,
        spread_price=0.0,
        vwap_slippage_price=0.0,
    )
    fusion_cost = TimeMachineCostModel(
        commission_per_lot_round_turn_usd=4.50,
        lots=1.0,
        risk_per_lot_usd=100.0,
        spread_price=0.00010,
        vwap_slippage_price=0.00004,
    )

    res_zero = tm.simulate_candles(
        symbol="EURUSD",
        side=Side.BUY,
        entry_price=1.10000,
        stop_price=1.09900,
        target_price=1.10200,
        candles=candles,
        horizon_bars=36,
        cost_model=zero_cost,
    )
    res_cost = tm.simulate_candles(
        symbol="EURUSD",
        side=Side.BUY,
        entry_price=1.10000,
        stop_price=1.09900,
        target_price=1.10200,
        candles=candles,
        horizon_bars=36,
        cost_model=fusion_cost,
    )

    assert res_cost.replays == 10_000
    assert pytest.approx(res_cost.commission_cost_r, abs=1e-9) == 0.045
    assert pytest.approx(res_cost.spread_slippage_cost_r, abs=1e-9) == 0.140
    assert pytest.approx(res_cost.round_trip_cost_r, abs=1e-9) == 0.185

    # Exact deduction: net_avg_R must equal gross_avg_R - round_trip_cost_r
    assert pytest.approx(res_cost.gross_avg_R, abs=1e-9) == res_zero.net_avg_R
    assert pytest.approx(res_cost.net_avg_R, abs=1e-9) == res_cost.gross_avg_R - 0.185
    assert res_cost.cvar_95_R <= res_cost.var_95_R
    assert res_cost.cvar_95_R <= -1.0  # In worst 5% tail (stops hit), net loss is -1.0R - 0.185R = -1.185R

    # Verify strict 0.5% ($50 on $10,000) Quarter-Kelly cap
    assert 0.0 <= res_cost.recommended_risk_pct <= 0.5
    assert 0.0 <= res_cost.recommended_risk_usd <= 50.0


def test_correlation_shield_blocks_3x_short_usd_concentration_trap() -> None:
    """Validate factor decomposition and 1.0% single-currency/factor cap on Long EURUSD + GBPUSD + XAUUSD."""
    assert decompose_symbol_factors("EURUSD", Side.BUY) == {"EUR": +1.0, "USD": -1.0}
    assert decompose_symbol_factors("GBPUSD", Side.BUY) == {"GBP": +1.0, "USD": -1.0}
    assert decompose_symbol_factors("XAUUSD", Side.BUY) == {"Gold": +1.0, "USD": -1.0}

    shield = CorrelationShield(
        enabled=True,
        max_factor_risk_pct=1.0,
        account_equity_usd=10_000.0,
    )

    # Trade 1: Long EURUSD risking 0.5% ($50) -> Short USD -0.5% (ALLOWED)
    t1 = PositionRiskSpec(symbol="EURUSD", side=Side.BUY, risk_pct=0.5, risk_usd=50.0)
    v1 = shield.evaluate_proposed_trade(t1, open_positions=[])
    assert v1.allowed is True
    assert pytest.approx(v1.factor_exposures_after_pct["USD"]) == -0.5

    # Trade 2: Long GBPUSD risking 0.5% ($50) -> Cumulative Short USD = -1.0% (ALLOWED, right at 1.0% cap)
    t2 = PositionRiskSpec(symbol="GBPUSD", side=Side.BUY, risk_pct=0.5, risk_usd=50.0)
    v2 = shield.evaluate_proposed_trade(t2, open_positions=[t1])
    assert v2.allowed is True
    assert pytest.approx(v2.factor_exposures_after_pct["USD"]) == -1.0

    # Trade 3: Long XAUUSD risking 0.5% ($50) -> Cumulative Short USD = -1.5% (> 1.0% cap -> BLOCKED!)
    t3 = PositionRiskSpec(symbol="XAUUSD", side=Side.BUY, risk_pct=0.5, risk_usd=50.0)
    v3 = shield.evaluate_proposed_trade(t3, open_positions=[t1, t2])
    assert v3.allowed is False
    assert "USD" in v3.breached_factors
    assert pytest.approx(v3.factor_exposures_after_pct["USD"]) == -1.5
    assert "CORRELATION_SHIELD_BLOCK" in v3.reason

    # Conversely, a trade that REDUCES Short USD exposure (e.g., Long USDJPY = +0.5% USD) is ALLOWED!
    t_hedge = PositionRiskSpec(symbol="USDJPY", side=Side.BUY, risk_pct=0.5, risk_usd=50.0)
    v_hedge = shield.evaluate_proposed_trade(t_hedge, open_positions=[t1, t2])
    assert v_hedge.allowed is True
    assert pytest.approx(v_hedge.factor_exposures_after_pct["USD"]) == -0.5
