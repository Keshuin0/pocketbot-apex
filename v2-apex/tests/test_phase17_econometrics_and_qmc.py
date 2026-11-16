"""Phase 3 (Omega v3.0 — Microstructure, Econometrics, GARCH/Sobol QMC & Convex Kelly) Tests:
1. Joint Loss Tail Dependence Coefficient converges to ~0.0 (< 0.15) on 10,000 independent standard normals.
2. Anis-Lloyd (1976) de-biased tail-sliced Hurst exponent centers near 0.50 (0.45 <= H <= 0.55) on white noise.
3. TimeMachineV2 uses half_n = 4096 with zero Sobol power-of-2 warnings, asymmetric barriers reduce p_target
   vs zero-spread barriers, and slippage_calibration_multiplier = 2.0 increases cost vs 1.0.
4. OrderBookEngine (Stoikov micro-price, rolling CKS OFI, VPIN toxicity), MomentumDOMGate (tier delegation &
   sweep/retest without 1.5x ATR trap), MTF forming-bar exclusion, Contagion MAD/UTC, and Convex Portfolio Kelly.
"""

from __future__ import annotations

import datetime
import warnings

import numpy as np
import pytest

from apex.feeds.ctrader_feed import OrderBookEngine
from apex.feeds.models import Candle, DOMAnalytics, Side
from apex.quant.contagion_oracle import ContagionOracle
from apex.quant.correlation_shield import (
    CorrelationShield,
    compute_ledoit_wolf_and_tail_correlation,
    compute_tail_crash_correlation,
)
from apex.quant.momentum_dom import (
    InstrumentTier,
    MomentumDOMGate,
    classify_instrument_tier,
)
from apex.quant.mtf_confluence import MTFConfluenceFilter
from apex.quant.regime import MarketRegime, RegimeClassifier, compute_hurst_exponent
from apex.quant.timemachine_v2 import BlockBootstrapTimeMachine, TimeMachineCostModel


def _make_candles(
    symbol: str,
    tf: str,
    closes: list[float],
    spread: float = 0.0002,
    start_ts: float = 1_763_000_000.0,
    step_s: float = 300.0,
) -> list[Candle]:
    out: list[Candle] = []
    prev = closes[0]
    for i, c in enumerate(closes):
        op = prev if i > 0 else c - 0.0001
        hi = max(op, c) + spread
        lo = min(op, c) - spread
        out.append(
            Candle(
                symbol=symbol,
                timeframe=tf,
                timestamp_utc=start_ts + i * step_s,
                open=op,
                high=hi,
                low=lo,
                close=c,
                volume=100.0,
                source="mt5",
            )
        )
        prev = c
    return out


def test_tail_crash_correlation_converges_to_zero_on_independent_normals() -> None:
    """On 10,000 independent standard-normal pairs, compute_tail_crash_correlation returns ~0.0 (< 0.15)."""
    rng = np.random.default_rng(2026)
    z1 = rng.standard_normal(10_000)
    z2 = rng.standard_normal(10_000)

    lam = compute_tail_crash_correlation(z1, z2, quantile=0.05)
    assert 0.0 <= lam < 0.15, f"Expected near-zero tail dependence on independent normals, got {lam:.4f}"

    # Verify directional standardization in compute_ledoit_wolf_and_tail_correlation as well
    rho_lw, rho_tail = compute_ledoit_wolf_and_tail_correlation(
        np.column_stack([z1, z2 * 10.0]),  # distinct scales must be standardized before shrinkage
        directions=np.array([1.0, -1.0]),
    )
    assert abs(float(rho_lw[0, 1])) < 0.05
    assert 0.0 <= float(rho_tail[0, 1]) < 0.15


def test_anis_lloyd_debiased_hurst_centers_near_half_on_gaussian_white_noise() -> None:
    """On pure Gaussian white noise, Anis-Lloyd debiased Hurst centers near 0.50 (0.45 <= H <= 0.55)."""
    hs: list[float] = []
    for seed in range(25):
        rng = np.random.default_rng(100 + seed)
        rets = rng.normal(0.0, 0.005, size=500)
        prices = 100.0 * np.exp(np.cumsum(rets))
        hs.append(compute_hurst_exponent(prices))

    mean_h = float(np.mean(hs))
    median_h = float(np.median(hs))
    assert 0.45 <= mean_h <= 0.55, f"Expected white-noise mean Hurst in [0.45, 0.55], got {mean_h:.4f}"
    assert 0.45 <= median_h <= 0.55, f"Expected white-noise median Hurst in [0.45, 0.55], got {median_h:.4f}"


def test_timemachine_v2_sobol_8192_no_warnings_asymmetric_barriers_and_slippage_multiplier() -> None:
    """Verify replays=8192 (half_n=4096, 0 Sobol warnings), asymmetric barrier tightening, and 2x slippage cost."""
    rng = np.random.default_rng(17)
    log_rets = rng.normal(0.00003, 0.00045, size=2200)
    closes = list(1.1000 * np.exp(np.cumsum(log_rets)))
    candles = _make_candles("EURUSD", "M5", closes, spread=0.00018)

    tm = BlockBootstrapTimeMachine(seed=19)
    assert tm.replays == 8_192  # half_n == 4096 == 2**12

    entry = closes[-1]
    stop = entry - 0.0010
    target = entry + 0.0020

    cm_zero = TimeMachineCostModel(
        commission_per_lot_round_turn_usd=0.0,
        risk_per_lot_usd=100.0,
        spread_price=0.0,
        vwap_slippage_price=0.0,
        asymmetric_barriers=False,
    )
    cm_asym = TimeMachineCostModel(
        commission_per_lot_round_turn_usd=4.5,
        risk_per_lot_usd=100.0,
        spread_price=0.00025,
        vwap_slippage_price=0.00005,
        slippage_calibration_multiplier=1.0,
        asymmetric_barriers=True,
    )
    cm_slip2x = TimeMachineCostModel(
        commission_per_lot_round_turn_usd=4.5,
        risk_per_lot_usd=100.0,
        spread_price=0.00025,
        vwap_slippage_price=0.00005,
        slippage_calibration_multiplier=2.0,
        asymmetric_barriers=True,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        res_zero = tm.simulate_candles("EURUSD", Side.BUY, entry, stop, target, candles, horizon_bars=36, cost_model=cm_zero)
        res_asym = tm.simulate_candles("EURUSD", Side.BUY, entry, stop, target, candles, horizon_bars=36, cost_model=cm_asym)
        res_2x = tm.simulate_candles("EURUSD", Side.BUY, entry, stop, target, candles, horizon_bars=36, cost_model=cm_slip2x)

    sobol_warns = [w for w in caught if "Sobol" in str(w.message) or "power of 2" in str(w.message)]
    assert len(sobol_warns) == 0, f"Unexpected Sobol power-of-2 warnings: {[str(w.message) for w in sobol_warns]}"
    assert res_asym.p_target <= res_zero.p_target
    assert res_asym.p_target < res_zero.p_target
    assert res_2x.round_trip_cost_r > res_asym.round_trip_cost_r


def test_microstructure_ofi_vpin_microprice_mtf_contagion_adx_and_convex_kelly() -> None:
    """Verify Phase 3 microstructure, tier mapping, forming-H1 exclusion, MAD contagion, and Convex Kelly."""
    # 1. OrderBookEngine: Stoikov micro-price, rolling CKS OFI, and VPIN toxicity
    book = OrderBookEngine("EURUSD", pip_size=0.0001, digits=5, top_n_levels=5)
    book.set_levels(
        bids=[(1.10000, 30.0), (1.09990, 20.0)],
        asks=[(1.10020, 10.0), (1.10030, 20.0)],
    )
    # Stoikov micro-price = (10*1.10000 + 30*1.10020) / 40 = 1.10015
    assert book.compute_micro_price() == pytest.approx(1.10015, abs=1e-6)

    # Repeated heavy bid queue additions vs ask cancellations push OFI positive and fill VPIN buckets
    for step in range(8):
        book.set_levels(
            bids=[(1.10000 + step * 0.00001, 120.0 + step * 25.0), (1.09990, 50.0)],
            asks=[(1.10020 + step * 0.00001, 5.0), (1.10030, 5.0)],
        )
    dom = book.compute_analytics()
    assert dom.ofi > 0.25
    assert dom.vpin > 0.70

    # 2. SmartAliasMapper tier delegation + uniform M5/M15 polarity & VPIN toxicity block
    assert classify_instrument_tier("EURUSDm") == InstrumentTier.TIGHT_SPREAD_M5
    assert classify_instrument_tier("Goldman Sachs Group") == InstrumentTier.WIDE_SPREAD_M15
    assert classify_instrument_tier("NQ") == InstrumentTier.TIGHT_SPREAD_M5

    m5_candles = _make_candles("EURUSD", "M5", [1.1000 + i * 0.00005 for i in range(20)], spread=0.0001)
    # Replace last bar with a healthy 0.6x ATR liquidity sweep + reclaim (eliminates 1.5x ATR chase trap)
    sweep_candles = list(m5_candles[:-1]) + [
        Candle(
            symbol="EURUSD",
            timeframe="M5",
            timestamp_utc=m5_candles[-1].timestamp_utc,
            open=1.10085,
            high=1.10105,
            low=1.09950,
            close=1.10095,
            volume=150.0,
            source="mt5",
        )
    ]
    gate = MomentumDOMGate(dom_obi_enabled=True, max_vpin=0.75)
    clean_dom = DOMAnalytics(
        symbol="EURUSD",
        timestamp_utc=sweep_candles[-1].timestamp_utc,
        top_n_levels=5,
        bid_volume_top_n=120.0,
        ask_volume_top_n=40.0,
        obi=0.50,
        best_bid=1.10094,
        best_ask=1.10096,
        spread_pips=0.2,
        micro_price=1.100955,
        ofi=0.35,
        vpin=0.40,
    )
    v_sweep = gate.evaluate("EURUSD", Side.BUY, 1.10096, 1.10000, sweep_candles, 0.00002, dom_analytics=clean_dom)
    assert v_sweep.passed is True
    assert v_sweep.setup_type == "LIQUIDITY_SWEEP_RECLAIM"

    # Extreme VPIN toxicity (> 0.75) must block even with bullish bar & positive OBI
    toxic_dom = DOMAnalytics(
        symbol="EURUSD",
        timestamp_utc=sweep_candles[-1].timestamp_utc,
        top_n_levels=5,
        bid_volume_top_n=120.0,
        ask_volume_top_n=40.0,
        obi=0.50,
        best_bid=1.10094,
        best_ask=1.10096,
        spread_pips=0.2,
        micro_price=1.100955,
        ofi=0.35,
        vpin=0.92,
    )
    v_toxic = gate.evaluate("EURUSD", Side.BUY, 1.10096, 1.10000, sweep_candles, 0.00002, dom_analytics=toxic_dom)
    assert v_toxic.passed is False
    assert any("vpin_toxicity_extreme" in r for r in v_toxic.reasons)

    # 3. MTF forming H1 bar exclusion: spike on still-forming H1[-1] must NOT self-block completed breakout
    now_ts = 1_763_100_000.0
    m5_trig = _make_candles("EURUSD", "M5", [1.0950 + i * 0.0002 for i in range(30)], start_ts=now_ts - 29 * 300, step_s=300)
    h1_hist = _make_candles("EURUSD", "H1", [1.0800 + i * 0.0003 for i in range(64)], start_ts=now_ts - 64 * 3600, step_s=3600)
    # Overwrite H1[-1] as the open forming bar covering now_ts with high=1.1010 close to entry=1.1008
    forming_h1 = Candle(
        symbol="EURUSD",
        timeframe="H1",
        timestamp_utc=now_ts - 600.0,
        open=1.0985,
        high=1.1010,
        low=1.0980,
        close=1.1008,
        volume=500.0,
        source="mt5",
    )
    mtf = MTFConfluenceFilter()
    v_mtf = mtf.evaluate("EURUSD", Side.BUY, entry_price=1.1008, atr_m5=0.0004, m5_candles=m5_trig, h1_candles=h1_hist + [forming_h1])
    assert v_mtf.h1_trend_aligned is True
    assert v_mtf.h1_swing_high < 1.1010

    # 4. RegimeClassifier Wilder ADX(14) & Hawkes + Contagion naive UTC & MAD-only outlier robustness
    rc = RegimeClassifier(adx_period=14)
    diag = rc.classify_candles("EURUSD", m5_trig)
    assert diag.adx > 20.0
    assert diag.hawkes_intensity_ratio >= 0.5
    assert diag.regime == MarketRegime.TRENDING_EXPANSION

    naive_rollover = datetime.datetime(2026, 11, 12, 22, 15)  # naive interpreted strictly as UTC
    is_roll, _ = ContagionOracle.is_rollover_illiquidity_window(naive_rollover)
    assert is_roll is True
    # MAD denominator stays tight even with one prior outlier in history
    series_with_old_outlier = [0.0002, -0.0002] * 10 + [0.0250] + [0.0001, -0.0001] * 4 + [0.0040]
    assert ContagionOracle._series_latest_z(series_with_old_outlier) > 5.0

    # 5. Convex Portfolio Kelly Optimization respects 1.0% USD cap & allocates non-zero weights cleanly
    shield = CorrelationShield(max_factor_risk_pct=1.0)
    opt = shield.optimize_portfolio_weights(
        candidates=[
            ("EURUSD", Side.BUY, 0.45, 0.50),
            ("GBPUSD", Side.BUY, 0.42, 0.50),
            ("AUDUSD", Side.BUY, 0.38, 0.50),
        ],
        max_concurrent_positions=5,
    )
    # All three are Short USD -> sum of allocations must keep |USD| <= 1.0%
    usd_risk = sum(opt[s] for s in ("EURUSD", "GBPUSD", "AUDUSD"))
    assert usd_risk <= 1.0 + 1e-4
    assert usd_risk > 0.0

    # 6. Verify RegimeClassifier VOLATILITY_SHOCK is reachable via historical_spreads
    shock_diag = rc.classify_candles(
        "EURUSD",
        m5_trig,
        current_spread=0.00040,
        historical_spreads=[0.00010] * 10,
    )
    assert shock_diag.regime == MarketRegime.VOLATILITY_SHOCK
    assert shock_diag.allow_breakout_entries is False
