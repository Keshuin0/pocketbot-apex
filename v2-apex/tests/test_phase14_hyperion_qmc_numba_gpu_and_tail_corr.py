"""Phase 14 (v2.7.0 — Apex Hyperion) Verification Tests:
1. Numerical parity across CUDA_TENSOR (if present), NUMBA_LLVM_AVX, and NUMPY_SIMD + FP64 5-digit FX precision
2. Owen-Scrambled Sobol + Antithetic Variates variance reduction vs naive RNG
3. Empirical crossover router & zero-crash OOM failover to NUMBA_LLVM_AVX
4. Ledoit-Wolf shrinkage & Lower-Tail Crash Co-Movement (rho_tail >= 0.72 blocks hidden panic correlation)
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from apex.compute.hyperion_fabric import ComputeBackend, HyperionComputeFabric, get_hyperion_fabric
from apex.feeds.models import Candle, Side
from apex.quant.correlation_shield import (
    CorrelationShield,
    PositionRiskSpec,
    compute_ledoit_wolf_and_tail_correlation,
)
from apex.quant.timemachine_v2 import BlockBootstrapTimeMachine, TimeMachineCostModel


def test_hyperion_numerical_parity_and_fp64_fx_precision(tmp_path) -> None:
    """CUDA_TENSOR (if available), NUMBA_LLVM_AVX, and NUMPY_SIMD must agree within tight tolerance on 5-digit FX."""
    fabric = HyperionComputeFabric(semantic_memory_path=tmp_path / "sem.json")
    rng = np.random.default_rng(123)
    b, n, h = 2, 2000, 24
    hi = 1.0 + np.abs(rng.normal(0.0004, 0.0002, size=(b, n, h)))
    lo = 1.0 - np.abs(rng.normal(0.0004, 0.0002, size=(b, n, h)))
    cl = 1.0 + rng.normal(0.0, 0.0002, size=(b, n, h))
    tu = rng.random((b, n, h))

    # 5-decimal FX prices (EURUSD ~1.08543, USDJPY ~154.321)
    entry = np.array([1.08543, 154.321], dtype=np.float64)
    stop = np.array([1.08393, 154.621], dtype=np.float64)
    tgt = np.array([1.08843, 153.721], dtype=np.float64)
    one_r = np.array([0.00150, 0.300], dtype=np.float64)
    tgt_r = np.array([2.0, 2.0], dtype=np.float64)
    cost_r = np.array([0.045, 0.045], dtype=np.float64)
    is_long = np.array([True, False], dtype=np.bool_)

    res_np = fabric.simulate_paths_batch(
        hi_ratios=hi, lo_ratios=lo, cl_ratios=cl, tie_uniforms=tu,
        entry_prices=entry, stop_prices=stop, target_prices=tgt,
        one_r_dists=one_r, target_rs=tgt_r, cost_rs=cost_r, is_longs=is_long,
        force_backend=ComputeBackend.NUMPY_SIMD,
    )
    if fabric.has_numba:
        res_nb = fabric.simulate_paths_batch(
            hi_ratios=hi, lo_ratios=lo, cl_ratios=cl, tie_uniforms=tu,
            entry_prices=entry, stop_prices=stop, target_prices=tgt,
            one_r_dists=one_r, target_rs=tgt_r, cost_rs=cost_r, is_longs=is_long,
            force_backend=ComputeBackend.NUMBA_LLVM_AVX,
        )
        assert np.allclose(res_np.net_avg_r, res_nb.net_avg_r, atol=1e-9)
        assert np.allclose(res_np.cvar_95_r, res_nb.cvar_95_r, atol=1e-9)

    if fabric.has_cuda:
        res_gpu = fabric.simulate_paths_batch(
            hi_ratios=hi, lo_ratios=lo, cl_ratios=cl, tie_uniforms=tu,
            entry_prices=entry, stop_prices=stop, target_prices=tgt,
            one_r_dists=one_r, target_rs=tgt_r, cost_rs=cost_r, is_longs=is_long,
            force_backend=ComputeBackend.CUDA_TENSOR,
        )
        assert np.allclose(res_np.net_avg_r, res_gpu.net_avg_r, atol=5e-3)
        assert np.allclose(res_np.prob_profit, res_gpu.prob_profit, atol=5e-3)


def test_qmc_sobol_antithetic_reduces_variance_vs_naive_rng() -> None:
    """Owen-Scrambled Sobol + Antithetic Variates must yield lower cross-seed variance than naive RNG."""
    rng_data = np.random.default_rng(7)
    candles: list[Candle] = []
    px = 1.1000
    for i in range(120):
        chg = float(rng_data.normal(0.00005, 0.0005))
        op = px
        cl = max(0.5, op + chg)
        hi = max(op, cl) + 0.0003
        lo = min(op, cl) - 0.0003
        candles.append(
            Candle(
                symbol="EURUSD",
                timeframe="M5",
                timestamp_utc=1_700_000_000.0 + i * 300,
                open=op,
                high=hi,
                low=lo,
                close=cl,
                volume=100.0,
                source="mt5",
            )
        )
        px = cl

    cm = TimeMachineCostModel(commission_per_lot_round_turn_usd=4.5, risk_per_lot_usd=100.0, spread_price=0.0001, vwap_slippage_price=0.00002)
    qmc_vals = []
    rng_vals = []
    for s in (11, 22, 33, 44, 55, 66):
        tm_qmc = BlockBootstrapTimeMachine(replays=1024, seed=s, use_qmc_sobol=True)
        tm_rng = BlockBootstrapTimeMachine(replays=1024, seed=s, use_qmc_sobol=False)
        qmc_vals.append(tm_qmc.simulate_candles("EURUSD", Side.BUY, px, px - 0.0015, px + 0.0030, candles, horizon_bars=24, cost_model=cm).net_avg_R)
        rng_vals.append(tm_rng.simulate_candles("EURUSD", Side.BUY, px, px - 0.0015, px + 0.0030, candles, horizon_bars=24, cost_model=cm).net_avg_R)

    assert float(np.std(qmc_vals)) <= float(np.std(rng_vals)) + 0.02


def test_crossover_router_and_oom_failover(tmp_path) -> None:
    """Micro-tasks route to CPU/Numba, and GPU OOM fails over cleanly in <1ms without crashing."""
    fabric = HyperionComputeFabric(semantic_memory_path=tmp_path / "sem.json")
    fabric.n_crossover_paths = 50_000
    assert fabric._select_backend(10_000) in (ComputeBackend.NUMBA_LLVM_AVX, ComputeBackend.NUMPY_SIMD)

    # Simulate GPU OOM failure -> must transparently fail over to Numba/NumPy
    fabric.has_cuda = True
    hi = np.full((1, 1000, 8), 1.001, dtype=np.float64)
    lo = np.full((1, 1000, 8), 0.999, dtype=np.float64)
    cl = np.full((1, 1000, 8), 1.000, dtype=np.float64)
    tu = np.full((1, 1000, 8), 0.5, dtype=np.float64)

    with patch.object(fabric, "_run_cuda_batch", side_effect=RuntimeError("CUDA out of memory")):
        res = fabric.simulate_paths_batch(
            hi_ratios=hi, lo_ratios=lo, cl_ratios=cl, tie_uniforms=tu,
            entry_prices=np.array([100.0]), stop_prices=np.array([98.0]), target_prices=np.array([104.0]),
            one_r_dists=np.array([2.0]), target_rs=np.array([2.0]), cost_rs=np.array([0.05]),
            is_longs=np.array([True]),
            force_backend=ComputeBackend.CUDA_TENSOR,
        )
    assert res.backend_used in (ComputeBackend.NUMBA_LLVM_AVX, ComputeBackend.NUMPY_SIMD)
    assert fabric.failover_count >= 1
    assert res.net_avg_r.shape == (1,)


def test_tail_crash_comovement_blocks_hidden_panic_correlation() -> None:
    """rho_tail >= 0.72 must block Long NAS100 + Long BTCUSD even when calm Pearson rho < 0.40."""
    rng = np.random.default_rng(99)
    n_calm = 180
    # Independent calm noise (rho ~ 0)
    r_nas = rng.normal(0.0005, 0.004, size=n_calm)
    r_btc = rng.normal(0.0005, 0.008, size=n_calm)
    # Inject 10 simultaneous -3 sigma panic crash bars
    crash_idx = np.array([15, 35, 55, 75, 95, 115, 135, 155, 165, 175])
    r_nas[crash_idx] = -0.025 + rng.normal(0.0, 0.0005, size=len(crash_idx))
    r_btc[crash_idx] = -0.050 + rng.normal(0.0, 0.0010, size=len(crash_idx))

    rho_lw, rho_tail = compute_ledoit_wolf_and_tail_correlation(np.column_stack([r_nas, r_btc]))
    assert float(rho_tail[0, 1]) >= 0.72

    shield = CorrelationShield(max_factor_risk_pct=1.0, max_empirical_corr=0.72)
    verdict = shield.evaluate_proposed_trade(
        proposed=PositionRiskSpec(symbol="BTCUSD", side=Side.BUY, risk_pct=0.40),
        open_positions=[PositionRiskSpec(symbol="NAS100", side=Side.BUY, risk_pct=0.40)],
        return_histories={"NAS100": r_nas, "BTCUSD": r_btc},
    )
    assert verdict.allowed is False
    assert "TAIL_CORR_NAS100" in verdict.breached_factors
