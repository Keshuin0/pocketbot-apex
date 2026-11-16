"""Phase 4 Verification Gate: Zero-Copy Hyperion Batching, DirectML Failover, 5x5 Grid & Neural Alpha."""

from __future__ import annotations

import datetime
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apex.ai.sleep_consolidation import SleepWakeConsolidator  # noqa: E402
from apex.compute import hyperion_fabric as hf_mod  # noqa: E402
from apex.compute.hyperion_fabric import (  # noqa: E402
    ComputeBackend,
    HyperionComputeFabric,
    build_historical_ratio_tensor,
)
from apex.ledger.walkforward_db import WalkForwardLedgerDB  # noqa: E402
from apex.quant.neural_alpha import (  # noqa: E402
    MultiResolutionWindow,
    NeuralAlphaForecaster,
    compute_realized_bipower_variation,
)


@dataclass
class _StubCandle:
    high: float
    low: float
    close: float


def _synthetic_candles(n: int = 85, drift: float = 0.0004, vol: float = 0.0012, seed: int = 11) -> list[_StubCandle]:
    rng = np.random.default_rng(seed)
    px = 1.1000
    bars: list[_StubCandle] = []
    for _ in range(n):
        ret = drift + vol * float(rng.normal())
        nxt = max(0.01, px * float(np.exp(ret)))
        wick = abs(vol * float(rng.normal())) * px + 0.0002 * px
        hi = max(px, nxt) + wick
        lo = max(0.005, min(px, nxt) - wick)
        bars.append(_StubCandle(high=hi, low=lo, close=nxt))
        px = nxt
    return bars


def test_simulate_paths_batch_b4_parity_and_no_silent_numpy_fallback(tmp_path: Path) -> None:
    """B=4 must match 4 sequential B=1 calls and DirectML unavailable never falls silently to NUMPY_SIMD."""
    fabric = HyperionComputeFabric(semantic_memory_path=tmp_path / "sem.json")
    assert fabric.has_cuda or fabric.has_numba, "Expected at least Numba or CUDA in test environment"

    rng = np.random.default_rng(2026)
    b_cnt, n_paths, h_bars = 4, 256, 16
    hi = np.clip(1.0 + np.abs(rng.normal(0.0012, 0.0008, size=(b_cnt, n_paths, h_bars))), 1.0001, 1.05)
    lo = np.clip(1.0 - np.abs(rng.normal(0.0012, 0.0008, size=(b_cnt, n_paths, h_bars))), 0.95, 0.9999)
    cl = np.clip(1.0 + rng.normal(0.0001, 0.0007, size=(b_cnt, n_paths, h_bars)), lo, hi)
    tu = rng.uniform(0.01, 0.99, size=(b_cnt, n_paths, h_bars))

    entry = np.array([1.1000, 1.2500, 1950.0, 62000.0], dtype=np.float64)
    one_r = np.array([0.0020, 0.0030, 8.0, 350.0], dtype=np.float64)
    is_long = np.array([True, False, True, False], dtype=np.bool_)
    stop = np.where(is_long, entry - one_r, entry + one_r)
    tgt_r = np.array([2.0, 2.5, 1.8, 3.0], dtype=np.float64)
    tgt = np.where(is_long, entry + tgt_r * one_r, entry - tgt_r * one_r)
    cost_r = np.array([0.05, 0.07, 0.04, 0.06], dtype=np.float64)

    target_backend = ComputeBackend.CUDA_TENSOR if fabric.has_cuda else ComputeBackend.NUMBA_LLVM_AVX
    batch_res = fabric.simulate_paths_batch(
        hi_ratios=hi,
        lo_ratios=lo,
        cl_ratios=cl,
        tie_uniforms=tu,
        entry_prices=entry,
        stop_prices=stop,
        target_prices=tgt,
        one_r_dists=one_r,
        target_rs=tgt_r,
        cost_rs=cost_r,
        is_longs=is_long,
        force_backend=target_backend,
    )
    assert batch_res.backend_used == target_backend
    assert batch_res.backend_used != ComputeBackend.NUMPY_SIMD

    # Sequential B=1 parity check
    for b in range(b_cnt):
        seq_res = fabric.simulate_paths_batch(
            hi_ratios=hi[b : b + 1],
            lo_ratios=lo[b : b + 1],
            cl_ratios=cl[b : b + 1],
            tie_uniforms=tu[b : b + 1],
            entry_prices=entry[b : b + 1],
            stop_prices=stop[b : b + 1],
            target_prices=tgt[b : b + 1],
            one_r_dists=one_r[b : b + 1],
            target_rs=tgt_r[b : b + 1],
            cost_rs=cost_r[b : b + 1],
            is_longs=is_long[b : b + 1],
            force_backend=target_backend,
        )
        assert np.isclose(batch_res.p_target[b], seq_res.p_target[0], atol=1e-7)
        assert np.isclose(batch_res.p_stop[b], seq_res.p_stop[0], atol=1e-7)
        assert np.isclose(batch_res.p_timeout[b], seq_res.p_timeout[0], atol=1e-7)
        assert np.isclose(batch_res.net_avg_r[b], seq_res.net_avg_r[0], atol=1e-6)
        assert np.isclose(batch_res.cvar_95_r[b], seq_res.cvar_95_r[0], atol=1e-6)
        assert np.allclose(batch_res.net_r_out[b], seq_res.net_r_out[0], atol=1e-6)

    # Also check return_full_paths=False produces identical stats with empty net_r_out
    if fabric.has_cuda:
        compact_res = fabric.simulate_paths_batch(
            hi_ratios=hi,
            lo_ratios=lo,
            cl_ratios=cl,
            tie_uniforms=tu,
            entry_prices=entry,
            stop_prices=stop,
            target_prices=tgt,
            one_r_dists=one_r,
            target_rs=tgt_r,
            cost_rs=cost_r,
            is_longs=is_long,
            force_backend=ComputeBackend.CUDA_TENSOR,
            return_full_paths=False,
        )
        assert compact_res.net_r_out.shape == (b_cnt, 0)
        assert np.allclose(compact_res.net_avg_r, batch_res.net_avg_r, atol=1e-9)
        assert np.allclose(compact_res.cvar_95_r, batch_res.cvar_95_r, atol=1e-9)

    # DirectML selected when DirectML is unavailable must route to CUDA or Numba, NEVER NUMPY_SIMD
    dml_redirect = fabric.simulate_paths_batch(
        hi_ratios=hi,
        lo_ratios=lo,
        cl_ratios=cl,
        tie_uniforms=tu,
        entry_prices=entry,
        stop_prices=stop,
        target_prices=tgt,
        one_r_dists=one_r,
        target_rs=tgt_r,
        cost_rs=cost_r,
        is_longs=is_long,
        force_backend=ComputeBackend.DIRECTML_GPU,
    )
    assert dml_redirect.backend_used in (
        ComputeBackend.CUDA_TENSOR,
        ComputeBackend.DIRECTML_GPU,
        ComputeBackend.NUMBA_LLVM_AVX,
    )
    assert dml_redirect.backend_used != ComputeBackend.NUMPY_SIMD


def test_directml_pytorch_dispatch_path_executes_when_available(monkeypatch, tmp_path: Path) -> None:
    """Verify DirectML branch executes PyTorch tensor ops on privateuseone:0 rather than dead fallthrough."""
    used_devices: list[str] = []

    class _FakeTensor:
        def __init__(self, arr: np.ndarray, device: str = "privateuseone:0") -> None:
            self.arr = np.asarray(arr)
            self.device = device

        def unsqueeze(self, dim: int) -> _FakeTensor:
            return _FakeTensor(np.expand_dims(self.arr, axis=dim), self.device)

        def any(self) -> _FakeTensor:
            return _FakeTensor(np.array(bool(np.any(self.arr))), self.device)

        def item(self):
            return self.arr.item()

        def cpu(self) -> _FakeTensor:
            return self

        def numpy(self) -> np.ndarray:
            return self.arr

        def __getitem__(self, item) -> _FakeTensor:
            return _FakeTensor(self.arr[item], self.device)

        def __eq__(self, other) -> _FakeTensor:
            o = other.arr if isinstance(other, _FakeTensor) else other
            return _FakeTensor(self.arr == o, self.device)

        def __lt__(self, other) -> _FakeTensor:
            o = other.arr if isinstance(other, _FakeTensor) else other
            return _FakeTensor(self.arr < o, self.device)

        def __le__(self, other) -> _FakeTensor:
            o = other.arr if isinstance(other, _FakeTensor) else other
            return _FakeTensor(self.arr <= o, self.device)

        def __gt__(self, other) -> _FakeTensor:
            o = other.arr if isinstance(other, _FakeTensor) else other
            return _FakeTensor(self.arr > o, self.device)

        def __ge__(self, other) -> _FakeTensor:
            o = other.arr if isinstance(other, _FakeTensor) else other
            return _FakeTensor(self.arr >= o, self.device)

        def __and__(self, other) -> _FakeTensor:
            o = other.arr if isinstance(other, _FakeTensor) else other
            return _FakeTensor(self.arr & o, self.device)

        def __or__(self, other) -> _FakeTensor:
            o = other.arr if isinstance(other, _FakeTensor) else other
            return _FakeTensor(self.arr | o, self.device)

        def __invert__(self) -> _FakeTensor:
            return _FakeTensor(~self.arr, self.device)

        def __mul__(self, other) -> _FakeTensor:
            o = other.arr if isinstance(other, _FakeTensor) else other
            return _FakeTensor(self.arr * o, self.device)

        def __add__(self, other) -> _FakeTensor:
            o = other.arr if isinstance(other, _FakeTensor) else other
            return _FakeTensor(self.arr + o, self.device)

        def __sub__(self, other) -> _FakeTensor:
            o = other.arr if isinstance(other, _FakeTensor) else other
            return _FakeTensor(self.arr - o, self.device)

        def __truediv__(self, other) -> _FakeTensor:
            o = other.arr if isinstance(other, _FakeTensor) else other
            return _FakeTensor(self.arr / o, self.device)

    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, *args):
            return False

    fake_torch = types.SimpleNamespace(
        float32=np.float32,
        int32=np.int32,
        device=lambda s: (used_devices.append(str(s)) or str(s)),
        no_grad=lambda: _NoGrad(),
        as_tensor=lambda a, dtype=None, device=None: _FakeTensor(np.asarray(a, dtype=dtype), str(device)),
        ones=lambda shape, dtype=None, device=None: _FakeTensor(np.ones(shape, dtype=dtype), str(device)),
        zeros=lambda shape, dtype=None, device=None: _FakeTensor(np.zeros(shape, dtype=dtype), str(device)),
        where=lambda c, x, y: _FakeTensor(
            np.where(
                c.arr if isinstance(c, _FakeTensor) else c,
                x.arr if isinstance(x, _FakeTensor) else x,
                y.arr if isinstance(y, _FakeTensor) else y,
            )
        ),
        abs=lambda x: _FakeTensor(np.abs(x.arr if isinstance(x, _FakeTensor) else x)),
        clamp=lambda x, min=None, max=None: _FakeTensor(np.clip(x.arr if isinstance(x, _FakeTensor) else x, min, max)),
    )

    monkeypatch.setattr(hf_mod, "torch", fake_torch)
    monkeypatch.setattr(hf_mod, "_HAS_DIRECTML", True)
    fabric = HyperionComputeFabric(semantic_memory_path=tmp_path / "sem_dml.json")
    fabric.has_directml = True
    fabric._directml_device = None

    rng = np.random.default_rng(7)
    hi = np.clip(1.0 + np.abs(rng.normal(0.001, 0.0005, size=(2, 64, 8))), 1.0001, 1.05)
    lo = np.clip(1.0 - np.abs(rng.normal(0.001, 0.0005, size=(2, 64, 8))), 0.95, 0.9999)
    cl = np.clip(1.0 + rng.normal(0.0, 0.0004, size=(2, 64, 8)), lo, hi)
    tu = rng.uniform(0.05, 0.95, size=(2, 64, 8))

    res_dml = fabric.simulate_paths_batch(
        hi_ratios=hi,
        lo_ratios=lo,
        cl_ratios=cl,
        tie_uniforms=tu,
        entry_prices=np.array([1.10, 1.20]),
        stop_prices=np.array([1.095, 1.205]),
        target_prices=np.array([1.11, 1.19]),
        one_r_dists=np.array([0.005, 0.005]),
        target_rs=np.array([2.0, 2.0]),
        cost_rs=np.array([0.05, 0.05]),
        is_longs=np.array([True, False]),
        force_backend=ComputeBackend.DIRECTML_GPU,
    )
    res_np = fabric.simulate_paths_batch(
        hi_ratios=hi,
        lo_ratios=lo,
        cl_ratios=cl,
        tie_uniforms=tu,
        entry_prices=np.array([1.10, 1.20]),
        stop_prices=np.array([1.095, 1.205]),
        target_prices=np.array([1.11, 1.19]),
        one_r_dists=np.array([0.005, 0.005]),
        target_rs=np.array([2.0, 2.0]),
        cost_rs=np.array([0.05, 0.05]),
        is_longs=np.array([True, False]),
        force_backend=ComputeBackend.NUMPY_SIMD,
    )
    assert res_dml.backend_used == ComputeBackend.DIRECTML_GPU
    assert "privateuseone:0" in used_devices
    assert np.allclose(res_dml.net_avg_r, res_np.net_avg_r, atol=1e-5)
    assert np.allclose(res_dml.p_target, res_np.p_target, atol=1e-9)


def test_zero_copy_5x5_grid_optimization_and_cross_backend_parity(tmp_path: Path) -> None:
    """5x5 (SL*, TP*) grid search batches B=12 Stage-2 symbols in one pass and stays within valid bounds."""
    fabric = HyperionComputeFabric(semantic_memory_path=tmp_path / "sem_grid.json")
    b_cnt = 12
    candles_list = [
        _synthetic_candles(n=81, drift=0.0005 if i % 2 == 0 else -0.0005, vol=0.0011, seed=100 + i)
        for i in range(b_cnt)
    ]
    hist_ratios, hist_lengths = build_historical_ratio_tensor(candles_list, max_bars=80)
    assert hist_ratios.shape == (b_cnt, 80, 4)
    assert hist_ratios.dtype == np.float32
    assert hist_ratios.nbytes < 32_768  # ~15 KB compact H2D footprint

    entry = np.array([c[-1].close for c in candles_list], dtype=np.float64)
    atr = entry * 0.0018
    spr = entry * 0.0001
    cost = entry * 0.0002
    is_long = np.array([i % 2 == 0 for i in range(b_cnt)], dtype=np.bool_)

    best_backend = ComputeBackend.CUDA_TENSOR if fabric.has_cuda else ComputeBackend.NUMBA_LLVM_AVX
    grid_res = fabric.optimize_sl_tp_grid_batch(
        hist_ratios=hist_ratios,
        hist_lengths=hist_lengths,
        entry_prices=entry,
        atr_values=atr,
        spread_values=spr,
        cost_prices=cost,
        is_longs=is_long,
        n_paths=1024,
        horizon_bars=24,
        block_size=4,
        force_backend=best_backend,
    )
    assert grid_res.backend_used == best_backend
    assert grid_res.scalars_6.shape == (b_cnt, 6)
    assert grid_res.sl_grid.shape == (5,)
    assert grid_res.tp_grid.shape == (5,)
    assert np.all((grid_res.optimal_sl_atr >= 1.0) & (grid_res.optimal_sl_atr <= 2.5))
    assert np.all((grid_res.optimal_tp_r >= 1.5) & (grid_res.optimal_tp_r <= 4.0))
    assert np.all((grid_res.prob_tp >= 0.0) & (grid_res.prob_tp <= 1.0))
    assert np.all((grid_res.prob_sl >= 0.0) & (grid_res.prob_sl <= 1.0))
    assert np.allclose(grid_res.scalars_6[:, 0], grid_res.optimal_sl_atr)
    assert np.allclose(grid_res.scalars_6[:, 1], grid_res.optimal_tp_r)
    assert np.allclose(grid_res.scalars_6[:, 4], grid_res.mean_r)
    assert np.allclose(grid_res.scalars_6[:, 5], grid_res.cvar_5)

    # Compare CUDA against Numba parity on identical inputs & Sobol tables
    if fabric.has_cuda and fabric.has_numba:
        grid_nb = fabric.optimize_sl_tp_grid_batch(
            hist_ratios=hist_ratios,
            hist_lengths=hist_lengths,
            entry_prices=entry,
            atr_values=atr,
            spread_values=spr,
            cost_prices=cost,
            is_longs=is_long,
            n_paths=1024,
            horizon_bars=24,
            block_size=4,
            force_backend=ComputeBackend.NUMBA_LLVM_AVX,
        )
        assert np.allclose(grid_res.mean_r, grid_nb.mean_r, atol=0.02)
        assert np.allclose(grid_res.cvar_5, grid_nb.cvar_5, atol=0.05)


def test_hyperion_compute_profile_persistence_survives_sleep_consolidation(tmp_path: Path) -> None:
    """hyperion_compute_profile persists across reloads and is preserved by sleep consolidation."""
    mem_path = tmp_path / "semantic_memory.json"
    db = WalkForwardLedgerDB(tmp_path / "ledger.db")

    fabric1 = HyperionComputeFabric(semantic_memory_path=mem_path)
    prof = fabric1.calibrate_crossover(persist=True)
    assert mem_path.exists()

    raw_before = json.loads(mem_path.read_text(encoding="utf-8"))
    assert "hyperion_compute_profile" in raw_before

    consolidator = SleepWakeConsolidator(ledger_db=db, memory_path=mem_path)
    consolidator.phase_b_consolidate_semantic_memory(
        trigger_reason="PHASE4_TEST",
        purged_cache_entries=0,
        canary_passed=True,
        now_utc=datetime.datetime(2026, 11, 15, 22, 10, tzinfo=datetime.timezone.utc).timestamp(),
    )

    raw_after = json.loads(mem_path.read_text(encoding="utf-8"))
    assert "hyperion_compute_profile" in raw_after
    assert raw_after["hyperion_compute_profile"]["n_crossover_paths"] == prof.n_crossover_paths

    fabric2 = HyperionComputeFabric(semantic_memory_path=mem_path)
    assert fabric2.crossover_profile is not None
    assert fabric2.n_crossover_paths == prof.n_crossover_paths


def test_neural_alpha_mamba_kan_quantiles_latency_and_fallback() -> None:
    """Mamba-3 / Temporal-KAN outputs ordered (Q10, Q50, Q90) in <4ms per batch and honors shocks/disabling."""
    t = np.arange(40, dtype=np.float64)
    bull_win = MultiResolutionWindow(
        symbol="EURUSD",
        m1_closes=1.1000 * np.exp(0.00025 * t),
        m5_closes=1.1000 * np.exp(0.00040 * t),
        m15_closes=1.1000 * np.exp(0.00050 * t),
        h1_closes=1.1000 * np.exp(0.00060 * t),
        ofi=0.72,
        contagion_shock_z=0.3,
    )
    bear_win = MultiResolutionWindow(
        symbol="GBPUSD",
        m1_closes=1.2500 * np.exp(-0.00025 * t),
        m5_closes=1.2500 * np.exp(-0.00040 * t),
        m15_closes=1.2500 * np.exp(-0.00050 * t),
        h1_closes=1.2500 * np.exp(-0.00060 * t),
        ofi=-0.72,
        contagion_shock_z=0.3,
    )
    shock_win = MultiResolutionWindow(
        symbol="XAUUSD",
        m1_closes=2000.0 * np.exp(0.00025 * t),
        m5_closes=2000.0 * np.exp(0.00040 * t),
        m15_closes=2000.0 * np.exp(0.00050 * t),
        h1_closes=2000.0 * np.exp(0.00060 * t),
        ofi=0.72,
        contagion_shock_z=5.5,
    )

    forecaster = NeuralAlphaForecaster(enabled=True, prefer_gpu=True)
    # Warm single pass then measure B=16 batch latency
    _ = forecaster.forecast_batch([bull_win, bear_win, shock_win])
    batch = [bull_win, bear_win, shock_win] * 5 + [bull_win]  # B=16
    res = forecaster.forecast_batch(batch)

    assert len(res.forecasts) == 16
    assert res.batch_latency_ms < 4.0
    assert np.all(res.q10 < res.q50)
    assert np.all(res.q50 < res.q90)
    assert res.forecasts[0].q50 > 0.0
    assert res.forecasts[1].q50 < 0.0
    assert res.forecasts[2].interval_width > res.forecasts[0].interval_width

    # Realized bipower variation detects jump spike separately from smooth diffusion
    smooth_r = np.full(30, 0.0005, dtype=np.float64)
    spiked_r = smooth_r.copy()
    spiked_r[15] = 0.0500
    bpv_smooth, jmp_smooth = compute_realized_bipower_variation(smooth_r)
    bpv_spike, jmp_spike = compute_realized_bipower_variation(spiked_r)
    assert jmp_smooth < 0.05
    assert jmp_spike > 0.50
    assert bpv_spike < 0.01  # BPV remains bounded despite single 100x jump

    # Split-conformal widening
    old_width = res.forecasts[0].interval_width
    forecaster.update_conformal_calibration(
        realized_returns=np.full(20, 0.05, dtype=np.float64),
        pred_q10=np.full(20, -0.002),
        pred_q90=np.full(20, 0.002),
        alpha=0.20,
    )
    res_cal = forecaster.forecast_symbol(bull_win)
    assert res_cal.interval_width > old_width

    # Clean disabled mode
    disabled = NeuralAlphaForecaster(enabled=False)
    out_dis = disabled.forecast_symbol(bull_win)
    assert out_dis.enabled is False
    assert out_dis.backend_used == "DISABLED"
    assert out_dis.q10 == 0.0 and out_dis.q50 == 0.0 and out_dis.q90 == 0.0
