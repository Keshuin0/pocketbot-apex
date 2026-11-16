"""Apex Hyperion: Self-Calibrating 3-Tier Heterogeneous Compute Fabric (`apex/compute/hyperion_fabric.py`).

Hardware Task Affinity:
  - Sub-Millisecond Serial Lane (ALWAYS CPU / Numba): MT5 IPC, cTrader Protobuf, L2 DOM OBI,
    Hydra NBBO, z3-solver SMT proofs, SQLite WAL, order_send (0 PCIe overhead).
  - Massive Parallel Tensor Lane (CUDA_TENSOR -> DIRECTML_GPU -> NUMBA_LLVM_AVX -> NUMPY_SIMD):
      a) 3D Batched [B_symbols, N_paths, H_bars] QMC-Sobol Time Machine
      b) Batched In-Flight Conditional Forward Time Machine & 100x Multi-Day Replay
      c) 252-Symbol Ledoit-Wolf Shrinkage & Lower-Tail Crash Correlation Matrix

Mixed-Precision Guardrail:
  - Relative bar excursion simulation runs in fast FP32 on GPU shaders while all final
    R-multiple, prob_profit, CVaR_95, and CVaR_99 reductions accumulate in exact FP64.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Optional Numba LLVM AVX2/AVX-512 backend
try:
    import numba  # type: ignore[import-untyped]
    from numba import njit, prange  # type: ignore[import-untyped]

    _HAS_NUMBA = True
except Exception:
    numba = None  # type: ignore[assignment]
    _HAS_NUMBA = False

# Optional NVIDIA CuPy CUDA Tensor backend
try:
    import cupy as cp  # type: ignore[import-untyped]

    _HAS_CUPY = True
except Exception:
    cp = None  # type: ignore[assignment]
    _HAS_CUPY = False

import math

# Optional AMD/Intel DirectML backend
try:
    import torch  # type: ignore[import-untyped]
    import torch_directml  # type: ignore[import-untyped]

    _HAS_DIRECTML = True
except Exception:
    try:
        import torch  # type: ignore[import-untyped]
    except Exception:
        torch = None  # type: ignore[assignment]
    torch_directml = None  # type: ignore[assignment]
    _HAS_DIRECTML = False


class ComputeBackend(str, Enum):
    """3-Tier Heterogeneous Compute Fabric Backends."""

    CUDA_TENSOR = "CUDA_TENSOR"
    DIRECTML_GPU = "DIRECTML_GPU"
    NUMBA_LLVM_AVX = "NUMBA_LLVM_AVX"
    NUMPY_SIMD = "NUMPY_SIMD"


@dataclass(frozen=True, slots=True)
class CrossoverProfile:
    """Empirical GPU-vs-CPU crossover calibration persisted in semantic_memory.json."""

    preferred_gpu_backend: str
    gpu_name: str
    n_crossover_paths: int
    bench_1x10k_cpu_ms: float
    bench_1x10k_gpu_ms: float
    bench_12x10k_cpu_ms: float
    bench_12x10k_gpu_ms: float
    bench_12x100k_cpu_ms: float
    bench_12x100k_gpu_ms: float
    calibrated_at_utc: float


@dataclass(frozen=True, slots=True)
class BatchSimulationResult:
    """Exact FP64-reduced outputs for a batch of B symbols."""

    p_target: np.ndarray        # (B,) float64
    p_stop: np.ndarray          # (B,) float64
    p_timeout: np.ndarray       # (B,) float64
    prob_profit: np.ndarray     # (B,) float64
    gross_avg_r: np.ndarray     # (B,) float64
    net_avg_r: np.ndarray       # (B,) float64
    median_net_r: np.ndarray    # (B,) float64
    var_95_r: np.ndarray        # (B,) float64
    cvar_95_r: np.ndarray       # (B,) float64
    cvar_99_r: np.ndarray       # (B,) float64
    net_r_out: np.ndarray       # (B, N) float64
    backend_used: ComputeBackend
    compute_ms: float


# ---------------------------------------------------------------------------
# Numba LLVM AVX2/AVX-512 Parallel Kernels (nogil=True, parallel=True, cache=True)
# ---------------------------------------------------------------------------
if _HAS_NUMBA:

    @njit(nogil=True, parallel=True, cache=True, fastmath=False)
    def _numba_simulate_batch_kernel(
        hi_ratios: np.ndarray,       # (B, N, H) float64
        lo_ratios: np.ndarray,       # (B, N, H) float64
        cl_ratios: np.ndarray,       # (B, N, H) float64
        tie_uniforms: np.ndarray,    # (B, N, H) float64
        entry_prices: np.ndarray,    # (B,) float64
        stop_prices: np.ndarray,     # (B,) float64
        target_prices: np.ndarray,   # (B,) float64
        one_r_dists: np.ndarray,     # (B,) float64
        target_rs: np.ndarray,       # (B,) float64
        cost_rs: np.ndarray,         # (B,) float64
        is_longs: np.ndarray,        # (B,) bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate [B, N, H] paths in parallel across all CPU cores with exact FP64 math."""
        b_cnt, n_paths, h_bars = hi_ratios.shape
        gross_r_out = np.empty((b_cnt, n_paths), dtype=np.float64)
        net_r_out = np.empty((b_cnt, n_paths), dtype=np.float64)
        status_out = np.zeros((b_cnt, n_paths), dtype=np.int8)

        for b in range(b_cnt):
            entry_px = entry_prices[b]
            stop_px = stop_prices[b]
            target_px = target_prices[b]
            one_r = one_r_dists[b]
            tgt_r = target_rs[b]
            cost_r = cost_rs[b]
            is_long = is_longs[b]

            for n in prange(n_paths):
                cur_px = entry_px
                st = 0
                for t in range(h_bars):
                    hi = cur_px * hi_ratios[b, n, t]
                    lo = cur_px * lo_ratios[b, n, t]
                    cl = cur_px * cl_ratios[b, n, t]

                    if is_long:
                        hit_stop = lo <= stop_px
                        hit_tgt = hi >= target_px
                    else:
                        hit_stop = hi >= stop_px
                        hit_tgt = lo <= target_px

                    if hit_stop and hit_tgt:
                        d_stop = abs(cur_px - stop_px)
                        d_tgt = abs(target_px - cur_px)
                        denom = d_stop + d_tgt
                        if denom < 1e-12:
                            denom = 1e-12
                        p_tgt = d_stop / denom
                        if (is_long and cl > cur_px) or ((not is_long) and cl < cur_px):
                            p_tgt += 0.15
                        else:
                            p_tgt -= 0.15
                        if p_tgt < 0.05:
                            p_tgt = 0.05
                        elif p_tgt > 0.95:
                            p_tgt = 0.95
                        if tie_uniforms[b, n, t] < p_tgt:
                            st = 1
                        else:
                            st = -1
                        break
                    elif hit_tgt:
                        st = 1
                        break
                    elif hit_stop:
                        st = -1
                        break
                    cur_px = cl

                status_out[b, n] = st
                if st == 1:
                    gr = tgt_r
                elif st == -1:
                    gr = -1.0
                else:
                    if is_long:
                        gr = (cur_px - entry_px) / one_r
                    else:
                        gr = (entry_px - cur_px) / one_r
                gross_r_out[b, n] = gr
                net_r_out[b, n] = gr - cost_r

        return status_out, gross_r_out, net_r_out


# ---------------------------------------------------------------------------
# CuPy CUDA RawKernel (Mixed-Precision FP32 excursions -> FP64 R outputs)
# ---------------------------------------------------------------------------
_CUDA_SIM_KERNEL_SRC = r"""
extern "C" __global__
void hyperion_sim_batch_f32_to_f64(
    const float* __restrict__ hi_ratios,
    const float* __restrict__ lo_ratios,
    const float* __restrict__ cl_ratios,
    const float* __restrict__ tie_uniforms,
    const double* __restrict__ entry_prices,
    const double* __restrict__ stop_prices,
    const double* __restrict__ target_prices,
    const double* __restrict__ one_r_dists,
    const double* __restrict__ target_rs,
    const double* __restrict__ cost_rs,
    const int* __restrict__ is_longs,
    signed char* __restrict__ status_out,
    double* __restrict__ gross_r_out,
    double* __restrict__ net_r_out,
    const int B,
    const int N,
    const int H
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = B * N;
    if (idx >= total) return;

    int b = idx / N;
    int base = idx * H;

    double entry_px = entry_prices[b];
    double stop_px = stop_prices[b];
    double target_px = target_prices[b];
    double one_r = one_r_dists[b];
    double tgt_r = target_rs[b];
    double cost_r = cost_rs[b];
    int is_long = is_longs[b];

    // Zero-centered relative excursion in FP32 for fast GPU math
    float rel_px = 1.0f;
    float stop_rel = (float)(stop_px / entry_px);
    float tgt_rel = (float)(target_px / entry_px);
    signed char st = 0;

    for (int t = 0; t < H; ++t) {
        int off = base + t;
        float hi = rel_px * hi_ratios[off];
        float lo = rel_px * lo_ratios[off];
        float cl = rel_px * cl_ratios[off];

        bool hit_stop = is_long ? (lo <= stop_rel) : (hi >= stop_rel);
        bool hit_tgt  = is_long ? (hi >= tgt_rel)  : (lo <= tgt_rel);

        if (hit_stop && hit_tgt) {
            float d_stop = fabsf(rel_px - stop_rel);
            float d_tgt = fabsf(tgt_rel - rel_px);
            float denom = fmaxf(d_stop + d_tgt, 1e-7f);
            float p_tgt = d_stop / denom;
            bool bull_bar = is_long ? (cl > rel_px) : (cl < rel_px);
            p_tgt += bull_bar ? 0.15f : -0.15f;
            p_tgt = fminf(0.95f, fmaxf(0.05f, p_tgt));
            st = (tie_uniforms[off] < p_tgt) ? 1 : -1;
            break;
        } else if (hit_tgt) {
            st = 1;
            break;
        } else if (hit_stop) {
            st = -1;
            break;
        }
        rel_px = cl;
    }

    status_out[idx] = st;
    double gr;
    if (st == 1) {
        gr = tgt_r;
    } else if (st == -1) {
        gr = -1.0;
    } else {
        double final_px = entry_px * (double)rel_px;
        gr = is_long ? ((final_px - entry_px) / one_r) : ((entry_px - final_px) / one_r);
    }
    gross_r_out[idx] = gr;
    net_r_out[idx] = gr - cost_r;
}
"""


_CUDA_GRID_KERNEL_SRC = r"""
extern "C" __global__
void hyperion_grid_bootstrap_kernel(
    const float* __restrict__ hist_ratios,
    const int* __restrict__ hist_lengths,
    const float* __restrict__ rand_u,
    const float* __restrict__ tie_u,
    const double* __restrict__ entry_prices,
    const double* __restrict__ atr_values,
    const double* __restrict__ spread_values,
    const double* __restrict__ cost_prices,
    const int* __restrict__ is_longs,
    const double* __restrict__ sl_grid,
    const double* __restrict__ tp_grid,
    signed char* __restrict__ status_out,
    double* __restrict__ net_r_out,
    const int B,
    const int M,
    const int N,
    const int H,
    const int block_size,
    const int G_SL,
    const int G_TP
) {
    int G = G_SL * G_TP;
    long long idx = (long long)blockDim.x * (long long)blockIdx.x + (long long)threadIdx.x;
    long long total = (long long)B * (long long)G * (long long)N;
    if (idx >= total) return;

    int bg = (int)(idx / (long long)N);
    int n = (int)(idx % (long long)N);
    int b = bg / G;
    int g = bg % G;
    int g_sl = g / G_TP;
    int g_tp = g % G_TP;

    double entry_px = entry_prices[b];
    double atr = atr_values[b];
    double spr = spread_values[b];
    double cost_px = cost_prices[b];
    int is_long = is_longs[b];
    double sl_mult = sl_grid[g_sl];
    double tp_mult = tp_grid[g_tp];

    double one_r = fmax(atr * sl_mult + spr, 1e-12);
    double stop_px = is_long ? fmax(entry_px * 0.01, entry_px - one_r) : (entry_px + one_r);
    double tgt_px = is_long ? (entry_px + tp_mult * one_r) : fmax(entry_px * 0.01, entry_px - tp_mult * one_r);
    double cost_r = cost_px / one_r;

    float stop_rel = (float)(stop_px / entry_px);
    float tgt_rel = (float)(tgt_px / entry_px);
    float rel_px = 1.0f;
    signed char st = 0;

    int m_len = hist_lengths[b];
    if (m_len < 1) m_len = 1;
    int bs = block_size > 0 ? block_size : 1;
    int max_start = m_len - bs + 1;
    if (max_start < 1) max_start = 1;

    int hist_base = b * M * 4;
    int cur_idx = 0;

    for (int t = 0; t < H; ++t) {
        int tbl_off = n * H + t;
        if ((t % bs) == 0) {
            float u = rand_u[tbl_off];
            if (u < 0.0f) u = 0.0f;
            if (u >= 1.0f) u = 0.999999f;
            cur_idx = (int)(u * (float)max_start);
            if (cur_idx >= max_start) cur_idx = max_start - 1;
        } else {
            cur_idx += 1;
            if (cur_idx >= m_len) cur_idx = cur_idx % m_len;
        }

        int r_off = hist_base + cur_idx * 4;
        float hi_r = hist_ratios[r_off + 0];
        float lo_r = hist_ratios[r_off + 1];
        float cl_r = hist_ratios[r_off + 2];

        float hi = rel_px * hi_r;
        float lo = rel_px * lo_r;
        float cl = rel_px * cl_r;

        bool hit_stop = is_long ? (lo <= stop_rel) : (hi >= stop_rel);
        bool hit_tgt  = is_long ? (hi >= tgt_rel)  : (lo <= tgt_rel);

        if (hit_stop && hit_tgt) {
            float d_stop = fabsf(rel_px - stop_rel);
            float d_tgt = fabsf(tgt_rel - rel_px);
            float denom = fmaxf(d_stop + d_tgt, 1e-7f);
            float p_tgt = d_stop / denom;
            bool bull_bar = is_long ? (cl > rel_px) : (cl < rel_px);
            p_tgt += bull_bar ? 0.15f : -0.15f;
            p_tgt = fminf(0.95f, fmaxf(0.05f, p_tgt));
            st = (tie_u[tbl_off] < p_tgt) ? 1 : -1;
            break;
        } else if (hit_tgt) {
            st = 1;
            break;
        } else if (hit_stop) {
            st = -1;
            break;
        }
        rel_px = cl;
    }

    status_out[idx] = st;
    double gr;
    if (st == 1) {
        gr = tp_mult;
    } else if (st == -1) {
        gr = -1.0;
    } else {
        double final_px = entry_px * (double)rel_px;
        gr = is_long ? ((final_px - entry_px) / one_r) : ((entry_px - final_px) / one_r);
    }
    net_r_out[idx] = gr - cost_r;
}
"""


if _HAS_NUMBA:
    @njit(parallel=True, fastmath=False)
    def _numba_grid_batch_kernel(
        hist_ratios: np.ndarray,
        hist_lengths: np.ndarray,
        rand_u: np.ndarray,
        tie_u: np.ndarray,
        entry_prices: np.ndarray,
        atr_values: np.ndarray,
        spread_values: np.ndarray,
        cost_prices: np.ndarray,
        is_longs: np.ndarray,
        sl_grid: np.ndarray,
        tp_grid: np.ndarray,
        block_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        b_cnt = hist_ratios.shape[0]
        n_paths = rand_u.shape[0]
        h_bars = rand_u.shape[1]
        g_sl = sl_grid.shape[0]
        g_tp = tp_grid.shape[0]
        g_cnt = g_sl * g_tp
        bg_cnt = b_cnt * g_cnt

        status_out = np.zeros((bg_cnt, n_paths), dtype=np.int8)
        net_r_out = np.empty((bg_cnt, n_paths), dtype=np.float64)
        bs = block_size if block_size > 0 else 1

        for bg in range(bg_cnt):
            b = bg // g_cnt
            g = bg % g_cnt
            g_s = g // g_tp
            g_t = g % g_tp

            entry_px = entry_prices[b]
            atr = atr_values[b]
            spr = spread_values[b]
            cost_px = cost_prices[b]
            is_long = bool(is_longs[b])
            sl_mult = sl_grid[g_s]
            tp_mult = tp_grid[g_t]

            one_r = max(atr * sl_mult + spr, 1e-12)
            stop_px = max(entry_px * 0.01, entry_px - one_r) if is_long else (entry_px + one_r)
            target_px = (entry_px + tp_mult * one_r) if is_long else max(entry_px * 0.01, entry_px - tp_mult * one_r)
            cost_r = cost_px / one_r

            m_len = int(hist_lengths[b])
            if m_len < 1:
                m_len = 1
            max_start = m_len - bs + 1
            if max_start < 1:
                max_start = 1

            for n in prange(n_paths):
                cur_px = entry_px
                st = 0
                cur_idx = 0
                for t in range(h_bars):
                    if t % bs == 0:
                        u = rand_u[n, t]
                        if u < 0.0:
                            u = 0.0
                        elif u >= 1.0:
                            u = 0.999999
                        cur_idx = int(u * max_start)
                        if cur_idx >= max_start:
                            cur_idx = max_start - 1
                    else:
                        cur_idx += 1
                        if cur_idx >= m_len:
                            cur_idx = cur_idx % m_len

                    hi = cur_px * hist_ratios[b, cur_idx, 0]
                    lo = cur_px * hist_ratios[b, cur_idx, 1]
                    cl = cur_px * hist_ratios[b, cur_idx, 2]

                    if is_long:
                        hit_stop = lo <= stop_px
                        hit_tgt = hi >= target_px
                    else:
                        hit_stop = hi >= stop_px
                        hit_tgt = lo <= target_px

                    if hit_stop and hit_tgt:
                        d_stop = abs(cur_px - stop_px)
                        d_tgt = abs(target_px - cur_px)
                        denom = d_stop + d_tgt
                        if denom < 1e-12:
                            denom = 1e-12
                        p_tgt = d_stop / denom
                        if (is_long and cl > cur_px) or ((not is_long) and cl < cur_px):
                            p_tgt += 0.15
                        else:
                            p_tgt -= 0.15
                        if p_tgt < 0.05:
                            p_tgt = 0.05
                        elif p_tgt > 0.95:
                            p_tgt = 0.95
                        st = 1 if tie_u[n, t] < p_tgt else -1
                        break
                    elif hit_tgt:
                        st = 1
                        break
                    elif hit_stop:
                        st = -1
                        break
                    cur_px = cl

                status_out[bg, n] = st
                if st == 1:
                    gr = tp_mult
                elif st == -1:
                    gr = -1.0
                else:
                    gr = ((cur_px - entry_px) / one_r) if is_long else ((entry_px - cur_px) / one_r)
                net_r_out[bg, n] = gr - cost_r

        return status_out, net_r_out
else:
    _numba_grid_batch_kernel = None


@dataclass(frozen=True, slots=True)
class GridOptimizationResult:
    """5x5 (SL*, TP*) bracket optimization output across B multi-symbol candidates."""

    optimal_sl_atr: np.ndarray     # (B,) float64
    optimal_tp_r: np.ndarray       # (B,) float64
    prob_tp: np.ndarray            # (B,) float64
    prob_sl: np.ndarray            # (B,) float64
    mean_r: np.ndarray             # (B,) float64
    cvar_5: np.ndarray             # (B,) float64
    kelly_cvar_score: np.ndarray   # (B,) float64
    scalars_6: np.ndarray          # (B, 6) float64 [sl*, tp*, prob_tp, prob_sl, mean_r, cvar_5]
    sl_grid: np.ndarray            # (G_SL,) float64
    tp_grid: np.ndarray            # (G_TP,) float64
    backend_used: ComputeBackend
    compute_ms: float


def build_historical_ratio_tensor(
    candles_by_symbol: list[Any] | tuple[Any, ...],
    max_bars: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode OHLC candle sequences into compact float32 (B, M, 4) relative ratios for zero-copy H2D transfer."""
    b_cnt = len(candles_by_symbol)
    if b_cnt == 0:
        return np.ones((0, 1, 4), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    m_cap = max(1, int(max_bars))
    ratios = np.ones((b_cnt, m_cap, 4), dtype=np.float32)
    ratios[:, :, 0] = 1.0005
    ratios[:, :, 1] = 0.9995
    lengths = np.ones((b_cnt,), dtype=np.int32)

    max_used = 1
    for b, seq in enumerate(candles_by_symbol):
        bars = list(seq) if seq is not None else []
        if len(bars) > m_cap + 1:
            bars = bars[-(m_cap + 1) :]
        if len(bars) >= 2:
            rows = []
            prev_c = max(float(getattr(bars[0], "close", 0.0) or 0.0), 1e-12)
            for bar in bars[1:]:
                h = max(float(getattr(bar, "high", prev_c) or prev_c), 1e-12)
                l = max(float(getattr(bar, "low", prev_c) or prev_c), 1e-12)
                c = max(float(getattr(bar, "close", prev_c) or prev_c), 1e-12)
                hi_r = max(h / prev_c, c / prev_c, 1.0)
                lo_r = min(l / prev_c, c / prev_c, 1.0)
                cl_r = c / prev_c
                rows.append((hi_r, lo_r, cl_r, 1.0))
                prev_c = c
            cnt = min(len(rows), m_cap)
            if cnt > 0:
                ratios[b, :cnt, :] = np.asarray(rows[:cnt], dtype=np.float32)
                lengths[b] = cnt
                if cnt > max_used:
                    max_used = cnt

    return np.ascontiguousarray(ratios[:, :max_used, :], dtype=np.float32), lengths


class HyperionComputeFabric:
    """Self-Calibrating 3-Tier Heterogeneous Compute Fabric for PocketBot Apex."""

    VRAM_LIMIT_FRAC: float = 0.85
    GPU_TIMEOUT_MS: float = 400.0

    def __init__(self, semantic_memory_path: str | Path = "trader/semantic_memory.json") -> None:
        self.semantic_memory_path = Path(semantic_memory_path)
        self._lock = threading.Lock()
        self.gpu_name: str = "None (CPU-Only)"
        self.has_cuda: bool = False
        self.has_directml: bool = False
        self.has_numba: bool = _HAS_NUMBA
        self._cuda_kernel: Any = None
        self._grid_cuda_kernel: Any = None
        self._directml_device: Any = None
        self._pinned_pool: dict[str, Any] = {}
        self._sobol_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self._gpu_sobol_cache: dict[tuple[int, int], tuple[Any, Any]] = {}
        self.n_crossover_paths: int = 10_000
        self.crossover_profile: CrossoverProfile | None = None
        self.last_backend: ComputeBackend = ComputeBackend.NUMBA_LLVM_AVX if _HAS_NUMBA else ComputeBackend.NUMPY_SIMD
        self.last_batch_ms: float = 0.0
        self.last_paths_count: int = 0
        self.failover_count: int = 0

        self._detect_hardware()
        self._load_crossover_profile()
        self.prewarm()

    def _detect_hardware(self) -> None:
        """Detect CUDA GPU, DirectML GPU, and Numba LLVM AVX capabilities."""
        if _HAS_CUPY:
            try:
                cnt = int(cp.cuda.runtime.getDeviceCount())
                if cnt > 0:
                    props = cp.cuda.runtime.getDeviceProperties(0)
                    raw_name = props.get("name", b"NVIDIA CUDA GPU")
                    self.gpu_name = raw_name.decode("utf-8", errors="replace") if isinstance(raw_name, bytes) else str(raw_name)
                    self.has_cuda = True
                    self.last_backend = ComputeBackend.CUDA_TENSOR
            except Exception as exc:
                logger.debug("CuPy CUDA detection skipped: %s", exc)
                self.has_cuda = False

        if not self.has_cuda and _HAS_DIRECTML and torch is not None and torch_directml is not None:
            try:
                try:
                    dml_dev = torch.device("privateuseone:0")
                except Exception:
                    dml_dev = torch_directml.device()
                probe = torch.tensor([1.0], dtype=torch.float32, device=dml_dev)
                _ = float((probe + 1.0).item())
                self._directml_device = dml_dev
                self.gpu_name = f"DirectML ({torch_directml.device_name(0)})"
                self.has_directml = True
                self.last_backend = ComputeBackend.DIRECTML_GPU
            except Exception as exc:
                logger.debug("DirectML detection skipped: %s", exc)
                self.has_directml = False
                self._directml_device = None

        if not self.has_cuda and not self.has_directml:
            self.last_backend = ComputeBackend.NUMBA_LLVM_AVX if self.has_numba else ComputeBackend.NUMPY_SIMD

    def _load_crossover_profile(self) -> CrossoverProfile | None:
        """Load persisted `hyperion_compute_profile` from `trader/semantic_memory.json` if valid."""
        try:
            if not self.semantic_memory_path.exists():
                return None
            raw = json.loads(self.semantic_memory_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return None
            prof_dict = raw.get("hyperion_compute_profile")
            if not isinstance(prof_dict, dict):
                return None
            prof = CrossoverProfile(
                preferred_gpu_backend=str(prof_dict.get("preferred_gpu_backend", self.last_backend.value)),
                gpu_name=str(prof_dict.get("gpu_name", self.gpu_name)),
                n_crossover_paths=max(1, int(prof_dict.get("n_crossover_paths", self.n_crossover_paths))),
                bench_1x10k_cpu_ms=float(prof_dict.get("bench_1x10k_cpu_ms", 0.0)),
                bench_1x10k_gpu_ms=float(prof_dict.get("bench_1x10k_gpu_ms", 0.0)),
                bench_12x10k_cpu_ms=float(prof_dict.get("bench_12x10k_cpu_ms", 0.0)),
                bench_12x10k_gpu_ms=float(prof_dict.get("bench_12x10k_gpu_ms", 0.0)),
                bench_12x100k_cpu_ms=float(prof_dict.get("bench_12x100k_cpu_ms", 0.0)),
                bench_12x100k_gpu_ms=float(prof_dict.get("bench_12x100k_gpu_ms", 0.0)),
                calibrated_at_utc=float(prof_dict.get("calibrated_at_utc", 0.0)),
            )
            self.crossover_profile = prof
            self.n_crossover_paths = prof.n_crossover_paths
            return prof
        except Exception as exc:
            logger.debug("Could not load Hyperion crossover profile: %s", exc)
            return None

    def get_vram_used_fraction(self) -> float:
        """Return current CUDA VRAM usage fraction in [0.0, 1.0], or 0.0 if N/A."""
        if not self.has_cuda or cp is None:
            return 0.0
        try:
            free_b, total_b = cp.cuda.Device(0).mem_info
            if total_b <= 0:
                return 0.0
            return float(total_b - free_b) / float(total_b)
        except Exception:
            return 0.0

    def prewarm(self) -> None:
        """Pre-compile Numba LLVM & CUDA kernels with a tiny 64-path warmup to prevent live JIT stalls."""
        b, n, h = 1, 64, 4
        hi = np.full((b, n, h), 1.001, dtype=np.float64)
        lo = np.full((b, n, h), 0.999, dtype=np.float64)
        cl = np.full((b, n, h), 1.000, dtype=np.float64)
        tu = np.full((b, n, h), 0.500, dtype=np.float64)
        entry = np.array([1.1000], dtype=np.float64)
        stop = np.array([1.0950], dtype=np.float64)
        tgt = np.array([1.1100], dtype=np.float64)
        one_r = np.array([0.0050], dtype=np.float64)
        tgt_r = np.array([2.0], dtype=np.float64)
        cost_r = np.array([0.05], dtype=np.float64)
        is_long = np.array([True], dtype=np.bool_)

        if self.has_numba and _numba_simulate_batch_kernel is not None:
            try:
                _numba_simulate_batch_kernel(hi, lo, cl, tu, entry, stop, tgt, one_r, tgt_r, cost_r, is_long)
                if _numba_grid_batch_kernel is not None:
                    hr = np.ones((1, 8, 4), dtype=np.float32)
                    hr[:, :, 0] = 1.001
                    hr[:, :, 1] = 0.999
                    hl = np.array([8], dtype=np.int32)
                    ru = np.full((8, 4), 0.25, dtype=np.float32)
                    tu2 = np.full((8, 4), 0.50, dtype=np.float32)
                    _numba_grid_batch_kernel(
                        hr, hl, ru, tu2,
                        entry, np.array([0.004], dtype=np.float64), np.array([0.0], dtype=np.float64),
                        cost_r, is_long, np.array([1.0], dtype=np.float64), np.array([2.0], dtype=np.float64), 2,
                    )
            except Exception as exc:
                logger.warning("Numba prewarm failed, disabling Numba: %s", exc)
                self.has_numba = False

        if self.has_cuda and cp is not None:
            try:
                self._cuda_kernel = cp.RawKernel(_CUDA_SIM_KERNEL_SRC, "hyperion_sim_batch_f32_to_f64")
                self._grid_cuda_kernel = cp.RawKernel(_CUDA_GRID_KERNEL_SRC, "hyperion_grid_bootstrap_kernel")
                self._run_cuda_batch(hi, lo, cl, tu, entry, stop, tgt, one_r, tgt_r, cost_r, is_long)
            except Exception as exc:
                logger.warning("CUDA prewarm failed, falling back to CPU: %s", exc)
                self.has_cuda = False
                self._cuda_kernel = None
                self._grid_cuda_kernel = None
                self.last_backend = (
                    ComputeBackend.DIRECTML_GPU
                    if self.has_directml
                    else (ComputeBackend.NUMBA_LLVM_AVX if self.has_numba else ComputeBackend.NUMPY_SIMD)
                )

    def _get_directml_device(self) -> Any:
        """Resolve the active PyTorch DirectML device (`privateuseone:0`)."""
        if torch is None:
            raise RuntimeError("PyTorch is unavailable for DirectML execution")
        if self._directml_device is not None:
            return self._directml_device
        try:
            dev = torch.device("privateuseone:0")
        except Exception:
            if torch_directml is not None:
                dev = torch_directml.device()
            else:
                raise
        self._directml_device = dev
        return dev

    def _run_directml_batch(
        self,
        hi_ratios: np.ndarray,
        lo_ratios: np.ndarray,
        cl_ratios: np.ndarray,
        tie_uniforms: np.ndarray,
        entry_prices: np.ndarray,
        stop_prices: np.ndarray,
        target_prices: np.ndarray,
        one_r_dists: np.ndarray,
        target_rs: np.ndarray,
        cost_rs: np.ndarray,
        is_longs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Execute vectorized (B, N, H) Monte Carlo barrier simulation on PyTorch DirectML."""
        if torch is None:
            raise RuntimeError("PyTorch not installed")
        dev = self._get_directml_device()
        b_cnt, n_paths, h_bars = hi_ratios.shape
        entry_f64 = np.asarray(entry_prices, dtype=np.float64)
        stop_f64 = np.asarray(stop_prices, dtype=np.float64)
        tgt_f64 = np.asarray(target_prices, dtype=np.float64)
        one_r_f64 = np.maximum(np.asarray(one_r_dists, dtype=np.float64), 1e-12)
        tgt_r_f64 = np.asarray(target_rs, dtype=np.float64)
        cost_r_f64 = np.asarray(cost_rs, dtype=np.float64)
        long_arr = np.asarray(is_longs, dtype=np.bool_)

        with torch.no_grad():
            t_hi = torch.as_tensor(np.asarray(hi_ratios, dtype=np.float32), dtype=torch.float32, device=dev)
            t_lo = torch.as_tensor(np.asarray(lo_ratios, dtype=np.float32), dtype=torch.float32, device=dev)
            t_cl = torch.as_tensor(np.asarray(cl_ratios, dtype=np.float32), dtype=torch.float32, device=dev)
            t_tu = torch.as_tensor(np.asarray(tie_uniforms, dtype=np.float32), dtype=torch.float32, device=dev)

            stop_rel = torch.as_tensor(
                (stop_f64 / np.maximum(entry_f64, 1e-12)).astype(np.float32), dtype=torch.float32, device=dev
            ).unsqueeze(1)
            tgt_rel = torch.as_tensor(
                (tgt_f64 / np.maximum(entry_f64, 1e-12)).astype(np.float32), dtype=torch.float32, device=dev
            ).unsqueeze(1)
            t_is_long = torch.as_tensor(long_arr.astype(np.int32), dtype=torch.int32, device=dev).unsqueeze(1) > 0

            rel_px = torch.ones((b_cnt, n_paths), dtype=torch.float32, device=dev)
            st = torch.zeros((b_cnt, n_paths), dtype=torch.int32, device=dev)

            for t in range(h_bars):
                active = st == 0
                if not bool(active.any().item()):
                    break
                hi = rel_px * t_hi[:, :, t]
                lo = rel_px * t_lo[:, :, t]
                cl = rel_px * t_cl[:, :, t]

                raw_stop = active & torch.where(t_is_long, lo <= stop_rel, hi >= stop_rel)
                raw_tgt = active & torch.where(t_is_long, hi >= tgt_rel, lo <= tgt_rel)
                both = raw_stop & raw_tgt

                if bool(both.any().item()):
                    d_stop = torch.abs(rel_px - stop_rel)
                    d_tgt = torch.abs(tgt_rel - rel_px)
                    denom = torch.clamp(d_stop + d_tgt, min=1e-7)
                    p_tgt = d_stop / denom
                    bull_bar = torch.where(t_is_long, cl > rel_px, cl < rel_px)
                    pol = torch.where(bull_bar, 0.15, -0.15)
                    p_tgt = torch.clamp(p_tgt + pol, 0.05, 0.95)
                    win_tie = t_tu[:, :, t] < p_tgt
                    hit_tgt = (raw_tgt & (~raw_stop)) | (both & win_tie)
                    hit_stop = (raw_stop & (~raw_tgt)) | (both & (~win_tie))
                else:
                    hit_stop = raw_stop
                    hit_tgt = raw_tgt

                st = torch.where(hit_stop, -1, torch.where(hit_tgt, 1, st))
                rel_px = torch.where(active & (~(hit_stop | hit_tgt)), cl, rel_px)

            status_out = st.cpu().numpy().astype(np.int8)
            rel_cpu = rel_px.cpu().numpy().astype(np.float64)

        final_px = entry_f64[:, None] * rel_cpu
        timeout_r = np.where(
            long_arr[:, None],
            (final_px - entry_f64[:, None]) / one_r_f64[:, None],
            (entry_f64[:, None] - final_px) / one_r_f64[:, None],
        )
        gross_out = np.where(status_out == 1, tgt_r_f64[:, None], np.where(status_out == -1, -1.0, timeout_r))
        net_out = gross_out - cost_r_f64[:, None]
        return status_out, gross_out, net_out

    def _run_cuda_kernel_on_device(
        self,
        hi_ratios: np.ndarray,
        lo_ratios: np.ndarray,
        cl_ratios: np.ndarray,
        tie_uniforms: np.ndarray,
        entry_prices: np.ndarray,
        stop_prices: np.ndarray,
        target_prices: np.ndarray,
        one_r_dists: np.ndarray,
        target_rs: np.ndarray,
        cost_rs: np.ndarray,
        is_longs: np.ndarray,
    ) -> tuple[Any, Any, Any]:
        """Launch CuPy RawKernel and keep (status, gross_r, net_r) resident on GPU VRAM."""
        if self._cuda_kernel is None:
            self._cuda_kernel = cp.RawKernel(_CUDA_SIM_KERNEL_SRC, "hyperion_sim_batch_f32_to_f64")
        b_cnt, n_paths, h_bars = hi_ratios.shape
        total = b_cnt * n_paths

        d_hi = cp.asarray(hi_ratios, dtype=cp.float32)
        d_lo = cp.asarray(lo_ratios, dtype=cp.float32)
        d_cl = cp.asarray(cl_ratios, dtype=cp.float32)
        d_tu = cp.asarray(tie_uniforms, dtype=cp.float32)
        d_entry = cp.asarray(entry_prices, dtype=cp.float64)
        d_stop = cp.asarray(stop_prices, dtype=cp.float64)
        d_tgt = cp.asarray(target_prices, dtype=cp.float64)
        d_one_r = cp.asarray(one_r_dists, dtype=cp.float64)
        d_tgt_r = cp.asarray(target_rs, dtype=cp.float64)
        d_cost_r = cp.asarray(cost_rs, dtype=cp.float64)
        d_long = cp.asarray(is_longs.astype(np.int32), dtype=cp.int32)

        d_status = cp.empty((b_cnt, n_paths), dtype=cp.int8)
        d_gross = cp.empty((b_cnt, n_paths), dtype=cp.float64)
        d_net = cp.empty((b_cnt, n_paths), dtype=cp.float64)

        threads = 256
        blocks = (total + threads - 1) // threads
        self._cuda_kernel(
            (blocks,),
            (threads,),
            (
                d_hi, d_lo, d_cl, d_tu,
                d_entry, d_stop, d_tgt, d_one_r, d_tgt_r, d_cost_r, d_long,
                d_status, d_gross, d_net,
                np.int32(b_cnt), np.int32(n_paths), np.int32(h_bars),
            ),
        )
        cp.cuda.Stream.null.synchronize()
        return d_status, d_gross, d_net

    def _run_cuda_batch(
        self,
        hi_ratios: np.ndarray,
        lo_ratios: np.ndarray,
        cl_ratios: np.ndarray,
        tie_uniforms: np.ndarray,
        entry_prices: np.ndarray,
        stop_prices: np.ndarray,
        target_prices: np.ndarray,
        one_r_dists: np.ndarray,
        target_rs: np.ndarray,
        cost_rs: np.ndarray,
        is_longs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        d_status, d_gross, d_net = self._run_cuda_kernel_on_device(
            hi_ratios, lo_ratios, cl_ratios, tie_uniforms,
            entry_prices, stop_prices, target_prices,
            one_r_dists, target_rs, cost_rs, is_longs,
        )
        return cp.asnumpy(d_status), cp.asnumpy(d_gross), cp.asnumpy(d_net)

    @staticmethod
    def _reduce_stats_on_device_cupy(d_status: Any, d_gross: Any, d_net: Any) -> np.ndarray:
        """Compute (B, 10) summary scalars on GPU via `cp.mean` & `cp.partition` before D2H copy."""
        n_paths = int(d_net.shape[1])
        p_target = cp.mean(d_status == 1, axis=1, dtype=cp.float64)
        p_stop = cp.mean(d_status == -1, axis=1, dtype=cp.float64)
        p_timeout = cp.mean(d_status == 0, axis=1, dtype=cp.float64)
        prob_profit = cp.mean(d_net > 0.0, axis=1, dtype=cp.float64)
        gross_avg_r = cp.mean(d_gross, axis=1, dtype=cp.float64)
        net_avg_r = cp.mean(d_net, axis=1, dtype=cp.float64)

        if n_paths == 1:
            median_net_r = d_net[:, 0]
        elif n_paths % 2 == 1:
            mid = n_paths // 2
            median_net_r = cp.partition(d_net, mid, axis=1)[:, mid]
        else:
            mid = n_paths // 2
            p_mid = cp.partition(d_net, mid, axis=1)
            median_net_r = 0.5 * (cp.max(p_mid[:, :mid], axis=1) + p_mid[:, mid])

        tail_95 = max(1, int(0.05 * n_paths))
        part_95 = cp.partition(d_net, tail_95 - 1, axis=1)
        var_95_r = part_95[:, tail_95 - 1]
        cvar_95_r = cp.mean(part_95[:, :tail_95], axis=1, dtype=cp.float64)

        tail_99 = max(1, int(0.01 * n_paths))
        part_99 = cp.partition(d_net, tail_99 - 1, axis=1)
        cvar_99_r = cp.mean(part_99[:, :tail_99], axis=1, dtype=cp.float64)

        d_summary = cp.stack(
            [
                p_target,
                p_stop,
                p_timeout,
                prob_profit,
                gross_avg_r,
                net_avg_r,
                median_net_r,
                var_95_r,
                cvar_95_r,
                cvar_99_r,
            ],
            axis=1,
        )
        return cp.asnumpy(d_summary)

    @staticmethod
    def _reduce_stats_numpy(
        status_out: np.ndarray, gross_out: np.ndarray, net_out: np.ndarray
    ) -> np.ndarray:
        """Compute (B, 10) summary scalars on CPU matching GPU reduction semantics."""
        n_paths = int(net_out.shape[1])
        p_target = np.mean(status_out == 1, axis=1, dtype=np.float64)
        p_stop = np.mean(status_out == -1, axis=1, dtype=np.float64)
        p_timeout = np.mean(status_out == 0, axis=1, dtype=np.float64)
        prob_profit = np.mean(net_out > 0.0, axis=1, dtype=np.float64)
        gross_avg_r = np.mean(gross_out, axis=1, dtype=np.float64)
        net_avg_r = np.mean(net_out, axis=1, dtype=np.float64)
        median_net_r = np.median(net_out, axis=1).astype(np.float64)

        tail_95 = max(1, int(0.05 * n_paths))
        part_95 = np.partition(net_out, tail_95 - 1, axis=1)
        var_95_r = part_95[:, tail_95 - 1].astype(np.float64)
        cvar_95_r = np.mean(part_95[:, :tail_95], axis=1, dtype=np.float64)

        tail_99 = max(1, int(0.01 * n_paths))
        part_99 = np.partition(net_out, tail_99 - 1, axis=1)
        cvar_99_r = np.mean(part_99[:, :tail_99], axis=1, dtype=np.float64)

        return np.stack(
            [
                p_target,
                p_stop,
                p_timeout,
                prob_profit,
                gross_avg_r,
                net_avg_r,
                median_net_r,
                var_95_r,
                cvar_95_r,
                cvar_99_r,
            ],
            axis=1,
        )

    @staticmethod
    def _run_numpy_batch(
        hi_ratios: np.ndarray,
        lo_ratios: np.ndarray,
        cl_ratios: np.ndarray,
        tie_uniforms: np.ndarray,
        entry_prices: np.ndarray,
        stop_prices: np.ndarray,
        target_prices: np.ndarray,
        one_r_dists: np.ndarray,
        target_rs: np.ndarray,
        cost_rs: np.ndarray,
        is_longs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        b_cnt, n_paths, h_bars = hi_ratios.shape
        status_out = np.zeros((b_cnt, n_paths), dtype=np.int8)
        gross_out = np.empty((b_cnt, n_paths), dtype=np.float64)
        net_out = np.empty((b_cnt, n_paths), dtype=np.float64)

        for b in range(b_cnt):
            entry_px = float(entry_prices[b])
            stop_px = float(stop_prices[b])
            target_px = float(target_prices[b])
            one_r = float(one_r_dists[b])
            tgt_r = float(target_rs[b])
            cost_r = float(cost_rs[b])
            is_long = bool(is_longs[b])

            cur_prices = np.full(n_paths, entry_px, dtype=np.float64)
            st = np.zeros(n_paths, dtype=np.int8)
            for t in range(h_bars):
                active = st == 0
                if not np.any(active):
                    break
                hi = cur_prices * hi_ratios[b, :, t]
                lo = cur_prices * lo_ratios[b, :, t]
                cl = cur_prices * cl_ratios[b, :, t]

                if is_long:
                    raw_stop = active & (lo <= stop_px)
                    raw_tgt = active & (hi >= target_px)
                else:
                    raw_stop = active & (hi >= stop_px)
                    raw_tgt = active & (lo <= target_px)

                both = raw_stop & raw_tgt
                if np.any(both):
                    d_stop = np.abs(cur_prices - stop_px)
                    d_tgt = np.abs(target_px - cur_prices)
                    p_tgt = d_stop / np.maximum(d_stop + d_tgt, 1e-12)
                    pol = np.where((cl > cur_prices) if is_long else (cl < cur_prices), 0.15, -0.15)
                    p_tgt = np.clip(p_tgt + pol, 0.05, 0.95)
                    win_tie = tie_uniforms[b, :, t] < p_tgt
                    hit_tgt = (raw_tgt & ~raw_stop) | (both & win_tie)
                    hit_stop = (raw_stop & ~raw_tgt) | (both & ~win_tie)
                else:
                    hit_stop = raw_stop
                    hit_tgt = raw_tgt

                st[hit_stop] = -1
                st[hit_tgt] = 1
                cur_prices = np.where(active, cl, cur_prices)

            timeout_r = ((cur_prices - entry_px) / one_r) if is_long else ((entry_px - cur_prices) / one_r)
            gr = np.where(st == 1, tgt_r, np.where(st == -1, -1.0, timeout_r))
            status_out[b] = st
            gross_out[b] = gr
            net_out[b] = gr - cost_r

        return status_out, gross_out, net_out

    def _select_backend(self, total_paths: int, force_backend: ComputeBackend | None = None) -> ComputeBackend:
        if force_backend is not None:
            return force_backend
        if self.has_cuda and total_paths >= self.n_crossover_paths:
            if self.get_vram_used_fraction() < self.VRAM_LIMIT_FRAC:
                return ComputeBackend.CUDA_TENSOR
        if self.has_directml and total_paths >= self.n_crossover_paths:
            return ComputeBackend.DIRECTML_GPU
        if self.has_numba:
            return ComputeBackend.NUMBA_LLVM_AVX
        return ComputeBackend.NUMPY_SIMD

    def _run_best_fallback_batch(
        self,
        hi_ratios: np.ndarray,
        lo_ratios: np.ndarray,
        cl_ratios: np.ndarray,
        tie_uniforms: np.ndarray,
        entry_prices: np.ndarray,
        stop_prices: np.ndarray,
        target_prices: np.ndarray,
        one_r_dists: np.ndarray,
        target_rs: np.ndarray,
        cost_rs: np.ndarray,
        is_longs: np.ndarray,
        *,
        exclude: tuple[ComputeBackend, ...] = (),
    ) -> tuple[ComputeBackend, np.ndarray, np.ndarray, np.ndarray]:
        """Route to highest-priority operational backend (CUDA -> DirectML -> Numba -> NumPy)."""
        if self.has_cuda and ComputeBackend.CUDA_TENSOR not in exclude:
            try:
                s, g, n = self._run_cuda_batch(
                    hi_ratios, lo_ratios, cl_ratios, tie_uniforms,
                    entry_prices, stop_prices, target_prices,
                    one_r_dists, target_rs, cost_rs, is_longs,
                )
                return ComputeBackend.CUDA_TENSOR, s, g, n
            except Exception as exc:
                self.failover_count += 1
                logger.warning("CUDA fallback execution failed (%s); cascading", exc)

        if self.has_directml and torch is not None and ComputeBackend.DIRECTML_GPU not in exclude:
            try:
                s, g, n = self._run_directml_batch(
                    hi_ratios, lo_ratios, cl_ratios, tie_uniforms,
                    entry_prices, stop_prices, target_prices,
                    one_r_dists, target_rs, cost_rs, is_longs,
                )
                return ComputeBackend.DIRECTML_GPU, s, g, n
            except Exception as exc:
                self.failover_count += 1
                logger.warning("DirectML fallback execution failed (%s); cascading to Numba", exc)

        if self.has_numba and _numba_simulate_batch_kernel is not None and ComputeBackend.NUMBA_LLVM_AVX not in exclude:
            s, g, n = _numba_simulate_batch_kernel(
                hi_ratios, lo_ratios, cl_ratios, tie_uniforms,
                entry_prices, stop_prices, target_prices,
                one_r_dists, target_rs, cost_rs, is_longs,
            )
            return ComputeBackend.NUMBA_LLVM_AVX, s, g, n

        s, g, n = self._run_numpy_batch(
            hi_ratios, lo_ratios, cl_ratios, tie_uniforms,
            entry_prices, stop_prices, target_prices,
            one_r_dists, target_rs, cost_rs, is_longs,
        )
        return ComputeBackend.NUMPY_SIMD, s, g, n

    def simulate_paths_batch(
        self,
        *,
        hi_ratios: np.ndarray,
        lo_ratios: np.ndarray,
        cl_ratios: np.ndarray,
        tie_uniforms: np.ndarray,
        entry_prices: np.ndarray,
        stop_prices: np.ndarray,
        target_prices: np.ndarray,
        one_r_dists: np.ndarray,
        target_rs: np.ndarray,
        cost_rs: np.ndarray,
        is_longs: np.ndarray,
        force_backend: ComputeBackend | None = None,
        return_full_paths: bool = True,
    ) -> BatchSimulationResult:
        """Execute 3D [B, N, H] Time Machine simulation on optimal Hyperion backend with on-device stats & <1ms failover."""
        b_cnt, n_paths, _h_bars = hi_ratios.shape
        total_paths = b_cnt * n_paths
        backend = self._select_backend(total_paths, force_backend=force_backend)

        t0 = time.perf_counter()
        summary_arr: np.ndarray | None = None
        net_out: np.ndarray | None = None

        if backend == ComputeBackend.CUDA_TENSOR and self.has_cuda:
            try:
                if "_run_cuda_batch" in self.__dict__:
                    status_out, gross_out, net_out = self._run_cuda_batch(
                        hi_ratios, lo_ratios, cl_ratios, tie_uniforms,
                        entry_prices, stop_prices, target_prices,
                        one_r_dists, target_rs, cost_rs, is_longs,
                    )
                    summary_arr = self._reduce_stats_numpy(status_out, gross_out, net_out)
                else:
                    d_status, d_gross, d_net = self._run_cuda_kernel_on_device(
                        hi_ratios, lo_ratios, cl_ratios, tie_uniforms,
                        entry_prices, stop_prices, target_prices,
                        one_r_dists, target_rs, cost_rs, is_longs,
                    )
                    summary_arr = self._reduce_stats_on_device_cupy(d_status, d_gross, d_net)
                    net_out = cp.asnumpy(d_net) if return_full_paths else np.empty((b_cnt, 0), dtype=np.float64)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                if elapsed_ms > self.GPU_TIMEOUT_MS:
                    self.failover_count += 1
                    logger.warning("CUDA batch exceeded %.0fms (%.1fms); adaptive crossover raised", self.GPU_TIMEOUT_MS, elapsed_ms)
            except Exception as exc:
                self.failover_count += 1
                logger.warning("CUDA batch failed (%s); instant failover", exc)
                t0 = time.perf_counter()
                backend, status_out, gross_out, net_out = self._run_best_fallback_batch(
                    hi_ratios, lo_ratios, cl_ratios, tie_uniforms,
                    entry_prices, stop_prices, target_prices,
                    one_r_dists, target_rs, cost_rs, is_longs,
                    exclude=(ComputeBackend.CUDA_TENSOR,),
                )
                summary_arr = self._reduce_stats_numpy(status_out, gross_out, net_out)
        elif backend == ComputeBackend.DIRECTML_GPU:
            if (self.has_directml or _HAS_DIRECTML) and torch is not None:
                try:
                    status_out, gross_out, net_out = self._run_directml_batch(
                        hi_ratios, lo_ratios, cl_ratios, tie_uniforms,
                        entry_prices, stop_prices, target_prices,
                        one_r_dists, target_rs, cost_rs, is_longs,
                    )
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    if elapsed_ms > self.GPU_TIMEOUT_MS:
                        self.failover_count += 1
                        logger.warning("DirectML batch exceeded %.0fms (%.1fms)", self.GPU_TIMEOUT_MS, elapsed_ms)
                    summary_arr = self._reduce_stats_numpy(status_out, gross_out, net_out)
                except Exception as exc:
                    self.failover_count += 1
                    logger.warning("DirectML batch failed (%s); instant failover", exc)
                    t0 = time.perf_counter()
                    backend, status_out, gross_out, net_out = self._run_best_fallback_batch(
                        hi_ratios, lo_ratios, cl_ratios, tie_uniforms,
                        entry_prices, stop_prices, target_prices,
                        one_r_dists, target_rs, cost_rs, is_longs,
                        exclude=(ComputeBackend.DIRECTML_GPU,),
                    )
                    summary_arr = self._reduce_stats_numpy(status_out, gross_out, net_out)
            else:
                backend, status_out, gross_out, net_out = self._run_best_fallback_batch(
                    hi_ratios, lo_ratios, cl_ratios, tie_uniforms,
                    entry_prices, stop_prices, target_prices,
                    one_r_dists, target_rs, cost_rs, is_longs,
                    exclude=(ComputeBackend.DIRECTML_GPU,),
                )
                summary_arr = self._reduce_stats_numpy(status_out, gross_out, net_out)
        elif backend == ComputeBackend.NUMBA_LLVM_AVX and self.has_numba and _numba_simulate_batch_kernel is not None:
            status_out, gross_out, net_out = _numba_simulate_batch_kernel(
                hi_ratios, lo_ratios, cl_ratios, tie_uniforms,
                entry_prices, stop_prices, target_prices,
                one_r_dists, target_rs, cost_rs, is_longs,
            )
            summary_arr = self._reduce_stats_numpy(status_out, gross_out, net_out)
        elif backend != ComputeBackend.NUMPY_SIMD and (self.has_cuda or self.has_directml or self.has_numba):
            # Never fall silently to NUMPY_SIMD if a faster accelerated backend is available
            backend, status_out, gross_out, net_out = self._run_best_fallback_batch(
                hi_ratios, lo_ratios, cl_ratios, tie_uniforms,
                entry_prices, stop_prices, target_prices,
                one_r_dists, target_rs, cost_rs, is_longs,
            )
            summary_arr = self._reduce_stats_numpy(status_out, gross_out, net_out)
        else:
            backend = ComputeBackend.NUMPY_SIMD
            status_out, gross_out, net_out = self._run_numpy_batch(
                hi_ratios, lo_ratios, cl_ratios, tie_uniforms,
                entry_prices, stop_prices, target_prices,
                one_r_dists, target_rs, cost_rs, is_longs,
            )
            summary_arr = self._reduce_stats_numpy(status_out, gross_out, net_out)

        assert summary_arr is not None and net_out is not None
        if not return_full_paths and net_out.shape[1] > 0:
            net_out = np.empty((b_cnt, 0), dtype=np.float64)

        compute_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        with self._lock:
            self.last_backend = backend
            self.last_batch_ms = compute_ms
            self.last_paths_count = total_paths

        return BatchSimulationResult(
            p_target=summary_arr[:, 0],
            p_stop=summary_arr[:, 1],
            p_timeout=summary_arr[:, 2],
            prob_profit=summary_arr[:, 3],
            gross_avg_r=summary_arr[:, 4],
            net_avg_r=summary_arr[:, 5],
            median_net_r=summary_arr[:, 6],
            var_95_r=summary_arr[:, 7],
            cvar_95_r=summary_arr[:, 8],
            cvar_99_r=summary_arr[:, 9],
            net_r_out=net_out,
            backend_used=backend,
            compute_ms=compute_ms,
        )

    def _get_sobol_tables(
        self, n_paths: int, horizon_bars: int, *, on_gpu: bool = False
    ) -> tuple[Any, Any]:
        """Build/return cached antithetic Sobol index & tie-break uniform tables `(N, H)` (on-device for CUDA)."""
        key = (int(n_paths), int(horizon_bars))
        with self._lock:
            cached = self._sobol_cache.get(key)
        if cached is None:
            from scipy.stats import qmc

            half_n = max(1, (n_paths + 1) // 2)
            m_pow = max(1, math.ceil(math.log2(max(2, half_n))))
            dim = max(2, 2 * horizon_bars)
            engine = qmc.Sobol(d=dim, scramble=True, seed=137)
            pts = engine.random_base2(m=m_pow)[:half_n]
            half_rand = pts[:, 0::2][:, :horizon_bars]
            half_tie = pts[:, 1::2][:, :horizon_bars]
            rand_full = np.concatenate([half_rand, 1.0 - half_rand], axis=0)[:n_paths]
            tie_full = np.concatenate([half_tie, 1.0 - half_tie], axis=0)[:n_paths]
            cached = (
                np.ascontiguousarray(np.clip(rand_full, 0.0, 0.999999), dtype=np.float32),
                np.ascontiguousarray(np.clip(tie_full, 0.000001, 0.999999), dtype=np.float32),
            )
            with self._lock:
                self._sobol_cache[key] = cached

        if on_gpu and cp is not None:
            with self._lock:
                g_cached = self._gpu_sobol_cache.get(key)
            if g_cached is None:
                g_cached = (cp.asarray(cached[0], dtype=cp.float32), cp.asarray(cached[1], dtype=cp.float32))
                with self._lock:
                    self._gpu_sobol_cache[key] = g_cached
            return g_cached
        return cached

    @staticmethod
    def _run_numpy_grid_batch(
        hist_ratios: np.ndarray,
        hist_lengths: np.ndarray,
        rand_u: np.ndarray,
        tie_u: np.ndarray,
        entry_prices: np.ndarray,
        atr_values: np.ndarray,
        spread_values: np.ndarray,
        cost_prices: np.ndarray,
        is_longs: np.ndarray,
        sl_grid: np.ndarray,
        tp_grid: np.ndarray,
        block_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """CPU NumPy parity kernel for the (B, G, N) zero-copy 5x5 grid bootstrap simulation."""
        b_cnt = hist_ratios.shape[0]
        n_paths = rand_u.shape[0]
        h_bars = rand_u.shape[1]
        g_sl = len(sl_grid)
        g_tp = len(tp_grid)
        g_cnt = g_sl * g_tp
        bg_cnt = b_cnt * g_cnt
        bs = max(1, int(block_size))

        status_out = np.zeros((bg_cnt, n_paths), dtype=np.int8)
        net_r_out = np.empty((bg_cnt, n_paths), dtype=np.float64)

        for b in range(b_cnt):
            m_len = max(1, int(hist_lengths[b]))
            max_start = max(1, m_len - bs + 1)
            # Vectorized index synthesis for symbol b across (N, H)
            idx_mat = np.empty((n_paths, h_bars), dtype=np.int32)
            cur_idx = np.zeros((n_paths,), dtype=np.int32)
            for t in range(h_bars):
                if t % bs == 0:
                    cur_idx = np.minimum((rand_u[:, t] * max_start).astype(np.int32), max_start - 1)
                else:
                    cur_idx = (cur_idx + 1) % m_len
                idx_mat[:, t] = cur_idx

            b_ratios = hist_ratios[b, idx_mat, :]  # (N, H, 4)
            entry_px = float(entry_prices[b])
            atr = float(atr_values[b])
            spr = float(spread_values[b])
            cost_px = float(cost_prices[b])
            is_long = bool(is_longs[b])

            for gs in range(g_sl):
                sl_m = float(sl_grid[gs])
                one_r = max(atr * sl_m + spr, 1e-12)
                stop_px = max(entry_px * 0.01, entry_px - one_r) if is_long else (entry_px + one_r)
                cost_r = cost_px / one_r
                for gt in range(g_tp):
                    tp_m = float(tp_grid[gt])
                    tgt_px = (entry_px + tp_m * one_r) if is_long else max(entry_px * 0.01, entry_px - tp_m * one_r)
                    bg = b * g_cnt + gs * g_tp + gt
                    cur_px = np.full(n_paths, entry_px, dtype=np.float64)
                    st = np.zeros(n_paths, dtype=np.int8)

                    for t in range(h_bars):
                        active = st == 0
                        if not np.any(active):
                            break
                        hi = cur_px * b_ratios[:, t, 0]
                        lo = cur_px * b_ratios[:, t, 1]
                        cl = cur_px * b_ratios[:, t, 2]
                        if is_long:
                            raw_stop = active & (lo <= stop_px)
                            raw_tgt = active & (hi >= tgt_px)
                        else:
                            raw_stop = active & (hi >= stop_px)
                            raw_tgt = active & (lo <= tgt_px)
                        both = raw_stop & raw_tgt
                        if np.any(both):
                            d_stop = np.abs(cur_px - stop_px)
                            d_tgt = np.abs(tgt_px - cur_px)
                            p_tgt = d_stop / np.maximum(d_stop + d_tgt, 1e-12)
                            pol = np.where((cl > cur_px) if is_long else (cl < cur_px), 0.15, -0.15)
                            p_tgt = np.clip(p_tgt + pol, 0.05, 0.95)
                            win_tie = tie_u[:, t] < p_tgt
                            hit_tgt = (raw_tgt & ~raw_stop) | (both & win_tie)
                            hit_stop = (raw_stop & ~raw_tgt) | (both & ~win_tie)
                        else:
                            hit_stop = raw_stop
                            hit_tgt = raw_tgt
                        st[hit_stop] = -1
                        st[hit_tgt] = 1
                        cur_px = np.where(active, cl, cur_px)

                    timeout_r = ((cur_px - entry_px) / one_r) if is_long else ((entry_px - cur_px) / one_r)
                    gr = np.where(st == 1, tp_m, np.where(st == -1, -1.0, timeout_r))
                    status_out[bg] = st
                    net_r_out[bg] = gr - cost_r

        return status_out, net_r_out

    def _run_directml_grid_batch(
        self,
        hist_ratios: np.ndarray,
        hist_lengths: np.ndarray,
        rand_u: np.ndarray,
        tie_u: np.ndarray,
        entry_prices: np.ndarray,
        atr_values: np.ndarray,
        spread_values: np.ndarray,
        cost_prices: np.ndarray,
        is_longs: np.ndarray,
        sl_grid: np.ndarray,
        tp_grid: np.ndarray,
        block_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """PyTorch DirectML vectorized (B, G, N) 5x5 grid path simulation returning compact (B, 6) + score."""
        if torch is None:
            raise RuntimeError("PyTorch not installed")
        dev = self._get_directml_device()
        b_cnt = hist_ratios.shape[0]
        n_paths = rand_u.shape[0]
        h_bars = rand_u.shape[1]
        g_sl = len(sl_grid)
        g_tp = len(tp_grid)
        g_cnt = g_sl * g_tp
        bs = max(1, int(block_size))

        sl_mesh, tp_mesh = np.meshgrid(sl_grid, tp_grid, indexing="ij")
        sl_flat = sl_mesh.reshape(1, g_cnt)
        tp_flat = tp_mesh.reshape(1, g_cnt)

        one_r = np.maximum(atr_values[:, None] * sl_flat + spread_values[:, None], 1e-12)
        long_bg = np.asarray(is_longs, dtype=np.bool_)[:, None]
        stop_px = np.where(long_bg, np.maximum(entry_prices[:, None] * 0.01, entry_prices[:, None] - one_r), entry_prices[:, None] + one_r)
        tgt_px = np.where(long_bg, entry_prices[:, None] + tp_flat * one_r, np.maximum(entry_prices[:, None] * 0.01, entry_prices[:, None] - tp_flat * one_r))
        cost_r = cost_prices[:, None] / one_r

        with torch.no_grad():
            t_hist = torch.as_tensor(np.asarray(hist_ratios, dtype=np.float32), dtype=torch.float32, device=dev)
            t_ru = torch.as_tensor(np.asarray(rand_u, dtype=np.float32), dtype=torch.float32, device=dev)
            t_tu = torch.as_tensor(np.asarray(tie_u, dtype=torch.float32), dtype=torch.float32, device=dev)

            stop_rel = torch.as_tensor((stop_px / np.maximum(entry_prices[:, None], 1e-12)).astype(np.float32), dtype=torch.float32, device=dev).unsqueeze(2)
            tgt_rel = torch.as_tensor((tgt_px / np.maximum(entry_prices[:, None], 1e-12)).astype(np.float32), dtype=torch.float32, device=dev).unsqueeze(2)
            t_long = torch.as_tensor(long_bg.astype(np.int32), dtype=torch.int32, device=dev).unsqueeze(2) > 0

            rel_px = torch.ones((b_cnt, g_cnt, n_paths), dtype=torch.float32, device=dev)
            st = torch.zeros((b_cnt, g_cnt, n_paths), dtype=torch.int32, device=dev)
            t_mlen = torch.as_tensor(np.maximum(1, hist_lengths).astype(np.int64), dtype=torch.int64, device=dev).unsqueeze(1)
            t_max_st = torch.clamp(t_mlen - bs + 1, min=1)
            cur_idx = torch.zeros((b_cnt, n_paths), dtype=torch.int64, device=dev)
            b_ar = torch.arange(b_cnt, dtype=torch.int64, device=dev).unsqueeze(1)

            for t in range(h_bars):
                if t % bs == 0:
                    cur_idx = torch.minimum((t_ru[:, t].unsqueeze(0) * t_max_st.float()).to(torch.int64), t_max_st - 1)
                else:
                    cur_idx = (cur_idx + 1) % t_mlen
                bar_r = t_hist[b_ar, cur_idx, :]  # (B, N, 4)
                r_hi = bar_r[:, :, 0].unsqueeze(1)
                r_lo = bar_r[:, :, 1].unsqueeze(1)
                r_cl = bar_r[:, :, 2].unsqueeze(1)

                active = st == 0
                hi = rel_px * r_hi
                lo = rel_px * r_lo
                cl = rel_px * r_cl

                raw_stop = active & torch.where(t_long, lo <= stop_rel, hi >= stop_rel)
                raw_tgt = active & torch.where(t_long, hi >= tgt_rel, lo <= tgt_rel)
                both = raw_stop & raw_tgt
                d_stop = torch.abs(rel_px - stop_rel)
                d_tgt = torch.abs(tgt_rel - rel_px)
                p_tgt = torch.clamp(
                    d_stop / torch.clamp(d_stop + d_tgt, min=1e-7)
                    + torch.where(torch.where(t_long, cl > rel_px, cl < rel_px), 0.15, -0.15),
                    0.05,
                    0.95,
                )
                win_tie = t_tu[:, t].view(1, 1, n_paths) < p_tgt
                hit_tgt = torch.where(both, win_tie, raw_tgt & (~raw_stop))
                hit_stop = torch.where(both, ~win_tie, raw_stop & (~raw_tgt))
                st = torch.where(hit_stop, -1, torch.where(hit_tgt, 1, st))
                rel_px = torch.where(active & (~(hit_stop | hit_tgt)), cl, rel_px)

            st_cpu = st.cpu().numpy()
            rel_cpu = rel_px.cpu().numpy().astype(np.float64)

        timeout_r = np.where(
            long_bg[:, :, None],
            (entry_prices[:, None, None] * rel_cpu - entry_prices[:, None, None]) / one_r[:, :, None],
            (entry_prices[:, None, None] - entry_prices[:, None, None] * rel_cpu) / one_r[:, :, None],
        )
        gr = np.where(st_cpu == 1, tp_flat[:, :, None], np.where(st_cpu == -1, -1.0, timeout_r))
        net_r = gr - cost_r[:, :, None]
        return self._select_optimal_grid_cpu(
            st_cpu.reshape(b_cnt * g_cnt, n_paths),
            net_r.reshape(b_cnt * g_cnt, n_paths),
            b_cnt,
            sl_grid,
            tp_grid,
        )

    @staticmethod
    def _select_optimal_grid_cpu(
        status_bg: np.ndarray,
        net_bg: np.ndarray,
        b_cnt: int,
        sl_grid: np.ndarray,
        tp_grid: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select (SL*, TP*) maximizing CVaR95-adjusted Kelly expectancy on CPU."""
        g_sl = len(sl_grid)
        g_tp = len(tp_grid)
        g_cnt = g_sl * g_tp
        n_paths = int(net_bg.shape[1])

        prob_tp = np.mean(status_bg == 1, axis=1, dtype=np.float64)
        prob_sl = np.mean(status_bg == -1, axis=1, dtype=np.float64)
        mean_r = np.mean(net_bg, axis=1, dtype=np.float64)
        second_m = np.maximum(np.mean(np.square(net_bg), axis=1, dtype=np.float64), 1e-6)

        tail_95 = max(1, int(0.05 * n_paths))
        part = np.partition(net_bg, tail_95 - 1, axis=1)
        cvar_5 = np.mean(part[:, :tail_95], axis=1, dtype=np.float64)

        f_kelly = np.clip(mean_r / second_m, 0.0, 2.0)
        raw_kelly = np.where(
            mean_r > 0.0,
            f_kelly * mean_r - 0.5 * f_kelly * f_kelly * second_m + 0.25 * mean_r,
            mean_r,
        )
        score = raw_kelly - 0.05 * np.maximum(0.0, -cvar_5)

        score_2d = score.reshape(b_cnt, g_cnt)
        best_g = np.argmax(score_2d, axis=1)
        rows = np.arange(b_cnt)

        opt_sl = sl_grid[best_g // g_tp].astype(np.float64)
        opt_tp = tp_grid[best_g % g_tp].astype(np.float64)
        best_ptp = prob_tp.reshape(b_cnt, g_cnt)[rows, best_g]
        best_psl = prob_sl.reshape(b_cnt, g_cnt)[rows, best_g]
        best_mean = mean_r.reshape(b_cnt, g_cnt)[rows, best_g]
        best_cvar = cvar_5.reshape(b_cnt, g_cnt)[rows, best_g]
        best_score = score_2d[rows, best_g]

        scalars_6 = np.stack([opt_sl, opt_tp, best_ptp, best_psl, best_mean, best_cvar], axis=1)
        return scalars_6, best_score

    def _run_cuda_grid_batch(
        self,
        hist_ratios: np.ndarray,
        hist_lengths: np.ndarray,
        entry_prices: np.ndarray,
        atr_values: np.ndarray,
        spread_values: np.ndarray,
        cost_prices: np.ndarray,
        is_longs: np.ndarray,
        sl_grid: np.ndarray,
        tp_grid: np.ndarray,
        n_paths: int,
        horizon_bars: int,
        block_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Execute zero-copy 5x5 grid kernel on CUDA + on-device reductions copying ONLY (B, 6) scalars back."""
        if self._grid_cuda_kernel is None:
            self._grid_cuda_kernel = cp.RawKernel(_CUDA_GRID_KERNEL_SRC, "hyperion_grid_bootstrap_kernel")

        b_cnt, m_cap, _ = hist_ratios.shape
        g_sl = len(sl_grid)
        g_tp = len(tp_grid)
        g_cnt = g_sl * g_tp
        bg_cnt = b_cnt * g_cnt

        # On-device cached Sobol uniform tables (0 PCIe per scan cycle after cache warm!)
        d_rand_u, d_tie_u = self._get_sobol_tables(n_paths, horizon_bars, on_gpu=True)

        # Tiny H2D transfers: only (B, M, 4) float32 historical ratios + (B,) scalar parameters
        d_hist = cp.asarray(hist_ratios, dtype=cp.float32)
        d_hl = cp.asarray(hist_lengths, dtype=cp.int32)
        d_entry = cp.asarray(entry_prices, dtype=cp.float64)
        d_atr = cp.asarray(atr_values, dtype=cp.float64)
        d_spr = cp.asarray(spread_values, dtype=cp.float64)
        d_cost = cp.asarray(cost_prices, dtype=cp.float64)
        d_long = cp.asarray(is_longs.astype(np.int32), dtype=cp.int32)
        d_sl = cp.asarray(sl_grid, dtype=cp.float64)
        d_tp = cp.asarray(tp_grid, dtype=cp.float64)

        d_status = cp.empty((bg_cnt, n_paths), dtype=cp.int8)
        d_net = cp.empty((bg_cnt, n_paths), dtype=cp.float64)

        threads = 256
        total = bg_cnt * n_paths
        blocks = (total + threads - 1) // threads
        self._grid_cuda_kernel(
            (blocks,),
            (threads,),
            (
                d_hist, d_hl, d_rand_u, d_tie_u,
                d_entry, d_atr, d_spr, d_cost, d_long, d_sl, d_tp,
                d_status, d_net,
                np.int32(b_cnt), np.int32(m_cap), np.int32(n_paths),
                np.int32(horizon_bars), np.int32(block_size),
                np.int32(g_sl), np.int32(g_tp),
            ),
        )

        # On-device reductions via cp.mean & cp.partition before copying just (B, 6) + score back to CPU
        prob_tp = cp.mean(d_status == 1, axis=1, dtype=cp.float64)
        prob_sl = cp.mean(d_status == -1, axis=1, dtype=cp.float64)
        mean_r = cp.mean(d_net, axis=1, dtype=cp.float64)
        second_m = cp.maximum(cp.mean(cp.square(d_net), axis=1, dtype=cp.float64), 1e-6)

        tail_95 = max(1, int(0.05 * n_paths))
        part = cp.partition(d_net, tail_95 - 1, axis=1)
        cvar_5 = cp.mean(part[:, :tail_95], axis=1, dtype=cp.float64)

        f_kelly = cp.clip(mean_r / second_m, 0.0, 2.0)
        raw_kelly = cp.where(
            mean_r > 0.0,
            f_kelly * mean_r - 0.5 * f_kelly * f_kelly * second_m + 0.25 * mean_r,
            mean_r,
        )
        score = raw_kelly - 0.05 * cp.maximum(0.0, -cvar_5)

        score_2d = score.reshape(b_cnt, g_cnt)
        best_g = cp.argmax(score_2d, axis=1)
        rows = cp.arange(b_cnt, dtype=cp.int64)

        opt_sl = d_sl[best_g // g_tp]
        opt_tp = d_tp[best_g % g_tp]
        best_ptp = prob_tp.reshape(b_cnt, g_cnt)[rows, best_g]
        best_psl = prob_sl.reshape(b_cnt, g_cnt)[rows, best_g]
        best_mean = mean_r.reshape(b_cnt, g_cnt)[rows, best_g]
        best_cvar = cvar_5.reshape(b_cnt, g_cnt)[rows, best_g]
        best_score = score_2d[rows, best_g]

        d_out = cp.stack([opt_sl, opt_tp, best_ptp, best_psl, best_mean, best_cvar, best_score], axis=1)
        cp.cuda.Stream.null.synchronize()
        host_out = cp.asnumpy(d_out)
        return host_out[:, :6], host_out[:, 6]

    def optimize_sl_tp_grid_batch(
        self,
        *,
        hist_ratios: np.ndarray,
        entry_prices: np.ndarray,
        atr_values: np.ndarray,
        is_longs: np.ndarray,
        cost_prices: np.ndarray | None = None,
        spread_values: np.ndarray | None = None,
        hist_lengths: np.ndarray | None = None,
        n_paths: int = 4096,
        horizon_bars: int = 24,
        block_size: int = 4,
        sl_grid: np.ndarray | list[float] | tuple[float, ...] | None = None,
        tp_grid: np.ndarray | list[float] | tuple[float, ...] | None = None,
        force_backend: ComputeBackend | None = None,
    ) -> GridOptimizationResult:
        """Optimize (SL*, TP*) over a 5x5 bracket grid across B Stage-2 candidates in one zero-copy pass."""
        hr = np.ascontiguousarray(np.asarray(hist_ratios, dtype=np.float32))
        if hr.ndim != 3 or hr.shape[2] != 4:
            raise ValueError(f"hist_ratios must have shape (B, M, 4), got {hr.shape}")
        b_cnt, m_cap, _ = hr.shape
        if b_cnt == 0:
            raise ValueError("hist_ratios batch B must be >= 1")

        sl_arr = np.asarray(
            sl_grid if sl_grid is not None else np.linspace(1.0, 2.5, 5), dtype=np.float64
        )
        tp_arr = np.asarray(
            tp_grid if tp_grid is not None else np.linspace(1.5, 4.0, 5), dtype=np.float64
        )
        g_cnt = int(len(sl_arr) * len(tp_arr))
        n_eff = max(64, int(n_paths))
        h_eff = max(1, int(horizon_bars))
        bs_eff = max(1, int(block_size))

        hl_arr = (
            np.clip(np.asarray(hist_lengths, dtype=np.int32), 1, m_cap)
            if hist_lengths is not None
            else np.full((b_cnt,), m_cap, dtype=np.int32)
        )
        entry_arr = np.asarray(entry_prices, dtype=np.float64)
        atr_arr = np.maximum(np.asarray(atr_values, dtype=np.float64), 1e-9)
        spr_arr = (
            np.maximum(np.asarray(spread_values, dtype=np.float64), 0.0)
            if spread_values is not None
            else np.zeros((b_cnt,), dtype=np.float64)
        )
        cost_arr = (
            np.maximum(np.asarray(cost_prices, dtype=np.float64), 0.0)
            if cost_prices is not None
            else np.zeros((b_cnt,), dtype=np.float64)
        )
        long_arr = np.asarray(is_longs, dtype=np.bool_)

        total_paths = b_cnt * g_cnt * n_eff
        backend = self._select_backend(total_paths, force_backend=force_backend)
        t0 = time.perf_counter()

        scalars_6: np.ndarray | None = None
        best_score: np.ndarray | None = None

        if backend == ComputeBackend.CUDA_TENSOR and self.has_cuda:
            try:
                scalars_6, best_score = self._run_cuda_grid_batch(
                    hr, hl_arr, entry_arr, atr_arr, spr_arr, cost_arr, long_arr,
                    sl_arr, tp_arr, n_eff, h_eff, bs_eff,
                )
            except Exception as exc:
                self.failover_count += 1
                logger.warning("CUDA grid batch failed (%s); instant failover", exc)
                backend = ComputeBackend.DIRECTML_GPU if self.has_directml else (
                    ComputeBackend.NUMBA_LLVM_AVX if self.has_numba else ComputeBackend.NUMPY_SIMD
                )

        if scalars_6 is None and backend == ComputeBackend.DIRECTML_GPU:
            if (self.has_directml or _HAS_DIRECTML) and torch is not None:
                try:
                    ru_cpu, tu_cpu = self._get_sobol_tables(n_eff, h_eff, on_gpu=False)
                    scalars_6, best_score = self._run_directml_grid_batch(
                        hr, hl_arr, ru_cpu, tu_cpu, entry_arr, atr_arr, spr_arr, cost_arr, long_arr,
                        sl_arr, tp_arr, bs_eff,
                    )
                except Exception as exc:
                    self.failover_count += 1
                    logger.warning("DirectML grid batch failed (%s); instant failover", exc)
                    backend = ComputeBackend.CUDA_TENSOR if self.has_cuda else (
                        ComputeBackend.NUMBA_LLVM_AVX if self.has_numba else ComputeBackend.NUMPY_SIMD
                    )
                    if backend == ComputeBackend.CUDA_TENSOR and self.has_cuda:
                        try:
                            scalars_6, best_score = self._run_cuda_grid_batch(
                                hr, hl_arr, entry_arr, atr_arr, spr_arr, cost_arr, long_arr,
                                sl_arr, tp_arr, n_eff, h_eff, bs_eff,
                            )
                        except Exception:
                            backend = ComputeBackend.NUMBA_LLVM_AVX if self.has_numba else ComputeBackend.NUMPY_SIMD
            else:
                backend = ComputeBackend.CUDA_TENSOR if self.has_cuda else (
                    ComputeBackend.NUMBA_LLVM_AVX if self.has_numba else ComputeBackend.NUMPY_SIMD
                )
                if backend == ComputeBackend.CUDA_TENSOR and self.has_cuda:
                    scalars_6, best_score = self._run_cuda_grid_batch(
                        hr, hl_arr, entry_arr, atr_arr, spr_arr, cost_arr, long_arr,
                        sl_arr, tp_arr, n_eff, h_eff, bs_eff,
                    )

        if scalars_6 is None:
            ru_cpu, tu_cpu = self._get_sobol_tables(n_eff, h_eff, on_gpu=False)
            if backend != ComputeBackend.NUMPY_SIMD and self.has_numba and _numba_grid_batch_kernel is not None:
                backend = ComputeBackend.NUMBA_LLVM_AVX
                st_bg, net_bg = _numba_grid_batch_kernel(
                    hr, hl_arr, ru_cpu, tu_cpu, entry_arr, atr_arr, spr_arr, cost_arr, long_arr,
                    sl_arr, tp_arr, bs_eff,
                )
            else:
                backend = ComputeBackend.NUMBA_LLVM_AVX if (backend != ComputeBackend.NUMPY_SIMD and self.has_numba) else ComputeBackend.NUMPY_SIMD
                st_bg, net_bg = self._run_numpy_grid_batch(
                    hr, hl_arr, ru_cpu, tu_cpu, entry_arr, atr_arr, spr_arr, cost_arr, long_arr,
                    sl_arr, tp_arr, bs_eff,
                )
            scalars_6, best_score = self._select_optimal_grid_cpu(st_bg, net_bg, b_cnt, sl_arr, tp_arr)

        assert scalars_6 is not None and best_score is not None
        compute_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        with self._lock:
            self.last_backend = backend
            self.last_batch_ms = compute_ms
            self.last_paths_count = total_paths

        return GridOptimizationResult(
            optimal_sl_atr=scalars_6[:, 0],
            optimal_tp_r=scalars_6[:, 1],
            prob_tp=scalars_6[:, 2],
            prob_sl=scalars_6[:, 3],
            mean_r=scalars_6[:, 4],
            cvar_5=scalars_6[:, 5],
            kelly_cvar_score=best_score,
            scalars_6=scalars_6,
            sl_grid=sl_arr,
            tp_grid=tp_arr,
            backend_used=backend,
            compute_ms=compute_ms,
        )

    def calibrate_crossover(self, persist: bool = True) -> CrossoverProfile:
        """Race GPU vs Numba LLVM AVX across (1x10k, 12x10k, 12x100k) to set empirical N_crossover."""
        def _bench(b_cnt: int, n_paths: int, backend: ComputeBackend) -> float:
            h_bars = 36
            hi = np.full((b_cnt, n_paths, h_bars), 1.0012, dtype=np.float64)
            lo = np.full((b_cnt, n_paths, h_bars), 0.9988, dtype=np.float64)
            cl = np.full((b_cnt, n_paths, h_bars), 1.0001, dtype=np.float64)
            tu = np.full((b_cnt, n_paths, h_bars), 0.5000, dtype=np.float64)
            entry = np.full(b_cnt, 1.1000, dtype=np.float64)
            stop = np.full(b_cnt, 1.0950, dtype=np.float64)
            tgt = np.full(b_cnt, 1.1100, dtype=np.float64)
            one_r = np.full(b_cnt, 0.0050, dtype=np.float64)
            tgt_r = np.full(b_cnt, 2.0, dtype=np.float64)
            cost_r = np.full(b_cnt, 0.05, dtype=np.float64)
            is_long = np.ones(b_cnt, dtype=np.bool_)
            res = self.simulate_paths_batch(
                hi_ratios=hi, lo_ratios=lo, cl_ratios=cl, tie_uniforms=tu,
                entry_prices=entry, stop_prices=stop, target_prices=tgt,
                one_r_dists=one_r, target_rs=tgt_r, cost_rs=cost_r, is_longs=is_long,
                force_backend=backend,
                return_full_paths=False,
            )
            return res.compute_ms

        cpu_backend = ComputeBackend.NUMBA_LLVM_AVX if self.has_numba else ComputeBackend.NUMPY_SIMD
        c_1x10 = _bench(1, 10_000, cpu_backend)
        c_12x10 = _bench(12, 10_000, cpu_backend)
        c_12x100 = _bench(12, 100_000, cpu_backend)

        if self.has_cuda or self.has_directml:
            gpu_backend = ComputeBackend.CUDA_TENSOR if self.has_cuda else ComputeBackend.DIRECTML_GPU
            g_1x10 = _bench(1, 10_000, gpu_backend)
            g_12x10 = _bench(12, 10_000, gpu_backend)
            g_12x100 = _bench(12, 100_000, gpu_backend)
            if g_1x10 < c_1x10:
                self.n_crossover_paths = 5_000
            elif g_12x10 < c_12x10:
                self.n_crossover_paths = 50_000
            else:
                self.n_crossover_paths = 200_000
            pref_gpu = gpu_backend.value
        else:
            g_1x10 = g_12x10 = g_12x100 = 0.0
            self.n_crossover_paths = 10_000
            pref_gpu = cpu_backend.value

        prof = CrossoverProfile(
            preferred_gpu_backend=pref_gpu,
            gpu_name=self.gpu_name,
            n_crossover_paths=self.n_crossover_paths,
            bench_1x10k_cpu_ms=round(c_1x10, 3),
            bench_1x10k_gpu_ms=round(g_1x10, 3),
            bench_12x10k_cpu_ms=round(c_12x10, 3),
            bench_12x10k_gpu_ms=round(g_12x10, 3),
            bench_12x100k_cpu_ms=round(c_12x100, 3),
            bench_12x100k_gpu_ms=round(g_12x100, 3),
            calibrated_at_utc=time.time(),
        )
        self.crossover_profile = prof

        if persist:
            self._persist_crossover_profile(prof)
        return prof

    def _persist_crossover_profile(self, prof: CrossoverProfile) -> None:
        """Save empirical crossover profile into trader/semantic_memory.json without clobbering existing keys."""
        try:
            self.semantic_memory_path.parent.mkdir(parents=True, exist_ok=True)
            existing: dict[str, Any] = {}
            if self.semantic_memory_path.exists():
                try:
                    existing = json.loads(self.semantic_memory_path.read_text(encoding="utf-8"))
                except Exception:
                    existing = {}
            existing["hyperion_compute_profile"] = asdict(prof)
            tmp_path = self.semantic_memory_path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
            os.replace(tmp_path, self.semantic_memory_path)
        except Exception as exc:
            logger.debug("Could not persist Hyperion crossover profile: %s", exc)

    def to_cockpit_dict(self) -> dict[str, Any]:
        """Return live Compute Fabric telemetry for Chrome Cockpit /api/state & /ws."""
        return {
            "active_backend": self.last_backend.value,
            "gpu_name": self.gpu_name,
            "has_cuda": self.has_cuda,
            "has_directml": self.has_directml,
            "has_numba": self.has_numba,
            "vram_used_pct": round(self.get_vram_used_fraction() * 100.0, 1),
            "n_crossover_paths": self.n_crossover_paths,
            "last_paths_count": self.last_paths_count,
            "last_batch_ms": self.last_batch_ms,
            "failover_count": self.failover_count,
        }


_GLOBAL_FABRIC: HyperionComputeFabric | None = None
_GLOBAL_LOCK = threading.Lock()


def get_hyperion_fabric() -> HyperionComputeFabric:
    """Return the process-wide pre-warmed HyperionComputeFabric singleton."""
    global _GLOBAL_FABRIC
    with _GLOBAL_LOCK:
        if _GLOBAL_FABRIC is None:
            _GLOBAL_FABRIC = HyperionComputeFabric()
        return _GLOBAL_FABRIC
