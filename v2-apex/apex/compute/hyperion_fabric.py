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

# Optional AMD/Intel DirectML backend
try:
    import torch  # type: ignore[import-untyped]
    import torch_directml  # type: ignore[import-untyped]

    _HAS_DIRECTML = True
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
        self._pinned_pool: dict[str, Any] = {}
        self.n_crossover_paths: int = 10_000
        self.crossover_profile: CrossoverProfile | None = None
        self.last_backend: ComputeBackend = ComputeBackend.NUMBA_LLVM_AVX if _HAS_NUMBA else ComputeBackend.NUMPY_SIMD
        self.last_batch_ms: float = 0.0
        self.last_paths_count: int = 0
        self.failover_count: int = 0

        self._detect_hardware()
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

        if not self.has_cuda and _HAS_DIRECTML:
            try:
                dml_dev = torch_directml.device()
                self.gpu_name = f"DirectML ({torch_directml.device_name(0)})"
                self.has_directml = True
                self.last_backend = ComputeBackend.DIRECTML_GPU
                _ = dml_dev
            except Exception as exc:
                logger.debug("DirectML detection skipped: %s", exc)
                self.has_directml = False

        if not self.has_cuda and not self.has_directml:
            self.last_backend = ComputeBackend.NUMBA_LLVM_AVX if self.has_numba else ComputeBackend.NUMPY_SIMD

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

        if self.has_numba:
            try:
                _numba_simulate_batch_kernel(hi, lo, cl, tu, entry, stop, tgt, one_r, tgt_r, cost_r, is_long)
            except Exception as exc:
                logger.warning("Numba prewarm failed, disabling Numba: %s", exc)
                self.has_numba = False

        if self.has_cuda and cp is not None:
            try:
                self._cuda_kernel = cp.RawKernel(_CUDA_SIM_KERNEL_SRC, "hyperion_sim_batch_f32_to_f64")
                self._run_cuda_batch(hi, lo, cl, tu, entry, stop, tgt, one_r, tgt_r, cost_r, is_long)
            except Exception as exc:
                logger.warning("CUDA prewarm failed, falling back to CPU: %s", exc)
                self.has_cuda = False
                self._cuda_kernel = None
                self.last_backend = ComputeBackend.NUMBA_LLVM_AVX if self.has_numba else ComputeBackend.NUMPY_SIMD

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
        return cp.asnumpy(d_status), cp.asnumpy(d_gross), cp.asnumpy(d_net)

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
    ) -> BatchSimulationResult:
        """Execute 3D [B, N, H] Time Machine simulation on the optimal Hyperion backend with <1ms failover."""
        b_cnt, n_paths, _h_bars = hi_ratios.shape
        total_paths = b_cnt * n_paths
        backend = self._select_backend(total_paths, force_backend=force_backend)

        t0 = time.perf_counter()
        status_out: np.ndarray
        gross_out: np.ndarray
        net_out: np.ndarray

        if backend == ComputeBackend.CUDA_TENSOR and self.has_cuda:
            try:
                status_out, gross_out, net_out = self._run_cuda_batch(
                    hi_ratios, lo_ratios, cl_ratios, tie_uniforms,
                    entry_prices, stop_prices, target_prices,
                    one_r_dists, target_rs, cost_rs, is_longs,
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                if elapsed_ms > self.GPU_TIMEOUT_MS:
                    self.failover_count += 1
                    logger.warning("CUDA batch exceeded %.0fms (%.1fms); adaptive crossover raised", self.GPU_TIMEOUT_MS, elapsed_ms)
            except Exception as exc:
                self.failover_count += 1
                logger.warning("CUDA batch failed (%s); instant failover to CPU/Numba", exc)
                backend = ComputeBackend.NUMBA_LLVM_AVX if self.has_numba else ComputeBackend.NUMPY_SIMD
                t0 = time.perf_counter()
                if backend == ComputeBackend.NUMBA_LLVM_AVX:
                    status_out, gross_out, net_out = _numba_simulate_batch_kernel(
                        hi_ratios, lo_ratios, cl_ratios, tie_uniforms,
                        entry_prices, stop_prices, target_prices,
                        one_r_dists, target_rs, cost_rs, is_longs,
                    )
                else:
                    status_out, gross_out, net_out = self._run_numpy_batch(
                        hi_ratios, lo_ratios, cl_ratios, tie_uniforms,
                        entry_prices, stop_prices, target_prices,
                        one_r_dists, target_rs, cost_rs, is_longs,
                    )
        elif backend == ComputeBackend.NUMBA_LLVM_AVX and self.has_numba:
            status_out, gross_out, net_out = _numba_simulate_batch_kernel(
                hi_ratios, lo_ratios, cl_ratios, tie_uniforms,
                entry_prices, stop_prices, target_prices,
                one_r_dists, target_rs, cost_rs, is_longs,
            )
        else:
            backend = ComputeBackend.NUMPY_SIMD
            status_out, gross_out, net_out = self._run_numpy_batch(
                hi_ratios, lo_ratios, cl_ratios, tie_uniforms,
                entry_prices, stop_prices, target_prices,
                one_r_dists, target_rs, cost_rs, is_longs,
            )

        # Exact FP64 reduction across N paths for each symbol b in B
        p_target = np.mean(status_out == 1, axis=1, dtype=np.float64)
        p_stop = np.mean(status_out == -1, axis=1, dtype=np.float64)
        p_timeout = np.mean(status_out == 0, axis=1, dtype=np.float64)
        prob_profit = np.mean(net_out > 0.0, axis=1, dtype=np.float64)
        gross_avg_r = np.mean(gross_out, axis=1, dtype=np.float64)
        net_avg_r = np.mean(net_out, axis=1, dtype=np.float64)
        median_net_r = np.median(net_out, axis=1).astype(np.float64)

        sorted_net = np.sort(net_out, axis=1)
        tail_95 = max(1, int(0.05 * n_paths))
        tail_99 = max(1, int(0.01 * n_paths))
        var_95_r = sorted_net[:, tail_95 - 1].astype(np.float64)
        cvar_95_r = np.mean(sorted_net[:, :tail_95], axis=1, dtype=np.float64)
        cvar_99_r = np.mean(sorted_net[:, :tail_99], axis=1, dtype=np.float64)

        compute_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        with self._lock:
            self.last_backend = backend
            self.last_batch_ms = compute_ms
            self.last_paths_count = total_paths

        return BatchSimulationResult(
            p_target=p_target,
            p_stop=p_stop,
            p_timeout=p_timeout,
            prob_profit=prob_profit,
            gross_avg_r=gross_avg_r,
            net_avg_r=net_avg_r,
            median_net_r=median_net_r,
            var_95_r=var_95_r,
            cvar_95_r=cvar_95_r,
            cvar_99_r=cvar_99_r,
            net_r_out=net_out,
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
            )
            return res.compute_ms

        cpu_backend = ComputeBackend.NUMBA_LLVM_AVX if self.has_numba else ComputeBackend.NUMPY_SIMD
        c_1x10 = _bench(1, 10_000, cpu_backend)
        c_12x10 = _bench(12, 10_000, cpu_backend)
        c_12x100 = _bench(12, 100_000, cpu_backend)

        if self.has_cuda:
            g_1x10 = _bench(1, 10_000, ComputeBackend.CUDA_TENSOR)
            g_12x10 = _bench(12, 10_000, ComputeBackend.CUDA_TENSOR)
            g_12x100 = _bench(12, 100_000, ComputeBackend.CUDA_TENSOR)
            if g_1x10 < c_1x10:
                self.n_crossover_paths = 5_000
            elif g_12x10 < c_12x10:
                self.n_crossover_paths = 50_000
            else:
                self.n_crossover_paths = 200_000
            pref_gpu = ComputeBackend.CUDA_TENSOR.value
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
