"""RTX 5090 Hot-Path Mamba-3 / Temporal-KAN Quantile Forecaster (`apex/quant/neural_alpha.py`).

Modular, zero-token, GPU-optional short-horizon quantile forecaster consuming synchronized
multi-resolution OHLCV features:
  - Log-returns across ``M1``, ``M5``, ``M15``, and ``H1`` windows
  - Barndorff-Nielsen & Shephard Realized Bipower Variation (continuous sigma vs jump ratio)
  - Level-2 Order-Flow Imbalance (``ofi`` in [-1, 1])
  - Cross-Asset Macro Contagion Shock Z-score (``contagion_shock_z``)

Runtime execution tiers:
  1. ``TORCH_CUDA``: PyTorch CUDA eager/compiled module on RTX 5090 when ``torch.cuda.is_available()``
  2. ``TORCH_CPU``: PyTorch CPU module when ``torch`` is installed without CUDA
  3. ``NUMPY_SSM_KAN``: Exact vectorized NumPy Selective State-Space + Temporal-KAN fallback when ``torch``
     is not installed (e.g. lean CI environments)
  4. ``DISABLED``: Clean zero-overhead disabled mode when ``enabled=False``
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch  # type: ignore[import-untyped]
    import torch.nn as nn  # type: ignore[import-untyped]

    _HAS_TORCH = True
except Exception:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _HAS_TORCH = False


@dataclass(frozen=True, slots=True)
class MultiResolutionWindow:
    """Synchronized multi-resolution feature bundle for one candidate symbol."""

    symbol: str
    m1_closes: Sequence[float] | np.ndarray = field(default_factory=tuple)
    m5_closes: Sequence[float] | np.ndarray = field(default_factory=tuple)
    m15_closes: Sequence[float] | np.ndarray = field(default_factory=tuple)
    h1_closes: Sequence[float] | np.ndarray = field(default_factory=tuple)
    ofi: float = 0.0
    contagion_shock_z: float = 0.0
    realized_bpv: float | None = None


@dataclass(frozen=True, slots=True)
class NeuralAlphaForecast:
    """Calibrated monotonic quantile return forecast ``Q10 <= Q50 <= Q90`` for a symbol."""

    symbol: str
    q10: float
    q50: float
    q90: float
    interval_width: float
    realized_bpv: float
    jump_ratio: float
    ofi: float
    contagion_shock_z: float
    confidence: float
    backend_used: str
    latency_ms: float
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class NeuralAlphaBatchResult:
    """Batched multi-symbol Mamba-3 / Temporal-KAN quantile output."""

    forecasts: list[NeuralAlphaForecast]
    q10: np.ndarray
    q50: np.ndarray
    q90: np.ndarray
    backend_used: str
    batch_latency_ms: float


def compute_log_returns(closes: Sequence[float] | np.ndarray) -> np.ndarray:
    """Compute finite sanitized log-returns from a sequence of close prices."""
    arr = np.asarray(closes, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    if arr.size < 2:
        return np.zeros((0,), dtype=np.float64)
    ratios = np.clip(arr[1:] / np.maximum(arr[:-1], 1e-12), 1e-6, 1e6)
    return np.log(ratios)


def compute_realized_bipower_variation(log_returns: Sequence[float] | np.ndarray) -> tuple[float, float]:
    """Compute Barndorff-Nielsen & Shephard per-bar continuous BPV sigma and jump ratio.

    Returns
    -------
    (bpv_sigma, jump_ratio)
        - ``bpv_sigma``: sqrt(BPV / N), jump-robust continuous volatility per return step.
        - ``jump_ratio``: ``max(0, RV - BPV) / max(RV, 1e-12)`` in ``[0.0, 1.0]``.
    """
    r = np.asarray(log_returns, dtype=np.float64).ravel()
    r = r[np.isfinite(r)]
    n = int(r.size)
    if n < 2:
        fallback = float(abs(r[0])) if n == 1 else 1e-4
        return max(fallback, 1e-6), 0.0

    rv = float(np.sum(np.square(r)))
    abs_r = np.abs(r)
    bpv_raw = (math.pi / 2.0) * (float(n) / float(n - 1)) * float(np.sum(abs_r[1:] * abs_r[:-1]))
    bpv = max(bpv_raw, 0.0)
    bpv_sigma = math.sqrt(max(bpv / float(n), 1e-12))
    jump_ratio = max(0.0, min(1.0, (rv - bpv) / max(rv, 1e-12)))
    return max(bpv_sigma, 1e-6), jump_ratio


def _tail_pad_returns(log_returns: np.ndarray, seq_len: int) -> np.ndarray:
    """Left-zero-pad or slice trailing ``seq_len`` returns cleanly."""
    out = np.zeros((seq_len,), dtype=np.float32)
    n = int(log_returns.size)
    if n <= 0:
        return out
    take = min(n, seq_len)
    out[-take:] = log_returns[-take:].astype(np.float32)
    return out


def _softplus_np(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _silu_np(x: np.ndarray) -> np.ndarray:
    sig = 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))
    return x * sig


def _structured_model_weights(
    d_in: int = 8, d_state: int = 16, n_knots: int = 5
) -> dict[str, np.ndarray]:
    """Build deterministic econometric Mamba-3 SSM + Temporal-KAN initialization matrices."""
    knots = np.linspace(-1.5, 1.5, n_knots, dtype=np.float32)

    # Selective delta projection gently boosts responsiveness on active flow & multi-TF returns
    w_dt = np.zeros((d_in, d_state), dtype=np.float32)
    w_dt[0:4, :] = 0.15
    w_dt[6, :] = 0.20
    b_dt = np.full((d_state,), -0.60, dtype=np.float32)

    # Diagonal decay rates spanning fast scalp modes (states 0..5) to slow trend modes (states 6..15)
    taus = np.linspace(2.0, 14.0, d_state, dtype=np.float32)
    log_a = np.log(1.0 / taus).astype(np.float32)

    # Input projection mixes (M1, M5, M15, H1) returns + OFI while damping extreme jumps & shock
    w_u = np.zeros((d_in, d_state), dtype=np.float32)
    for k in range(d_state):
        phase = (k + 1.0) / float(d_state)
        w_u[0, k] = 0.22 * (1.0 - 0.3 * phase)   # M1 fast return
        w_u[1, k] = 0.32                         # M5 primary signal
        w_u[2, k] = 0.26 * (0.5 + 0.5 * phase)   # M15 intermediate confirmation
        w_u[3, k] = 0.20 * phase                 # H1 structural drift
        w_u[6, k] = 0.35 * (0.7 + 0.3 * phase)   # Order-Flow Imbalance
    w_b = np.full((d_in, d_state), 0.12, dtype=np.float32)
    w_b[0:4, :] = 0.28
    w_b[7, :] = -0.18  # Contagion shock dampens input gating into directional drift

    # Output selector reads out trend state while reading jump & shock uncertainty
    w_c = np.zeros((d_in, d_state), dtype=np.float32)
    w_c[0:4, :] = 0.20
    w_c[4:6, :] = 0.12
    b_c = np.full((d_state,), 0.25, dtype=np.float32)

    # Temporal-KAN head: maps pooled latent z (d_state,) + macro (4,) -> (raw_mid, raw_down, raw_up)
    z_dim = d_state + 4  # latent SSM state + [bpv_norm, jump_ratio, ofi, shock_norm]
    w_kan_base = np.zeros((z_dim, 3), dtype=np.float32)
    w_kan_base[:d_state, 0] = 0.08 / float(d_state) * 16.0  # aggregate trend -> median q50
    w_kan_base[d_state + 2, 0] = 0.18                       # direct OFI tilt -> q50
    w_kan_base[d_state + 0, 1] = 0.45                       # high vol widens lower quantile spread
    w_kan_base[d_state + 1, 1] = 0.55                       # jump ratio widens lower tail asymmetrically
    w_kan_base[d_state + 3, 1] = 0.65                       # contagion shock widens lower tail risk
    w_kan_base[d_state + 0, 2] = 0.45                       # high vol widens upper quantile spread
    w_kan_base[d_state + 1, 2] = 0.35                       # jump ratio widens upper tail
    w_kan_base[d_state + 3, 2] = 0.30                       # contagion widens upper uncertainty too

    # Radial B-spline coefficients (z_dim, n_knots, 3) capturing smooth non-linear saturation
    w_kan_rbf = np.zeros((z_dim, n_knots, 3), dtype=np.float32)
    for k_idx, c in enumerate(knots):
        w_kan_rbf[:d_state, k_idx, 0] = 0.025 * math.tanh(float(c))
        w_kan_rbf[d_state:, k_idx, 1] = 0.08 * math.exp(-0.5 * float(c) ** 2)
        w_kan_rbf[d_state:, k_idx, 2] = 0.07 * math.exp(-0.5 * float(c) ** 2)

    b_out = np.array([0.0, 0.25, 0.25], dtype=np.float32)

    return {
        "knots": knots,
        "w_dt": w_dt,
        "b_dt": b_dt,
        "log_a": log_a,
        "w_u": w_u,
        "w_b": w_b,
        "w_c": w_c,
        "b_c": b_c,
        "w_kan_base": w_kan_base,
        "w_kan_rbf": w_kan_rbf,
        "b_out": b_out,
    }


if _HAS_TORCH:
    class _TorchMamba3TemporalKAN(nn.Module):
        """PyTorch Mamba-3 Selective State-Space + Temporal-KAN Quantile Module."""

        def __init__(self, weights: dict[str, np.ndarray]) -> None:
            super().__init__()
            self.register_buffer("knots", torch.tensor(weights["knots"], dtype=torch.float32))
            self.w_dt = nn.Parameter(torch.tensor(weights["w_dt"], dtype=torch.float32))
            self.b_dt = nn.Parameter(torch.tensor(weights["b_dt"], dtype=torch.float32))
            self.log_a = nn.Parameter(torch.tensor(weights["log_a"], dtype=torch.float32))
            self.w_u = nn.Parameter(torch.tensor(weights["w_u"], dtype=torch.float32))
            self.w_b = nn.Parameter(torch.tensor(weights["w_b"], dtype=torch.float32))
            self.w_c = nn.Parameter(torch.tensor(weights["w_c"], dtype=torch.float32))
            self.b_c = nn.Parameter(torch.tensor(weights["b_c"], dtype=torch.float32))
            self.w_kan_base = nn.Parameter(torch.tensor(weights["w_kan_base"], dtype=torch.float32))
            self.w_kan_rbf = nn.Parameter(torch.tensor(weights["w_kan_rbf"], dtype=torch.float32))
            self.b_out = nn.Parameter(torch.tensor(weights["b_out"], dtype=torch.float32))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x shape: (B, L, D=8)
            bsz, seq_len, _ = x.shape
            d_state = self.log_a.shape[0]
            h = torch.zeros((bsz, d_state), dtype=x.dtype, device=x.device)
            a_neg = -torch.exp(self.log_a)

            dt_all = torch.nn.functional.softplus(x @ self.w_dt + self.b_dt)
            u_all = torch.nn.functional.silu(x @ self.w_u)
            b_all = torch.tanh(x @ self.w_b)
            c_all = torch.tanh(x @ self.w_c + self.b_c)

            y_last = h
            for t in range(seq_len):
                dt_t = dt_all[:, t, :]
                decay = torch.exp(torch.clamp(a_neg * dt_t, min=-10.0, max=-1e-4))
                h = decay * h + (dt_t * b_all[:, t, :]) * u_all[:, t, :]
                y_last = c_all[:, t, :] * h

            macro = x[:, -1, 4:8]  # [bpv_norm, jump_ratio, ofi, shock_norm]
            z = torch.cat([y_last, macro], dim=-1)  # (B, d_state + 4)

            base_term = torch.nn.functional.silu(z) @ self.w_kan_base
            # RBF B-spline expansion: (B, z_dim, n_knots)
            diff = z.unsqueeze(-1) - self.knots.view(1, 1, -1)
            phi = torch.exp(-0.5 * (diff / 0.75) ** 2)
            rbf_term = torch.einsum("bzk,zko->bo", phi, self.w_kan_rbf)
            return base_term + rbf_term + self.b_out  # (B, 3): [raw_mid, raw_down, raw_up]


class NeuralAlphaForecaster:
    """GPU-optional Mamba-3 / Temporal-KAN Quantile Forecaster for Apex Hyperion Hot Path."""

    QUANTILE_Z_10: float = 1.28155

    def __init__(
        self,
        *,
        enabled: bool = True,
        prefer_gpu: bool = True,
        seq_len: int = 16,
        horizon_bars: int = 6,
        require_torch: bool = False,
        conformal_q_offset: float = 0.0,
    ) -> None:
        self.enabled = bool(enabled)
        self.prefer_gpu = bool(prefer_gpu)
        self.seq_len = max(4, int(seq_len))
        self.horizon_bars = max(1, int(horizon_bars))
        self.require_torch = bool(require_torch)
        self.conformal_q_offset = max(0.0, float(conformal_q_offset))
        self._lock = threading.Lock()
        self._weights = _structured_model_weights(d_in=8, d_state=16, n_knots=5)
        self._torch_model: Any = None
        self._torch_device: Any = None
        self.backend_used: str = "DISABLED"
        self.last_batch_ms: float = 0.0
        self.last_batch_size: int = 0

        self._init_backend()

    def _init_backend(self) -> None:
        if not self.enabled:
            self.backend_used = "DISABLED"
            return

        if _HAS_TORCH and torch is not None:
            try:
                use_cuda = bool(self.prefer_gpu and torch.cuda.is_available())
                dev = torch.device("cuda" if use_cuda else "cpu")
                model = _TorchMamba3TemporalKAN(self._weights).to(dev)
                model.eval()
                with torch.no_grad():
                    _ = model(torch.zeros((1, self.seq_len, 8), dtype=torch.float32, device=dev))
                self._torch_model = model
                self._torch_device = dev
                self.backend_used = "TORCH_CUDA" if use_cuda else "TORCH_CPU"
                return
            except Exception as exc:
                logger.warning("PyTorch NeuralAlpha init failed (%s); falling back", exc)

        if self.require_torch:
            self.enabled = False
            self.backend_used = "DISABLED"
        else:
            self.backend_used = "NUMPY_SSM_KAN"

    def _build_batch_tensors(
        self, windows: Sequence[MultiResolutionWindow]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Encode B multi-resolution windows into normalized (B, L, 8) feature tensor."""
        b_cnt = len(windows)
        l_seq = self.seq_len
        x_arr = np.zeros((b_cnt, l_seq, 8), dtype=np.float32)
        bpv_arr = np.empty((b_cnt,), dtype=np.float64)
        jump_arr = np.empty((b_cnt,), dtype=np.float64)
        ofi_arr = np.empty((b_cnt,), dtype=np.float64)
        shock_arr = np.empty((b_cnt,), dtype=np.float64)

        for i, win in enumerate(windows):
            r_m1 = compute_log_returns(win.m1_closes)
            r_m5 = compute_log_returns(win.m5_closes)
            r_m15 = compute_log_returns(win.m15_closes)
            r_h1 = compute_log_returns(win.h1_closes)

            base_r = r_m5 if r_m5.size >= 4 else (r_m1 if r_m1.size >= 2 else r_h1)
            bpv_sigma, jump_ratio = compute_realized_bipower_variation(base_r)
            if win.realized_bpv is not None and math.isfinite(float(win.realized_bpv)) and float(win.realized_bpv) > 0.0:
                bpv_sigma = max(float(win.realized_bpv), 1e-6)

            inv_scale = 1.0 / max(bpv_sigma, 1e-5)
            x_arr[i, :, 0] = np.clip(_tail_pad_returns(r_m1, l_seq) * inv_scale, -6.0, 6.0)
            x_arr[i, :, 1] = np.clip(_tail_pad_returns(r_m5, l_seq) * inv_scale, -6.0, 6.0)
            x_arr[i, :, 2] = np.clip(_tail_pad_returns(r_m15, l_seq) * inv_scale, -6.0, 6.0)
            x_arr[i, :, 3] = np.clip(_tail_pad_returns(r_h1, l_seq) * inv_scale, -6.0, 6.0)

            ofi_clamped = max(-1.0, min(1.0, float(win.ofi) if math.isfinite(float(win.ofi)) else 0.0))
            shock_val = float(win.contagion_shock_z) if math.isfinite(float(win.contagion_shock_z)) else 0.0
            bpv_norm = math.tanh(bpv_sigma * 250.0)
            shock_norm = math.tanh(abs(shock_val) / 4.0)

            x_arr[i, :, 4] = bpv_norm
            x_arr[i, :, 5] = jump_ratio
            x_arr[i, :, 6] = ofi_clamped
            x_arr[i, :, 7] = shock_norm

            bpv_arr[i] = bpv_sigma
            jump_arr[i] = jump_ratio
            ofi_arr[i] = ofi_clamped
            shock_arr[i] = shock_val

        return x_arr, bpv_arr, jump_arr, ofi_arr, shock_arr

    def _forward_numpy(self, x: np.ndarray) -> np.ndarray:
        """Vectorized NumPy Selective State-Space + Temporal-KAN forward pass matching PyTorch module."""
        bsz, seq_len, _ = x.shape
        w = self._weights
        d_state = w["log_a"].shape[0]
        h = np.zeros((bsz, d_state), dtype=np.float32)
        a_neg = -np.exp(w["log_a"])

        dt_all = _softplus_np(x @ w["w_dt"] + w["b_dt"])
        u_all = _silu_np(x @ w["w_u"])
        b_all = np.tanh(x @ w["w_b"])
        c_all = np.tanh(x @ w["w_c"] + w["b_c"])

        y_last = h
        for t in range(seq_len):
            dt_t = dt_all[:, t, :]
            decay = np.exp(np.clip(a_neg * dt_t, -10.0, -1e-4))
            h = decay * h + (dt_t * b_all[:, t, :]) * u_all[:, t, :]
            y_last = c_all[:, t, :] * h

        macro = x[:, -1, 4:8]
        z = np.concatenate([y_last, macro], axis=-1)

        base_term = _silu_np(z) @ w["w_kan_base"]
        diff = z[:, :, None] - w["knots"][None, None, :]
        phi = np.exp(-0.5 * np.square(diff / 0.75))
        rbf_term = np.einsum("bzk,zko->bo", phi, w["w_kan_rbf"])
        return base_term + rbf_term + w["b_out"]

    def update_conformal_calibration(
        self,
        realized_returns: Sequence[float] | np.ndarray,
        pred_q10: Sequence[float] | np.ndarray,
        pred_q90: Sequence[float] | np.ndarray,
        alpha: float = 0.20,
    ) -> float:
        """Fit finite-sample split-conformal symmetric interval offset for target (1 - alpha) coverage."""
        y = np.asarray(realized_returns, dtype=np.float64)
        lo = np.asarray(pred_q10, dtype=np.float64)
        hi = np.asarray(pred_q90, dtype=np.float64)
        n = min(y.size, lo.size, hi.size)
        if n < 5:
            return self.conformal_q_offset
        scores = np.maximum(lo[:n] - y[:n], y[:n] - hi[:n])
        q_idx = min(n - 1, max(0, int(math.ceil((1.0 - alpha) * (n + 1))) - 1))
        offset = max(0.0, float(np.partition(scores, q_idx)[q_idx]))
        with self._lock:
            self.conformal_q_offset = offset
        return offset

    def forecast_batch(self, windows: Sequence[MultiResolutionWindow]) -> NeuralAlphaBatchResult:
        """Compute calibrated `(Q10, Q50, Q90)` return quantiles for B symbols in one batch pass."""
        t0 = time.perf_counter()
        b_cnt = len(windows)
        if b_cnt == 0:
            return NeuralAlphaBatchResult(
                forecasts=[],
                q10=np.zeros((0,), dtype=np.float64),
                q50=np.zeros((0,), dtype=np.float64),
                q90=np.zeros((0,), dtype=np.float64),
                backend_used=self.backend_used,
                batch_latency_ms=0.0,
            )

        if not self.enabled or self.backend_used == "DISABLED":
            forecasts = [
                NeuralAlphaForecast(
                    symbol=win.symbol,
                    q10=0.0,
                    q50=0.0,
                    q90=0.0,
                    interval_width=0.0,
                    realized_bpv=0.0,
                    jump_ratio=0.0,
                    ofi=float(win.ofi),
                    contagion_shock_z=float(win.contagion_shock_z),
                    confidence=0.0,
                    backend_used="DISABLED",
                    latency_ms=0.0,
                    enabled=False,
                )
                for win in windows
            ]
            zeros = np.zeros((b_cnt,), dtype=np.float64)
            return NeuralAlphaBatchResult(
                forecasts=forecasts,
                q10=zeros,
                q50=zeros.copy(),
                q90=zeros.copy(),
                backend_used="DISABLED",
                batch_latency_ms=0.0,
            )

        x_arr, bpv_arr, jump_arr, ofi_arr, shock_arr = self._build_batch_tensors(windows)
        if self._torch_model is not None and torch is not None:
            with torch.no_grad():
                xt = torch.as_tensor(x_arr, dtype=torch.float32, device=self._torch_device)
                raw = self._torch_model(xt).detach().cpu().numpy().astype(np.float64)
        else:
            raw = self._forward_numpy(x_arr).astype(np.float64)

        horizon_scale = bpv_arr * math.sqrt(float(self.horizon_bars))
        # Shock attenuation shrinks median directional tilt toward zero during contagion spikes
        shock_attenuation = 1.0 / (1.0 + 0.15 * np.square(np.maximum(0.0, np.abs(shock_arr) - 1.5)))
        q50 = np.tanh(raw[:, 0]) * horizon_scale * 0.85 * shock_attenuation

        down_spread = (_softplus_np(raw[:, 1]) + self.QUANTILE_Z_10 * 0.75) * horizon_scale + self.conformal_q_offset
        up_spread = (_softplus_np(raw[:, 2]) + self.QUANTILE_Z_10 * 0.75) * horizon_scale + self.conformal_q_offset

        q10 = q50 - np.maximum(down_spread, 1e-7)
        q90 = q50 + np.maximum(up_spread, 1e-7)
        widths = np.maximum(q90 - q10, 1e-7)
        # Confidence normalized via directional signal-to-spread ratio & penalized by jump/shock uncertainty
        snr = np.abs(q50) / widths
        conf = np.clip(snr * (1.0 - 0.35 * jump_arr) * shock_attenuation * 2.0, 0.05, 0.99)

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        per_sym_ms = round(elapsed_ms / float(b_cnt), 3)

        forecasts = [
            NeuralAlphaForecast(
                symbol=windows[i].symbol,
                q10=round(float(q10[i]), 7),
                q50=round(float(q50[i]), 7),
                q90=round(float(q90[i]), 7),
                interval_width=round(float(widths[i]), 7),
                realized_bpv=round(float(bpv_arr[i]), 7),
                jump_ratio=round(float(jump_arr[i]), 4),
                ofi=round(float(ofi_arr[i]), 4),
                contagion_shock_z=round(float(shock_arr[i]), 4),
                confidence=round(float(conf[i]), 4),
                backend_used=self.backend_used,
                latency_ms=per_sym_ms,
                enabled=True,
            )
            for i in range(b_cnt)
        ]

        with self._lock:
            self.last_batch_ms = elapsed_ms
            self.last_batch_size = b_cnt

        return NeuralAlphaBatchResult(
            forecasts=forecasts,
            q10=q10,
            q50=q50,
            q90=q90,
            backend_used=self.backend_used,
            batch_latency_ms=elapsed_ms,
        )

    def forecast_symbol(self, window: MultiResolutionWindow) -> NeuralAlphaForecast:
        """Compute calibrated `(Q10, Q50, Q90)` return quantiles for a single symbol."""
        return self.forecast_batch([window]).forecasts[0]

    def to_cockpit_dict(self) -> dict[str, Any]:
        """Return live telemetry for Chrome Cockpit / TUI monitoring."""
        return {
            "enabled": self.enabled,
            "backend": self.backend_used,
            "last_batch_ms": self.last_batch_ms,
            "last_batch_size": self.last_batch_size,
            "conformal_q_offset": round(self.conformal_q_offset, 7),
        }
