"""10,000-Path Stationary Block-Bootstrap + Cost-Adjusted Time Machine (`apex/quant/timemachine_v2.py`).

Upgrades the Monte Carlo replay engine from naive single-bar shuffling to Politis & Romano (1994)
Stationary Block Bootstrap (geometric block lengths preserving multi-bar volatility clustering across
60 days of M5/M15 sessions), deducts realistic round-trip execution friction on EVERY simulated path
(broker commission such as Fusion Zero $4.50/lot round-turn + live VWAP DOM spread/slippage), and
computes:
  - Gross and Net Expected Value in R (`gross_avg_R`, `net_avg_R`)
  - Net Probability of Profit (`prob_profit`)
  - 95% Conditional Value-at-Risk (`cvar_95_R` — expected tail loss in worst 5% of paths)
  - Quarter-Kelly optimal risk fraction capped strictly at `0.5%` (`0.005`) max equity risk
    (`$50` on a `$10,000` account).
"""

from __future__ import annotations

import math
import threading
import zlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats.qmc import Sobol

from apex.compute.hyperion_fabric import get_hyperion_fabric
from apex.feeds.models import Candle, Side

_SOBOL_CACHE_LOCK = threading.Lock()
_SOBOL_POINT_CACHE: dict[tuple[int, int, int], np.ndarray] = {}


def _get_cached_sobol_points(half_n: int, sobol_dim: int, seed: int) -> np.ndarray:
    """Return (half_n, sobol_dim) Owen-scrambled Sobol points drawn strictly with power-of-2 count m."""
    m = int(math.ceil(math.log2(max(2, half_n))))
    key = (m, sobol_dim, int(seed))
    with _SOBOL_CACHE_LOCK:
        pts = _SOBOL_POINT_CACHE.get(key)
        if pts is None:
            sobol = Sobol(d=sobol_dim, scramble=True, seed=int(seed))
            pts = sobol.random_base2(m=m)
            if len(_SOBOL_POINT_CACHE) >= 64:
                _SOBOL_POINT_CACHE.pop(next(iter(_SOBOL_POINT_CACHE)))
            _SOBOL_POINT_CACHE[key] = pts
    return pts[:half_n]


@dataclass(frozen=True, slots=True)
class TimeMachineCostModel:
    """Round-trip institutional friction model applied to every simulated path.

    Attributes:
        commission_per_lot_round_turn_usd: Broker round-turn commission per 1.00 standard lot
            (default $4.50/lot for Fusion Markets Zero).
        lots: Position size in standard lots (e.g., 1.0 lot).
        risk_per_lot_usd: Monetary risk at 1R Stop-Loss per 1.00 standard lot in USD.
            Used to convert dollar commissions into exact R-multiples:
            ``commission_R = commission_per_lot_round_turn_usd / risk_per_lot_usd``.
        spread_price: Current bid-ask spread in price units.
        vwap_slippage_price: Expected round-trip VWAP DOM slippage in price units (entry + exit).
        slippage_calibration_multiplier: Adaptive execution slippage multiplier in [0.5, 3.0].
        asymmetric_barriers: Apply exact Ask0/Bid_t asymmetric barrier tightening inside path replay.
    """

    commission_per_lot_round_turn_usd: float = 4.50
    lots: float = 1.0
    risk_per_lot_usd: float = 100.0
    spread_price: float = 0.00010
    vwap_slippage_price: float = 0.00004
    slippage_calibration_multiplier: float = 1.0
    asymmetric_barriers: bool = False

    def compute_round_trip_cost_r(self, one_r_price_distance: float) -> float:
        """Compute total round-trip friction (commission + spread + calibrated VWAP DOM slippage) in R-multiples."""
        if one_r_price_distance <= 1e-12:
            raise ValueError(f"one_r_price_distance must be > 0, got {one_r_price_distance}")
        if self.risk_per_lot_usd <= 1e-12:
            raise ValueError(f"risk_per_lot_usd must be > 0, got {self.risk_per_lot_usd}")

        commission_r = max(0.0, float(self.commission_per_lot_round_turn_usd)) / float(self.risk_per_lot_usd)
        cal_mult = max(0.5, min(3.0, float(self.slippage_calibration_multiplier)))
        calibrated_slip = max(0.0, float(self.vwap_slippage_price)) * cal_mult
        microstructure_price_cost = max(0.0, float(self.spread_price)) + calibrated_slip
        microstructure_r = microstructure_price_cost / float(one_r_price_distance)
        return float(commission_r + microstructure_r)


@dataclass(frozen=True, slots=True)
class TimeMachineV2Result:
    """Comprehensive output of the 10,000-path Block-Bootstrap Cost-Adjusted Time Machine."""

    symbol: str
    side: Side
    entry_price: float
    stop_price: float
    target_price: float
    target_r: float
    replays: int
    horizon_bars: int
    mean_block_length: float
    round_trip_cost_r: float
    commission_cost_r: float
    spread_slippage_cost_r: float
    p_target: float
    p_stop: float
    p_timeout: float
    prob_profit: float
    gross_avg_R: float
    net_avg_R: float
    median_net_R: float
    cvar_95_R: float
    var_95_R: float
    full_kelly_pct: float
    quarter_kelly_pct: float
    recommended_risk_pct: float
    recommended_risk_usd: float
    account_equity_usd: float

    @property
    def net_avg_r(self) -> float:
        """Lowercase alias for `net_avg_R`."""
        return self.net_avg_R


class BlockBootstrapTimeMachine:
    """10,000-Path Stationary Block-Bootstrap Monte Carlo Engine (`apex/quant/timemachine_v2.py`)."""

    MAX_EQUITY_RISK_PCT: float = 0.5  # Strict 0.5% ($50 on $10,000) institutional risk cap

    def __init__(
        self,
        replays: int = 8_192,
        mean_block_length: float = 6.0,
        seed: int = 19,
        account_equity_usd: float = 10_000.0,
        max_risk_pct: float = 0.5,
        use_qmc_sobol: bool = True,
    ) -> None:
        if replays < 100:
            raise ValueError(f"replays must be >= 100, got {replays}")
        if mean_block_length < 1.0:
            raise ValueError(f"mean_block_length must be >= 1.0, got {mean_block_length}")

        self.replays: int = int(replays)
        self.mean_block_length: float = float(mean_block_length)
        self.seed: int = int(seed)
        self.account_equity_usd: float = float(account_equity_usd)
        self.max_risk_pct: float = min(float(max_risk_pct), self.MAX_EQUITY_RISK_PCT)
        self.use_qmc_sobol: bool = bool(use_qmc_sobol)

    @staticmethod
    def build_relative_bar_ratios(candles: Sequence[Candle]) -> np.ndarray:
        """Convert historical M5/M15 candles into contiguous relative bar ratios ``(o/pc, h/pc, l/pc, c/pc)``."""
        if len(candles) < 10:
            raise ValueError(f"Need at least 10 historical candles for block bootstrap, got {len(candles)}")

        n = len(candles) - 1
        closes = np.asarray([float(c.close) for c in candles], dtype=np.float64)
        opens = np.asarray([float(c.open) for c in candles[1:]], dtype=np.float64)
        highs = np.asarray([float(c.high) for c in candles[1:]], dtype=np.float64)
        lows = np.asarray([float(c.low) for c in candles[1:]], dtype=np.float64)
        cur_closes = closes[1:]
        prev_closes = np.where(closes[:-1] > 0.0, closes[:-1], np.maximum(opens, 1e-12))

        ratios = np.empty((n, 4), dtype=np.float64)
        ratios[:, 0] = opens / prev_closes
        ratios[:, 1] = highs / prev_closes
        ratios[:, 2] = lows / prev_closes
        ratios[:, 3] = cur_closes / prev_closes
        return ratios

    @classmethod
    def build_regime_conditional_garch_ratios(
        cls,
        candles: Sequence[Candle],
        current_hurst: float | None = None,
        current_er: float | None = None,
        current_vol_pct: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Apply intraday TOD deseasonalization, EWMA/GARCH vol-filtering, and regime kernel weighting."""
        ratios = cls.build_relative_bar_ratios(candles)
        n = len(ratios)
        if n < 60:
            return ratios, None

        log_rets = np.log(np.maximum(ratios[:, 3], 1e-12))
        timestamps = np.asarray([float(getattr(c, "timestamp_utc", 0.0) or 0.0) for c in candles[1:]], dtype=np.float64)

        # 1. Time-of-Day intraday volatility deseasonalization across 15m UTC buckets (96/day)
        tod_scale = np.ones(n, dtype=np.float64)
        if np.all(timestamps > 0.0):
            buckets = ((timestamps % 86400.0) // 900.0).astype(np.int64)
            abs_r = np.abs(log_rets)
            overall_med = max(float(np.median(abs_r)), 1e-8)
            bucket_meds = np.full(96, overall_med, dtype=np.float64)
            for b in range(96):
                mask = buckets == b
                if int(np.sum(mask)) >= 5:
                    bucket_meds[b] = max(float(np.median(abs_r[mask])), 1e-8)
            tod_raw = bucket_meds[np.clip(buckets, 0, 95)] / overall_med
            tod_scale = np.clip(tod_raw, 0.5, 2.0)

        deseason = log_rets / tod_scale

        # 2. Rolling GARCH(1,1) / EWMA conditional volatility normalization
        mean_r = float(np.mean(deseason))
        eps = deseason - mean_r
        lr_var = max(float(np.var(eps)), 1e-12)
        alpha_g, beta_g = 0.09, 0.90
        omega_g = max((1.0 - alpha_g - beta_g) * lr_var, 1e-14)
        cond_var = np.empty(n, dtype=np.float64)
        cond_var[0] = lr_var
        for i in range(1, n):
            cond_var[i] = omega_g + alpha_g * (eps[i - 1] ** 2) + beta_g * cond_var[i - 1]
        cond_vol = np.sqrt(np.maximum(cond_var, 1e-12))
        target_vol = float(cond_vol[-1])
        cur_tod = float(tod_scale[-1])
        vol_mult = np.clip((target_vol * cur_tod) / np.maximum(cond_vol * tod_scale, 1e-12), 0.55, 1.85)

        filtered = ratios.copy()
        # Rescale candle bar excursion around 1.0 proportionally while guaranteeing H >= O,C >= L > 0
        for col in range(4):
            filtered[:, col] = np.maximum(1.0 + (ratios[:, col] - 1.0) * vol_mult, 1e-6)
        filtered[:, 1] = np.maximum(filtered[:, 1], np.maximum(filtered[:, 0], filtered[:, 3]))
        filtered[:, 2] = np.minimum(filtered[:, 2], np.minimum(filtered[:, 0], filtered[:, 3]))

        # 3. Gaussian Kernel Regime-Similarity Block Weights K(||x_t - x_now||)
        weights: np.ndarray | None = None
        if n >= 120:
            win = 20
            local_vol = np.convolve(np.abs(log_rets), np.ones(win) / float(win), mode="same")
            local_net = np.abs(np.convolve(log_rets, np.ones(win), mode="same"))
            local_abs_sum = np.maximum(local_vol * float(win), 1e-12)
            local_er = np.clip(local_net / local_abs_sum, 0.0, 1.0)

            now_vol = (float(np.percentile(local_vol, np.clip(current_vol_pct, 0.0, 100.0)))
                       if current_vol_pct is not None else float(local_vol[-1]))
            now_er = float(current_er) if current_er is not None else float(local_er[-1])

            v_std = max(float(np.std(local_vol)), 1e-8)
            e_std = max(float(np.std(local_er)), 1e-4)
            dist_sq = ((local_vol - now_vol) / v_std) ** 2 + ((local_er - now_er) / e_std) ** 2
            # Smooth kernel blend with uniform floor so block diversity remains high
            k_raw = np.exp(-0.5 * np.minimum(dist_sq, 16.0))
            k_norm = 0.70 * (k_raw / max(float(np.sum(k_raw)), 1e-12)) + 0.30 / float(n)
            weights = k_norm / float(np.sum(k_norm))

        return filtered, weights

    def generate_stationary_block_indices(
        self,
        pool_length: int,
        horizon_bars: int,
        rng: np.random.Generator,
        use_qmc_sobol: bool | None = None,
        block_weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """Generate ``(replays, horizon_bars)`` indices via Politis & Romano Stationary Block Bootstrap.

        Uses cached power-of-2 Owen-Scrambled Sobol Low-Discrepancy QMC paired with
        Antithetic Coupling where mirror pairs share identical block-restart boundaries
        and invert only start/jump innovation ranks (``1 - u_block``).
        """
        qmc_on = self.use_qmc_sobol if use_qmc_sobol is None else bool(use_qmc_sobol)
        p_restart = 1.0 / max(self.mean_block_length, 1.0)
        indices = np.empty((self.replays, horizon_bars), dtype=np.int64)

        weight_cdf: np.ndarray | None = None
        if block_weights is not None and len(block_weights) == pool_length:
            w = np.maximum(np.asarray(block_weights, dtype=np.float64), 0.0)
            s = float(np.sum(w))
            if s > 0.0:
                weight_cdf = np.cumsum(w / s)
                weight_cdf[-1] = 1.0

        def _u_to_idx(u_arr: np.ndarray) -> np.ndarray:
            if weight_cdf is None:
                return np.clip((u_arr * pool_length).astype(np.int64), 0, pool_length - 1)
            return np.clip(np.searchsorted(weight_cdf, u_arr, side="right"), 0, pool_length - 1)

        if qmc_on:
            half_n = (self.replays + 1) // 2
            sobol_dim = min(max(1, horizon_bars), 64)
            u_half = _get_cached_sobol_points(half_n=half_n, sobol_dim=sobol_dim, seed=self.seed)
            if horizon_bars > sobol_dim:
                extra = rng.random((half_n, horizon_bars - sobol_dim))
                u_half = np.hstack([u_half, extra])

            # Share identical restart indicators across antithetic pairs; invert innovation rank (1 - u)
            u_pos_half = u_half
            u_pos_anti = 1.0 - u_half
            u_pos_all = np.vstack([u_pos_half, u_pos_anti])[: self.replays]

            indices[:, 0] = _u_to_idx(u_pos_all[:, 0])
            if horizon_bars > 1:
                # Dedicated shared restart draws + inverted jump rank on mirror pairs
                r_half = rng.random((half_n, horizon_bars - 1)) < p_restart
                restarts = np.vstack([r_half, r_half])[: self.replays]
                random_jumps = _u_to_idx(u_pos_all[:, 1:])
                for t in range(1, horizon_bars):
                    prev_idx = indices[:, t - 1]
                    cont_idx = (prev_idx + 1) % pool_length
                    indices[:, t] = np.where(restarts[:, t - 1], random_jumps[:, t - 1], cont_idx)
            return indices

        if weight_cdf is None:
            indices[:, 0] = rng.integers(0, pool_length, size=self.replays)
        else:
            indices[:, 0] = _u_to_idx(rng.random(self.replays))
        if horizon_bars > 1:
            restarts = rng.random((self.replays, horizon_bars - 1)) < p_restart
            if weight_cdf is None:
                random_jumps = rng.integers(0, pool_length, size=(self.replays, horizon_bars - 1))
            else:
                random_jumps = _u_to_idx(rng.random((self.replays, horizon_bars - 1)))
            for t in range(1, horizon_bars):
                prev_idx = indices[:, t - 1]
                cont_idx = (prev_idx + 1) % pool_length
                indices[:, t] = np.where(restarts[:, t - 1], random_jumps[:, t - 1], cont_idx)

        return indices

    @classmethod
    def compute_quarter_kelly_risk(
        cls,
        net_r_outcomes: np.ndarray,
        max_risk_pct: float = 0.5,
        account_equity_usd: float = 10_000.0,
    ) -> tuple[float, float, float, float]:
        """Compute Full Kelly %, Quarter-Kelly %, Capped Risk %, and Capped Risk USD from net R outcomes.

        Uses the continuous-outcome Kelly formulation:
        - Let $p = P(R_{\\text{net}} > 0)$, $q = 1 - p$,
          $W = \\mathbb{E}[R_{\\text{net}} \\mid R_{\\text{net}} > 0]$,
          $L = \\mathbb{E}[|R_{\\text{net}}| \\mid R_{\\text{net}} \\le 0]$.
        - Payoff odds ratio $b = W / L$.
        - Full Kelly fraction $f^* = p - q / b = \\frac{p \\cdot W - q \\cdot L}{W}$ (or zero if $\\mathbb{E}[R_{\\text{net}}] \\le 0$).
        - Quarter-Kelly fraction $f_{0.25} = 0.25 \\cdot f^*$ (expressed as % of equity),
          strictly capped at $\\min(\\text{max\\_risk\\_pct}, 0.5\\%)$.
        """
        mean_net_r = float(np.mean(net_r_outcomes))
        if mean_net_r <= 0.0:
            return 0.0, 0.0, 0.0, 0.0

        wins = net_r_outcomes[net_r_outcomes > 0.0]
        losses = net_r_outcomes[net_r_outcomes <= 0.0]
        if len(wins) == 0:
            return 0.0, 0.0, 0.0, 0.0
        if len(losses) == 0:
            cap_pct = min(float(max_risk_pct), cls.MAX_EQUITY_RISK_PCT)
            return 100.0, 25.0, cap_pct, round(account_equity_usd * (cap_pct / 100.0), 2)

        p_win = float(len(wins)) / float(len(net_r_outcomes))
        q_loss = 1.0 - p_win
        avg_win_r = float(np.mean(wins))
        avg_loss_r = max(float(np.mean(np.abs(losses))), 1e-6)
        b_odds = avg_win_r / avg_loss_r

        # Standard Kelly fraction expressed as percentage of bankroll (scaled so 1R = f% of equity)
        # In R-multiple space, variance-optimal Kelly fraction is E[R] / E[R^2] or edge/odds:
        second_moment = max(float(np.mean(np.square(net_r_outcomes))), 1e-6)
        kelly_edge_odds = max(0.0, p_win - (q_loss / max(b_odds, 1e-6)))
        kelly_continuous = max(0.0, mean_net_r / second_moment)
        # Use conservative minimum of edge/odds and mean/second-moment as % of account per trade
        full_kelly_pct = float(min(kelly_edge_odds, kelly_continuous) * 100.0)
        quarter_kelly_pct = 0.25 * full_kelly_pct

        strict_cap_pct = min(float(max_risk_pct), cls.MAX_EQUITY_RISK_PCT)
        recommended_pct = float(min(quarter_kelly_pct, strict_cap_pct))
        recommended_usd = round(float(account_equity_usd) * (recommended_pct / 100.0), 2)
        return full_kelly_pct, quarter_kelly_pct, recommended_pct, recommended_usd

    def simulate_from_ratios(
        self,
        symbol: str,
        side: Side,
        entry_price: float,
        stop_price: float,
        target_price: float,
        bar_ratios: np.ndarray,
        horizon_bars: int = 48,
        cost_model: TimeMachineCostModel | None = None,
        block_weights: np.ndarray | None = None,
        bar_ts: float = 0.0,
    ) -> TimeMachineV2Result:
        """Run Stationary Block-Bootstrap paths and deduct round-trip costs on every path."""
        if bar_ratios.ndim != 2 or bar_ratios.shape[1] != 4 or len(bar_ratios) < 5:
            raise ValueError("bar_ratios must be an (N, 4) array with N >= 5.")

        is_long = side == Side.BUY
        one_r_dist = (float(entry_price) - float(stop_price)) if is_long else (float(stop_price) - float(entry_price))
        if one_r_dist <= 0.0:
            raise ValueError(
                f"Invalid stop_price={stop_price} relative to entry_price={entry_price} for side={side.value}"
            )

        target_dist = (float(target_price) - float(entry_price)) if is_long else (float(entry_price) - float(target_price))
        if target_dist <= 0.0:
            raise ValueError(
                f"Invalid target_price={target_price} relative to entry_price={entry_price} for side={side.value}"
            )

        cm = cost_model if cost_model is not None else TimeMachineCostModel()
        cal_mult = max(0.5, min(3.0, float(getattr(cm, "slippage_calibration_multiplier", 1.0))))
        calibrated_slip = max(0.0, float(cm.vwap_slippage_price)) * cal_mult
        commission_r = max(0.0, float(cm.commission_per_lot_round_turn_usd)) / max(float(cm.risk_per_lot_usd), 1e-9)
        asym_on = bool(getattr(cm, "asymmetric_barriers", False)) and cm.spread_price > 0.0
        # When asymmetric_barriers models bid-ask spread in barrier geometry, do NOT subtract spread_price a second time
        spread_cost_px = 0.0 if asym_on else max(0.0, float(cm.spread_price))
        spread_slip_r = (spread_cost_px + calibrated_slip) / one_r_dist
        round_trip_cost_r = commission_r + spread_slip_r

        # Optional asymmetric Bid/Ask barrier geometry:
        # BUY entered at Ask0 with SL/TP on Bid_t: SL distance tightens by S_t/2, TP distance widens by S_t/2
        eff_stop = float(stop_price)
        eff_target = float(target_price)
        if asym_on:
            half_s = 0.5 * float(cm.spread_price)
            eff_sl_dist = max(one_r_dist - half_s, 1e-9)
            eff_tp_dist = target_dist + half_s
            if is_long:
                eff_stop = float(entry_price) - eff_sl_dist
                eff_target = float(entry_price) + eff_tp_dist
            else:
                eff_stop = float(entry_price) + eff_sl_dist
                eff_target = float(entry_price) - eff_tp_dist
        target_r = float(target_dist / one_r_dist)

        eff_seed = (
            (int(self.seed) ^ zlib.crc32(f"{symbol.upper()}:{int(round(float(bar_ts)))}".encode("utf-8"))) & 0x7FFFFFFF
            if float(bar_ts) > 0.0
            else int(self.seed)
        )
        rng = np.random.default_rng(eff_seed)
        prev_seed = self.seed
        self.seed = eff_seed
        try:
            idx_matrix = self.generate_stationary_block_indices(
                pool_length=len(bar_ratios),
                horizon_bars=int(horizon_bars),
                rng=rng,
                block_weights=block_weights,
            )
        finally:
            self.seed = prev_seed

        # Vectorized 3D [1, N, H] path evaluation via HyperionComputeFabric (GPU / Numba LLVM / NumPy)
        sampled_high_ratios = bar_ratios[idx_matrix, 1][np.newaxis, :, :]
        sampled_low_ratios = bar_ratios[idx_matrix, 2][np.newaxis, :, :]
        sampled_close_ratios = bar_ratios[idx_matrix, 3][np.newaxis, :, :]
        tie_uniforms = rng.random((1, self.replays, int(horizon_bars)))

        fabric = get_hyperion_fabric()
        batch_res = fabric.simulate_paths_batch(
            hi_ratios=sampled_high_ratios,
            lo_ratios=sampled_low_ratios,
            cl_ratios=sampled_close_ratios,
            tie_uniforms=tie_uniforms,
            entry_prices=np.array([float(entry_price)], dtype=np.float64),
            stop_prices=np.array([float(eff_stop)], dtype=np.float64),
            target_prices=np.array([float(eff_target)], dtype=np.float64),
            one_r_dists=np.array([float(one_r_dist)], dtype=np.float64),
            target_rs=np.array([float(target_r)], dtype=np.float64),
            cost_rs=np.array([float(round_trip_cost_r)], dtype=np.float64),
            is_longs=np.array([is_long], dtype=np.bool_),
        )

        p_target = float(batch_res.p_target[0])
        p_stop = float(batch_res.p_stop[0])
        p_timeout = float(batch_res.p_timeout[0])
        prob_profit = float(batch_res.prob_profit[0])

        gross_avg_r = float(batch_res.gross_avg_r[0])
        net_avg_r = float(batch_res.net_avg_r[0])
        median_net_r = float(batch_res.median_net_r[0])

        var_95_r = float(batch_res.var_95_r[0])
        cvar_95_r = float(batch_res.cvar_95_r[0])
        net_r = batch_res.net_r_out[0]

        full_k, quarter_k, rec_pct, rec_usd = self.compute_quarter_kelly_risk(
            net_r_outcomes=net_r,
            max_risk_pct=self.max_risk_pct,
            account_equity_usd=self.account_equity_usd,
        )

        return TimeMachineV2Result(
            symbol=symbol.upper(),
            side=side,
            entry_price=float(entry_price),
            stop_price=float(stop_price),
            target_price=float(target_price),
            target_r=target_r,
            replays=self.replays,
            horizon_bars=int(horizon_bars),
            mean_block_length=self.mean_block_length,
            round_trip_cost_r=round_trip_cost_r,
            commission_cost_r=commission_r,
            spread_slippage_cost_r=spread_slip_r,
            p_target=p_target,
            p_stop=p_stop,
            p_timeout=p_timeout,
            prob_profit=prob_profit,
            gross_avg_R=gross_avg_r,
            net_avg_R=net_avg_r,
            median_net_R=median_net_r,
            cvar_95_R=cvar_95_r,
            var_95_R=var_95_r,
            full_kelly_pct=full_k,
            quarter_kelly_pct=quarter_k,
            recommended_risk_pct=rec_pct,
            recommended_risk_usd=rec_usd,
            account_equity_usd=self.account_equity_usd,
        )

    def simulate_candles(
        self,
        symbol: str,
        side: Side,
        entry_price: float,
        stop_price: float,
        target_price: float,
        candles: Sequence[Candle],
        horizon_bars: int = 48,
        cost_model: TimeMachineCostModel | None = None,
        current_hurst: float | None = None,
        current_er: float | None = None,
        current_vol_pct: float | None = None,
    ) -> TimeMachineV2Result:
        """Run Stationary Block-Bootstrap paths directly from historical `Candle` objects."""
        ratios, block_weights = self.build_regime_conditional_garch_ratios(
            candles,
            current_hurst=current_hurst,
            current_er=current_er,
            current_vol_pct=current_vol_pct,
        )
        bar_ts = float(candles[-1].timestamp_utc) if candles else 0.0
        return self.simulate_from_ratios(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            bar_ratios=ratios,
            horizon_bars=horizon_bars,
            cost_model=cost_model,
            block_weights=block_weights,
            bar_ts=bar_ts,
        )
