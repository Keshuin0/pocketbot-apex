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

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from apex.feeds.models import Candle, Side


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
    """

    commission_per_lot_round_turn_usd: float = 4.50
    lots: float = 1.0
    risk_per_lot_usd: float = 100.0
    spread_price: float = 0.00010
    vwap_slippage_price: float = 0.00004

    def compute_round_trip_cost_r(self, one_r_price_distance: float) -> float:
        """Compute total round-trip friction (commission + spread + VWAP DOM slippage) in R-multiples."""
        if one_r_price_distance <= 1e-12:
            raise ValueError(f"one_r_price_distance must be > 0, got {one_r_price_distance}")
        if self.risk_per_lot_usd <= 1e-12:
            raise ValueError(f"risk_per_lot_usd must be > 0, got {self.risk_per_lot_usd}")

        commission_r = max(0.0, float(self.commission_per_lot_round_turn_usd)) / float(self.risk_per_lot_usd)
        microstructure_price_cost = max(0.0, float(self.spread_price)) + max(0.0, float(self.vwap_slippage_price))
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


class BlockBootstrapTimeMachine:
    """10,000-Path Stationary Block-Bootstrap Monte Carlo Engine (`apex/quant/timemachine_v2.py`)."""

    MAX_EQUITY_RISK_PCT: float = 0.5  # Strict 0.5% ($50 on $10,000) institutional risk cap

    def __init__(
        self,
        replays: int = 10_000,
        mean_block_length: float = 6.0,
        seed: int = 19,
        account_equity_usd: float = 10_000.0,
        max_risk_pct: float = 0.5,
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

    @staticmethod
    def build_relative_bar_ratios(candles: Sequence[Candle]) -> np.ndarray:
        """Convert historical M5/M15 candles into contiguous relative bar ratios `(o/pc, h/pc, l/pc, c/pc)`.

        Returns an `(N-1, 4)` float64 array where column 0=open_ratio, 1=high_ratio, 2=low_ratio, 3=close_ratio.
        """
        if len(candles) < 10:
            raise ValueError(f"Need at least 10 historical candles for block bootstrap, got {len(candles)}")

        n = len(candles) - 1
        ratios = np.empty((n, 4), dtype=np.float64)
        for i in range(n):
            prev_c = float(candles[i].close)
            if prev_c <= 0.0:
                prev_c = float(candles[i + 1].open)
            cur = candles[i + 1]
            ratios[i, 0] = float(cur.open) / prev_c
            ratios[i, 1] = float(cur.high) / prev_c
            ratios[i, 2] = float(cur.low) / prev_c
            ratios[i, 3] = float(cur.close) / prev_c
        return ratios

    def generate_stationary_block_indices(
        self,
        pool_length: int,
        horizon_bars: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate `(replays, horizon_bars)` indices via Politis & Romano Stationary Block Bootstrap.

        At step $t = 0$, draw a uniform starting index $I_0 \\in \\{0, \\dots, L-1\\}$.
        At each subsequent step $t > 0$:
          - With probability $1 - p$ (where $p = 1 / \\text{mean\\_block\\_length}$), continue the
            contiguous historical block: $I_t = (I_{t-1} + 1) \\bmod L$.
          - With probability $p$, start a new random block at $I_t \\sim \\text{Uniform}(0, L-1)$.
        This preserves multi-bar volatility clustering and GARCH-like serial dependence in bar ranges.
        """
        p_restart = 1.0 / max(self.mean_block_length, 1.0)
        indices = np.empty((self.replays, horizon_bars), dtype=np.int64)
        indices[:, 0] = rng.integers(0, pool_length, size=self.replays)

        if horizon_bars > 1:
            restarts = rng.random((self.replays, horizon_bars - 1)) < p_restart
            random_jumps = rng.integers(0, pool_length, size=(self.replays, horizon_bars - 1))
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
    ) -> TimeMachineV2Result:
        """Run 10,000 Stationary Block-Bootstrap paths and deduct round-trip costs on every path."""
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

        target_r = float(target_dist / one_r_dist)
        cm = cost_model if cost_model is not None else TimeMachineCostModel()
        commission_r = max(0.0, float(cm.commission_per_lot_round_turn_usd)) / max(float(cm.risk_per_lot_usd), 1e-9)
        spread_slip_r = (max(0.0, float(cm.spread_price)) + max(0.0, float(cm.vwap_slippage_price))) / one_r_dist
        round_trip_cost_r = commission_r + spread_slip_r

        rng = np.random.default_rng(self.seed)
        idx_matrix = self.generate_stationary_block_indices(
            pool_length=len(bar_ratios),
            horizon_bars=int(horizon_bars),
            rng=rng,
        )

        # Vectorized path evaluation across all 10,000 replays
        sampled_high_ratios = bar_ratios[idx_matrix, 1]
        sampled_low_ratios = bar_ratios[idx_matrix, 2]
        sampled_close_ratios = bar_ratios[idx_matrix, 3]

        n_paths = self.replays
        cur_prices = np.full(n_paths, float(entry_price), dtype=np.float64)
        # 0 = active, 1 = target hit first, -1 = stop hit first
        status = np.zeros(n_paths, dtype=np.int8)

        for t in range(int(horizon_bars)):
            active = status == 0
            if not np.any(active):
                break
            hi = cur_prices * sampled_high_ratios[:, t]
            lo = cur_prices * sampled_low_ratios[:, t]
            cl = cur_prices * sampled_close_ratios[:, t]

            if is_long:
                # Conservative rule: check stop before target on same bar
                hit_stop = active & (lo <= float(stop_price))
                hit_target = active & (~hit_stop) & (hi >= float(target_price))
            else:
                hit_stop = active & (hi >= float(stop_price))
                hit_target = active & (~hit_stop) & (lo <= float(target_price))

            status[hit_stop] = -1
            status[hit_target] = 1
            cur_prices = np.where(active, cl, cur_prices)

        if is_long:
            timeout_r = (cur_prices - float(entry_price)) / one_r_dist
        else:
            timeout_r = (float(entry_price) - cur_prices) / one_r_dist

        gross_r = np.where(
            status == 1,
            target_r,
            np.where(status == -1, -1.0, timeout_r),
        )
        # Deduct round-trip friction on EVERY simulated path
        net_r = gross_r - round_trip_cost_r

        p_target = float(np.mean(status == 1))
        p_stop = float(np.mean(status == -1))
        p_timeout = float(np.mean(status == 0))
        prob_profit = float(np.mean(net_r > 0.0))

        gross_avg_r = float(np.mean(gross_r))
        net_avg_r = float(np.mean(net_r))
        median_net_r = float(np.median(net_r))

        # 95% Value-at-Risk and 95% Conditional Value-at-Risk (Expected Shortfall in worst 5% tail)
        sorted_net_r = np.sort(net_r)
        tail_cutoff = max(1, int(0.05 * len(sorted_net_r)))
        var_95_r = float(sorted_net_r[tail_cutoff - 1])
        cvar_95_r = float(np.mean(sorted_net_r[:tail_cutoff]))

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
    ) -> TimeMachineV2Result:
        """Run 10,000 Stationary Block-Bootstrap paths directly from historical `Candle` objects."""
        ratios = self.build_relative_bar_ratios(candles)
        return self.simulate_from_ratios(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            bar_ratios=ratios,
            horizon_bars=horizon_bars,
            cost_model=cost_model,
        )
