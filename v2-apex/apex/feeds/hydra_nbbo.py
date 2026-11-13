r"""Apex Hydra — Synthetic Consolidated NBBO, All-In Effective Cost Engine & Single-Broker Spike Rejector (`apex/feeds/hydra_nbbo.py`).

Implements:
  1. **Self-Detecting 4-Topology Mesh (`MeshTopology` / `MeshOperatingMode`)**:
     - `AUTO_HYDRA_MESH` (Default):
         * Both cTrader & MT5 online -> `ACTIVE_ACTIVE_SOR` (Synthetic NBBO + Cross-Pollination + Best-Execution Routing).
         * Only MT5 online -> `MT5_SOLO`.
         * Only cTrader online -> `CTRADER_SOLO` (works headless on Windows or Linux VPS without MT5).
     - `CTRADER_EYES_MT5_HANDS`: Uses cTrader Demo Level 2 DOM/OBI microstructure to filter trades while executing 100% of orders on MT5.
     - `MT5_EYES_CTRADER_HANDS`: Uses MT5 price/regime feeds while routing 100% of executions to cTrader.
     - `SPLIT_EXECUTION`: Splits approved risk 50/50 across both venues in parallel when both are online.

  2. **Synthetic Consolidated NBBO & All-In Effective Cost Engine**:
     - Computes consolidated Best Bid ($\max(\text{Bid}_{\text{cT}}, \text{Bid}_{\text{MT5}})$),
       Best Ask ($\min(\text{Ask}_{\text{cT}}, \text{Ask}_{\text{MT5}})$), and Consolidated Mid-Price.
     - Computes per-venue All-In Effective Cost in price, pips, R-multiples, and USD:
       $$\text{effective\_cost} = \text{live\_spread} + \text{round\_turn\_commission\_in\_price} + \text{expected\_vwap\_dom\_slippage} + \text{latency\_penalty}$$
     - Smart Venue Selector picks the venue with the lowest `effective_cost` and valid margin / AutoTrading status,
       computing exact `pips_saved` and `usd_saved` versus the slower/wider venue.

  3. **Single-Broker Spike / Stop-Hunt Wick Rejector**:
     - Tracks rolling normal cross-venue basis ($\text{median}(|\text{Mid}_{\text{cT}} - \text{Mid}_{\text{MT5}}|)$)
       and automatically blocks breakout entries if one venue diverges by $> 2.5\times$ normal cross-venue basis.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import numpy as np

from apex.feeds.models import DOMAnalytics, Side, Tick


class MeshOperatingMode(str, Enum):
    """User-configurable Broker Mesh operating modes in `config/apex_config.json`."""

    AUTO_HYDRA_MESH = "AUTO_HYDRA_MESH"
    CTRADER_EYES_MT5_HANDS = "CTRADER_EYES_MT5_HANDS"
    MT5_EYES_CTRADER_HANDS = "MT5_EYES_CTRADER_HANDS"
    SPLIT_EXECUTION = "SPLIT_EXECUTION"


class ResolvedMeshTopology(str, Enum):
    """Live resolved execution topology based on active heartbeats & configured mode."""

    ACTIVE_ACTIVE_SOR = "ACTIVE_ACTIVE_SOR"
    MT5_SOLO = "MT5_SOLO"
    CTRADER_SOLO = "CTRADER_SOLO"
    CTRADER_EYES_MT5_HANDS = "CTRADER_EYES_MT5_HANDS"
    MT5_EYES_CTRADER_HANDS = "MT5_EYES_CTRADER_HANDS"
    SPLIT_EXECUTION = "SPLIT_EXECUTION"
    OFFLINE = "OFFLINE"


@dataclass(frozen=True, slots=True)
class VenueFeeProfile:
    """Fee, commission, and latency profile for a single execution venue."""

    venue: str  # "ctrader" or "mt5"
    commission_per_lot_round_turn_usd: float = 4.50
    pip_value_per_lot_usd: float = 10.0
    ping_ms: float = 15.0
    latency_penalty_pips_per_100ms: float = 0.05
    margin_and_autotrading_ok: bool = True


@dataclass(frozen=True, slots=True)
class VenueCostQuote:
    """Computed all-in effective execution cost for a single venue on a symbol."""

    venue: str
    symbol: str
    bid: float
    ask: float
    mid: float
    pip_size: float
    live_spread_price: float
    live_spread_pips: float
    commission_price: float
    commission_pips: float
    expected_vwap_slippage_price: float
    expected_vwap_slippage_pips: float
    latency_penalty_price: float
    latency_penalty_pips: float
    effective_cost_price: float
    effective_cost_pips: float
    effective_cost_r: float
    effective_cost_usd_per_lot: float
    margin_and_autotrading_ok: bool


@dataclass(frozen=True, slots=True)
class SyntheticNBBOResult:
    """Consolidated NBBO and Smart Order Router (SOR) venue selection verdict."""

    symbol: str
    topology: ResolvedMeshTopology
    best_bid: float
    best_bid_venue: str
    best_ask: float
    best_ask_venue: str
    consolidated_mid: float
    synthetic_spread_pips: float
    winning_venue: str  # "ctrader", "mt5", or "SPLIT_50_50"
    winning_cost_pips: float
    winning_cost_r: float
    runner_up_venue: str | None
    runner_up_cost_pips: float | None
    pips_saved: float
    usd_saved_per_lot: float
    ctrader_quote: VenueCostQuote | None
    mt5_quote: VenueCostQuote | None
    cross_pollinated_dom: DOMAnalytics | None
    single_broker_spike_blocked: bool
    spike_basis_ratio: float
    spike_reason: str


class SingleBrokerSpikeRejector:
    """Single-Broker Stop-Hunt Wick & Stale Feed Rejector.

    If venue A prints a breakout bar while venue B's mid-price diverges by `> 2.5x` the rolling
    normal cross-venue basis (`|mid_A - mid_B|`), blocks entry automatically until prices converge.
    """

    def __init__(
        self,
        max_basis_multiple: float = 2.5,
        rolling_window: int = 60,
        min_samples: int = 8,
        min_floor_pips: float = 0.25,
    ) -> None:
        self.max_basis_multiple: float = float(max_basis_multiple)
        self.rolling_window: int = int(rolling_window)
        self.min_samples: int = int(min_samples)
        self.min_floor_pips: float = float(min_floor_pips)
        self._basis_history: dict[str, deque[float]] = {}

    def reset(self, symbol: str | None = None) -> None:
        if symbol is None:
            self._basis_history.clear()
        else:
            self._basis_history.pop(symbol.upper(), None)

    def evaluate_basis(
        self,
        symbol: str,
        ctrader_tick: Tick | None,
        mt5_tick: Tick | None,
        pip_size: float = 0.0001,
    ) -> tuple[bool, float, str]:
        """Return `(is_blocked, basis_ratio, reason)` comparing cTrader and MT5 mid-prices."""
        if ctrader_tick is None or mt5_tick is None:
            return False, 1.0, "single_venue_online"

        norm_sym = symbol.upper()
        cur_basis = abs(ctrader_tick.mid - mt5_tick.mid)
        hist = self._basis_history.setdefault(norm_sym, deque(maxlen=self.rolling_window))

        floor_price = self.min_floor_pips * max(pip_size, 1e-9)

        if len(hist) < self.min_samples:
            hist.append(cur_basis)
            return False, 1.0, f"basis_warming_up_{len(hist)}/{self.min_samples}"

        arr = np.asarray(hist, dtype=np.float64)
        normal_basis = max(float(np.median(arr)), floor_price)
        ratio = float(cur_basis / normal_basis)

        if ratio > self.max_basis_multiple:
            cur_pips = cur_basis / max(pip_size, 1e-9)
            norm_pips = normal_basis / max(pip_size, 1e-9)
            reason = (
                f"SINGLE_BROKER_SPIKE_REJECTED: cross-venue basis {cur_pips:.2f}p "
                f"is {ratio:.2f}x normal basis ({norm_pips:.2f}p) [> {self.max_basis_multiple:.1f}x limit]"
            )
            return True, ratio, reason

        hist.append(cur_basis)
        return False, ratio, "ok"


class HydraNBBOEngine:
    """Synthetic Consolidated NBBO & All-In Effective Cost Engine (`apex/feeds/hydra_nbbo.py`)."""

    def __init__(
        self,
        ctrader_profile: VenueFeeProfile | None = None,
        mt5_profile: VenueFeeProfile | None = None,
        spike_rejector: SingleBrokerSpikeRejector | None = None,
    ) -> None:
        self.ctrader_profile: VenueFeeProfile = ctrader_profile or VenueFeeProfile(
            venue="ctrader",
            commission_per_lot_round_turn_usd=4.50,  # Fusion Markets Zero cTrader raw spread + $4.50 RT
            pip_value_per_lot_usd=10.0,
            ping_ms=12.0,
            latency_penalty_pips_per_100ms=0.05,
            margin_and_autotrading_ok=True,
        )
        self.mt5_profile: VenueFeeProfile = mt5_profile or VenueFeeProfile(
            venue="mt5",
            commission_per_lot_round_turn_usd=4.50,
            pip_value_per_lot_usd=10.0,
            ping_ms=18.0,
            latency_penalty_pips_per_100ms=0.05,
            margin_and_autotrading_ok=True,
        )
        self.spike_rejector: SingleBrokerSpikeRejector = spike_rejector or SingleBrokerSpikeRejector()
        self.symbol_slippage_multipliers: dict[str, dict[str, float]] = {}

    def apply_slippage_calibration(self, multipliers: dict[str, dict[str, float]]) -> None:
        """Apply Sleep-Wake consolidated realized slippage multipliers per symbol/venue."""
        self.symbol_slippage_multipliers = {
            sym.upper(): {v.lower(): max(0.5, min(3.0, float(m))) for v, m in venues.items()}
            for sym, venues in multipliers.items()
        }

    @staticmethod
    def resolve_topology(
        configured_mode: MeshOperatingMode | str,
        ctrader_online: bool,
        mt5_online: bool,
    ) -> ResolvedMeshTopology:
        """Resolve the active 4-Topology Mesh state from configured mode and live venue heartbeats."""
        mode_str = (
            configured_mode.value
            if isinstance(configured_mode, MeshOperatingMode)
            else str(configured_mode).upper().strip()
        )

        if not ctrader_online and not mt5_online:
            return ResolvedMeshTopology.OFFLINE

        if mode_str == MeshOperatingMode.CTRADER_EYES_MT5_HANDS.value:
            if ctrader_online and mt5_online:
                return ResolvedMeshTopology.CTRADER_EYES_MT5_HANDS
            return ResolvedMeshTopology.MT5_SOLO if mt5_online else ResolvedMeshTopology.CTRADER_SOLO

        if mode_str == MeshOperatingMode.MT5_EYES_CTRADER_HANDS.value:
            if ctrader_online and mt5_online:
                return ResolvedMeshTopology.MT5_EYES_CTRADER_HANDS
            return ResolvedMeshTopology.CTRADER_SOLO if ctrader_online else ResolvedMeshTopology.MT5_SOLO

        if mode_str == MeshOperatingMode.SPLIT_EXECUTION.value:
            if ctrader_online and mt5_online:
                return ResolvedMeshTopology.SPLIT_EXECUTION
            return ResolvedMeshTopology.CTRADER_SOLO if ctrader_online else ResolvedMeshTopology.MT5_SOLO

        # Default: AUTO_HYDRA_MESH
        if ctrader_online and mt5_online:
            return ResolvedMeshTopology.ACTIVE_ACTIVE_SOR
        if mt5_online:
            return ResolvedMeshTopology.MT5_SOLO
        return ResolvedMeshTopology.CTRADER_SOLO

    @staticmethod
    def compute_venue_cost_quote(
        tick: Tick,
        profile: VenueFeeProfile,
        pip_size: float = 0.0001,
        one_r_price_distance: float = 0.0010,
        expected_vwap_slippage_pips: float = 0.10,
    ) -> VenueCostQuote:
        """Compute `effective_cost = live_spread + round_turn_commission_in_price + expected_vwap_dom_slippage + latency_penalty`."""
        eff_pip = max(float(pip_size), 1e-9)
        eff_1r = max(float(one_r_price_distance), 1e-9)
        pip_val_usd = max(float(profile.pip_value_per_lot_usd), 1e-6)

        live_spread_price = max(0.0, float(tick.ask - tick.bid))
        live_spread_pips = live_spread_price / eff_pip

        commission_pips = max(0.0, float(profile.commission_per_lot_round_turn_usd)) / pip_val_usd
        commission_price = commission_pips * eff_pip

        slip_pips = max(0.0, float(expected_vwap_slippage_pips))
        slip_price = slip_pips * eff_pip

        latency_pips = max(0.0, float(profile.ping_ms) / 100.0) * max(
            0.0, float(profile.latency_penalty_pips_per_100ms)
        )
        latency_price = latency_pips * eff_pip

        eff_cost_pips = live_spread_pips + commission_pips + slip_pips + latency_pips
        eff_cost_price = live_spread_price + commission_price + slip_price + latency_price
        eff_cost_r = eff_cost_price / eff_1r
        eff_cost_usd_per_lot = eff_cost_pips * pip_val_usd

        return VenueCostQuote(
            venue=profile.venue,
            symbol=tick.symbol.upper(),
            bid=float(tick.bid),
            ask=float(tick.ask),
            mid=float(tick.mid),
            pip_size=eff_pip,
            live_spread_price=live_spread_price,
            live_spread_pips=live_spread_pips,
            commission_price=commission_price,
            commission_pips=commission_pips,
            expected_vwap_slippage_price=slip_price,
            expected_vwap_slippage_pips=slip_pips,
            latency_penalty_price=latency_price,
            latency_penalty_pips=latency_pips,
            effective_cost_price=eff_cost_price,
            effective_cost_pips=eff_cost_pips,
            effective_cost_r=eff_cost_r,
            effective_cost_usd_per_lot=eff_cost_usd_per_lot,
            margin_and_autotrading_ok=profile.margin_and_autotrading_ok,
        )

    def evaluate_symbol(
        self,
        symbol: str,
        configured_mode: MeshOperatingMode | str,
        ctrader_tick: Tick | None,
        mt5_tick: Tick | None,
        pip_size: float = 0.0001,
        one_r_price_distance: float = 0.0010,
        ctrader_vwap_slippage_pips: float = 0.10,
        mt5_vwap_slippage_pips: float = 0.15,
        ctrader_dom: DOMAnalytics | None = None,
        ctrader_profile_override: VenueFeeProfile | None = None,
        mt5_profile_override: VenueFeeProfile | None = None,
    ) -> SyntheticNBBOResult:
        """Compute Synthetic NBBO, All-In Effective Cost per venue, Cross-Pollinated DOM, and Spike Shield."""
        norm_sym = symbol.upper()
        ct_prof = ctrader_profile_override or self.ctrader_profile
        mt5_prof = mt5_profile_override or self.mt5_profile

        ct_online = ctrader_tick is not None
        mt5_online = mt5_tick is not None
        topology = self.resolve_topology(configured_mode, ctrader_online=ct_online, mt5_online=mt5_online)

        ct_quote = (
            self.compute_venue_cost_quote(
                tick=ctrader_tick,
                profile=ct_prof,
                pip_size=pip_size,
                one_r_price_distance=one_r_price_distance,
                expected_vwap_slippage_pips=ctrader_vwap_slippage_pips,
            )
            if ctrader_tick is not None
            else None
        )
        mt5_quote = (
            self.compute_venue_cost_quote(
                tick=mt5_tick,
                profile=mt5_prof,
                pip_size=pip_size,
                one_r_price_distance=one_r_price_distance,
                expected_vwap_slippage_pips=mt5_vwap_slippage_pips,
            )
            if mt5_tick is not None
            else None
        )

        # 1. Compute Synthetic Consolidated NBBO (Best Bid = max(bids), Best Ask = min(asks))
        if ct_quote is not None and mt5_quote is not None:
            if ct_quote.bid >= mt5_quote.bid:
                best_bid, best_bid_venue = ct_quote.bid, "ctrader"
            else:
                best_bid, best_bid_venue = mt5_quote.bid, "mt5"

            if ct_quote.ask <= mt5_quote.ask:
                best_ask, best_ask_venue = ct_quote.ask, "ctrader"
            else:
                best_ask, best_ask_venue = mt5_quote.ask, "mt5"
        elif ct_quote is not None:
            best_bid, best_bid_venue = ct_quote.bid, "ctrader"
            best_ask, best_ask_venue = ct_quote.ask, "ctrader"
        elif mt5_quote is not None:
            best_bid, best_bid_venue = mt5_quote.bid, "mt5"
            best_ask, best_ask_venue = mt5_quote.ask, "mt5"
        else:
            best_bid, best_bid_venue = 0.0, "offline"
            best_ask, best_ask_venue = 0.0, "offline"

        consolidated_mid = 0.5 * (best_bid + best_ask) if (best_bid > 0 and best_ask > 0) else 0.0
        syn_spread_pips = (best_ask - best_bid) / max(pip_size, 1e-9) if (best_bid > 0 and best_ask > 0) else 0.0

        # 2. Evaluate Single-Broker Spike / Stop-Hunt Wick Rejector
        spike_blocked, spike_ratio, spike_reason = self.spike_rejector.evaluate_basis(
            symbol=norm_sym,
            ctrader_tick=ctrader_tick,
            mt5_tick=mt5_tick,
            pip_size=pip_size,
        )

        # 3. Smart Venue Selector across topologies
        winning_venue = "offline"
        winning_cost_pips = 0.0
        winning_cost_r = 0.0
        runner_up_venue: str | None = None
        runner_up_cost_pips: float | None = None
        pips_saved = 0.0
        usd_saved_per_lot = 0.0

        if topology == ResolvedMeshTopology.CTRADER_EYES_MT5_HANDS and mt5_quote is not None:
            winning_venue = "mt5"
            winning_cost_pips = mt5_quote.effective_cost_pips
            winning_cost_r = mt5_quote.effective_cost_r
            if ct_quote is not None:
                runner_up_venue = "ctrader"
                runner_up_cost_pips = ct_quote.effective_cost_pips
                pips_saved = max(0.0, ct_quote.effective_cost_pips - mt5_quote.effective_cost_pips)
                usd_saved_per_lot = max(0.0, ct_quote.effective_cost_usd_per_lot - mt5_quote.effective_cost_usd_per_lot)

        elif topology == ResolvedMeshTopology.MT5_EYES_CTRADER_HANDS and ct_quote is not None:
            winning_venue = "ctrader"
            winning_cost_pips = ct_quote.effective_cost_pips
            winning_cost_r = ct_quote.effective_cost_r
            if mt5_quote is not None:
                runner_up_venue = "mt5"
                runner_up_cost_pips = mt5_quote.effective_cost_pips
                pips_saved = max(0.0, mt5_quote.effective_cost_pips - ct_quote.effective_cost_pips)
                usd_saved_per_lot = max(0.0, mt5_quote.effective_cost_usd_per_lot - ct_quote.effective_cost_usd_per_lot)

        elif topology == ResolvedMeshTopology.SPLIT_EXECUTION and ct_quote is not None and mt5_quote is not None:
            winning_venue = "SPLIT_50_50"
            winning_cost_pips = 0.5 * (ct_quote.effective_cost_pips + mt5_quote.effective_cost_pips)
            winning_cost_r = 0.5 * (ct_quote.effective_cost_r + mt5_quote.effective_cost_r)
            worse_pips = max(ct_quote.effective_cost_pips, mt5_quote.effective_cost_pips)
            worse_usd = max(ct_quote.effective_cost_usd_per_lot, mt5_quote.effective_cost_usd_per_lot)
            avg_usd = 0.5 * (ct_quote.effective_cost_usd_per_lot + mt5_quote.effective_cost_usd_per_lot)
            pips_saved = max(0.0, worse_pips - winning_cost_pips)
            usd_saved_per_lot = max(0.0, worse_usd - avg_usd)

        elif ct_quote is not None and mt5_quote is not None:
            # ACTIVE_ACTIVE_SOR: pick lowest effective_cost among venues with valid margin & AutoTrading
            ct_ok = ct_quote.margin_and_autotrading_ok
            mt5_ok = mt5_quote.margin_and_autotrading_ok

            if ct_ok and mt5_ok:
                if ct_quote.effective_cost_pips <= mt5_quote.effective_cost_pips:
                    winning_venue = "ctrader"
                    winning_cost_pips = ct_quote.effective_cost_pips
                    winning_cost_r = ct_quote.effective_cost_r
                    runner_up_venue = "mt5"
                    runner_up_cost_pips = mt5_quote.effective_cost_pips
                    pips_saved = mt5_quote.effective_cost_pips - ct_quote.effective_cost_pips
                    usd_saved_per_lot = mt5_quote.effective_cost_usd_per_lot - ct_quote.effective_cost_usd_per_lot
                else:
                    winning_venue = "mt5"
                    winning_cost_pips = mt5_quote.effective_cost_pips
                    winning_cost_r = mt5_quote.effective_cost_r
                    runner_up_venue = "ctrader"
                    runner_up_cost_pips = ct_quote.effective_cost_pips
                    pips_saved = ct_quote.effective_cost_pips - mt5_quote.effective_cost_pips
                    usd_saved_per_lot = ct_quote.effective_cost_usd_per_lot - mt5_quote.effective_cost_usd_per_lot
            elif ct_ok:
                winning_venue = "ctrader"
                winning_cost_pips = ct_quote.effective_cost_pips
                winning_cost_r = ct_quote.effective_cost_r
                runner_up_venue = "mt5"
                runner_up_cost_pips = mt5_quote.effective_cost_pips
            else:
                winning_venue = "mt5"
                winning_cost_pips = mt5_quote.effective_cost_pips
                winning_cost_r = mt5_quote.effective_cost_r
                runner_up_venue = "ctrader"
                runner_up_cost_pips = ct_quote.effective_cost_pips

        elif ct_quote is not None:
            winning_venue = "ctrader"
            winning_cost_pips = ct_quote.effective_cost_pips
            winning_cost_r = ct_quote.effective_cost_r

        elif mt5_quote is not None:
            winning_venue = "mt5"
            winning_cost_pips = mt5_quote.effective_cost_pips
            winning_cost_r = mt5_quote.effective_cost_r

        return SyntheticNBBOResult(
            symbol=norm_sym,
            topology=topology,
            best_bid=best_bid,
            best_bid_venue=best_bid_venue,
            best_ask=best_ask,
            best_ask_venue=best_ask_venue,
            consolidated_mid=consolidated_mid,
            synthetic_spread_pips=syn_spread_pips,
            winning_venue=winning_venue,
            winning_cost_pips=winning_cost_pips,
            winning_cost_r=winning_cost_r,
            runner_up_venue=runner_up_venue,
            runner_up_cost_pips=runner_up_cost_pips,
            pips_saved=max(0.0, float(pips_saved)),
            usd_saved_per_lot=max(0.0, float(usd_saved_per_lot)),
            ctrader_quote=ct_quote,
            mt5_quote=mt5_quote,
            cross_pollinated_dom=ctrader_dom,
            single_broker_spike_blocked=spike_blocked,
            spike_basis_ratio=spike_ratio,
            spike_reason=spike_reason,
        )
