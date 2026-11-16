"""Autonomous Multi-Agent In-Flight Position Orchestrator (`apex/execution/position_manager.py`).

Manages every open Demo/Live position across its full post-entry lifecycle:
  - Rule A: Anti-Panic Grace Window (min_hold_bars=2 + ±0.65*ATR noise cone + Z3 Shield 8).
  - Rule B: Continuous Conditional Forward Time Machine ("futuring") on each closed bar.
  - Rule C: Dynamic Profit-Ratchet (+1.0R Free-Roll Lock, +1.5R Bank vs. 3.5R Super-Momentum Runner).
  - Rule D: Zombie Time-Decay (bars>=18, |R|<0.20, ER<0.15) & Friday 20:45 UTC Weekend-Gap Shield.
  - Closed-Deal Autopsy: Reconciles MT5 history deals with MFE_R & MAE_R into WalkForwardLedgerDB.
"""

from __future__ import annotations

import datetime
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Sequence

try:
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover - headless/cTrader-only nodes
    mt5 = None  # type: ignore[assignment]

from apex.ai.aegis_verifier import AegisVerifier
from apex.feeds.alias_mapper import SmartAliasMapper
from apex.feeds.models import AssetClass, Candle, Side
from apex.ledger.walkforward_db import WalkForwardLedgerDB
from apex.quant.regime import MarketRegime, RegimeClassifier
from apex.quant.timemachine_v2 import BlockBootstrapTimeMachine, TimeMachineCostModel

logger = logging.getLogger("apex.execution.position_manager")

_TIMEFRAME_SECONDS = {
    "M1": 60,
    "M5": 300,
    "M15": 900,
    "M30": 1800,
    "H1": 3600,
}


class PositionAction(str, Enum):
    """Canonical in-flight position action strings."""

    HOLD = "HOLD"
    RATCHET_SLTP = "RATCHET_SLTP"
    EARLY_EXIT = "EARLY_EXIT"


@dataclass
class InFlightPositionState:
    """Persistent in-flight tracking state for a single open position ticket."""

    ticket: int
    symbol: str
    side: Side
    timeframe: str
    entry_price: float
    initial_sl: float
    initial_tp: float
    one_r_distance: float
    opened_at_utc: float
    lots: float = 0.01
    mfe_r: float = 0.0
    mae_r: float = 0.0
    last_evaluated_bar_ts: float = 0.0
    ratchet_stage: str = "GRACE_LOCK"
    forward_prob_tp: float = 0.50
    forward_ev_r: float = 0.0
    z3_status: str = "SHIELD8_ARMED"
    bars_held: int = 0
    grace_locked: bool = True
    last_mark_price: float = 0.0


@dataclass(frozen=True, slots=True)
class PositionDecision:
    """Output of evaluating one open position on a cycle."""

    ticket: int
    symbol: str
    action: str  # HOLD | RATCHET_SLTP | EARLY_EXIT
    ratchet_stage: str
    new_sl: Optional[float]
    new_tp: Optional[float]
    current_r: float
    forward_ev_r: float
    forward_prob_tp: float
    mfe_r: float
    mae_r: float
    bars_held: int
    grace_locked: bool
    z3_status: str
    reason: str


class AutonomousPositionOrchestrator:
    """Multi-agent in-flight trajectory & profit-ratchet manager for open positions."""

    def __init__(
        self,
        *,
        regime_classifier: Optional[RegimeClassifier] = None,
        time_machine: Optional[BlockBootstrapTimeMachine] = None,
        ledger: Optional[WalkForwardLedgerDB] = None,
        min_hold_bars: int = 2,
        noise_cone_atr: float = 0.65,
        flash_crash_shock_z: float = 4.2,
        zombie_bars: int = 18,
        weekend_cutoff_utc_hhmm: tuple[int, int] = (20, 45),
        sltp_sender: Optional[Callable[[int, str, float, float], int]] = None,
        close_sender: Optional[Callable[[int, str, Side, float, float], int]] = None,
        feed_router: Optional[Any] = None,
    ) -> None:
        self.regime_classifier = regime_classifier or RegimeClassifier()
        self.time_machine = time_machine or BlockBootstrapTimeMachine(replays=2000, seed=19)
        self.ledger = ledger
        self.feed_router = feed_router
        self.min_hold_bars = max(1, int(min_hold_bars))
        self.noise_cone_atr = float(noise_cone_atr)
        self.flash_crash_shock_z = float(flash_crash_shock_z)
        self.zombie_bars = int(zombie_bars)
        self.weekend_cutoff_utc_hhmm = weekend_cutoff_utc_hhmm
        self._sltp_sender = sltp_sender
        self._close_sender = close_sender
        self._alias_mapper = SmartAliasMapper()
        self.tracked: dict[int, InFlightPositionState] = {}

    @staticmethod
    def is_weekend_shield_active(symbol: str, now_utc: float, cutoff_hhmm: tuple[int, int] = (20, 45)) -> bool:
        """True on Friday >= 20:45 UTC for non-crypto assets."""
        mapper = SmartAliasMapper()
        asset_class = mapper.classify_asset_class(symbol)
        if asset_class == AssetClass.CRYPTO:
            return False
        dt = datetime.datetime.fromtimestamp(float(now_utc), tz=datetime.timezone.utc)
        if dt.weekday() != 4:  # 4 = Friday
            return False
        return (dt.hour * 60 + dt.minute) >= (cutoff_hhmm[0] * 60 + cutoff_hhmm[1])

    @staticmethod
    def _never_widen_sl(side: Side, current_sl: float, candidate_sl: float) -> float:
        """Enforce monotonic SL tightening (never move SL backward into worse risk)."""
        if current_sl <= 0.0:
            return candidate_sl
        if side == Side.BUY:
            return max(current_sl, candidate_sl)
        return min(current_sl, candidate_sl)

    @staticmethod
    def _is_broker_ack_ok(retcode: int) -> bool:
        """Accept MT5 TRADE_RETCODE_DONE (10009), PLACED (10008), or cTrader/generic 0."""
        return int(retcode) in (0, 10008, 10009)

    def _register_or_update_state(
        self,
        *,
        ticket: int,
        symbol: str,
        side: Side,
        timeframe: str,
        entry_price: float,
        current_sl: float,
        current_tp: float,
        mark_price: float,
        opened_at_utc: float,
        lots: float,
        now_utc: float,
        bars_held_override: Optional[int] = None,
        initial_one_r: Optional[float] = None,
    ) -> tuple[InFlightPositionState, float]:
        st = self.tracked.get(ticket)
        resolved_1r = (
            float(initial_one_r)
            if (initial_one_r is not None and float(initial_one_r) > 1e-9)
            else (abs(entry_price - current_sl) if current_sl > 0.0 else max(entry_price * 0.002, 1e-5))
        )
        if st is None:
            st = InFlightPositionState(
                ticket=ticket,
                symbol=symbol,
                side=side,
                timeframe=timeframe.upper(),
                entry_price=entry_price,
                initial_sl=current_sl if current_sl > 0.0 else (entry_price - resolved_1r if side == Side.BUY else entry_price + resolved_1r),
                initial_tp=current_tp if current_tp > 0.0 else (entry_price + 2.0 * resolved_1r if side == Side.BUY else entry_price - 2.0 * resolved_1r),
                one_r_distance=max(resolved_1r, 1e-6),
                opened_at_utc=opened_at_utc,
                lots=lots,
            )
            self.tracked[ticket] = st
        elif initial_one_r is not None and float(initial_one_r) > 1e-9:
            st.one_r_distance = max(float(initial_one_r), 1e-6)

        st.last_mark_price = float(mark_price)
        signed_diff = (mark_price - st.entry_price) if side == Side.BUY else (st.entry_price - mark_price)
        current_r = signed_diff / max(st.one_r_distance, 1e-6)
        st.mfe_r = max(st.mfe_r, current_r)
        st.mae_r = min(st.mae_r, current_r)

        if bars_held_override is not None:
            st.bars_held = max(0, int(bars_held_override))
        else:
            tf_sec = _TIMEFRAME_SECONDS.get(st.timeframe, 300)
            st.bars_held = max(0, int((now_utc - st.opened_at_utc) // tf_sec))

        return st, current_r

    def evaluate_position(
        self,
        *,
        ticket: int,
        symbol: str,
        side: Side,
        entry_price: float,
        current_sl: float,
        current_tp: float,
        mark_price: float,
        candles: Optional[Sequence[Candle]] = None,
        candles_m5: Optional[Sequence[Candle]] = None,
        spread_price: float = 0.0,
        commission_price_equiv: float = 0.0,
        dom_obi: Optional[float] = None,
        cross_asset_shock_z: float = 0.0,
        timeframe: str = "M5",
        opened_at_utc: Optional[float] = None,
        lots: float = 0.01,
        now_utc: Optional[float] = None,
        bars_held_override: Optional[int] = None,
        bars_held: Optional[int] = None,
        initial_one_r: Optional[float] = None,
        force_evaluate: bool = False,
        execute_broker_actions: bool = False,
    ) -> PositionDecision:
        """Evaluate one open position across Zones 1-3 + Rules A-D and verify SLTP/close ACK."""
        candles_seq = list(candles if candles is not None else (candles_m5 or ()))
        eff_bars_held = bars_held if bars_held is not None else bars_held_override
        now_ts = float(now_utc if now_utc is not None else time.time())
        open_ts = float(opened_at_utc if opened_at_utc is not None else now_ts)
        st, current_r = self._register_or_update_state(
            ticket=ticket,
            symbol=symbol,
            side=side,
            timeframe=timeframe,
            entry_price=entry_price,
            current_sl=current_sl,
            current_tp=current_tp,
            mark_price=mark_price,
            opened_at_utc=open_ts,
            lots=lots,
            now_utc=now_ts,
            bars_held_override=eff_bars_held,
            initial_one_r=initial_one_r,
        )

        candles = candles_seq

        # Determine ATR & latest bar timestamp
        atr_val = st.one_r_distance / 1.5
        hurst_val = 0.55
        er_val = 0.40
        regime_val = MarketRegime.TRENDING_EXPANSION
        latest_bar_ts = float(candles[-1].timestamp_utc) if candles else now_ts

        if len(candles) >= 20:
            diag = self.regime_classifier.classify_candles(symbol=symbol, candles=candles, current_spread=spread_price)
            atr_val = max(diag.current_atr, 1e-6)
            hurst_val = diag.hurst_exponent
            er_val = diag.kaufman_er
            regime_val = diag.regime

        abs_move_atr = abs(mark_price - st.entry_price) / max(atr_val, 1e-6)
        weekend_active = self.is_weekend_shield_active(symbol, now_ts, self.weekend_cutoff_utc_hhmm)
        in_grace = (st.bars_held < self.min_hold_bars) and (abs_move_atr <= self.noise_cone_atr)
        st.grace_locked = in_grace

        # Rule A: Anti-Panic Grace Window & Tick-Noise Immunity
        if in_grace and cross_asset_shock_z <= self.flash_crash_shock_z and not weekend_active:
            _, _, z3_msg = AegisVerifier.verify_z3_inflight_trajectory(
                proposed_action="HOLD",
                bars_held=st.bars_held,
                min_hold_bars=self.min_hold_bars,
                abs_move_atr=abs_move_atr,
                cross_asset_shock_z=cross_asset_shock_z,
                forward_ev_r=st.forward_ev_r,
                current_r=current_r,
                invalidation_signals=0,
            )
            st.ratchet_stage = "GRACE_LOCK"
            st.z3_status = z3_msg
            return PositionDecision(
                ticket=ticket,
                symbol=symbol,
                action="HOLD",
                ratchet_stage="GRACE_LOCK",
                new_sl=None,
                new_tp=None,
                current_r=round(current_r, 4),
                forward_ev_r=round(st.forward_ev_r, 4),
                forward_prob_tp=round(st.forward_prob_tp, 4),
                mfe_r=round(st.mfe_r, 4),
                mae_r=round(st.mae_r, 4),
                bars_held=st.bars_held,
                grace_locked=True,
                z3_status=z3_msg,
                reason=f"Rule A Grace Lock ({st.bars_held}/{self.min_hold_bars} bars, move={abs_move_atr:.2f} ATR)",
            )

        # Rule D (Immediate Friday Weekend Shield)
        if weekend_active:
            _, _, z3_msg = AegisVerifier.verify_z3_inflight_trajectory(
                proposed_action="EARLY_EXIT",
                bars_held=st.bars_held,
                min_hold_bars=self.min_hold_bars,
                abs_move_atr=abs_move_atr,
                cross_asset_shock_z=cross_asset_shock_z,
                forward_ev_r=st.forward_ev_r,
                current_r=current_r,
                invalidation_signals=0,
                is_weekend_close=True,
            )
            st.ratchet_stage = "WEEKEND_SHIELD"
            st.z3_status = z3_msg
            if execute_broker_actions:
                self._dispatch_close(ticket, symbol, side, lots, mark_price)
            return PositionDecision(
                ticket=ticket,
                symbol=symbol,
                action="EARLY_EXIT",
                ratchet_stage="WEEKEND_SHIELD",
                new_sl=None,
                new_tp=None,
                current_r=round(current_r, 4),
                forward_ev_r=round(st.forward_ev_r, 4),
                forward_prob_tp=round(st.forward_prob_tp, 4),
                mfe_r=round(st.mfe_r, 4),
                mae_r=round(st.mae_r, 4),
                bars_held=st.bars_held,
                grace_locked=False,
                z3_status=z3_msg,
                reason="Rule D Friday Weekend-Gap Shield (>=20:45 UTC)",
            )

        # Rule B: Continuous Conditional Forward Time Machine on new closed bar (or forced)
        should_run_futuring = force_evaluate or (latest_bar_ts > st.last_evaluated_bar_ts)
        if should_run_futuring and len(candles) >= 30:
            st.last_evaluated_bar_ts = latest_bar_ts
            sim_sl = current_sl
            sim_tp = current_tp if current_tp > 0.0 else st.initial_tp
            valid_bracket = (
                (side == Side.BUY and 0.0 < sim_sl < mark_price < sim_tp)
                or (side == Side.SELL and 0.0 < sim_tp < mark_price < sim_sl)
            )
            if valid_bracket:
                cost_model = TimeMachineCostModel(
                    commission_per_lot_round_turn_usd=4.50,
                    lots=lots,
                    risk_per_lot_usd=100.0,
                    spread_price=spread_price,
                    vwap_slippage_price=0.0,
                )
                tm = self.time_machine.simulate_candles(
                    symbol=symbol,
                    side=side,
                    entry_price=mark_price,
                    stop_price=sim_sl,
                    target_price=sim_tp,
                    candles=candles,
                    horizon_bars=24,
                    cost_model=cost_model,
                )
                st.forward_ev_r = float(tm.net_avg_R)
                st.forward_prob_tp = float(tm.prob_profit)

        # Count independent thesis-invalidation signals
        invalidation_signals = 0
        if st.forward_ev_r < -0.25:
            invalidation_signals += 1
        if regime_val in (MarketRegime.VOLATILITY_SHOCK, MarketRegime.MEAN_REVERTING_CHOP) and er_val < 0.22:
            invalidation_signals += 1
        if candles:
            last_c = candles[-1]
            opposing_bar = (
                (side == Side.BUY and last_c.close < last_c.open and abs(last_c.close - last_c.open) >= 0.8 * atr_val)
                or (side == Side.SELL and last_c.close > last_c.open and abs(last_c.close - last_c.open) >= 0.8 * atr_val)
            )
            if opposing_bar:
                invalidation_signals += 1
        if dom_obi is not None:
            if (side == Side.BUY and dom_obi <= -0.25) or (side == Side.SELL and dom_obi >= 0.25):
                invalidation_signals += 1
        if cross_asset_shock_z > 3.0:
            invalidation_signals += 1

        # Flash-crash emergency OR 3-signal Thesis Invalidation OR Zombie Time-Decay
        is_zombie = (st.bars_held >= self.zombie_bars) and (abs(current_r) < 0.20) and (er_val < 0.15) and (st.forward_ev_r <= 0.0)
        wants_exit = (
            (cross_asset_shock_z > self.flash_crash_shock_z)
            or is_zombie
            or (st.forward_ev_r < -0.25 and invalidation_signals >= 3)
        )
        if wants_exit:
            z3_ok, _, z3_msg = AegisVerifier.verify_z3_inflight_trajectory(
                proposed_action="EARLY_EXIT",
                bars_held=st.bars_held,
                min_hold_bars=self.min_hold_bars,
                abs_move_atr=abs_move_atr,
                cross_asset_shock_z=cross_asset_shock_z,
                forward_ev_r=st.forward_ev_r,
                current_r=current_r,
                invalidation_signals=invalidation_signals,
                is_zombie_decay=is_zombie,
            )
            st.z3_status = z3_msg
            if z3_ok:
                stage = "FLASH_EXIT" if cross_asset_shock_z > self.flash_crash_shock_z else ("ZOMBIE_EXIT" if is_zombie else "THESIS_EXIT")
                if execute_broker_actions:
                    ret = self._dispatch_close(ticket, symbol, side, lots, mark_price)
                    if not self._is_broker_ack_ok(ret):
                        st.ratchet_stage = f"{stage}_RETRY"
                        return PositionDecision(
                            ticket=ticket,
                            symbol=symbol,
                            action=PositionAction.HOLD.value,
                            ratchet_stage=st.ratchet_stage,
                            new_sl=None,
                            new_tp=None,
                            current_r=round(current_r, 4),
                            forward_ev_r=round(st.forward_ev_r, 4),
                            forward_prob_tp=round(st.forward_prob_tp, 4),
                            mfe_r=round(st.mfe_r, 4),
                            mae_r=round(st.mae_r, 4),
                            bars_held=st.bars_held,
                            grace_locked=False,
                            z3_status=z3_msg,
                            reason=f"{stage} broker close failed (retcode={ret}); holding for retry.",
                        )
                st.ratchet_stage = stage
                return PositionDecision(
                    ticket=ticket,
                    symbol=symbol,
                    action=PositionAction.EARLY_EXIT.value,
                    ratchet_stage=stage,
                    new_sl=None,
                    new_tp=None,
                    current_r=round(current_r, 4),
                    forward_ev_r=round(st.forward_ev_r, 4),
                    forward_prob_tp=round(st.forward_prob_tp, 4),
                    mfe_r=round(st.mfe_r, 4),
                    mae_r=round(st.mae_r, 4),
                    bars_held=st.bars_held,
                    grace_locked=False,
                    z3_status=z3_msg,
                    reason=f"{stage}: fwd_ev={st.forward_ev_r:+.2f}R, sigs={invalidation_signals}, ER={er_val:.2f}",
                )

        # Zone 1 Defensive Early Exit / Risk Compression (negative excursion outside Grace Lock)
        min_sep = max(float(spread_price) * 2.0, 0.05 * st.one_r_distance, 1e-6)
        if not in_grace and current_r <= -0.25 and (st.forward_prob_tp < 0.35 or st.forward_ev_r < -0.15):
            if st.forward_ev_r <= -0.35 and st.forward_prob_tp < 0.25 and invalidation_signals >= 2:
                z3_ok, _, z3_msg = AegisVerifier.verify_z3_inflight_trajectory(
                    proposed_action="EARLY_EXIT",
                    bars_held=st.bars_held,
                    min_hold_bars=self.min_hold_bars,
                    abs_move_atr=abs_move_atr,
                    cross_asset_shock_z=cross_asset_shock_z,
                    forward_ev_r=st.forward_ev_r,
                    current_r=current_r,
                    invalidation_signals=max(invalidation_signals, 2),
                    is_zombie_decay=False,
                )
                st.z3_status = z3_msg
                if z3_ok:
                    if execute_broker_actions:
                        ret = self._dispatch_close(ticket, symbol, side, lots, mark_price)
                        if not self._is_broker_ack_ok(ret):
                            return PositionDecision(
                                ticket=ticket,
                                symbol=symbol,
                                action=PositionAction.HOLD.value,
                                ratchet_stage="DEFENSIVE_RETRY",
                                new_sl=None,
                                new_tp=None,
                                current_r=round(current_r, 4),
                                forward_ev_r=round(st.forward_ev_r, 4),
                                forward_prob_tp=round(st.forward_prob_tp, 4),
                                mfe_r=round(st.mfe_r, 4),
                                mae_r=round(st.mae_r, 4),
                                bars_held=st.bars_held,
                                grace_locked=False,
                                z3_status=z3_msg,
                                reason=f"DEFENSIVE_EXIT close rejected (retcode={ret})",
                            )
                    st.ratchet_stage = "DEFENSIVE_EXIT"
                    return PositionDecision(
                        ticket=ticket,
                        symbol=symbol,
                        action=PositionAction.EARLY_EXIT.value,
                        ratchet_stage="DEFENSIVE_EXIT",
                        new_sl=None,
                        new_tp=None,
                        current_r=round(current_r, 4),
                        forward_ev_r=round(st.forward_ev_r, 4),
                        forward_prob_tp=round(st.forward_prob_tp, 4),
                        mfe_r=round(st.mfe_r, 4),
                        mae_r=round(st.mae_r, 4),
                        bars_held=st.bars_held,
                        grace_locked=False,
                        z3_status=z3_msg,
                        reason=f"Zone 1 DEFENSIVE_EXIT: fwd_ev={st.forward_ev_r:+.2f}R, p_tp={st.forward_prob_tp:.2f}",
                    )

        # Zone 1 Compression, Zone 2 Smooth Profit-Surfing & Zone 3 Dynamic TP Sliding
        new_sl: Optional[float] = None
        new_tp: Optional[float] = None
        stage = "HOLD"

        if current_r >= 1.5:
            super_momentum = (hurst_val >= 0.60) and (er_val >= 0.55)
            one_r_lock = (st.entry_price + st.one_r_distance) if side == Side.BUY else (st.entry_price - st.one_r_distance)
            if super_momentum:
                stage = "RUNNER_3_5R"
                # Zone 3 smooth dynamic TP sliding in [3.5R, 4.5R]
                tp_mult = min(4.5, 3.5 + max(0.0, (current_r - 2.5) * 0.50))
                expanded_tp = (st.entry_price + tp_mult * st.one_r_distance) if side == Side.BUY else (st.entry_price - tp_mult * st.one_r_distance)
                if side == Side.BUY:
                    swing_trail = min(float(c.low) for c in candles[-3:]) if len(candles) >= 3 else (mark_price - 1.8 * atr_val)
                    atr_trail = mark_price - 1.8 * atr_val
                    cand_sl = min(mark_price - min_sep, max(one_r_lock, min(swing_trail, atr_trail)))
                    cand_tp = max(current_tp, expanded_tp)
                else:
                    swing_trail = max(float(c.high) for c in candles[-3:]) if len(candles) >= 3 else (mark_price + 1.8 * atr_val)
                    atr_trail = mark_price + 1.8 * atr_val
                    cand_sl = max(mark_price + min_sep, min(one_r_lock, max(swing_trail, atr_trail)))
                    cand_tp = min(current_tp, expanded_tp) if current_tp > 0.0 else expanded_tp
                new_sl = self._never_widen_sl(side, current_sl, cand_sl)
                new_tp = cand_tp
            else:
                stage = "BANK_1R"
                # Lock +1.0R at 1.5R..2.0R and surf smoothly higher beyond +2.0R
                lock_r = 1.0 + max(0.0, 0.55 * (current_r - 2.0))
                bank_sl = (st.entry_price + lock_r * st.one_r_distance) if side == Side.BUY else (st.entry_price - lock_r * st.one_r_distance)
                bank_sl = min(mark_price - min_sep, bank_sl) if side == Side.BUY else max(mark_price + min_sep, bank_sl)
                new_sl = self._never_widen_sl(side, current_sl, bank_sl)
                new_tp = current_tp if current_tp > 0.0 else st.initial_tp
        elif current_r >= 1.0:
            stage = "FREE_ROLL_1R"
            fee_offset = max(spread_price + commission_price_equiv, 0.0) + max(0.05, 0.85 * (current_r - 1.0)) * st.one_r_distance
            be_sl = (st.entry_price + fee_offset) if side == Side.BUY else (st.entry_price - fee_offset)
            be_sl = min(mark_price - min_sep, be_sl) if side == Side.BUY else max(mark_price + min_sep, be_sl)
            new_sl = self._never_widen_sl(side, current_sl, be_sl)
            new_tp = current_tp if current_tp > 0.0 else st.initial_tp
        elif current_r >= 0.75:
            stage = "PROFIT_SURF"
            frac = (current_r - 0.75) / 0.25
            surf_r = -0.35 * (1.0 - max(0.0, min(1.0, frac)))
            surf_sl = (st.entry_price + surf_r * st.one_r_distance) if side == Side.BUY else (st.entry_price - surf_r * st.one_r_distance)
            surf_sl = min(mark_price - min_sep, surf_sl) if side == Side.BUY else max(mark_price + min_sep, surf_sl)
            new_sl = self._never_widen_sl(side, current_sl, surf_sl)
            new_tp = current_tp if current_tp > 0.0 else st.initial_tp
        elif not in_grace and current_r <= -0.25 and (st.forward_prob_tp < 0.35 or st.forward_ev_r < -0.15):
            stage = "DEFENSIVE_COMPRESS"
            lock_r = max(-0.70, current_r - 0.22)
            def_sl = (st.entry_price + lock_r * st.one_r_distance) if side == Side.BUY else (st.entry_price - lock_r * st.one_r_distance)
            def_sl = min(mark_price - min_sep, def_sl) if side == Side.BUY else max(mark_price + min_sep, def_sl)
            new_sl = self._never_widen_sl(side, current_sl, def_sl)
            new_tp = current_tp if current_tp > 0.0 else st.initial_tp

        hyst_min = max(float(spread_price) * 0.25, 0.02 * st.one_r_distance, 1e-6)
        sl_changed = new_sl is not None and abs(new_sl - current_sl) >= hyst_min
        tp_changed = new_tp is not None and abs(new_tp - current_tp) >= hyst_min
        wants_ratchet = new_sl is not None and (sl_changed or tp_changed)

        z3_ok, _, z3_msg = AegisVerifier.verify_z3_inflight_trajectory(
            proposed_action="RATCHET_SLTP" if wants_ratchet else "HOLD",
            bars_held=st.bars_held,
            min_hold_bars=self.min_hold_bars,
            abs_move_atr=abs_move_atr,
            cross_asset_shock_z=cross_asset_shock_z,
            forward_ev_r=st.forward_ev_r,
            current_r=current_r,
            invalidation_signals=invalidation_signals,
            side=side.value,
            entry_price=st.entry_price,
            sl_old=current_sl if wants_ratchet else None,
            sl_new=new_sl if wants_ratchet else None,
            mark_price=mark_price if wants_ratchet else None,
            tp_old=current_tp if wants_ratchet else None,
            tp_new=(new_tp or current_tp) if wants_ratchet else None,
        )
        st.z3_status = z3_msg
        st.ratchet_stage = stage

        if wants_ratchet and z3_ok and new_sl is not None:
            if execute_broker_actions:
                ret = self._dispatch_sltp(ticket, symbol, new_sl, new_tp or current_tp)
                if not self._is_broker_ack_ok(ret):
                    logger.warning("SL/TP amend failed for %s ticket=%d (retcode=%s)", symbol, ticket, ret)
                    return PositionDecision(
                        ticket=ticket,
                        symbol=symbol,
                        action=PositionAction.HOLD.value,
                        ratchet_stage=f"{stage}_RETRY",
                        new_sl=None,
                        new_tp=None,
                        current_r=round(current_r, 4),
                        forward_ev_r=round(st.forward_ev_r, 4),
                        forward_prob_tp=round(st.forward_prob_tp, 4),
                        mfe_r=round(st.mfe_r, 4),
                        mae_r=round(st.mae_r, 4),
                        bars_held=st.bars_held,
                        grace_locked=False,
                        z3_status=z3_msg,
                        reason=f"Broker SL/TP amend rejected (retcode={ret}); holding.",
                    )
            return PositionDecision(
                ticket=ticket,
                symbol=symbol,
                action=PositionAction.RATCHET_SLTP.value,
                ratchet_stage=stage,
                new_sl=new_sl,
                new_tp=new_tp,
                current_r=round(current_r, 4),
                forward_ev_r=round(st.forward_ev_r, 4),
                forward_prob_tp=round(st.forward_prob_tp, 4),
                mfe_r=round(st.mfe_r, 4),
                mae_r=round(st.mae_r, 4),
                bars_held=st.bars_held,
                grace_locked=False,
                z3_status=z3_msg,
                reason=f"Rule C {stage}: SL->{new_sl:.5f}, TP->{(new_tp or current_tp):.5f}",
            )

        return PositionDecision(
            ticket=ticket,
            symbol=symbol,
            action="HOLD",
            ratchet_stage=stage,
            new_sl=None,
            new_tp=None,
            current_r=round(current_r, 4),
            forward_ev_r=round(st.forward_ev_r, 4),
            forward_prob_tp=round(st.forward_prob_tp, 4),
            mfe_r=round(st.mfe_r, 4),
            mae_r=round(st.mae_r, 4),
            bars_held=st.bars_held,
            grace_locked=False,
            z3_status=z3_msg,
            reason=f"Hold ({stage}, FwdEV={st.forward_ev_r:+.2f}R, P(TP)={st.forward_prob_tp * 100:.0f}%)",
        )

    def _dispatch_sltp(self, ticket: int, symbol: str, sl: float, tp: float) -> int:
        if self._sltp_sender is not None:
            return int(self._sltp_sender(ticket, symbol, sl, tp))

        feed_router = getattr(self, "feed_router", None)
        ct = getattr(feed_router, "ctrader", None) if feed_router is not None else None
        mt5_pos = mt5.positions_get(ticket=int(ticket)) if (mt5 is not None and getattr(mt5, "positions_get", None)) else None
        if not mt5_pos and ct is not None and ct.is_connected and hasattr(ct, "amend_position_sltp"):
            res = ct.amend_position_sltp(position_id=int(ticket), stop_loss=float(sl), take_profit=float(tp), symbol=symbol)
            return 10009 if (isinstance(res, dict) and res.get("ok")) else -1

        if mt5 is None:
            return -1
        info = mt5.symbol_info(symbol) if getattr(mt5, "symbol_info", None) else None
        digits = int(getattr(info, "digits", 5) or 5) if info is not None else 5
        req = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": int(ticket),
            "symbol": symbol,
            "sl": round(float(sl), digits),
            "tp": round(float(tp), digits),
            "magic": 202611,
            "comment": "APEX_RATCHET",
        }
        res = mt5.order_send(req)
        return int(getattr(res, "retcode", -1)) if res is not None else -1

    def _dispatch_close(self, ticket: int, symbol: str, side: Side, lots: float, price: float) -> int:
        if self._close_sender is not None:
            return int(self._close_sender(ticket, symbol, side, lots, price))

        feed_router = getattr(self, "feed_router", None)
        ct = getattr(feed_router, "ctrader", None) if feed_router is not None else None
        mt5_pos = mt5.positions_get(ticket=int(ticket)) if (mt5 is not None and getattr(mt5, "positions_get", None)) else None
        if not mt5_pos and ct is not None and ct.is_connected and hasattr(ct, "close_position"):
            res = ct.close_position(position_id=int(ticket), volume_lots=float(lots), symbol=symbol)
            return 10009 if (isinstance(res, dict) and res.get("ok")) else -1

        if mt5 is None:
            return -1
        info = mt5.symbol_info(symbol)
        digits = int(getattr(info, "digits", 5) or 5) if info is not None else 5
        filling = int(mt5.ORDER_FILLING_FOK)
        if info is not None:
            fm = int(getattr(info, "filling_mode", 0))
            filling = int(mt5.ORDER_FILLING_FOK) if (fm & 1) else (int(mt5.ORDER_FILLING_IOC) if (fm & 2) else int(mt5.ORDER_FILLING_RETURN))
        last_retcode = -1
        for fill_mode in dict.fromkeys(
            [
                filling,
                int(mt5.ORDER_FILLING_IOC),
                int(mt5.ORDER_FILLING_FOK),
                int(mt5.ORDER_FILLING_RETURN),
            ]
        ):
            req = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": int(ticket),
                "symbol": symbol,
                "volume": float(lots),
                "type": int(mt5.ORDER_TYPE_SELL if side == Side.BUY else mt5.ORDER_TYPE_BUY),
                "price": round(float(price), digits),
                "deviation": 50,
                "magic": 202611,
                "comment": "APEX_INFLIGHT_EXIT",
                "type_time": int(mt5.ORDER_TIME_GTC),
                "type_filling": fill_mode,
            }
            res = mt5.order_send(req)
            last_retcode = int(getattr(res, "retcode", -1)) if res is not None else -1
            if last_retcode == int(getattr(mt5, "TRADE_RETCODE_DONE", 10009)):
                return last_retcode
        return last_retcode

    def reconcile_closed_deals(
        self,
        open_tickets: set[int] | None,
        deals_fetcher: Optional[Callable[[int], Sequence[Any]]] = None,
    ) -> list[dict[str, Any]]:
        """Detect tracked positions no longer open on broker and log MFE_R/MAE_R autopsy."""
        open_set = set(open_tickets or ())
        closed_tickets = [t for t in list(self.tracked.keys()) if t not in open_set]
        reconciled: list[dict[str, Any]] = []
        for ticket in closed_tickets:
            st = self.tracked.get(ticket)
            if st is None:
                continue
            exit_px: float | None = None
            realized_pnl: float = 0.0
            venue_used: str = "mt5"

            if deals_fetcher is not None:
                deals = list(deals_fetcher(ticket) or ())
                exit_entry_code = int(getattr(mt5, "DEAL_ENTRY_OUT", 1)) if mt5 is not None else 1
                exit_deals = [d for d in deals if int(getattr(d, "entry", -1)) == exit_entry_code]
                if not exit_deals:
                    self.tracked.pop(ticket, None)
                    continue
                d = exit_deals[-1]
                exit_px = float(getattr(d, "price", st.entry_price) or st.entry_price)
                realized_pnl = float(getattr(d, "profit", 0.0)) + float(getattr(d, "commission", 0.0)) + float(getattr(d, "swap", 0.0))
            else:
                if mt5 is not None and getattr(mt5, "history_deals_get", None):
                    deals = list(mt5.history_deals_get(position=ticket) or ())
                    exit_entry_code = int(getattr(mt5, "DEAL_ENTRY_OUT", 1))
                    exit_deals = [d for d in deals if int(getattr(d, "entry", -1)) == exit_entry_code]
                    if exit_deals:
                        d = exit_deals[-1]
                        exit_px = float(getattr(d, "price", st.entry_price) or st.entry_price)
                        realized_pnl = float(getattr(d, "profit", 0.0)) + float(getattr(d, "commission", 0.0)) + float(getattr(d, "swap", 0.0))

                # cTrader closed-deal reconciliation when ticket is absent from MT5 deals
                if exit_px is None and self.feed_router is not None:
                    ct = getattr(self.feed_router, "ctrader", None)
                    if ct is not None and (ct.is_connected or getattr(ct, "_recent_execution_events", None)):
                        ct_events = list(getattr(ct, "_recent_execution_events", ()) or ())
                        for ev in reversed(ct_events):
                            if int(ev.get("position_id", 0) or 0) == int(ticket):
                                px_ev = float(ev.get("executed_price", 0.0) or 0.0)
                                if px_ev > 0.0:
                                    exit_px = px_ev
                                    venue_used = "ctrader"
                                    break
                        if exit_px is None and hasattr(ct, "get_open_positions"):
                            ct_open_ids = {int(p.get("position_id", 0) or 0) for p in (ct.get_open_positions() or ())}
                            if int(ticket) not in ct_open_ids:
                                tick_ct = ct.get_latest_tick(st.symbol) if hasattr(ct, "get_latest_tick") else None
                                if tick_ct is not None:
                                    exit_px = float(tick_ct.bid if st.side == Side.BUY else tick_ct.ask)
                                    venue_used = "ctrader"
                                elif st.last_mark_price > 0.0:
                                    exit_px = float(st.last_mark_price)
                                    venue_used = "ctrader"

            if exit_px is None:
                self.tracked.pop(ticket, None)
                continue

            self.tracked.pop(ticket, None)
            signed = (exit_px - st.entry_price) if st.side == Side.BUY else (st.entry_price - exit_px)
            realized_r = signed / max(st.one_r_distance, 1e-6)
            if self.ledger is not None:
                self.ledger.close_trade_with_autopsy(
                    order_id=str(ticket),
                    exit_price=exit_px,
                    realized_net_r=round(realized_r, 4),
                    realized_pnl=round(realized_pnl, 2),
                    mfe_r=round(st.mfe_r, 4),
                    mae_r=round(st.mae_r, 4),
                    symbol=st.symbol,
                    side=st.side.value,
                    entry_price=st.entry_price,
                    stop_loss=st.initial_sl,
                    take_profit=st.initial_tp,
                    lots=st.lots,
                    venue=venue_used,
                )
            reconciled.append(
                {
                    "ticket": ticket,
                    "symbol": st.symbol,
                    "exit_price": exit_px,
                    "realized_pnl": round(realized_pnl, 2),
                    "realized_r": round(realized_r, 4),
                    "mfe_r": round(st.mfe_r, 4),
                    "mae_r": round(st.mae_r, 4),
                }
            )
        return reconciled

    def get_cockpit_trajectories(self) -> dict[str, dict[str, Any]]:
        """Return live In-Flight Trajectory metadata keyed by canonical symbol for the Chrome Cockpit."""
        out: dict[str, dict[str, Any]] = {}
        for st in self.tracked.values():
            can = self._alias_mapper.canonicalize(st.symbol).upper()
            out[can] = {
                "ticket": st.ticket,
                "ratchet_stage": st.ratchet_stage,
                "forward_prob_tp": round(st.forward_prob_tp * 100.0, 1),
                "forward_ev_r": round(st.forward_ev_r, 2),
                "mfe_r": round(st.mfe_r, 2),
                "mae_r": round(st.mae_r, 2),
                "bars_held": st.bars_held,
                "grace_locked": st.grace_locked,
                "z3_status": st.z3_status,
            }
        return out
