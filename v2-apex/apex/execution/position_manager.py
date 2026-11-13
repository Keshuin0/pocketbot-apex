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
from typing import Any, Callable, Optional, Sequence

import MetaTrader5 as mt5

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
    ) -> None:
        self.regime_classifier = regime_classifier or RegimeClassifier()
        self.time_machine = time_machine or BlockBootstrapTimeMachine(replays=2000, seed=19)
        self.ledger = ledger
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
    ) -> tuple[InFlightPositionState, float]:
        st = self.tracked.get(ticket)
        if st is None:
            one_r = abs(entry_price - current_sl) if current_sl > 0.0 else max(entry_price * 0.002, 1e-5)
            st = InFlightPositionState(
                ticket=ticket,
                symbol=symbol,
                side=side,
                timeframe=timeframe.upper(),
                entry_price=entry_price,
                initial_sl=current_sl if current_sl > 0.0 else (entry_price - one_r if side == Side.BUY else entry_price + one_r),
                initial_tp=current_tp if current_tp > 0.0 else (entry_price + 2.0 * one_r if side == Side.BUY else entry_price - 2.0 * one_r),
                one_r_distance=max(one_r, 1e-6),
                opened_at_utc=opened_at_utc,
                lots=lots,
            )
            self.tracked[ticket] = st

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
        candles: Sequence[Candle],
        spread_price: float = 0.0,
        commission_price_equiv: float = 0.0,
        dom_obi: Optional[float] = None,
        cross_asset_shock_z: float = 0.0,
        timeframe: str = "M5",
        opened_at_utc: Optional[float] = None,
        lots: float = 0.01,
        now_utc: Optional[float] = None,
        bars_held_override: Optional[int] = None,
        force_evaluate: bool = False,
        execute_broker_actions: bool = False,
    ) -> PositionDecision:
        """Evaluate one open position across Rules A-D and optionally dispatch SLTP/close."""
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
            bars_held_override=bars_held_override,
        )

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
                st.ratchet_stage = stage
                if execute_broker_actions:
                    self._dispatch_close(ticket, symbol, side, lots, mark_price)
                return PositionDecision(
                    ticket=ticket,
                    symbol=symbol,
                    action="EARLY_EXIT",
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

        # Rule C: Dynamic Profit-Ratchet & Super-Momentum Runner Mode
        new_sl: Optional[float] = None
        new_tp: Optional[float] = None
        stage = "HOLD"

        if current_r >= 1.5:
            super_momentum = (hurst_val >= 0.60) and (er_val >= 0.55)
            one_r_lock = (st.entry_price + st.one_r_distance) if side == Side.BUY else (st.entry_price - st.one_r_distance)
            if super_momentum:
                stage = "RUNNER_3_5R"
                expanded_tp = (st.entry_price + 3.5 * st.one_r_distance) if side == Side.BUY else (st.entry_price - 3.5 * st.one_r_distance)
                if side == Side.BUY:
                    swing_trail = min(float(c.low) for c in candles[-3:]) if len(candles) >= 3 else (mark_price - 1.8 * atr_val)
                    atr_trail = mark_price - 1.8 * atr_val
                    cand_sl = max(one_r_lock, min(swing_trail, atr_trail))
                    cand_tp = max(current_tp, expanded_tp)
                else:
                    swing_trail = max(float(c.high) for c in candles[-3:]) if len(candles) >= 3 else (mark_price + 1.8 * atr_val)
                    atr_trail = mark_price + 1.8 * atr_val
                    cand_sl = min(one_r_lock, max(swing_trail, atr_trail))
                    cand_tp = min(current_tp, expanded_tp) if current_tp > 0.0 else expanded_tp
                new_sl = self._never_widen_sl(side, current_sl, cand_sl)
                new_tp = cand_tp
            else:
                stage = "BANK_1R"
                new_sl = self._never_widen_sl(side, current_sl, one_r_lock)
                new_tp = current_tp if current_tp > 0.0 else st.initial_tp
        elif current_r >= 1.0:
            stage = "FREE_ROLL_1R"
            fee_offset = max(spread_price + commission_price_equiv, 0.0) + 0.05 * st.one_r_distance
            be_sl = (st.entry_price + fee_offset) if side == Side.BUY else (st.entry_price - fee_offset)
            new_sl = self._never_widen_sl(side, current_sl, be_sl)
            new_tp = current_tp if current_tp > 0.0 else st.initial_tp

        _, _, z3_msg = AegisVerifier.verify_z3_inflight_trajectory(
            proposed_action="HOLD",
            bars_held=st.bars_held,
            min_hold_bars=self.min_hold_bars,
            abs_move_atr=abs_move_atr,
            cross_asset_shock_z=cross_asset_shock_z,
            forward_ev_r=st.forward_ev_r,
            current_r=current_r,
            invalidation_signals=invalidation_signals,
        )
        st.z3_status = z3_msg
        st.ratchet_stage = stage

        if new_sl is not None and (abs(new_sl - current_sl) > 1e-7 or (new_tp is not None and abs(new_tp - current_tp) > 1e-7)):
            if execute_broker_actions:
                self._dispatch_sltp(ticket, symbol, new_sl, new_tp or current_tp)
            return PositionDecision(
                ticket=ticket,
                symbol=symbol,
                action="RATCHET_SLTP",
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
        req = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": int(ticket),
            "symbol": symbol,
            "sl": float(sl),
            "tp": float(tp),
            "magic": 202611,
            "comment": "APEX_RATCHET",
        }
        res = mt5.order_send(req)
        return int(getattr(res, "retcode", -1)) if res is not None else -1

    def _dispatch_close(self, ticket: int, symbol: str, side: Side, lots: float, price: float) -> int:
        if self._close_sender is not None:
            return int(self._close_sender(ticket, symbol, side, lots, price))
        info = mt5.symbol_info(symbol)
        filling = int(mt5.ORDER_FILLING_FOK)
        if info is not None:
            fm = int(getattr(info, "filling_mode", 0))
            filling = int(mt5.ORDER_FILLING_FOK) if (fm & 1) else (int(mt5.ORDER_FILLING_IOC) if (fm & 2) else int(mt5.ORDER_FILLING_RETURN))
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": int(ticket),
            "symbol": symbol,
            "volume": float(lots),
            "type": int(mt5.ORDER_TYPE_SELL if side == Side.BUY else mt5.ORDER_TYPE_BUY),
            "price": float(price),
            "deviation": 50,
            "magic": 202611,
            "comment": "APEX_INFLIGHT_EXIT",
            "type_time": int(mt5.ORDER_TIME_GTC),
            "type_filling": filling,
        }
        res = mt5.order_send(req)
        return int(getattr(res, "retcode", -1)) if res is not None else -1

    def reconcile_closed_deals(
        self,
        open_tickets: set[int],
        deals_fetcher: Optional[Callable[[int], Sequence[Any]]] = None,
    ) -> list[dict[str, Any]]:
        """Detect tracked positions no longer open on broker and log MFE_R/MAE_R autopsy."""
        closed_tickets = [t for t in list(self.tracked.keys()) if t not in open_tickets]
        reconciled: list[dict[str, Any]] = []
        for ticket in closed_tickets:
            st = self.tracked.pop(ticket)
            deals = deals_fetcher(ticket) if deals_fetcher is not None else (mt5.history_deals_get(position=ticket) or ())
            exit_deals = [d for d in deals if int(getattr(d, "entry", -1)) == int(getattr(mt5, "DEAL_ENTRY_OUT", 1))]
            if not exit_deals:
                continue
            d = exit_deals[-1]
            exit_px = float(getattr(d, "price", st.entry_price) or st.entry_price)
            realized_pnl = float(getattr(d, "profit", 0.0)) + float(getattr(d, "commission", 0.0)) + float(getattr(d, "swap", 0.0))
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
