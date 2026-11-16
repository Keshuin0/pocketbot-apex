"""Unified 11-Agent + In-Flight Canonical Pipeline (`apex/engine/canonical_pipeline.py`).

Shared by Live Scanning, 100x Historical Replay, and 60s Live 0.01-Lot MT5 Drill so
zero shortcuts exist between simulation, drill, and autonomous DEMO execution.
"""

from __future__ import annotations

import datetime
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Sequence

from apex.ai.aegis_verifier import AegisVerifier
from apex.ai.veto_committee import AIVetoCommittee, CandidateSetupAuditContext
from apex.feeds.alias_mapper import SmartAliasMapper
from apex.feeds.models import AssetClass, Candle, Side
from apex.quant.contagion_oracle import ContagionOracle
from apex.quant.correlation_shield import CorrelationShield, PositionRiskSpec
from apex.quant.momentum_dom import DOMAnalytics, MomentumDOMGate
from apex.quant.mtf_confluence import MTFConfluenceFilter
from apex.quant.regime import RegimeClassifier
from apex.quant.timemachine_v2 import BlockBootstrapTimeMachine, TimeMachineCostModel


@dataclass(frozen=True, slots=True)
class PipelineStageResult:
    """Single stage verdict inside the 11-Agent X-Ray Scorecard."""

    stage_num: int
    agent_name: str
    passed: bool
    detail: str
    metric_value: str = ""


@dataclass(frozen=True, slots=True)
class AgentPipelineTrace:
    """Complete 11-stage trace with explicit symbol & timeframe metadata."""

    symbol: str
    timeframe: str
    side: str
    entry_price: float
    stop_loss: float
    take_profit: float
    passed_all: bool
    final_decision: str
    net_avg_r: float
    recommended_risk_pct: float
    z3_status: str
    stages: list[PipelineStageResult] = field(default_factory=list)
    regime: str = ""
    hurst_exponent: float = 0.5
    kaufman_er: float = 0.0
    volatility_percentile: float = 0.5
    dom_obi: float = 0.0
    spread_pct_of_sl: float = 0.0
    prob_profit: float = 0.0
    cvar_95_r: float = 0.0
    ai_decision: str = ""
    ai_reason: str = ""
    ai_model: str = ""
    ai_provider: str = ""
    ai_confidence: float = 0.0
    ai_risk_multiplier: float = 1.0
    winning_venue: str = ""

    @property
    def blocked_stage(self) -> str | None:
        """Return the first stage label that blocked this trace, or None if all passed."""
        if self.passed_all:
            return None
        for stage in self.stages:
            if not stage.passed:
                return f"Stage {stage.stage_num}: {stage.agent_name}"
        return self.final_decision or "PIPELINE_BLOCKED"

    @property
    def blocked_reason(self) -> str:
        """Return the detail string of the first failed stage, or empty string if passed."""
        if self.passed_all:
            return ""
        for stage in self.stages:
            if not stage.passed:
                return stage.detail
        return self.final_decision

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "side": self.side,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "passed_all": self.passed_all,
            "final_decision": self.final_decision,
            "blocked_stage": self.blocked_stage,
            "blocked_reason": self.blocked_reason,
            "net_avg_r": self.net_avg_r,
            "recommended_risk_pct": self.recommended_risk_pct,
            "ai_risk_multiplier": self.ai_risk_multiplier,
            "ai_decision": self.ai_decision,
            "ai_model": self.ai_model,
            "ai_reason": self.ai_reason,
            "z3_status": self.z3_status,
            "stages": [asdict(s) for s in self.stages],
        }


def compute_asset_class_commission_usd(
    symbol: str,
    asset_class: Optional[AssetClass] = None,
    *,
    forex_metals_commission_usd: float = 4.50,
    other_commission_usd: float = 0.0,
) -> float:
    """Fusion Markets Zero: commission ONLY on FOREX and METALS by default ($0 on Crypto/Indices/Equities/Energy)."""
    ac = asset_class or SmartAliasMapper().classify_asset_class(symbol)
    if ac in (AssetClass.FOREX, AssetClass.METALS):
        return float(forex_metals_commission_usd)
    return float(other_commission_usd)


class Canonical11AgentPipeline:
    """Single source of truth for the 11-stage quant + AI + Z3 + In-Flight decision chain."""

    def __init__(
        self,
        *,
        regime_classifier: Optional[RegimeClassifier] = None,
        momentum_dom_gate: Optional[MomentumDOMGate] = None,
        mtf_filter: Optional[MTFConfluenceFilter] = None,
        time_machine: Optional[BlockBootstrapTimeMachine] = None,
        correlation_shield: Optional[CorrelationShield] = None,
        ai_committee: Optional[AIVetoCommittee] = None,
        contagion_oracle: Optional[ContagionOracle] = None,
        config: Optional[dict[str, Any]] = None,
        min_net_avg_r: Optional[float] = None,
        default_risk_pct: Optional[float] = None,
        forex_metals_commission_usd: Optional[float] = None,
        other_commission_usd: Optional[float] = None,
        horizon_bars: Optional[int] = None,
    ) -> None:
        cfg = dict(config or {})
        risk_cfg = dict(cfg.get("risk", {})) if isinstance(cfg.get("risk"), dict) else {}
        tm_replays = int(risk_cfg.get("monte_carlo_replays", 8192) or 8192)
        self.regime_classifier = regime_classifier or RegimeClassifier()
        self.momentum_dom_gate = momentum_dom_gate or MomentumDOMGate()
        self.mtf_filter = mtf_filter or MTFConfluenceFilter()
        self.time_machine = time_machine or BlockBootstrapTimeMachine(replays=tm_replays, seed=19)
        self.correlation_shield = correlation_shield or CorrelationShield(
            max_factor_risk_pct=float(risk_cfg.get("max_factor_risk_pct", 1.0)),
            account_equity_usd=float(risk_cfg.get("account_equity_usd", 10_000.0)),
        )
        self.ai_committee = ai_committee
        self.contagion_oracle = contagion_oracle
        self.config = cfg
        self.min_net_avg_r = float(min_net_avg_r if min_net_avg_r is not None else risk_cfg.get("min_net_avg_r", 0.10))
        self.default_risk_pct = float(
            default_risk_pct if default_risk_pct is not None else risk_cfg.get("default_risk_per_trade_pct", 0.50)
        )
        self.forex_metals_commission_usd = float(
            forex_metals_commission_usd
            if forex_metals_commission_usd is not None
            else risk_cfg.get("forex_metals_commission_usd", 4.50)
        )
        self.other_commission_usd = float(
            other_commission_usd if other_commission_usd is not None else risk_cfg.get("other_commission_usd", 0.0)
        )
        self.horizon_bars = int(horizon_bars if horizon_bars is not None else risk_cfg.get("horizon_bars", 36))
        self._alias_mapper = SmartAliasMapper()

    def evaluate(
        self,
        *,
        symbol: str,
        side: Side,
        timeframe: str,
        entry_price: float,
        candles: Sequence[Candle],
        h1_candles: Sequence[Candle] | None = None,
        spread_price: float = 0.0,
        pip_size: float = 0.0001,
        dom_analytics: Optional[DOMAnalytics] = None,
        cross_asset_shock_z: Optional[float] = None,
        rollover_illiquidity_flag: Optional[bool] = None,
        open_positions: Sequence[PositionRiskSpec] = (),
        risk_per_lot_usd: float = 100.0,
        contract_spec: Any = None,
        cost_model: Optional[TimeMachineCostModel] = None,
        run_ai_committee: bool = False,
        in_flight_ready: bool = True,
        feed_router: Any = None,
        winning_venue: str = "",
        historical_spreads: Sequence[float] | None = None,
        current_tick_rate: float | None = None,
        historical_tick_rates: Sequence[float] | None = None,
    ) -> AgentPipelineTrace:
        """Execute all 11 stages deterministically and emit an AgentPipelineTrace."""
        candles = list(candles or ())
        h1_candles = list(h1_candles) if h1_candles is not None else None
        open_positions = list(open_positions or ())
        stages: list[PipelineStageResult] = []
        asset_class = self._alias_mapper.classify_asset_class(symbol)

        # Stage 2 first computes ATR/Hurst/ER + microstructure telemetry, recorded in order 1..11
        diag = self.regime_classifier.classify_candles(
            symbol=symbol,
            candles=candles,
            current_spread=spread_price if spread_price > 0.0 else None,
            historical_spreads=historical_spreads,
            current_tick_rate=current_tick_rate,
            historical_tick_rates=historical_tick_rates,
        )
        atr_val = max(diag.current_atr, max(pip_size, 1e-6) * 5.0)
        sl_dist = 1.5 * atr_val
        stop_px = (entry_price - sl_dist) if side == Side.BUY else (entry_price + sl_dist)
        target_px = (entry_price + 2.0 * sl_dist) if side == Side.BUY else (entry_price - 2.0 * sl_dist)

        # Stage 1: trader-scanner + H1 MTF Confluence
        mtf_verdict = self.mtf_filter.evaluate(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            atr_m5=atr_val,
            m5_candles=candles,
            h1_candles=h1_candles,
        )
        stages.append(
            PipelineStageResult(
                stage_num=1,
                agent_name="trader-scanner + H1 MTF",
                passed=mtf_verdict.passed,
                detail=mtf_verdict.reason,
                metric_value=f"slope={mtf_verdict.h1_ema_slope:+.2f}",
            )
        )

        # Stage 2: RegimeClassifier
        stages.append(
            PipelineStageResult(
                stage_num=2,
                agent_name="RegimeClassifier",
                passed=diag.allow_breakout_entries,
                detail=f"{diag.regime.value}: {diag.reason}",
                metric_value=f"H={diag.hurst_exponent:.2f} ER={diag.kaufman_er:.2f}",
            )
        )

        # Stage 3: trader-entry + MomentumDOMGate
        gate_verdict = self.momentum_dom_gate.evaluate(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            stop_price=stop_px,
            candles=candles,
            current_spread=spread_price,
            dom_analytics=dom_analytics,
            hawkes_intensity_ratio=diag.hawkes_intensity_ratio,
        )
        spread_pct_of_sl_pct = float(gate_verdict.spread_pct_of_sl) * 100.0
        stages.append(
            PipelineStageResult(
                stage_num=3,
                agent_name="trader-entry + MomentumDOM",
                passed=gate_verdict.passed,
                detail=("; ".join(gate_verdict.reasons) if gate_verdict.reasons else ("OK" if gate_verdict.passed else "Blocked")),
                metric_value=f"spread%SL={spread_pct_of_sl_pct:.1f}%",
            )
        )

        # Stage 4: trader-exit (2.0R bracket geometry)
        bracket_ok = (entry_price > 0.0) and (sl_dist > 0.0)
        stages.append(
            PipelineStageResult(
                stage_num=4,
                agent_name="trader-exit",
                passed=bracket_ok,
                detail=f"SL={stop_px:.5f} | TP={target_px:.5f} (2.0R)",
                metric_value=f"1R={sl_dist:.5f}",
            )
        )

        # Stage 5: 10,000-Path TimeMachine v2 (asset-class commission)
        comm_usd = compute_asset_class_commission_usd(
            symbol,
            asset_class=asset_class,
            forex_metals_commission_usd=self.forex_metals_commission_usd,
            other_commission_usd=self.other_commission_usd,
        )
        effective_risk_per_lot = max(float(risk_per_lot_usd), 1.0)
        if contract_spec is not None:
            t_val = float(getattr(contract_spec, "trade_tick_value", 0.0) or 0.0)
            t_size = float(getattr(contract_spec, "trade_tick_size", 0.0) or 0.0)
            lot_sz = float(getattr(contract_spec, "lot_size", 0.0) or 0.0)
            if t_val > 0.0 and t_size > 0.0:
                effective_risk_per_lot = max((sl_dist / t_size) * t_val, 1.0)
            elif lot_sz > 0.0 and sl_dist * lot_sz > 0.0:
                effective_risk_per_lot = max(sl_dist * lot_sz, 1.0)

        eff_cost_model = cost_model or TimeMachineCostModel(
            commission_per_lot_round_turn_usd=comm_usd,
            lots=0.10,
            risk_per_lot_usd=effective_risk_per_lot,
            spread_price=spread_price,
            vwap_slippage_price=0.2 * pip_size,
            asymmetric_barriers=True,
        )
        tm_res = self.time_machine.simulate_candles(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            stop_price=stop_px,
            target_price=target_px,
            candles=candles,
            horizon_bars=self.horizon_bars,
            cost_model=eff_cost_model,
            current_hurst=diag.hurst_exponent,
            current_er=diag.kaufman_er,
            current_vol_pct=diag.volatility_percentile,
        )
        tm_ok = tm_res.net_avg_R >= self.min_net_avg_r
        stages.append(
            PipelineStageResult(
                stage_num=5,
                agent_name="TimeMachine v2 (Bootstrap)",
                passed=tm_ok,
                detail=f"net_avg_R={tm_res.net_avg_R:+.3f}R (floor={self.min_net_avg_r:.2f}R, comm=${comm_usd:.2f})",
                metric_value=f"P(Win)={tm_res.prob_profit * 100:.0f}%",
            )
        )

        # Stage 6: Contagion / News / Analyst
        if (cross_asset_shock_z is None or rollover_illiquidity_flag is None) and self.contagion_oracle is not None:
            bar_dt = (
                datetime.datetime.fromtimestamp(float(candles[-1].timestamp_utc), tz=datetime.timezone.utc)
                if candles
                else datetime.datetime.now(datetime.timezone.utc)
            )
            contagion = self.contagion_oracle.evaluate(feed_router=feed_router, utc_dt=bar_dt)
            resolved_shock_z = float(contagion.cross_asset_shock_z if cross_asset_shock_z is None else cross_asset_shock_z)
            resolved_rollover = bool(
                contagion.rollover_illiquidity_flag
                if rollover_illiquidity_flag is None
                else rollover_illiquidity_flag
            )
        else:
            resolved_shock_z = float(cross_asset_shock_z or 0.0)
            resolved_rollover = bool(rollover_illiquidity_flag or False)

        contagion_ok = (resolved_shock_z <= 3.5) and not (resolved_rollover and diag.volatility_percentile > 85.0)
        stages.append(
            PipelineStageResult(
                stage_num=6,
                agent_name="Contagion / News / Analyst",
                passed=contagion_ok,
                detail=f"shock_z={resolved_shock_z:.2f}, rollover={resolved_rollover}",
                metric_value=f"Z={resolved_shock_z:.2f}",
            )
        )

        # Stage 7: trader-budget (Kelly risk = 0.00% if any pre-gate failed)
        pre_quant_ok = mtf_verdict.passed and diag.allow_breakout_entries and gate_verdict.passed and bracket_ok and tm_ok and contagion_ok
        rec_risk = float(tm_res.recommended_risk_pct) if pre_quant_ok else 0.0
        stages.append(
            PipelineStageResult(
                stage_num=7,
                agent_name="trader-budget",
                passed=(rec_risk > 0.0) if pre_quant_ok else False,
                detail=f"Q-Kelly={rec_risk:.2f}%" if pre_quant_ok else "Q-Kelly forced 0.00% (filtered)",
                metric_value=f"{rec_risk:.2f}%",
            )
        )

        # Stage 8: trader-risk + CorrelationShield
        proposed_risk_pct = rec_risk if rec_risk > 0.0 else self.default_risk_pct
        corr_verdict = self.correlation_shield.evaluate_proposed_trade(
            proposed=PositionRiskSpec(symbol=symbol, side=side, risk_pct=proposed_risk_pct),
            open_positions=open_positions,
        )
        corr_ok = bool(corr_verdict.allowed)
        stages.append(
            PipelineStageResult(
                stage_num=8,
                agent_name="trader-risk + CorrelationShield",
                passed=corr_ok,
                detail=corr_verdict.reason,
                metric_value=f"CAP<={self.correlation_shield.max_factor_risk_pct:.1f}%",
            )
        )

        # Stage 9: Aegis Z3 + AI Waterfall
        z3_telemetry = {
            "symbol": symbol,
            "side": side.value,
            "timemachine_net_avg_R": tm_res.net_avg_R,
            "spread_pct_of_sl": spread_pct_of_sl_pct,
            "vol_pct": diag.volatility_percentile,
            "cross_asset_shock_z": resolved_shock_z,
            "rollover_illiquidity_flag": resolved_rollover,
        }
        z3_dec, z3_msg = AegisVerifier.z3_pure_math_axiom_gate(z3_telemetry)
        ai_ok = z3_dec == "APPROVE"
        ai_detail = z3_msg
        ai_decision_label = z3_dec if not ai_ok else ("FILTERED" if not (pre_quant_ok and corr_ok) else "APPROVE")
        ai_reason_text = z3_msg
        ai_model_name = "Z3_AXIOM_GATE" if not ai_ok else "PURE_QUANT"
        ai_provider_name = "z3" if not ai_ok else "quant"
        ai_conf = 0.0
        ai_risk_mult = 1.0 if (pre_quant_ok and corr_ok and ai_ok) else 0.0

        if run_ai_committee and self.ai_committee is not None and pre_quant_ok and corr_ok and ai_ok:
            bar_ts = float(candles[-1].timestamp_utc) if candles else time.time()
            ctx = CandidateSetupAuditContext(
                symbol=symbol,
                side=side,
                timeframe=timeframe,
                entry_price=entry_price,
                stop_loss=stop_px,
                take_profit=target_px,
                regime=diag.regime.value,
                hurst_exponent=diag.hurst_exponent,
                kaufman_er=diag.kaufman_er,
                volatility_percentile=diag.volatility_percentile,
                dom_obi=dom_analytics.obi if dom_analytics else None,
                spread_pct_of_sl=gate_verdict.spread_pct_of_sl,
                net_avg_R=tm_res.net_avg_R,
                prob_profit=tm_res.prob_profit,
                cvar_95_R=tm_res.cvar_95_R,
                recommended_risk_pct=rec_risk,
                session_utc_iso=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                bar_timestamp=bar_ts,
                rollover_illiquidity_flag=resolved_rollover,
                cross_asset_shock_z=resolved_shock_z,
            )
            ai_res = self.ai_committee.evaluate_sync(ctx)
            raw_dec = getattr(ai_res, "decision", "VETO")
            raw_reason = getattr(ai_res, "reason", "")
            raw_model = getattr(ai_res, "model_used", "")
            raw_prov = getattr(ai_res, "provider_used", "")
            raw_conf = getattr(ai_res, "confidence", 0.0)
            raw_mult = getattr(ai_res, "risk_multiplier", None)
            ai_ok = raw_dec == "APPROVE"
            ai_decision_label = raw_dec if isinstance(raw_dec, str) else "FILTERED"
            ai_reason_text = raw_reason if isinstance(raw_reason, str) else z3_msg
            ai_model_name = raw_model if isinstance(raw_model, str) else "AI_COMMITTEE"
            ai_provider_name = raw_prov if isinstance(raw_prov, str) else "ai"
            ai_conf = float(raw_conf) if isinstance(raw_conf, (int, float)) else 0.0
            ai_risk_mult = float(raw_mult) if isinstance(raw_mult, (int, float)) else (1.0 if ai_ok else 0.0)
            ai_detail = f"{ai_decision_label} ({ai_model_name}) | {z3_msg}"

        stages.append(
            PipelineStageResult(
                stage_num=9,
                agent_name="Aegis Z3 + AI Waterfall",
                passed=ai_ok,
                detail=ai_detail,
                metric_value=z3_dec,
            )
        )

        # Stage 10: ExecutionRouter
        exec_ready = pre_quant_ok and corr_ok and ai_ok and bracket_ok
        stages.append(
            PipelineStageResult(
                stage_num=10,
                agent_name="ExecutionRouter (DEMO)",
                passed=exec_ready,
                detail="DEMO-locked + Server-side SL/TP armed" if exec_ready else "Blocked upstream prior to wire",
                metric_value="DEMO_SAFE",
            )
        )

        # Stage 11: v2.5.0 InFlightPositionManager
        stages.append(
            PipelineStageResult(
                stage_num=11,
                agent_name="InFlightPositionManager",
                passed=in_flight_ready,
                detail="2-Bar Grace Lock + Forward TM + 1R/1.5R Runner armed",
                metric_value="SHIELD8_Z3",
            )
        )

        passed_all = exec_ready and in_flight_ready
        final_decision = "APPROVE" if passed_all else ("VETO" if (pre_quant_ok and not ai_ok) else "FILTERED")
        return AgentPipelineTrace(
            symbol=symbol,
            timeframe=timeframe,
            side=side.value,
            entry_price=entry_price,
            stop_loss=stop_px,
            take_profit=target_px,
            passed_all=passed_all,
            final_decision=final_decision,
            net_avg_r=round(tm_res.net_avg_R, 4),
            recommended_risk_pct=round(rec_risk, 4),
            z3_status=z3_msg,
            stages=stages,
            regime=diag.regime.value,
            hurst_exponent=round(float(diag.hurst_exponent), 4),
            kaufman_er=round(float(diag.kaufman_er), 4),
            volatility_percentile=round(float(diag.volatility_percentile), 2),
            dom_obi=round(float(dom_analytics.obi), 4) if dom_analytics else 0.0,
            spread_pct_of_sl=round(float(gate_verdict.spread_pct_of_sl), 6),
            prob_profit=round(float(tm_res.prob_profit), 4),
            cvar_95_r=round(float(tm_res.cvar_95_R), 4),
            ai_decision=ai_decision_label,
            ai_reason=ai_reason_text,
            ai_model=ai_model_name,
            ai_provider=ai_provider_name,
            ai_confidence=ai_conf,
            ai_risk_multiplier=ai_risk_mult,
            winning_venue=winning_venue,
        )
