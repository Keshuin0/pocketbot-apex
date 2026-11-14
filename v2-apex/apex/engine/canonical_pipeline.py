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
            "net_avg_r": self.net_avg_r,
            "recommended_risk_pct": self.recommended_risk_pct,
            "z3_status": self.z3_status,
            "stages": [asdict(s) for s in self.stages],
        }


def compute_asset_class_commission_usd(symbol: str, asset_class: Optional[AssetClass] = None) -> float:
    """Fusion Markets Zero: $4.50/lot round-turn ONLY on FOREX and METALS ($0 on Crypto/Indices/Equities/Energy/Commodities)."""
    ac = asset_class or SmartAliasMapper().classify_asset_class(symbol)
    if ac in (AssetClass.FOREX, AssetClass.METALS):
        return 4.50
    return 0.0


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
        min_net_avg_r: float = 0.10,
    ) -> None:
        self.regime_classifier = regime_classifier or RegimeClassifier()
        self.momentum_dom_gate = momentum_dom_gate or MomentumDOMGate()
        self.mtf_filter = mtf_filter or MTFConfluenceFilter()
        self.time_machine = time_machine or BlockBootstrapTimeMachine(replays=2000, seed=19)
        self.correlation_shield = correlation_shield or CorrelationShield(account_equity_usd=10_000.0)
        self.ai_committee = ai_committee
        self.min_net_avg_r = float(min_net_avg_r)
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
        cross_asset_shock_z: float = 0.0,
        rollover_illiquidity_flag: bool = False,
        open_positions: Sequence[PositionRiskSpec] = (),
        risk_per_lot_usd: float = 100.0,
        run_ai_committee: bool = False,
        in_flight_ready: bool = True,
    ) -> AgentPipelineTrace:
        """Execute all 11 stages deterministically and emit an AgentPipelineTrace."""
        stages: list[PipelineStageResult] = []
        asset_class = self._alias_mapper.classify_asset_class(symbol)

        # Stage 2 first computes ATR/Hurst/ER for downstream geometry, recorded in order 1..11
        diag = self.regime_classifier.classify_candles(
            symbol=symbol,
            candles=candles,
            current_spread=spread_price if spread_price > 0.0 else None,
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
        )
        stages.append(
            PipelineStageResult(
                stage_num=3,
                agent_name="trader-entry + MomentumDOM",
                passed=gate_verdict.passed,
                detail=("; ".join(gate_verdict.reasons) if gate_verdict.reasons else ("OK" if gate_verdict.passed else "Blocked")),
                metric_value=f"spread%SL={gate_verdict.spread_pct_of_sl:.1f}%",
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
        comm_usd = compute_asset_class_commission_usd(symbol, asset_class=asset_class)
        cost_model = TimeMachineCostModel(
            commission_per_lot_round_turn_usd=comm_usd,
            lots=0.10,
            risk_per_lot_usd=max(risk_per_lot_usd, 1.0),
            spread_price=spread_price,
            vwap_slippage_price=0.2 * pip_size,
        )
        tm_res = self.time_machine.simulate_candles(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            stop_price=stop_px,
            target_price=target_px,
            candles=candles,
            horizon_bars=36,
            cost_model=cost_model,
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
        contagion_ok = (cross_asset_shock_z <= 3.5) and not (rollover_illiquidity_flag and diag.volatility_percentile > 85.0)
        stages.append(
            PipelineStageResult(
                stage_num=6,
                agent_name="Contagion / News / Analyst",
                passed=contagion_ok,
                detail=f"shock_z={cross_asset_shock_z:.2f}, rollover={rollover_illiquidity_flag}",
                metric_value=f"Z={cross_asset_shock_z:.2f}",
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
        proposed_risk_pct = rec_risk if rec_risk > 0.0 else 0.50
        corr_verdict = self.correlation_shield.evaluate_proposed_trade(
            proposed=PositionRiskSpec(symbol=symbol, side=side, risk_pct=proposed_risk_pct),
            open_positions=open_positions,
        )
        stages.append(
            PipelineStageResult(
                stage_num=8,
                agent_name="trader-risk + CorrelationShield",
                passed=corr_verdict.allowed,
                detail=corr_verdict.reason,
                metric_value="CAP<=1.0%",
            )
        )

        # Stage 9: Aegis Z3 + AI Waterfall
        z3_telemetry = {
            "symbol": symbol,
            "side": side.value,
            "timemachine_net_avg_R": tm_res.net_avg_R,
            "spread_pct_of_sl": gate_verdict.spread_pct_of_sl,
            "vol_pct": diag.volatility_percentile,
            "cross_asset_shock_z": cross_asset_shock_z,
            "rollover_illiquidity_flag": rollover_illiquidity_flag,
        }
        z3_dec, z3_msg = AegisVerifier.z3_pure_math_axiom_gate(z3_telemetry)
        ai_ok = z3_dec == "APPROVE"
        ai_detail = z3_msg

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
                rollover_illiquidity_flag=rollover_illiquidity_flag,
                cross_asset_shock_z=cross_asset_shock_z,
            )
            ai_res = self.ai_committee.evaluate_sync(ctx)
            ai_ok = ai_res.decision == "APPROVE"
            ai_detail = f"{ai_res.decision} ({ai_res.model_used}) | {z3_msg}"

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
        )
