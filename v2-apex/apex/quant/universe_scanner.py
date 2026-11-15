"""Two-Stage Full-Market Funnel Scanner (`apex/quant/universe_scanner.py`).

Stage A: Fast sub-second universe sweep across ALL available broker catalog symbols ($0 AI token cost).
         Filters out closed sessions and abnormal spreads; ranks candidates by momentum/ER mover scores.
Stage B: Deep institutional quantitative pipeline on Top Movers (Regime + DOM Gate + 10,000-Path Time Machine + AI Veto).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from apex.ai.aegis_verifier import AegisVerifier
from apex.ai.veto_committee import AIVetoCommittee, CandidateSetupAuditContext
from apex.cockpit.tui_dashboard import BrokerExecutionEvent, SymbolCockpitRow
from apex.engine.canonical_pipeline import (
    AgentPipelineTrace,
    PipelineStageResult,
    compute_asset_class_commission_usd,
)
from apex.feeds.alias_mapper import SmartAliasMapper
from apex.feeds.dual_router import MarketFeedRouter
from apex.feeds.models import AssetClass, DiscoveredSymbol, Side, Tick
from apex.ledger.walkforward_db import WalkForwardLedgerDB
from apex.quant.contagion_oracle import ContagionOracle, ContagionTelemetry
from apex.quant.momentum_dom import (
    InstrumentTier,
    MomentumDOMGate,
    classify_instrument_tier,
)
from apex.quant.mtf_confluence import MTFConfluenceFilter
from apex.quant.regime import RegimeClassifier, compute_hurst_exponent
from apex.quant.timemachine_v2 import BlockBootstrapTimeMachine, TimeMachineCostModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CandidatePreScore:
    """Stage A pre-scoring result for an active symbol."""

    symbol: str
    canonical_name: str
    asset_class: AssetClass
    tier: InstrumentTier
    timeframe: str
    bid: float
    ask: float
    spread_pips: float
    impulse_body_ratio: float
    kaufman_er: float
    hurst_estimate: float
    composite_mover_score: float
    is_open: bool
    status_reason: str = ""


@dataclass(frozen=True, slots=True)
class UniverseScanSummary:
    """Summary metrics of a complete two-stage universe scan."""

    timestamp_utc: float
    total_scanned: int
    open_scanned: int
    closed_skipped: int
    candidates_advanced: int
    top_candidates: tuple[str, ...] = field(default_factory=tuple)
    asset_class_counts: dict[str, int] = field(default_factory=dict)


class UniverseScanner:
    """Two-Stage Full-Market Funnel Scanner for PocketBot Apex.

    Discovers all available broker symbols (e.g. 251 symbols across Fusion Markets MT5/cTrader),
    runs sub-second Stage A pre-scoring to filter closed markets and rank top active movers,
    and subjects top movers to the full institutional Stage B pipeline.
    """

    def __init__(
        self,
        feed_router: MarketFeedRouter,
        regime_classifier: RegimeClassifier,
        momentum_dom_gate: MomentumDOMGate,
        time_machine: BlockBootstrapTimeMachine,
        ai_committee: AIVetoCommittee,
        ledger: WalkForwardLedgerDB,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.feed_router: MarketFeedRouter = feed_router
        self.regime_classifier: RegimeClassifier = regime_classifier
        self.momentum_dom_gate: MomentumDOMGate = momentum_dom_gate
        self.time_machine: BlockBootstrapTimeMachine = time_machine
        self.ai_committee: AIVetoCommittee = ai_committee
        self.ledger: WalkForwardLedgerDB = ledger
        self.config: dict[str, Any] = config or {}

        scanner_cfg = self.config.get("scanner", {})
        self.max_candidates_per_cycle: int = int(scanner_cfg.get("max_candidates_per_cycle", 12))
        self.auto_enable_market_watch: bool = bool(scanner_cfg.get("auto_enable_market_watch", True))
        self.max_tick_age_seconds: float = float(scanner_cfg.get("max_tick_age_seconds", 120.0))
        self.min_candles_required: int = int(scanner_cfg.get("min_candles_required", 30))
        self.min_net_avg_R: float = float(scanner_cfg.get("min_net_avg_R", 0.10))

        self.alias_mapper: SmartAliasMapper = SmartAliasMapper()
        self.contagion_oracle: ContagionOracle = ContagionOracle()
        mtf_cfg = self.config.get("filters", {}).get("mtf_confluence", {})
        self.mtf_filter: MTFConfluenceFilter = MTFConfluenceFilter(
            enabled=bool(mtf_cfg.get("enabled", True)),
            ceiling_atr_buffer=float(mtf_cfg.get("ceiling_atr_buffer", 0.30)),
            max_chase_atr_multiple=float(mtf_cfg.get("max_chase_atr_multiple", 2.2)),
        )
        self.last_contagion: ContagionTelemetry | None = None
        self.recent_audit_events: list[BrokerExecutionEvent] = []
        self._cached_catalog: list[DiscoveredSymbol] = []
        self._last_catalog_discovery_time: float = 0.0

    def discover_catalog(self, force: bool = False) -> list[DiscoveredSymbol]:
        """Fetch or refresh the full broker catalog across active feeds."""
        return self.discover_or_get_catalog(force=force)

    def get_grouped_catalog(self, force: bool = False) -> dict[str, list[str]]:
        """Return the broker catalog grouped by AssetClass for the Chrome Cockpit picker."""
        catalog_items = self.discover_catalog(force=force)
        grouped: dict[str, list[str]] = {}
        for item in catalog_items:
            ac = item.asset_class.value
            grouped.setdefault(ac, []).append(item.canonical_name or item.name)
        for ac in grouped:
            grouped[ac] = sorted(set(grouped[ac]))
        return grouped

    @staticmethod
    def infer_candidate_side(
        candles: Sequence[Any],
        h1_candles: Sequence[Any] | None = None,
    ) -> Side:
        """Dynamically infer Side.BUY vs Side.SELL from M5/M15 breakout, bar polarity, and H1 EMA slope."""
        if not candles or len(candles) < 3:
            return Side.BUY

        latest_c = candles[-1]
        is_bearish_bar = float(latest_c.close) < float(latest_c.open)
        is_bullish_bar = float(latest_c.close) > float(latest_c.open)

        closes = np.asarray([float(c.close) for c in candles], dtype=np.float64)
        alpha = 2.0 / 21.0
        ema20 = float(closes[0])
        ema_hist: list[float] = [ema20]
        for px in closes[1:]:
            ema20 = alpha * float(px) + (1.0 - alpha) * ema20
            ema_hist.append(ema20)
        m5_slope_down = ema_hist[-1] < ema_hist[-min(5, len(ema_hist))]
        m5_bearish = (float(closes[-1]) < ema_hist[-1]) and m5_slope_down

        h1_bearish = False
        h1_bullish = False
        if h1_candles and len(h1_candles) >= 10:
            h1_closes = np.asarray([float(c.close) for c in h1_candles], dtype=np.float64)
            h1_ema = float(h1_closes[0])
            h1_hist: list[float] = [h1_ema]
            for px in h1_closes[1:]:
                h1_ema = alpha * float(px) + (1.0 - alpha) * h1_ema
                h1_hist.append(h1_ema)
            h1_slope = h1_hist[-1] - h1_hist[-min(5, len(h1_hist))]
            h1_bearish = (float(h1_closes[-1]) < h1_hist[-1]) and (h1_slope < 0.0)
            h1_bullish = (float(h1_closes[-1]) > h1_hist[-1]) and (h1_slope > 0.0)

        prior = candles[-13:-1] if len(candles) >= 14 else candles[:-1]
        prior_hi = max(float(c.high) for c in prior)
        prior_lo = min(float(c.low) for c in prior)
        broke_down = float(latest_c.close) < prior_lo
        broke_up = float(latest_c.close) > prior_hi

        if broke_down and not broke_up:
            return Side.SELL
        if broke_up and not broke_down:
            return Side.BUY
        if is_bearish_bar and (m5_bearish or h1_bearish):
            return Side.SELL
        if is_bullish_bar and (not m5_bearish or h1_bullish):
            return Side.BUY
        if h1_bearish or m5_bearish:
            return Side.SELL
        return Side.BUY

    def discover_or_get_catalog(self, force: bool = False) -> list[DiscoveredSymbol]:
        """Fetch or refresh the full broker catalog across active feeds."""
        now = time.time()
        # Refresh catalog every 30 minutes or when forced / empty
        if force or not self._cached_catalog or (now - self._last_catalog_discovery_time > 1800.0):
            catalog = self.feed_router.discover_catalog(auto_enable_market_watch=self.auto_enable_market_watch)
            if catalog:
                self._cached_catalog = catalog
                self._last_catalog_discovery_time = now
        return list(self._cached_catalog)

    def is_symbol_market_open(
        self,
        symbol: str,
        tick: Tick | None,
        now_utc: float | None = None,
    ) -> tuple[bool, str]:
        """Determine if a symbol is open and actively trading."""
        mt5_client = getattr(self.feed_router, "mt5", None)
        if mt5_client is not None and getattr(mt5_client, "_initialized", False):
            check_fn = getattr(mt5_client, "is_symbol_trade_mode_full", None)
            if callable(check_fn) and not check_fn(symbol):
                return False, "TRADE_MODE_CLOSED"

        if tick is None:
            return False, "NO_TICK"

        if tick.bid <= 0.0 or tick.ask <= 0.0:
            return False, "ZERO_PRICE"

        if tick.ask < tick.bid:
            return False, "CROSSED_SPREAD"

        current_time = now_utc if now_utc is not None else time.time()
        tick_age = abs(current_time - tick.timestamp_utc)
        if tick_age > self.max_tick_age_seconds:
            return False, f"STALE_TICK_{int(tick_age)}s"

        return True, "ACTIVE"

    def pre_score_symbol(
        self,
        symbol_info: DiscoveredSymbol,
        now_utc: float | None = None,
    ) -> CandidatePreScore:
        """Stage A: Sub-second pre-score for a single symbol ($0 token cost)."""
        sym = symbol_info.name
        tier = classify_instrument_tier(sym)
        tf = "M5" if tier == InstrumentTier.TIGHT_SPREAD_M5 else "M15"

        spec = self.feed_router.get_contract_spec(sym)
        pip_size = spec.pip_size if spec else (0.01 if "JPY" in sym or "XAU" in sym else 0.0001)

        tick, _ = self.feed_router.get_tick(sym)
        is_open, status_reason = self.is_symbol_market_open(sym, tick, now_utc=now_utc)

        if not is_open or tick is None:
            return CandidatePreScore(
                symbol=sym,
                canonical_name=symbol_info.canonical_name,
                asset_class=symbol_info.asset_class,
                tier=tier,
                timeframe=tf,
                bid=tick.bid if tick else 0.0,
                ask=tick.ask if tick else 0.0,
                spread_pips=(tick.spread / max(pip_size, 1e-9)) if tick else 0.0,
                impulse_body_ratio=0.0,
                kaufman_er=0.0,
                hurst_estimate=0.50,
                composite_mover_score=0.0,
                is_open=False,
                status_reason=status_reason,
            )

        spread_pips = tick.spread / max(pip_size, 1e-9)

        # Fetch recent bars for momentum pre-scoring
        candles = self.feed_router.get_candles(sym, timeframe=tf, count=50)
        if len(candles) < self.min_candles_required:
            return CandidatePreScore(
                symbol=sym,
                canonical_name=symbol_info.canonical_name,
                asset_class=symbol_info.asset_class,
                tier=tier,
                timeframe=tf,
                bid=tick.bid,
                ask=tick.ask,
                spread_pips=spread_pips,
                impulse_body_ratio=0.0,
                kaufman_er=0.0,
                hurst_estimate=0.50,
                composite_mover_score=0.0,
                is_open=True,
                status_reason="INSUFFICIENT_BARS",
            )

        closes = np.array([c.close for c in candles], dtype=np.float64)
        highs = np.array([c.high for c in candles], dtype=np.float64)
        lows = np.array([c.low for c in candles], dtype=np.float64)

        # Kaufman Efficiency Ratio
        net_change = abs(closes[-1] - closes[0])
        sum_abs_diff = float(np.sum(np.abs(np.diff(closes))))
        er = float(net_change / sum_abs_diff) if sum_abs_diff > 0.0 else 0.0

        # ATR calculation (14 periods)
        trs = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])),
        )
        atr = float(np.mean(trs[-14:])) if len(trs) >= 14 else (float(np.mean(trs)) if len(trs) > 0 else 1e-4)
        safe_atr = max(atr, pip_size * 2.0, 1e-6)

        # Latest candle impulse body ratio
        latest_c = candles[-1]
        body = abs(latest_c.close - latest_c.open)
        body_ratio = float(body / safe_atr)

        # Fast Hurst exponent via logarithmic variance
        if len(closes) >= 30:
            hurst = float(compute_hurst_exponent(closes))
        else:
            hurst = 0.50

        # Spread penalty: tighter spread & lower spread/ATR = higher score (penalize wide-spread exotics)
        spread_to_atr = tick.spread / max(safe_atr, 1e-9)
        spread_factor = 1.0 / (
            1.0 + max(0.0, spread_pips - 2.0) * 0.18 + max(0.0, spread_to_atr - 0.12) * 8.0
        )
        if spread_pips > 12.0:
            spread_factor *= 0.35

        # Penalize MEAN_REVERTING_CHOP (H < 0.50 or ER < 0.18) so TRENDING_EXPANSION majors/indices rank Top 12
        if hurst < 0.50 or er < 0.18:
            regime_multiplier = 0.20
        elif hurst >= 0.55 and er >= 0.25:
            regime_multiplier = 1.35
        else:
            regime_multiplier = 1.0

        # Composite Mover Score:
        # 40% Kaufman ER + 30% Impulse Body Ratio + 30% Volatility/Return * Spread Quality * Regime Quality
        pct_return_score = min(abs(closes[-1] - closes[0]) / max(closes[0], 1e-6) * 1000.0, 50.0)
        mover_score = (
            (er * 40.0)
            + (min(body_ratio, 3.0) * 10.0)
            + (pct_return_score * 0.4)
            + (max(0.0, hurst - 0.50) * 40.0)
        ) * spread_factor * regime_multiplier

        return CandidatePreScore(
            symbol=sym,
            canonical_name=symbol_info.canonical_name,
            asset_class=symbol_info.asset_class,
            tier=tier,
            timeframe=tf,
            bid=tick.bid,
            ask=tick.ask,
            spread_pips=spread_pips,
            impulse_body_ratio=body_ratio,
            kaufman_er=er,
            hurst_estimate=hurst,
            composite_mover_score=float(mover_score),
            is_open=True,
            status_reason="ACTIVE_MOVER",
        )

    def sweep_universe(
        self,
        target_symbols: Sequence[DiscoveredSymbol] | Sequence[str] | None = None,
        now_utc: float | None = None,
    ) -> tuple[list[CandidatePreScore], UniverseScanSummary]:
        """Stage A: Fast sub-second universe sweep across all catalog symbols."""
        if target_symbols is None:
            catalog = self.discover_or_get_catalog()
            if not catalog:
                # Fallback to configured symbols if catalog not yet discovered
                default_symbols = self.config.get("symbols", ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "BTCUSD"])
                catalog = [
                    DiscoveredSymbol(
                        name=s,
                        canonical_name=self.alias_mapper.canonicalize(s),
                        asset_class=self.alias_mapper.classify_asset_class(s),
                        path="",
                        digits=5 if "JPY" not in s else 3,
                        point=0.00001 if "JPY" not in s else 0.001,
                        trade_mode=4,
                        is_visible=True,
                        source="default",
                    )
                    for s in default_symbols
                ]
        elif isinstance(target_symbols[0], str):
            catalog = [
                DiscoveredSymbol(
                    name=str(s),
                    canonical_name=self.alias_mapper.canonicalize(str(s)),
                    asset_class=self.alias_mapper.classify_asset_class(str(s)),
                    path="",
                    digits=5,
                    point=0.00001,
                    trade_mode=4,
                    is_visible=True,
                    source="provided",
                )
                for s in target_symbols
            ]
        else:
            catalog = list(target_symbols)

        prescores: list[CandidatePreScore] = []
        open_count = 0
        closed_count = 0
        asset_counts: dict[str, int] = {}

        for item in catalog:
            ac_str = item.asset_class.value
            asset_counts[ac_str] = asset_counts.get(ac_str, 0) + 1

            ps = self.pre_score_symbol(item, now_utc=now_utc)
            prescores.append(ps)
            if ps.is_open:
                open_count += 1
            else:
                closed_count += 1

        # Rank open symbols by composite mover score descending
        open_scores = [ps for ps in prescores if ps.is_open and ps.status_reason != "INSUFFICIENT_BARS"]
        open_scores.sort(key=lambda x: x.composite_mover_score, reverse=True)

        effective_candidates = open_scores if open_scores else prescores
        top_selected = effective_candidates if target_symbols is not None else effective_candidates[: self.max_candidates_per_cycle]
        top_names = tuple(c.symbol for c in top_selected)

        summary = UniverseScanSummary(
            timestamp_utc=now_utc if now_utc is not None else time.time(),
            total_scanned=len(catalog),
            open_scanned=open_count,
            closed_skipped=closed_count,
            candidates_advanced=len(top_selected),
            top_candidates=top_names,
            asset_class_counts=asset_counts,
        )

        return (open_scores if open_scores else prescores), summary

    def evaluate_candidate_pipeline(
        self,
        candidate: CandidatePreScore,
        cost_model: TimeMachineCostModel | None = None,
    ) -> SymbolCockpitRow:
        """Stage B: Full institutional quantitative pipeline on a single top candidate."""
        sym = candidate.symbol
        tf = candidate.timeframe
        pip_size = max(0.01 if "JPY" in sym or "XAU" in sym else 0.0001, 1e-6)
        spec = self.feed_router.get_contract_spec(sym)
        if spec:
            pip_size = max(spec.pip_size, 1e-9)

        tick, _ = self.feed_router.get_tick(sym)
        candles = self.feed_router.get_candles(sym, timeframe=tf, count=200)
        dom = self.feed_router.get_dom_analytics(sym)
        q_state = self.ledger.get_quarantine_state(sym)

        if tick is None or len(candles) < self.min_candles_required:
            return SymbolCockpitRow(
                symbol=sym,
                tier=candidate.tier.value,
                timeframe=tf,
                bid=candidate.bid,
                ask=candidate.ask,
                spread_pips=candidate.spread_pips,
                regime="NO_DATA",
                hurst=candidate.hurst_estimate,
                kaufman_er=candidate.kaufman_er,
                obi=dom.obi if dom else None,
                net_avg_r=None,
                rec_risk_pct=None,
                ai_verdict="STANDBY",
                quarantine_mode=q_state.mode.value,
                profit_factor=q_state.profit_factor,
            )

        spread_px = tick.spread
        spread_pips_atomic = round(max(spread_px / pip_size, 0.0), 2)
        diag = self.regime_classifier.classify_candles(
            symbol=sym,
            candles=candles,
            current_spread=spread_px,
        )

        h1_candles = self.feed_router.get_candles(sym, timeframe="H1", count=40)
        side = self.infer_candidate_side(candles, h1_candles)

        # Symmetric 1.5x ATR stop and 2.0R target
        atr_val = max(diag.current_atr, pip_size * 5.0)
        sl_dist = 1.5 * atr_val
        entry_px = tick.ask if side == Side.BUY else tick.bid
        stop_px = (entry_px - sl_dist) if side == Side.BUY else (entry_px + sl_dist)
        target_px = (entry_px + 2.0 * sl_dist) if side == Side.BUY else (entry_px - 2.0 * sl_dist)

        nbbo = self.feed_router.evaluate_nbbo(
            symbol=sym,
            one_r_price_distance=sl_dist,
            side=side,
        )
        effective_cost_pips = max(float(nbbo.winning_cost_pips), spread_pips_atomic)
        if nbbo.pips_saved > 0.0:
            best_venue_str = f"{nbbo.winning_venue.upper()} ({effective_cost_pips:.1f}p, -{nbbo.pips_saved:.1f}p)"
        else:
            best_venue_str = f"{nbbo.winning_venue.upper()} ({effective_cost_pips:.1f}p)"

        gate_verdict = self.momentum_dom_gate.evaluate(
            symbol=sym,
            side=side,
            entry_price=entry_px,
            stop_price=stop_px,
            candles=candles,
            current_spread=spread_px,
            dom_analytics=dom,
        )

        commission_usd = compute_asset_class_commission_usd(sym, candidate.asset_class)
        risk_per_lot_usd = 100.0
        if spec is not None:
            if getattr(spec, "trade_tick_value", 0.0) > 0.0 and getattr(spec, "trade_tick_size", 0.0) > 0.0:
                risk_per_lot_usd = max((sl_dist / spec.trade_tick_size) * spec.trade_tick_value, 1.0)
            elif spec.lot_size > 0.0:
                # For low-priced USD-quoted crypto/CFDs (e.g. ADAUSD), sl_dist * lot_size gives true $ risk/lot
                est = sl_dist * spec.lot_size
                if est > 0.0:
                    risk_per_lot_usd = max(est, 1.0)

        eff_cost_model = cost_model or TimeMachineCostModel(
            commission_per_lot_round_turn_usd=commission_usd,
            lots=0.10,
            risk_per_lot_usd=risk_per_lot_usd,
            spread_price=spread_px,
            vwap_slippage_price=0.2 * pip_size,
        )
        tm_res = self.time_machine.simulate_candles(
            symbol=sym,
            side=side,
            entry_price=entry_px,
            stop_price=stop_px,
            target_price=target_px,
            candles=candles,
            horizon_bars=36,
            cost_model=eff_cost_model,
        )

        mtf_verdict = self.mtf_filter.evaluate(
            symbol=sym,
            side=side,
            entry_price=entry_px,
            atr_m5=atr_val,
            m5_candles=candles,
            h1_candles=h1_candles,
        )

        passed_all_quant = (
            diag.allow_breakout_entries
            and gate_verdict.passed
            and mtf_verdict.passed
            and tm_res.net_avg_R >= self.min_net_avg_R
        )

        self.ledger.log_scan_event(
            symbol=sym,
            timeframe=tf,
            regime=diag.regime.value,
            hurst=diag.hurst_exponent,
            kaufman_er=diag.kaufman_er,
            vol_pct=diag.volatility_percentile,
            dom_obi=dom.obi if dom else None,
            spread_pct_of_sl=gate_verdict.spread_pct_of_sl,
            passed_quant=passed_all_quant,
            detail=f"{side.value} | {diag.reason}",
        )

        if self.last_contagion is None:
            self.last_contagion = self.contagion_oracle.evaluate(feed_router=self.feed_router)
        shock_z = float(self.last_contagion.cross_asset_shock_z)
        rollover_flag = bool(self.last_contagion.rollover_illiquidity_flag)

        z3_telemetry = {
            "symbol": sym,
            "side": side.value,
            "timemachine_net_avg_R": tm_res.net_avg_R,
            "spread_pct_of_sl": gate_verdict.spread_pct_of_sl,
            "vol_pct": diag.volatility_percentile,
            "cross_asset_shock_z": shock_z,
            "rollover_illiquidity_flag": rollover_flag,
        }
        z3_dec, z3_msg = AegisVerifier.z3_pure_math_axiom_gate(z3_telemetry)
        z3_ok = z3_dec == "APPROVE"

        ai_label = "IDLE (Quant Filtered)"
        ai_ok = z3_ok
        ai_detail = z3_msg
        if passed_all_quant:
            latest_bar_ts = candles[-1].timestamp_utc if candles else 0.0
            ctx = CandidateSetupAuditContext(
                symbol=sym,
                side=side,
                timeframe=tf,
                entry_price=entry_px,
                stop_loss=stop_px,
                take_profit=target_px,
                regime=diag.regime.value,
                hurst_exponent=diag.hurst_exponent,
                kaufman_er=diag.kaufman_er,
                volatility_percentile=diag.volatility_percentile,
                dom_obi=dom.obi if dom else None,
                spread_pct_of_sl=gate_verdict.spread_pct_of_sl,
                net_avg_R=tm_res.net_avg_R,
                prob_profit=tm_res.prob_profit,
                cvar_95_R=tm_res.cvar_95_R,
                recommended_risk_pct=tm_res.recommended_risk_pct,
                session_utc_iso=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                bar_timestamp=latest_bar_ts,
                rollover_illiquidity_flag=rollover_flag,
                cross_asset_shock_z=shock_z,
            )
            ai_decision = self.ai_committee.evaluate_sync(ctx)
            ai_label = f"{ai_decision.decision} ({ai_decision.confidence:.2f} via {ai_decision.provider_used})"
            ai_ok = ai_decision.decision == "APPROVE"
            ai_detail = f"{ai_decision.decision} ({ai_decision.model_used}) | {z3_msg}"
            if ai_decision.decision == "VETO":
                z3_tag = "Z3:PASS" if ai_decision.aegis_verified else "Z3:BYPASS"
                self.recent_audit_events.append(
                    BrokerExecutionEvent(
                        timestamp_str=time.strftime("%H:%M:%S", time.gmtime()),
                        symbol=sym,
                        side=side.value,
                        lots=0.0,
                        price=entry_px,
                        stop_loss=stop_px,
                        take_profit=target_px,
                        venue=f"AI/{ai_decision.provider_used.upper()}",
                        status="VETO",
                        details=f"model={ai_decision.model_used} | {z3_tag} | {ai_decision.reason[:80]}",
                    )
                )
                if len(self.recent_audit_events) > 20:
                    self.recent_audit_events = self.recent_audit_events[-20:]

        rec_risk = float(tm_res.recommended_risk_pct) if passed_all_quant else 0.0
        contagion_ok = (shock_z <= 3.5) and not (rollover_flag and diag.volatility_percentile > 85.0)
        bracket_ok = (entry_px > 0.0) and (sl_dist > 0.0)
        tm_ok = tm_res.net_avg_R >= self.min_net_avg_R
        exec_ready = passed_all_quant and contagion_ok and ai_ok and bracket_ok

        trace_stages = [
            PipelineStageResult(1, "trader-scanner + H1 MTF", mtf_verdict.passed, mtf_verdict.reason, f"{side.value} slope={mtf_verdict.h1_ema_slope:+.2f}"),
            PipelineStageResult(2, "RegimeClassifier", diag.allow_breakout_entries, f"{diag.regime.value}: {diag.reason}", f"H={diag.hurst_exponent:.2f} ER={diag.kaufman_er:.2f}"),
            PipelineStageResult(3, "trader-entry + MomentumDOM", gate_verdict.passed, ("; ".join(gate_verdict.reasons) if gate_verdict.reasons else ("OK" if gate_verdict.passed else "Blocked")), f"spread%SL={gate_verdict.spread_pct_of_sl:.1f}%"),
            PipelineStageResult(4, "trader-exit", bracket_ok, f"{side.value} SL={stop_px:.5f} | TP={target_px:.5f} (2.0R)", f"1R={sl_dist:.5f}"),
            PipelineStageResult(5, "TimeMachine v2 (Bootstrap)", tm_ok, f"net_avg_R={tm_res.net_avg_R:+.3f}R (floor={self.min_net_avg_r if hasattr(self, 'min_net_avg_r') else self.min_net_avg_R:.2f}R, comm=${commission_usd:.2f})", f"P(Win)={tm_res.prob_profit * 100:.0f}%"),
            PipelineStageResult(6, "Contagion / News / Analyst", contagion_ok, f"shock_z={shock_z:.2f}, rollover={rollover_flag}", f"Z={shock_z:.2f}"),
            PipelineStageResult(7, "trader-budget", (rec_risk > 0.0) if passed_all_quant else False, f"Q-Kelly={rec_risk:.2f}%" if passed_all_quant else "Q-Kelly forced 0.00% (filtered)", f"{rec_risk:.2f}%"),
            PipelineStageResult(8, "trader-risk + CorrelationShield", q_state.mode.value == "ACTIVE", f"Quarantine={q_state.mode.value} PF={q_state.profit_factor:.2f}", "CAP<=1.0%"),
            PipelineStageResult(9, "Aegis Z3 + AI Waterfall", ai_ok, ai_detail, z3_dec),
            PipelineStageResult(10, "ExecutionRouter (DEMO)", exec_ready, "DEMO-locked + Server-side SL/TP armed" if exec_ready else "Blocked upstream prior to wire", "DEMO_SAFE"),
            PipelineStageResult(11, "InFlightPositionManager", True, "2-Bar Grace Lock + Forward TM + 1R/1.5R Runner armed", "SHIELD8_Z3"),
        ]
        blocked_stage = next((st.stage_num for st in trace_stages if not st.passed), None)
        trace_dict = AgentPipelineTrace(
            symbol=sym,
            timeframe=tf,
            side=side.value,
            entry_price=entry_px,
            stop_loss=stop_px,
            take_profit=target_px,
            passed_all=exec_ready,
            final_decision="APPROVE" if exec_ready else ("VETO" if (passed_all_quant and not ai_ok) else "FILTERED"),
            net_avg_r=round(tm_res.net_avg_R, 4),
            recommended_risk_pct=round(rec_risk, 4),
            z3_status=z3_msg,
            stages=trace_stages,
        ).to_dict()
        trace_dict["blocked_stage"] = blocked_stage

        return SymbolCockpitRow(
            symbol=sym,
            tier=candidate.tier.value,
            timeframe=tf,
            bid=tick.bid,
            ask=tick.ask,
            spread_pips=spread_pips_atomic,
            regime=diag.regime.value,
            hurst=diag.hurst_exponent,
            kaufman_er=diag.kaufman_er,
            obi=dom.obi if dom else None,
            net_avg_r=tm_res.net_avg_R,
            rec_risk_pct=rec_risk,
            ai_verdict=ai_label,
            quarantine_mode=q_state.mode.value,
            profit_factor=q_state.profit_factor,
            best_venue=best_venue_str,
            side=side,
            entry_price=entry_px,
            stop_loss=stop_px,
            take_profit=target_px,
            pipeline_trace=trace_dict,
        )

    def scan_cycle(
        self,
        target_symbols: Sequence[DiscoveredSymbol] | Sequence[str] | None = None,
        now_utc: float | None = None,
    ) -> tuple[list[SymbolCockpitRow], UniverseScanSummary]:
        """Execute a full two-stage funnel scan cycle."""
        # Recompute fresh ground-truth contagion & rollover telemetry on every cycle
        self.last_contagion = self.contagion_oracle.evaluate(feed_router=self.feed_router)
        candidates, summary = self.sweep_universe(target_symbols=target_symbols, now_utc=now_utc)
        top_candidates = candidates if target_symbols is not None else candidates[: self.max_candidates_per_cycle]

        rows: list[SymbolCockpitRow] = []
        for candidate in top_candidates:
            row = self.evaluate_candidate_pipeline(candidate)
            rows.append(row)

        rows.sort(
            key=lambda r: (
                0 if (r.regime == "TRENDING_EXPANSION" and (r.rec_risk_pct or 0.0) > 0.0) else 1,
                -(r.net_avg_r if r.net_avg_r is not None else -999.0),
            )
        )
        return rows, summary
