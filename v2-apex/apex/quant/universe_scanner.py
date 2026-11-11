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

from apex.ai.veto_committee import AIVetoCommittee, CandidateSetupAuditContext
from apex.cockpit.tui_dashboard import SymbolCockpitRow
from apex.feeds.alias_mapper import SmartAliasMapper
from apex.feeds.dual_router import MarketFeedRouter
from apex.feeds.models import AssetClass, Candle, DiscoveredSymbol, Side, Tick
from apex.ledger.walkforward_db import WalkForwardLedgerDB
from apex.quant.momentum_dom import (
    InstrumentTier,
    MomentumDOMGate,
    classify_instrument_tier,
)
from apex.quant.regime import MarketRegime, RegimeClassifier, compute_hurst_exponent
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
        self.max_tick_age_seconds: float = float(scanner_cfg.get("max_tick_age_seconds", 180.0))
        self.min_candles_required: int = int(scanner_cfg.get("min_candles_required", 30))

        self.alias_mapper: SmartAliasMapper = SmartAliasMapper()
        self._cached_catalog: list[DiscoveredSymbol] = []
        self._last_catalog_discovery_time: float = 0.0

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

        # Spread penalty: tighter spread = higher score
        spread_factor = 1.0 / (1.0 + max(0.0, spread_pips - 2.0) * 0.1)

        # Composite Mover Score:
        # 40% Kaufman ER + 30% Impulse Body Ratio + 30% Volatility/Return * Spread Quality
        pct_return_score = min(abs(closes[-1] - closes[0]) / max(closes[0], 1e-6) * 1000.0, 50.0)
        mover_score = (
            (er * 40.0)
            + (min(body_ratio, 3.0) * 10.0)
            + (pct_return_score * 0.4)
            + (max(0.0, hurst - 0.50) * 40.0)
        ) * spread_factor

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
        diag = self.regime_classifier.classify_candles(
            symbol=sym,
            candles=candles,
            current_spread=spread_px,
        )

        # Standard 1.5x ATR stop and 2.0R target
        atr_val = max(diag.current_atr, pip_size * 5.0)
        sl_dist = 1.5 * atr_val
        entry_px = tick.ask
        stop_px = entry_px - sl_dist
        target_px = entry_px + 2.0 * sl_dist

        nbbo = self.feed_router.evaluate_nbbo(
            symbol=sym,
            one_r_price_distance=sl_dist,
            side=Side.BUY,
        )
        if nbbo.pips_saved > 0.0:
            best_venue_str = f"{nbbo.winning_venue.upper()} (-{nbbo.pips_saved:.1f}p)"
        else:
            best_venue_str = f"{nbbo.winning_venue.upper()} ({nbbo.winning_cost_pips:.1f}p)"

        gate_verdict = self.momentum_dom_gate.evaluate(
            symbol=sym,
            side=Side.BUY,
            entry_price=entry_px,
            stop_price=stop_px,
            candles=candles,
            current_spread=spread_px,
            dom_analytics=dom,
        )

        eff_cost_model = cost_model or TimeMachineCostModel(
            commission_per_lot_round_turn_usd=4.50,
            lots=0.10,
            risk_per_lot_usd=100.0,
            spread_price=spread_px,
            vwap_slippage_price=0.2 * pip_size,
        )
        tm_res = self.time_machine.simulate_candles(
            symbol=sym,
            side=Side.BUY,
            entry_price=entry_px,
            stop_price=stop_px,
            target_price=target_px,
            candles=candles,
            horizon_bars=36,
            cost_model=eff_cost_model,
        )

        passed_all_quant = (
            diag.allow_breakout_entries
            and gate_verdict.passed
            and tm_res.net_avg_R > 0.0
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
            detail=diag.reason,
        )

        ai_label = "IDLE (Quant Filtered)"
        if passed_all_quant:
            ctx = CandidateSetupAuditContext(
                symbol=sym,
                side=Side.BUY,
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
            )
            ai_decision = self.ai_committee.evaluate_sync(ctx)
            ai_label = f"{ai_decision.decision} ({ai_decision.confidence:.2f} via {ai_decision.provider_used})"

        return SymbolCockpitRow(
            symbol=sym,
            tier=candidate.tier.value,
            timeframe=tf,
            bid=tick.bid,
            ask=tick.ask,
            spread_pips=candidate.spread_pips,
            regime=diag.regime.value,
            hurst=diag.hurst_exponent,
            kaufman_er=diag.kaufman_er,
            obi=dom.obi if dom else None,
            net_avg_r=tm_res.net_avg_R,
            rec_risk_pct=tm_res.recommended_risk_pct,
            ai_verdict=ai_label,
            quarantine_mode=q_state.mode.value,
            profit_factor=q_state.profit_factor,
            best_venue=best_venue_str,
        )

    def scan_cycle(
        self,
        target_symbols: Sequence[DiscoveredSymbol] | Sequence[str] | None = None,
        now_utc: float | None = None,
    ) -> tuple[list[SymbolCockpitRow], UniverseScanSummary]:
        """Execute a full two-stage funnel scan cycle."""
        candidates, summary = self.sweep_universe(target_symbols=target_symbols, now_utc=now_utc)
        top_candidates = candidates if target_symbols is not None else candidates[: self.max_candidates_per_cycle]

        rows: list[SymbolCockpitRow] = []
        for candidate in top_candidates:
            row = self.evaluate_candidate_pipeline(candidate)
            rows.append(row)

        return rows, summary
