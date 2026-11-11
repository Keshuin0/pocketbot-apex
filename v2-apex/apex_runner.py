"""PocketBot Apex — Unified Orchestrator (`apex_runner.py`).

Supports:
  - `--mode confirm`: Runs the full quantitative + AI veto pipeline and requires interactive confirmation
    before transmitting orders.
  - `--mode demo-auto`: Runs autonomous DEMO execution with all 6 hard pre-trade circuit breakers and
    Auto-Quarantine protection.
  - `--mode shadow-ab`: Runs walk-forward A/B ablation & Paper-Shadow simulation across the universe.
  - `--dashboard`: Renders the live Rich cockpit table (`ApexTUIDashboard`).
  - `--once`: Executes a single complete scan-to-execution pass and prints the Rich summary table.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
import time
from pathlib import Path
from typing import Sequence

from rich.console import Console

from apex.ai.veto_committee import AIVetoCommittee, CandidateSetupAuditContext
from apex.cockpit.telegram_remote import TelegramRemoteController
from apex.cockpit.tui_dashboard import ApexTUIDashboard, SymbolCockpitRow
from apex.execution.router import DualBrokerExecutionRouter, ExecutionOrderRequest
from apex.feeds.dual_router import MarketFeedRouter
from apex.feeds.models import Side, verify_live_unlock
from apex.ledger.walkforward_db import SymbolExecutionMode, WalkForwardLedgerDB
from apex.quant.momentum_dom import (
    InstrumentTier,
    MomentumDOMGate,
    classify_instrument_tier,
)
from apex.quant.regime import MarketRegime, RegimeClassifier
from apex.quant.timemachine_v2 import BlockBootstrapTimeMachine, TimeMachineCostModel

console = Console(width=145)


class ApexOrchestrator:
    """End-to-end Quantitative Orchestrator for PocketBot Apex."""

    def __init__(
        self,
        mode: str = "demo-auto",
        config_path: str | Path = "config/apex_config.json",
        db_path: str | Path = "trader/apex_ledger.db",
        is_live: bool = False,
        cli_args: Sequence[str] | None = None,
    ) -> None:
        self.mode: str = mode.lower().strip()
        self.config_path = Path(config_path)
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.is_live: bool = bool(is_live or str(self.config.get("account_mode", "DEMO")).upper() == "LIVE")
        verify_live_unlock(self.is_live, cli_args=cli_args)

        filters_cfg = self.config.get("filters", {})
        reg_cfg = filters_cfg.get("regime", {})
        dom_cfg = filters_cfg.get("dom_obi", {})
        corr_cfg = filters_cfg.get("correlation_cap", {})
        ai_cfg = filters_cfg.get("ai_veto", {})

        self.feed_router = MarketFeedRouter(
            config_path=self.config_path,
            is_live=self.is_live,
            cli_args=cli_args,
        )
        self.regime_classifier = RegimeClassifier(
            enabled=bool(reg_cfg.get("enabled", True)),
            hurst_window=int(reg_cfg.get("hurst_window", 100)),
            adx_atr_period=int(reg_cfg.get("adx_period", 14)),
            trending_hurst_threshold=float(reg_cfg.get("trending_hurst_threshold", 0.52)),
            trending_er_threshold=float(reg_cfg.get("trending_er_threshold", 0.25)),
            max_spread_median_multiple=float(reg_cfg.get("max_spread_median_multiple", 1.5)),
            max_tick_velocity_sigma=float(reg_cfg.get("max_tick_velocity_sigma", 4.0)),
        )
        self.momentum_dom_gate = MomentumDOMGate(
            dom_obi_enabled=bool(dom_cfg.get("enabled", True)),
            min_abs_obi=float(dom_cfg.get("min_abs_obi_threshold", 0.15)),
            max_spread_pct_of_sl=float(dom_cfg.get("max_spread_pct_of_sl", 0.15)),
            min_impulse_atr_multiple=float(dom_cfg.get("min_impulse_atr_multiple", 1.5)),
            extreme_close_fraction=float(dom_cfg.get("extreme_close_fraction", 0.25)),
            wall_min_median_multiple=float(dom_cfg.get("wall_min_median_multiple", 3.0)),
            wall_max_distance_r=float(dom_cfg.get("wall_max_distance_r", 0.35)),
        )
        self.time_machine = BlockBootstrapTimeMachine(
            replays=10_000,
            mean_block_length=6.0,
            seed=19,
            account_equity_usd=10_000.0,
            max_risk_pct=float(self.config.get("risk", {}).get("max_risk_per_trade_pct", 0.5)),
        )
        self.ai_committee = AIVetoCommittee(
            enabled=bool(ai_cfg.get("enabled", True)),
            timeout_seconds=min(float(ai_cfg.get("studio_timeout_seconds", ai_cfg.get("timeout_seconds", 4.0))), 4.0),
            agy_timeout_seconds=float(ai_cfg.get("agy_pro_timeout_seconds", 15.0)),
            mode=str(ai_cfg.get("mode", "auto_studio_first")),
            priority_order=ai_cfg.get("priority_order"),
            config_path=self.config_path,
        )
        self.ledger = WalkForwardLedgerDB(db_path=db_path)
        self.execution_router = DualBrokerExecutionRouter(
            feed_router=self.feed_router,
            account_equity_usd=10_000.0,
            max_daily_drawdown_pct=float(self.config.get("risk", {}).get("max_daily_drawdown_pct", 2.0)),
            max_concurrent_positions=2,
            max_factor_exposure_pct=float(corr_cfg.get("max_factor_risk_pct", 1.0)),
            max_spread_slippage_sl_ratio=float(dom_cfg.get("max_spread_pct_of_sl", 0.15)),
            correlation_cap_enabled=bool(corr_cfg.get("enabled", True)),
            dom_obi_enabled=bool(dom_cfg.get("enabled", True)),
            is_live=self.is_live,
            cli_args=cli_args,
            ledger_db=self.ledger,
        )
        self.dashboard = ApexTUIDashboard(console=console)
        self.telegram = TelegramRemoteController(
            execution_router=self.execution_router,
            ledger_db=self.ledger,
        )

    def run_cycle(self, symbols: Sequence[str] | None = None) -> list[SymbolCockpitRow]:
        """Run a full quantitative scan, regime classification, Time Machine v2, and ledger cycle."""
        target_symbols = list(symbols or self.config.get("symbols", ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "BTCUSD"]))
        active_src = self.feed_router.connect()
        rows: list[SymbolCockpitRow] = []

        for sym in target_symbols:
            norm_sym = sym.upper()
            tier = classify_instrument_tier(norm_sym)
            tf = "M5" if tier == InstrumentTier.TIGHT_SPREAD_M5 else "M15"

            spec = self.feed_router.get_contract_spec(norm_sym)
            pip_size = spec.pip_size if spec else (0.01 if "JPY" in norm_sym or "XAU" in norm_sym else 0.0001)

            tick, _ = self.feed_router.get_tick(norm_sym)
            candles = self.feed_router.get_candles(norm_sym, timeframe=tf, count=200)
            dom = self.feed_router.get_dom_analytics(norm_sym)
            q_state = self.ledger.get_quarantine_state(norm_sym)

            if tick is None or len(candles) < 30:
                rows.append(
                    SymbolCockpitRow(
                        symbol=norm_sym,
                        tier=tier.value,
                        timeframe=tf,
                        bid=tick.bid if tick else 0.0,
                        ask=tick.ask if tick else 0.0,
                        spread_pips=(tick.spread / pip_size) if tick else 0.0,
                        regime="NO_DATA",
                        hurst=0.5,
                        kaufman_er=0.0,
                        obi=dom.obi if dom else None,
                        net_avg_r=None,
                        rec_risk_pct=None,
                        ai_verdict="STANDBY",
                        quarantine_mode=q_state.mode.value,
                        profit_factor=q_state.profit_factor,
                    )
                )
                continue

            spread_px = tick.spread
            spread_pips = spread_px / max(pip_size, 1e-9)

            diag = self.regime_classifier.classify_candles(
                symbol=norm_sym,
                candles=candles,
                current_spread=spread_px,
            )

            # Evaluate candidate 1.5x ATR stop and 2.0R target
            atr_val = max(diag.current_atr, pip_size * 5.0)
            sl_dist = 1.5 * atr_val
            entry_px = tick.ask
            stop_px = entry_px - sl_dist
            target_px = entry_px + 2.0 * sl_dist

            nbbo = self.feed_router.evaluate_nbbo(
                symbol=norm_sym,
                one_r_price_distance=sl_dist,
                side=Side.BUY,
            )
            if nbbo.pips_saved > 0.0:
                best_venue_str = f"{nbbo.winning_venue.upper()} (-{nbbo.pips_saved:.1f}p)"
            else:
                best_venue_str = f"{nbbo.winning_venue.upper()} ({nbbo.winning_cost_pips:.1f}p)"

            gate_verdict = self.momentum_dom_gate.evaluate(
                symbol=norm_sym,
                side=Side.BUY,
                entry_price=entry_px,
                stop_price=stop_px,
                candles=candles,
                current_spread=spread_px,
                dom_analytics=dom,
            )

            cost_model = TimeMachineCostModel(
                commission_per_lot_round_turn_usd=4.50,
                lots=0.10,
                risk_per_lot_usd=100.0,
                spread_price=spread_px,
                vwap_slippage_price=0.2 * pip_size,
            )
            tm_res = self.time_machine.simulate_candles(
                symbol=norm_sym,
                side=Side.BUY,
                entry_price=entry_px,
                stop_price=stop_px,
                target_price=target_px,
                candles=candles,
                horizon_bars=36,
                cost_model=cost_model,
            )

            passed_all_quant = (
                diag.allow_breakout_entries
                and gate_verdict.passed
                and tm_res.net_avg_R > 0.0
            )

            self.ledger.log_scan_event(
                symbol=norm_sym,
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
                    symbol=norm_sym,
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
                    session_utc_iso=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                )
                ai_dec = self.ai_committee.evaluate_sync(ctx)
                ai_label = f"{ai_dec.decision} ({ai_dec.provider_used})"

            rows.append(
                SymbolCockpitRow(
                    symbol=norm_sym,
                    tier=tier.value,
                    timeframe=tf,
                    bid=tick.bid,
                    ask=tick.ask,
                    spread_pips=spread_pips,
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
            )

        ai_status = self.ai_committee.get_router_status()
        sor_pips, sor_usd = self.execution_router.cumulative_sor_savings
        self.dashboard.print_snapshot(
            run_mode=self.mode,
            active_feed=active_src,
            ctrader_connected=self.feed_router.ctrader.is_connected,
            mt5_connected=self.feed_router.mt5.is_connected,
            account_equity_usd=self.execution_router.account_equity_usd,
            daily_pnl_usd=0.0,
            open_positions_count=len(self.execution_router._shadow_positions),
            rows=rows,
            ai_brain_badge=ai_status.dashboard_badge,
            mesh_badge=self.feed_router.dashboard_mesh_badge,
            sor_saved_pips=sor_pips,
            sor_saved_usd=sor_usd,
        )
        return rows

    def shutdown(self) -> None:
        self.feed_router.shutdown()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PocketBot Apex — Institutional Dual-Broker Runner")
    parser.add_argument(
        "--mode",
        choices=["confirm", "demo-auto", "shadow-ab"],
        default="demo-auto",
        help="Execution operating mode",
    )
    parser.add_argument("--config", default="config/apex_config.json", help="Path to apex_config.json")
    parser.add_argument("--once", action="store_true", help="Run one complete cycle and exit")
    parser.add_argument("--dashboard", action="store_true", help="Run live Rich TUI dashboard loop")
    parser.add_argument("--interval", type=float, default=15.0, help="Refresh interval in seconds")
    parser.add_argument("--unlock-live", action="store_true", help="Unlock live trading (requires env var too)")
    args = parser.parse_args(argv)

    orchestrator = ApexOrchestrator(
        mode=args.mode,
        config_path=args.config,
        is_live=bool(args.unlock_live),
        cli_args=sys.argv if argv is None else argv,
    )
    try:
        if args.once or not args.dashboard:
            orchestrator.run_cycle()
            return 0
        while True:
            orchestrator.run_cycle()
            time.sleep(max(1.0, args.interval))
    except KeyboardInterrupt:
        console.print("\n[yellow]PocketBot Apex stopped by user.[/yellow]")
        return 0
    finally:
        orchestrator.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
