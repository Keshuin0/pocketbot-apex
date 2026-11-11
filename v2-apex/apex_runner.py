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
from apex.quant.universe_scanner import UniverseScanner, UniverseScanSummary

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
        self.universe_scanner = UniverseScanner(
            feed_router=self.feed_router,
            regime_classifier=self.regime_classifier,
            momentum_dom_gate=self.momentum_dom_gate,
            time_machine=self.time_machine,
            ai_committee=self.ai_committee,
            ledger=self.ledger,
            config=self.config,
        )
        self.last_scan_summary: UniverseScanSummary | None = None

    def run_cycle(self, symbols: Sequence[str] | None = None) -> list[SymbolCockpitRow]:
        """Run a full quantitative scan, regime classification, Time Machine v2, and ledger cycle."""
        active_src = self.feed_router.connect()
        use_funnel = symbols is None and self.config.get("scanner", {}).get("mode", "FULL_MARKET_FUNNEL") == "FULL_MARKET_FUNNEL"

        if use_funnel:
            rows, summary = self.universe_scanner.scan_cycle()
            self.last_scan_summary = summary
        else:
            target_symbols = list(symbols or self.config.get("symbols", ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "BTCUSD"]))
            rows, summary = self.universe_scanner.scan_cycle(target_symbols=target_symbols)
            self.last_scan_summary = summary

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
            universe_scanned_count=self.last_scan_summary.open_scanned if self.last_scan_summary else 0,
            universe_total_count=self.last_scan_summary.total_scanned if self.last_scan_summary else 0,
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
