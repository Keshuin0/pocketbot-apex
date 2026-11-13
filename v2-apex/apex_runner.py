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
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from rich.console import Console
from rich.live import Live

from apex.ai.aegis_verifier import AegisVerifier
from apex.ai.sleep_consolidation import SleepWakeConsolidator
from apex.ai.veto_committee import AIVetoCommittee, CandidateSetupAuditContext
from apex.cockpit.replay_engine import HistoricalReplayEngine
from apex.cockpit.telegram_remote import TelegramRemoteController
from apex.cockpit.tui_dashboard import ApexTUIDashboard, SymbolCockpitRow
from apex.cockpit.web_cockpit import WebCockpitServer, launch_chrome_app
from apex.execution.position_manager import AutonomousPositionOrchestrator
from apex.execution.router import DualBrokerExecutionRouter, ExecutionOrderRequest
from apex.feeds.dual_router import MarketFeedRouter
from apex.feeds.models import Side, verify_live_unlock
from apex.ledger.walkforward_db import SymbolExecutionMode, WalkForwardLedgerDB
from apex.quant.correlation_shield import PositionRiskSpec
from apex.quant.momentum_dom import (
    InstrumentTier,
    MomentumDOMGate,
    classify_instrument_tier,
)
from apex.quant.regime import MarketRegime, RegimeClassifier
from apex.quant.timemachine_v2 import BlockBootstrapTimeMachine, TimeMachineCostModel
from apex.quant.universe_scanner import UniverseScanner, UniverseScanSummary

logger = logging.getLogger(__name__)
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
        self.ledger = WalkForwardLedgerDB(db_path=db_path)
        self.ai_committee = AIVetoCommittee(
            enabled=bool(ai_cfg.get("enabled", True)),
            timeout_seconds=min(float(ai_cfg.get("studio_timeout_seconds", ai_cfg.get("timeout_seconds", 4.0))), 4.0),
            agy_timeout_seconds=float(ai_cfg.get("agy_pro_timeout_seconds", 15.0)),
            mode=str(ai_cfg.get("mode", "auto_studio_first")),
            priority_order=ai_cfg.get("priority_order"),
            config_path=self.config_path,
            ledger_db=self.ledger,
        )
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
        self.position_manager = AutonomousPositionOrchestrator(
            regime_classifier=self.regime_classifier,
            time_machine=self.time_machine,
            ledger=self.ledger,
        )
        self.web_cockpit: WebCockpitServer | None = None
        self.paused: bool = False
        self._bg_lock = threading.Lock()
        self._drill_running = False
        self._replay_running = False
        sleep_cfg = self.config.get("sleep_wake", {})
        self.sleep_consolidator = SleepWakeConsolidator(
            ledger_db=self.ledger,
            memory_path=Path(db_path).parent / "semantic_memory.json",
            rollover_utc_hour=int(sleep_cfg.get("rollover_utc_hour", 22)),
            rollover_utc_minute=int(sleep_cfg.get("rollover_utc_minute", 5)),
            stale_hours_threshold=float(sleep_cfg.get("stale_hours_threshold", 16.0)),
        )
        if bool(sleep_cfg.get("enabled", True)):
            due, reason = self.sleep_consolidator.should_run_sleep_cycle(is_startup=True)
            if due:
                self.sleep_consolidator.run_full_cycle(
                    trigger_reason=reason,
                    ai_committee=self.ai_committee,
                    feed_router=self.feed_router,
                    hydra_nbbo=getattr(self.feed_router, "nbbo_engine", None),
                )

    def run_cycle(
        self,
        symbols: Sequence[str] | None = None,
        live: Live | None = None,
        render_console_tui: bool = True,
    ) -> list[SymbolCockpitRow]:
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

        # Sync live account equity and currency
        self.execution_router.update_account_equity_from_broker()
        account_currency = (
            self.feed_router.mt5.get_account_currency()
            if self.feed_router.mt5.is_connected
            else "USD"
        )
        open_positions = self.execution_router.get_open_positions()

        # Compute floating daily P/L from connected broker
        daily_pnl = 0.0
        if self.feed_router.mt5.is_connected:
            acct_info = self.feed_router.mt5.get_account_info()
            eq = float(acct_info.get("equity", 0.0))
            bal = float(acct_info.get("balance", 0.0))
            if eq > 0.0 and bal > 0.0:
                daily_pnl = eq - bal

        # In-Flight Multi-Agent Trajectory Orchestrator & Closed-Deal Autopsy
        contagion_z_live = self.universe_scanner.last_contagion.cross_asset_shock_z if self.universe_scanner.last_contagion else 0.0
        if self.feed_router.mt5.is_connected:
            raw_mt5_positions = self.feed_router.mt5.get_open_positions()
            open_tickets = {int(p.get("ticket", 0)) for p in raw_mt5_positions if int(p.get("ticket", 0)) > 0}
            for p in raw_mt5_positions:
                tkt = int(p.get("ticket", 0))
                raw_sym = str(p.get("symbol", ""))
                if tkt <= 0 or not raw_sym:
                    continue
                is_buy = int(p.get("type", 0)) == 0
                pos_candles = self.feed_router.get_candles(raw_sym, timeframe="M5", count=60)
                pos_tick = self.feed_router.mt5.get_tick(raw_sym)
                self.position_manager.evaluate_position(
                    ticket=tkt,
                    symbol=raw_sym,
                    side=Side.BUY if is_buy else Side.SELL,
                    entry_price=float(p.get("price_open", 0.0)),
                    current_sl=float(p.get("sl", 0.0)),
                    current_tp=float(p.get("tp", 0.0)),
                    mark_price=float(p.get("price_current", p.get("price_open", 0.0))),
                    candles=pos_candles,
                    spread_price=pos_tick.spread if pos_tick else 0.0,
                    cross_asset_shock_z=contagion_z_live,
                    opened_at_utc=float(p.get("time", time.time())),
                    lots=float(p.get("volume", 0.01)),
                    execute_broker_actions=(self.mode == "demo-auto"),
                )
            self.position_manager.reconcile_closed_deals(open_tickets)

        # In demo-auto or shadow-ab mode, automatically dispatch approved rows
        if self.mode in ("demo-auto", "shadow-ab"):
            self._dispatch_approved_trades(rows, open_positions=open_positions)
            open_positions = self.execution_router.get_open_positions()

        ai_status = self.ai_committee.get_router_status()
        aegis_summary = self.ledger.get_aegis_integrity_summary()
        sleep_due, sleep_reason = self.sleep_consolidator.should_run_sleep_cycle(is_startup=False)
        if sleep_due and bool(self.config.get("sleep_wake", {}).get("enabled", True)):
            self.sleep_consolidator.run_full_cycle(
                trigger_reason=sleep_reason,
                ai_committee=self.ai_committee,
                feed_router=self.feed_router,
                hydra_nbbo=getattr(self.feed_router, "nbbo_engine", None),
            )
        mind_badge = self.sleep_consolidator.get_mind_state_badge()
        contagion_z = self.universe_scanner.last_contagion.cross_asset_shock_z if self.universe_scanner.last_contagion else 0.0
        sor_pips, sor_usd = self.execution_router.cumulative_sor_savings
        combined_events = sorted(
            list(self.universe_scanner.recent_audit_events) + list(self.execution_router.recent_broker_events),
            key=lambda e: e.timestamp_str,
        )[-5:]
        if render_console_tui:
            self.dashboard.print_snapshot(
                run_mode=self.mode,
                active_feed=active_src,
                ctrader_connected=self.feed_router.ctrader.is_connected,
                mt5_connected=self.feed_router.mt5.is_connected,
                account_equity_usd=self.execution_router.account_equity_usd,
                daily_pnl_usd=daily_pnl,
                open_positions_count=len(open_positions),
                rows=rows,
                ai_brain_badge=ai_status.model_quota_badge,
                mesh_badge=self.feed_router.dashboard_mesh_badge,
                sor_saved_pips=sor_pips,
                sor_saved_usd=sor_usd,
                universe_scanned_count=self.last_scan_summary.open_scanned if self.last_scan_summary else 0,
                universe_total_count=self.last_scan_summary.total_scanned if self.last_scan_summary else 0,
                account_currency=account_currency,
                broker_events=combined_events,
                aegis_badge=aegis_summary["badge"],
                contagion_shock_z=contagion_z,
                mind_state_badge=mind_badge,
                live=live,
            )
        if self.web_cockpit is not None:
            self._push_web_cockpit_snapshot(
                account_currency=account_currency,
                daily_pnl=daily_pnl,
                open_positions=open_positions,
                rows=rows,
                ai_status=ai_status,
                aegis_badge=aegis_summary["badge"],
                contagion_z=contagion_z,
                mind_badge=mind_badge,
                combined_events=combined_events,
            )
        return rows

    def attach_web_cockpit(self, server: WebCockpitServer) -> None:
        """Attach WebCockpitServer and wire localhost control button handlers."""
        self.web_cockpit = server
        server.set_control_handler("pause", self._handle_web_pause)
        server.set_control_handler("drill", self._handle_web_drill)
        server.set_control_handler("replay", self._handle_web_replay)
        server.set_control_handler("refresh_ai", self._handle_web_refresh_ai)
        server.set_control_handler("panic_flatten", self._handle_web_panic_flatten)

    def _handle_web_pause(self) -> dict[str, object]:
        self.paused = not self.paused
        if self.web_cockpit is not None:
            self.web_cockpit.push_state({"paused": self.paused})
        return {"paused": self.paused, "message": "Paused" if self.paused else "Resumed"}

    def _handle_web_drill(self) -> dict[str, object]:
        with self._bg_lock:
            if self._drill_running:
                return {"started": False, "message": "Live drill is already running."}
            self._drill_running = True

        def _worker() -> None:
            try:
                from tools.live_lifecycle_drill import run_live_lifecycle_drill

                cb = self.web_cockpit.update_drill_state if self.web_cockpit is not None else None
                run_live_lifecycle_drill(
                    hold_seconds=15,
                    db_path=self.ledger.db_path,
                    on_update=cb,
                )
            except Exception as exc:
                if self.web_cockpit is not None:
                    self.web_cockpit.update_drill_state({"status": "ERROR", "message": str(exc)})
            finally:
                with self._bg_lock:
                    self._drill_running = False

        threading.Thread(target=_worker, name="ApexWebDrillThread", daemon=True).start()
        return {"started": True, "message": "Started 60s Live 0.01-Lot MT5 Demo Drill."}

    def _handle_web_replay(self) -> dict[str, object]:
        with self._bg_lock:
            if self._replay_running:
                return {"started": False, "message": "100x Replay is already running."}
            self._replay_running = True

        def _worker() -> None:
            try:
                engine = HistoricalReplayEngine()
                cb = self.web_cockpit.update_replay_state if self.web_cockpit is not None else None
                engine.run_replay(feed_router=self.feed_router, on_update=cb)
            except Exception as exc:
                if self.web_cockpit is not None:
                    self.web_cockpit.update_replay_state({"status": "ERROR", "message": str(exc)})
            finally:
                with self._bg_lock:
                    self._replay_running = False

        threading.Thread(target=_worker, name="ApexWebReplayThread", daemon=True).start()
        return {"started": True, "message": "Started 100x Historical Market Replay."}

    def _handle_web_refresh_ai(self) -> dict[str, object]:
        report = self.sleep_consolidator.run_full_cycle(
            trigger_reason="MANUAL_WEB_COCKPIT",
            ai_committee=self.ai_committee,
            feed_router=self.feed_router,
            hydra_nbbo=getattr(self.feed_router, "nbbo_engine", None),
        )
        return {"message": f"Sleep-Wake consolidation complete (cycle #{report.cycle_number})."}

    def _handle_web_panic_flatten(self) -> dict[str, object]:
        res = self.execution_router.panic_flatten_all()
        if self.web_cockpit is not None:
            self.web_cockpit.push_state({"open_positions": [], "open_positions_count": 0})
        return {"message": f"PANIC FLATTEN executed: {res}", **res}

    def _push_web_cockpit_snapshot(
        self,
        *,
        account_currency: str,
        daily_pnl: float,
        open_positions: Sequence[PositionRiskSpec],
        rows: Sequence[SymbolCockpitRow],
        ai_status: object,
        aegis_badge: str,
        contagion_z: float,
        mind_badge: str,
        combined_events: Sequence[object],
    ) -> None:
        if self.web_cockpit is None:
            return
        now = time.time()
        selector = getattr(self.ai_committee, "model_selector", None)
        quarantined: list[str] = []
        if selector is not None:
            cooldowns = getattr(selector, "_family_cooldowns", {})
            quarantined = sorted(
                {f"{prov}:{fam}" for (prov, fam), (until_ts, _) in cooldowns.items() if until_ts > now}
            )

        trajectories = self.position_manager.get_cockpit_trajectories()
        serialized_positions = []
        for pos in open_positions:
            sym_key = str(getattr(pos, "symbol", "")).upper()
            traj = trajectories.get(sym_key, {})
            entry_px = float(getattr(pos, "entry_price", 0.0) or 0.0)
            curr_px = float(getattr(pos, "current_price", entry_px) or entry_px)
            sl_px = float(getattr(pos, "stop_loss", 0.0) or 0.0)
            tp_px = float(getattr(pos, "take_profit", 0.0) or 0.0)
            serialized_positions.append(
                {
                    "symbol": str(getattr(pos, "symbol", "")),
                    "side": str(getattr(getattr(pos, "side", ""), "value", getattr(pos, "side", ""))),
                    "lots": float(getattr(pos, "lots", 0.0) or 0.0),
                    "entry_price": entry_px,
                    "current_price": curr_px,
                    "stop_loss": sl_px,
                    "take_profit": tp_px,
                    "profit_usd": float(getattr(pos, "profit_usd", 0.0) or 0.0),
                    "ratchet_stage": traj.get("ratchet_stage", "ACTIVE"),
                    "forward_prob_tp": traj.get("forward_prob_tp", 50.0),
                    "forward_ev_r": traj.get("forward_ev_r", 0.0),
                    "mfe_r": traj.get("mfe_r", 0.0),
                    "mae_r": traj.get("mae_r", 0.0),
                    "grace_locked": traj.get("grace_locked", False),
                }
            )

        serialized_rows = []
        for r in rows:
            d = asdict(r)
            if r.side is not None:
                d["side"] = r.side.value
            serialized_rows.append(d)

        serialized_events = [asdict(ev) for ev in combined_events]

        self.web_cockpit.push_state(
            {
                "run_mode": self.mode.upper(),
                "mind_state_badge": mind_badge,
                "universe_scanned_count": self.last_scan_summary.open_scanned if self.last_scan_summary else 0,
                "universe_total_count": self.last_scan_summary.total_scanned if self.last_scan_summary else 0,
                "account_currency": account_currency,
                "account_equity": round(float(self.execution_router.account_equity_usd), 2),
                "daily_pnl": round(float(daily_pnl), 2),
                "open_positions_count": len(open_positions),
                "mesh_badge": self.feed_router.dashboard_mesh_badge,
                "verified_ai_model": str(getattr(ai_status, "active_model_label", "UNKNOWN")),
                "ai_brain_badge": str(getattr(ai_status, "model_quota_badge", "READY")),
                "quarantined_families": quarantined,
                "contagion_shock_z": round(float(contagion_z), 2),
                "aegis_badge": aegis_badge,
                "paused": self.paused,
                "open_positions": serialized_positions,
                "rows": serialized_rows,
                "broker_events": serialized_events,
            }
        )

    def _dispatch_approved_trades(
        self,
        rows: Sequence[SymbolCockpitRow],
        open_positions: Sequence[PositionRiskSpec],
    ) -> None:
        """Route approved candidates through the 6 circuit breakers to the execution router."""
        if len(open_positions) >= self.execution_router.max_concurrent_positions:
            return

        open_symbols = self.execution_router.get_open_symbols()
        min_net_r = float(self.config.get("scanner", {}).get("min_net_avg_R", 0.10))
        risk_pct = float(self.config.get("risk", {}).get("max_risk_per_trade_pct", 0.5))
        risk_amount = (
            (self.execution_router.account_equity_usd * (risk_pct / 100.0))
            if self.execution_router.account_equity_usd > 0
            else 50.0
        )

        for row in rows:
            if len(open_positions) >= self.execution_router.max_concurrent_positions:
                break

            # Filter: must have AI approval and meet the min_net_avg_R floor
            is_approved = "APPROVE" in row.ai_verdict.upper()
            quant_ok = is_approved and (row.net_avg_r >= min_net_r)
            diode = AegisVerifier.apply_subtract_only_diode(
                quant_approved=quant_ok,
                ai_decision="APPROVE" if is_approved else "VETO",
                requested_risk_multiplier=1.0,
                recommended_risk_pct=row.rec_risk_pct if row.rec_risk_pct is not None else risk_pct,
                sl_price=row.stop_loss,
                tp_price=row.take_profit,
                max_risk_pct=risk_pct,
            )
            if not diode.final_approved:
                continue

            can_sym = self.feed_router.mt5.alias_mapper.canonicalize(row.symbol)
            if can_sym.upper() in open_symbols or row.symbol.upper() in open_symbols:
                continue

            if row.entry_price <= 0.0 or diode.stop_loss <= 0.0 or diode.take_profit <= 0.0:
                continue

            effective_risk_amount = (
                (self.execution_router.account_equity_usd * (diode.final_risk_pct / 100.0))
                if self.execution_router.account_equity_usd > 0
                else risk_amount
            )

            # Compute compliant lot size
            lots = self.execution_router.compute_compliant_lot_size(
                symbol=row.symbol,
                entry_price=row.entry_price,
                stop_loss=diode.stop_loss,
                risk_usd=effective_risk_amount,
            )

            side = row.side if row.side is not None else Side.BUY
            order_req = ExecutionOrderRequest(
                symbol=row.symbol,
                side=side,
                lots=lots,
                entry_price=row.entry_price,
                stop_loss=diode.stop_loss,
                take_profit=diode.take_profit,
                risk_pct=diode.final_risk_pct,
                risk_usd=effective_risk_amount,
                force_paper_shadow=(self.mode == "shadow-ab"),
                comment="ApexAuto",
            )

            receipt = self.execution_router.execute_order(
                order=order_req,
                open_positions=open_positions,
            )
            logger.info("Dispatched order for %s: %s on %s", row.symbol, receipt.status, receipt.venue)
            if receipt.status in ("FILLED", "PAPER_SHADOW_FILLED"):
                open_symbols.add(can_sym.upper())
                open_symbols.add(row.symbol.upper())
                open_positions = self.execution_router.get_open_positions()

    def shutdown(self) -> None:
        self.feed_router.shutdown()


def format_web_cockpit_banner(cockpit_url: str, mode: str) -> str:
    """Return a compact 5-line startup banner for the Web Cockpit engine console."""
    return (
        "============================================================\n"
        f"  POCKETBOT APEX v2.4.1  |  Mode: {mode.upper()}\n"
        f"  Chrome Cockpit URL: {cockpit_url}\n"
        "  Keep this engine window minimized;\n"
        "  view & control everything in the Chrome Cockpit window.\n"
        "============================================================\n"
    )


def format_single_line_heartbeat(orchestrator: ApexOrchestrator) -> str:
    """Return a single <100-char carriage-return heartbeat line that never wraps or scrolls."""
    eq = orchestrator.execution_router.account_equity_usd
    scanned = orchestrator.last_scan_summary.open_scanned if orchestrator.last_scan_summary else 0
    total = orchestrator.last_scan_summary.total_scanned if orchestrator.last_scan_summary else 0
    state = "PAUSED" if orchestrator.paused else "SCANNING"
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] APEX v2.4.1 | {state} | {scanned}/{total} syms | Eq: {eq:,.2f}"
    return line[:96].ljust(96)


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
    parser.add_argument("--dashboard", action="store_true", help="Run live WebSocket Chrome App Cockpit loop")
    parser.add_argument("--tui-only", action="store_true", help="Fallback to Rich terminal TUI instead of Chrome Cockpit")
    parser.add_argument("--interval", type=float, default=5.0, help="Refresh interval in seconds")
    parser.add_argument("--web-port", type=int, default=8765, help="Localhost port for WebSocket Chrome App Cockpit")
    parser.add_argument("--no-gui", action="store_true", help="Do not auto-launch the Chrome/Edge cockpit window")
    parser.add_argument("--unlock-live", action="store_true", help="Unlock live trading (requires env var too)")
    args = parser.parse_args(argv)

    orchestrator = ApexOrchestrator(
        mode=args.mode,
        config_path=args.config,
        is_live=bool(args.unlock_live),
        cli_args=sys.argv if argv is None else argv,
    )
    web_server: WebCockpitServer | None = None
    try:
        if args.once or (not args.dashboard and not args.tui_only):
            orchestrator.run_cycle()
            return 0

        if args.tui_only:
            with Live(
                console=console,
                refresh_per_second=4,
                screen=True,
                vertical_overflow="crop",
            ) as live:
                while True:
                    if not orchestrator.paused:
                        orchestrator.run_cycle(live=live, render_console_tui=True)
                    time.sleep(max(1.0, args.interval))

        web_server = WebCockpitServer(port=int(args.web_port))
        orchestrator.attach_web_cockpit(web_server)
        cockpit_url = web_server.start()
        sys.stdout.write(format_web_cockpit_banner(cockpit_url, args.mode))
        sys.stdout.flush()
        launch_chrome_app(cockpit_url, no_gui=bool(args.no_gui))

        last_heartbeat = 0.0
        scan_interval = max(1.0, float(args.interval))
        while True:
            if not orchestrator.paused:
                orchestrator.run_cycle(render_console_tui=False)
            now = time.monotonic()
            if now - last_heartbeat >= 5.0:
                sys.stdout.write("\r" + format_single_line_heartbeat(orchestrator))
                sys.stdout.flush()
                last_heartbeat = now
            # Sleep in 1s steps so pause/shutdown stays responsive
            remaining = scan_interval
            while remaining > 0.0:
                step = min(1.0, remaining)
                time.sleep(step)
                remaining -= step
                now_tick = time.monotonic()
                if now_tick - last_heartbeat >= 5.0:
                    sys.stdout.write("\r" + format_single_line_heartbeat(orchestrator))
                    sys.stdout.flush()
                    last_heartbeat = now_tick
    except KeyboardInterrupt:
        sys.stdout.write("\nPocketBot Apex stopped by user.\n")
        sys.stdout.flush()
        return 0
    finally:
        if web_server is not None:
            web_server.stop()
        orchestrator.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
