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
import asyncio
import atexit
import ctypes
import datetime
import json
import logging
import os
import signal
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from rich.console import Console
from rich.live import Live

from apex.__version__ import DISPLAY_VERSION, __version__
from apex.ai.aegis_verifier import AegisVerifier
from apex.ai.sleep_consolidation import SleepWakeConsolidator
from apex.ai.veto_committee import AIVetoCommittee, CandidateSetupAuditContext
from apex.cockpit.replay_engine import HistoricalReplayEngine
from apex.cockpit.telegram_remote import TelegramRemoteController
from apex.cockpit.tui_dashboard import ApexTUIDashboard, SymbolCockpitRow
from apex.cockpit.web_cockpit import WebCockpitServer, launch_chrome_app
from apex.compute.hyperion_fabric import get_hyperion_fabric
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
console = Console()


@dataclass(frozen=True)
class ApexConfig:
    """Validated immutable configuration binding `config/apex_config.json` keys."""

    account_mode: str = "DEMO"
    symbols: tuple[str, ...] = ("EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "BTCUSD")
    regime_enabled: bool = True
    regime_hurst_window: int = 100
    regime_adx_period: int = 14
    regime_trending_hurst_threshold: float = 0.52
    regime_trending_er_threshold: float = 0.25
    regime_max_spread_median_multiple: float = 1.5
    regime_max_tick_velocity_sigma: float = 4.0
    dom_obi_enabled: bool = True
    dom_min_abs_obi_threshold: float = 0.15
    dom_max_spread_pct_of_sl: float = 0.15
    dom_min_impulse_atr_multiple: float = 1.5
    dom_extreme_close_fraction: float = 0.25
    dom_wall_min_median_multiple: float = 3.0
    dom_wall_max_distance_r: float = 0.35
    correlation_cap_enabled: bool = True
    correlation_max_factor_risk_pct: float = 1.0
    dual_feed_guard_enabled: bool = True
    ai_veto_enabled: bool = True
    ai_mode: str = "auto_studio_first"
    ai_studio_timeout_seconds: float = 4.0
    ai_agy_pro_timeout_seconds: float = 15.0
    ai_priority_order: tuple[str, ...] | None = None
    scanner_mode: str = "FULL_MARKET_FUNNEL"
    scanner_max_candidates_per_cycle: int = 12
    scanner_auto_enable_market_watch: bool = True
    scanner_max_tick_age_seconds: float = 120.0
    scanner_min_candles_required: int = 30
    scanner_min_net_avg_R: float = 0.10
    risk_account_equity_usd: float = 10_000.0
    risk_max_risk_per_trade_pct: float = 0.5
    risk_max_daily_drawdown_pct: float = 2.0
    risk_max_concurrent_positions: int = 2
    risk_monte_carlo_replays: int = 8192
    risk_mean_block_length: float = 6.0
    sleep_enabled: bool = True
    sleep_rollover_utc_hour: int = 22
    sleep_rollover_utc_minute: int = 5
    sleep_stale_hours_threshold: float = 16.0
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw_cfg: dict[str, Any]) -> "ApexConfig":
        """Parse and validate raw JSON config dictionary into immutable ApexConfig."""
        cfg = dict(raw_cfg or {})
        filters = dict(cfg.get("filters", {}))
        reg = dict(filters.get("regime", {}))
        dom = dict(filters.get("dom_obi", {}))
        corr = dict(filters.get("correlation_cap", {}))
        dfg = dict(filters.get("dual_feed_guard", {}))
        ai = dict(filters.get("ai_veto", {}))
        scn = dict(cfg.get("scanner", {}))
        rsk = dict(cfg.get("risk", {}))
        slp = dict(cfg.get("sleep_wake", {}))
        raw_prio = ai.get("priority_order")
        equity = float(rsk.get("account_equity_usd", 10_000.0))
        max_risk = float(rsk.get("max_risk_per_trade_pct", 0.5))
        max_dd = float(rsk.get("max_daily_drawdown_pct", 2.0))
        replays = int(rsk.get("monte_carlo_replays", 8192))
        if equity <= 0.0 or max_risk <= 0.0 or max_dd <= 0.0 or replays < 1:
            raise ValueError("ApexConfig risk parameters must be strictly positive")
        return cls(
            account_mode=str(cfg.get("account_mode", "DEMO")).upper(),
            symbols=tuple(str(s) for s in cfg.get("symbols", ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "BTCUSD"])),
            regime_enabled=bool(reg.get("enabled", True)),
            regime_hurst_window=int(reg.get("hurst_window", 100)),
            regime_adx_period=int(reg.get("adx_period", 14)),
            regime_trending_hurst_threshold=float(reg.get("trending_hurst_threshold", 0.52)),
            regime_trending_er_threshold=float(reg.get("trending_er_threshold", 0.25)),
            regime_max_spread_median_multiple=float(reg.get("max_spread_median_multiple", 1.5)),
            regime_max_tick_velocity_sigma=float(reg.get("max_tick_velocity_sigma", 4.0)),
            dom_obi_enabled=bool(dom.get("enabled", True)),
            dom_min_abs_obi_threshold=float(dom.get("min_abs_obi_threshold", 0.15)),
            dom_max_spread_pct_of_sl=float(dom.get("max_spread_pct_of_sl", 0.15)),
            dom_min_impulse_atr_multiple=float(dom.get("min_impulse_atr_multiple", 1.5)),
            dom_extreme_close_fraction=float(dom.get("extreme_close_fraction", 0.25)),
            dom_wall_min_median_multiple=float(dom.get("wall_min_median_multiple", 3.0)),
            dom_wall_max_distance_r=float(dom.get("wall_max_distance_r", 0.35)),
            correlation_cap_enabled=bool(corr.get("enabled", True)),
            correlation_max_factor_risk_pct=float(corr.get("max_factor_risk_pct", 1.0)),
            dual_feed_guard_enabled=bool(dfg.get("enabled", True)),
            ai_veto_enabled=bool(ai.get("enabled", True)),
            ai_mode=str(ai.get("mode", "auto_studio_first")),
            ai_studio_timeout_seconds=min(float(ai.get("studio_timeout_seconds", ai.get("timeout_seconds", 4.0))), 4.0),
            ai_agy_pro_timeout_seconds=float(ai.get("agy_pro_timeout_seconds", 15.0)),
            ai_priority_order=tuple(str(p) for p in raw_prio) if raw_prio else None,
            scanner_mode=str(scn.get("mode", "FULL_MARKET_FUNNEL")),
            scanner_max_candidates_per_cycle=int(scn.get("max_candidates_per_cycle", 12)),
            scanner_auto_enable_market_watch=bool(scn.get("auto_enable_market_watch", True)),
            scanner_max_tick_age_seconds=float(scn.get("max_tick_age_seconds", 120.0)),
            scanner_min_candles_required=int(scn.get("min_candles_required", 30)),
            scanner_min_net_avg_R=float(scn.get("min_net_avg_R", 0.10)),
            risk_account_equity_usd=equity,
            risk_max_risk_per_trade_pct=max_risk,
            risk_max_daily_drawdown_pct=max_dd,
            risk_max_concurrent_positions=max(int(rsk.get("max_concurrent_positions", 2)), 1),
            risk_monte_carlo_replays=replays,
            risk_mean_block_length=float(rsk.get("mean_block_length", 6.0)),
            sleep_enabled=bool(slp.get("enabled", True)),
            sleep_rollover_utc_hour=int(slp.get("rollover_utc_hour", 22)),
            sleep_rollover_utc_minute=int(slp.get("rollover_utc_minute", 5)),
            sleep_stale_hours_threshold=float(slp.get("stale_hours_threshold", 16.0)),
            raw=cfg,
        )


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
        self.apex_config: ApexConfig = ApexConfig.from_dict(self.config)
        self.is_live: bool = bool(is_live or self.apex_config.account_mode == "LIVE")
        verify_live_unlock(self.is_live, cli_args=cli_args)

        self.feed_router = MarketFeedRouter(
            config_path=self.config_path,
            is_live=self.is_live,
            cli_args=cli_args,
        )
        self.regime_classifier = RegimeClassifier(
            enabled=self.apex_config.regime_enabled,
            hurst_window=self.apex_config.regime_hurst_window,
            adx_atr_period=self.apex_config.regime_adx_period,
            trending_hurst_threshold=self.apex_config.regime_trending_hurst_threshold,
            trending_er_threshold=self.apex_config.regime_trending_er_threshold,
            max_spread_median_multiple=self.apex_config.regime_max_spread_median_multiple,
            max_tick_velocity_sigma=self.apex_config.regime_max_tick_velocity_sigma,
        )
        self.momentum_dom_gate = MomentumDOMGate(
            dom_obi_enabled=self.apex_config.dom_obi_enabled,
            min_abs_obi=self.apex_config.dom_min_abs_obi_threshold,
            max_spread_pct_of_sl=self.apex_config.dom_max_spread_pct_of_sl,
            min_impulse_atr_multiple=self.apex_config.dom_min_impulse_atr_multiple,
            extreme_close_fraction=self.apex_config.dom_extreme_close_fraction,
            wall_min_median_multiple=self.apex_config.dom_wall_min_median_multiple,
            wall_max_distance_r=self.apex_config.dom_wall_max_distance_r,
        )
        self.time_machine = BlockBootstrapTimeMachine(
            replays=self.apex_config.risk_monte_carlo_replays,
            mean_block_length=self.apex_config.risk_mean_block_length,
            seed=19,
            account_equity_usd=self.apex_config.risk_account_equity_usd,
            max_risk_pct=self.apex_config.risk_max_risk_per_trade_pct,
        )
        self.ledger = WalkForwardLedgerDB(db_path=db_path)
        self.ai_committee = AIVetoCommittee(
            enabled=self.apex_config.ai_veto_enabled,
            timeout_seconds=self.apex_config.ai_studio_timeout_seconds,
            agy_timeout_seconds=self.apex_config.ai_agy_pro_timeout_seconds,
            mode=self.apex_config.ai_mode,
            priority_order=list(self.apex_config.ai_priority_order) if self.apex_config.ai_priority_order else None,
            config_path=self.config_path,
            ledger_db=self.ledger,
        )
        self.execution_router = DualBrokerExecutionRouter(
            feed_router=self.feed_router,
            account_equity_usd=self.apex_config.risk_account_equity_usd,
            max_daily_drawdown_pct=self.apex_config.risk_max_daily_drawdown_pct,
            max_concurrent_positions=self.apex_config.risk_max_concurrent_positions,
            max_factor_exposure_pct=self.apex_config.correlation_max_factor_risk_pct,
            max_spread_slippage_sl_ratio=self.apex_config.dom_max_spread_pct_of_sl,
            correlation_cap_enabled=self.apex_config.correlation_cap_enabled,
            dom_obi_enabled=self.apex_config.dom_obi_enabled,
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
            correlation_shield=self.execution_router.correlation_shield,
        )
        self.last_scan_summary: UniverseScanSummary | None = None
        self.position_manager = AutonomousPositionOrchestrator(
            regime_classifier=self.regime_classifier,
            time_machine=self.time_machine,
            ledger=self.ledger,
            feed_router=self.feed_router,
        )
        self.web_cockpit: WebCockpitServer | None = None
        self.paused: bool = False
        self._bg_lock = threading.Lock()
        self._drill_running = False
        self._replay_running = False
        self._pending_confirmations: dict[str, dict[str, Any]] = {}
        self.confirm_input_fn: Optional[Callable[[str], str]] = None
        self._telegram_stop_event = threading.Event()
        self._telegram_thread: threading.Thread | None = None
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

    def _register_symbol_returns(self, symbol_list: Sequence[str]) -> None:
        """Feed recent closed-bar simple returns into CorrelationShield for covariance & tail checks."""
        for sym in symbol_list:
            if not sym:
                continue
            try:
                candles = self.feed_router.get_candles(str(sym), timeframe="M5", count=60)
                if len(candles) < 10:
                    continue
                rets = [
                    (candles[i].close - candles[i - 1].close) / candles[i - 1].close
                    for i in range(1, len(candles))
                    if candles[i - 1].close > 0.0
                ]
                if len(rets) >= 8:
                    self.execution_router.correlation_shield.register_return_series(str(sym), rets)
            except Exception as exc:
                logger.debug("Skipping return series warm-up for %s: %s", sym, exc)
                continue

    def run_cycle(
        self,
        symbols: Sequence[str] | None = None,
        live: Live | None = None,
        render_console_tui: bool = True,
    ) -> list[SymbolCockpitRow]:
        """Run a full quantitative scan, regime classification, Time Machine v2, and ledger cycle."""
        active_src = self.feed_router.connect()
        self.execution_router.update_account_equity_from_broker()
        open_positions = self.execution_router.get_open_positions()
        seed_syms = list(symbols or self.apex_config.symbols) + [p.symbol for p in open_positions]
        self._register_symbol_returns(seed_syms)

        # Sync live account equity, currency, and daily P/L BEFORE scan_cycle so budget gates see live DD
        account_currency = (
            self.feed_router.mt5.get_account_currency()
            if self.feed_router.mt5.is_connected
            else "USD"
        )
        closed_pnl_today = 0.0
        floating_pnl = 0.0
        if self.feed_router.mt5.is_connected:
            acct_info = self.feed_router.mt5.get_account_info()
            eq = float(acct_info.get("equity", 0.0) or 0.0)
            bal = float(acct_info.get("balance", 0.0) or 0.0)
            if eq > 0.0 and bal > 0.0:
                floating_pnl = eq - bal
            try:
                import MetaTrader5 as _mt5

                now_dt = datetime.datetime.now(datetime.timezone.utc)
                utc_start = now_dt.replace(hour=0, minute=0, second=0, microsecond=0)
                deals = _mt5.history_deals_get(
                    utc_start - datetime.timedelta(hours=12),
                    now_dt + datetime.timedelta(hours=14),
                )
                bal_type = int(getattr(_mt5, "DEAL_TYPE_BALANCE", 2))
                if deals:
                    start_ts = utc_start.timestamp()
                    for d in deals:
                        d_type = getattr(d, "type", -1)
                        if isinstance(d_type, int) and d_type == bal_type:
                            continue
                        raw_t = getattr(d, "time", 0)
                        if isinstance(raw_t, (int, float)) and raw_t > 0:
                            norm_ts = float(self.feed_router.mt5.normalize_server_timestamp_to_utc(float(raw_t)))
                            if norm_ts < start_ts:
                                continue
                        prof = getattr(d, "profit", 0.0)
                        swp = getattr(d, "swap", 0.0)
                        com = getattr(d, "commission", 0.0)
                        if isinstance(prof, (int, float)) and not isinstance(prof, bool):
                            closed_pnl_today += float(prof)
                        if isinstance(swp, (int, float)) and not isinstance(swp, bool):
                            closed_pnl_today += float(swp)
                        if isinstance(com, (int, float)) and not isinstance(com, bool):
                            closed_pnl_today += float(com)
            except Exception as exc:
                logger.debug("MT5 history_deals_get daily P/L skipped: %s", exc)
        else:
            floating_pnl = sum(float(getattr(p, "profit_usd", 0.0) or 0.0) for p in open_positions)

        daily_pnl = closed_pnl_today + floating_pnl
        self._last_closed_pnl_today_usd = closed_pnl_today
        self._last_floating_pnl_usd = floating_pnl
        self._last_daily_pnl_usd = daily_pnl

        eq_base = max(
            1e-6,
            float(
                self.execution_router.account_equity_usd
                or self.apex_config.risk_account_equity_usd
                or 10_000.0
            ),
        )
        daily_pnl_pct = (daily_pnl / eq_base) * 100.0
        remaining_daily_budget_pct = max(
            0.0,
            float(self.apex_config.risk_max_daily_drawdown_pct) - max(0.0, -daily_pnl_pct),
        )

        use_funnel = symbols is None and self.apex_config.scanner_mode == "FULL_MARKET_FUNNEL"

        if use_funnel:
            rows, summary = self.universe_scanner.scan_cycle(
                open_positions=open_positions,
                remaining_daily_budget_pct=remaining_daily_budget_pct,
            )
            self.last_scan_summary = summary
        else:
            target_symbols = list(symbols or self.apex_config.symbols)
            rows, summary = self.universe_scanner.scan_cycle(
                target_symbols=target_symbols,
                open_positions=open_positions,
                remaining_daily_budget_pct=remaining_daily_budget_pct,
            )
            self.last_scan_summary = summary

        self._register_symbol_returns([r.symbol for r in rows])
        self.last_matrix = list(rows)

        # In-Flight Multi-Agent Trajectory Orchestrator & Closed-Deal Autopsy (MT5 + cTrader)
        contagion_z_live = self.universe_scanner.last_contagion.cross_asset_shock_z if self.universe_scanner.last_contagion else 0.0
        open_tickets: set[int] = set()
        if self.feed_router.mt5.is_connected:
            raw_mt5_positions = self.feed_router.mt5.get_open_positions()
            for p in raw_mt5_positions:
                tkt = int(p.get("ticket", 0))
                raw_sym = str(p.get("symbol", ""))
                if tkt <= 0 or not raw_sym:
                    continue
                open_tickets.add(tkt)
                is_buy = int(p.get("type", 0)) == 0
                pos_candles = self.feed_router.get_candles(raw_sym, timeframe="M5", count=60)
                pos_tick = self.feed_router.mt5.get_tick(raw_sym)
                init_1r_mt5 = self.ledger.get_open_initial_one_r(order_id=str(tkt), symbol=raw_sym)
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
                    initial_one_r=init_1r_mt5,
                    execute_broker_actions=(self.mode == "demo-auto"),
                )
        if self.feed_router.ctrader.is_connected and hasattr(self.feed_router.ctrader, "get_open_positions"):
            for cp in self.feed_router.ctrader.get_open_positions():
                pid = int(cp.get("position_id", 0) or 0)
                raw_sym = str(cp.get("symbol", "") or "")
                if pid <= 0 or not raw_sym:
                    continue
                open_tickets.add(pid)
                is_buy = str(cp.get("side", "BUY")).upper() == "BUY"
                pos_candles = self.feed_router.get_candles(raw_sym, timeframe="M5", count=60)
                pos_tick = self.feed_router.ctrader.get_latest_tick(raw_sym)
                mark_px = (
                    (pos_tick.bid if is_buy else pos_tick.ask)
                    if pos_tick is not None
                    else float(cp.get("entry_price", 0.0) or 0.0)
                )
                init_1r_ct = self.ledger.get_open_initial_one_r(order_id=str(pid), symbol=raw_sym)
                self.position_manager.evaluate_position(
                    ticket=pid,
                    symbol=raw_sym,
                    side=Side.BUY if is_buy else Side.SELL,
                    entry_price=float(cp.get("entry_price", 0.0) or 0.0),
                    current_sl=float(cp.get("stop_loss", 0.0) or 0.0),
                    current_tp=float(cp.get("take_profit", 0.0) or 0.0),
                    mark_price=mark_px,
                    candles=pos_candles,
                    spread_price=pos_tick.spread if pos_tick else 0.0,
                    cross_asset_shock_z=contagion_z_live,
                    opened_at_utc=float(cp.get("open_timestamp_utc", time.time()) or time.time()),
                    lots=float(cp.get("lots", 0.01) or 0.01),
                    initial_one_r=init_1r_ct,
                    execute_broker_actions=(self.mode == "demo-auto"),
                )
        if self.feed_router.mt5.is_connected or self.feed_router.ctrader.is_connected:
            self.position_manager.reconcile_closed_deals(open_tickets)

        # Settle open PAPER_SHADOW trades against closed bar OHLC & free closed slots
        shadow_syms: set[str] = {
            str(sp.get("symbol", "")).upper()
            for sp in self.execution_router._shadow_positions
            if sp.get("symbol")
        } | {
            r.symbol.upper()
            for r in rows
            if getattr(r, "quarantine_mode", "") == SymbolExecutionMode.PAPER_SHADOW.value
        }
        settled_total = 0
        for s_sym in sorted(shadow_syms):
            try:
                s_bars = self.feed_router.get_candles(s_sym, timeframe="M5", count=2)
                if s_bars:
                    bar = s_bars[-1]
                    settled_total += self.ledger.settle_paper_shadow_trades(s_sym, bar.high, bar.low, bar.close)
            except Exception as exc:
                logger.debug("settle_paper_shadow_trades skipped for %s: %s", s_sym, exc)
        if settled_total > 0 or self.execution_router._shadow_positions:
            self.execution_router.prune_closed_shadow_positions()
            open_positions = self.execution_router.get_open_positions()

        # In demo-auto or shadow-ab mode, automatically dispatch approved rows;
        # in confirm mode, prompt interactively or queue tickets for WebSocket /api/control/confirm.
        if self.mode in ("demo-auto", "shadow-ab"):
            self._dispatch_approved_trades(rows, open_positions=open_positions)
            open_positions = self.execution_router.get_open_positions()
        elif self.mode == "confirm":
            self._process_confirm_mode_candidates(rows, open_positions=open_positions)
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
        server.set_control_handler("confirm", self._handle_web_confirm)

    def _handle_web_pause(self) -> dict[str, object]:
        with self._bg_lock:
            self.paused = not self.paused
            paused_now = self.paused
        if self.web_cockpit is not None:
            self.web_cockpit.push_state({"paused": paused_now})
        return {"paused": paused_now, "message": "Paused" if paused_now else "Resumed"}

    def _handle_web_drill(self, payload: dict[str, Any] | None = None) -> dict[str, object]:
        payload = payload or {}
        raw_sym = str(payload.get("symbol", "") or "").strip().upper()
        sym_override = None if raw_sym in {"", "AUTO_BEST_MOVER", "AUTO_TOP12"} else raw_sym
        drill_mode = str(payload.get("drill_mode", "plumbing") or "plumbing").strip().lower()
        if drill_mode not in {"strict", "plumbing"}:
            drill_mode = "plumbing"
        hold_secs = 15

        with self._bg_lock:
            if self._drill_running:
                return {"started": False, "message": "Live drill is already running."}
            self._drill_running = True

        def _worker() -> None:
            try:
                from tools.live_lifecycle_drill import run_live_lifecycle_drill

                cb = self.web_cockpit.update_drill_state if self.web_cockpit is not None else None
                run_live_lifecycle_drill(
                    hold_seconds=hold_secs,
                    symbol_override=sym_override,
                    db_path=self.ledger.db_path,
                    on_update=cb,
                    drill_mode=drill_mode,
                )
            except Exception as exc:
                if self.web_cockpit is not None:
                    self.web_cockpit.update_drill_state({"status": "ERROR", "message": str(exc)})
            finally:
                with self._bg_lock:
                    self._drill_running = False

        threading.Thread(target=_worker, name="ApexWebDrillThread", daemon=True).start()
        target_label = sym_override or "Auto Best Open Mover"
        return {"started": True, "message": f"Started {hold_secs}s Live Drill ({drill_mode.upper()}) on {target_label}."}

    def _resolve_replay_symbols(self, raw_sym: str) -> tuple[tuple[str, ...], str]:
        """Resolve cockpit selector into concrete replay symbols & display label."""
        from apex.cockpit.replay_engine import DEFAULT_REPLAY_SYMBOLS

        matrix = list(getattr(self, "last_matrix", None) or ())
        if raw_sym in {"", "AUTO_TOP12"}:
            active = [str(r.symbol).upper() for r in matrix if getattr(r, "symbol", None)]
            merged: list[str] = []
            for s in [*active, "BTCUSD", "ETHUSD", *DEFAULT_REPLAY_SYMBOLS]:
                if s and s not in merged:
                    merged.append(s)
            return tuple(merged[:12]), "Auto Top-12 Active Basket"
        if raw_sym == "AUTO_BEST_MOVER":
            if matrix:
                best = max(matrix, key=lambda r: abs(float(getattr(r, "net_avg_r", 0.0) or 0.0)))
                sym = str(getattr(best, "symbol", "") or "").upper()
                if sym:
                    return (sym,), f"Auto Best Open Mover ({sym})"
            return ("BTCUSD",), "Auto Best Open Mover (BTCUSD)"
        return (raw_sym,), raw_sym

    def _handle_web_replay(self, payload: dict[str, Any] | None = None) -> dict[str, object]:
        payload = payload or {}
        raw_sym = str(payload.get("symbol", "") or "").strip().upper()
        symbols_arg, target_label = self._resolve_replay_symbols(raw_sym)
        try:
            window_days = int(payload.get("window_days", 5))
        except (TypeError, ValueError):
            window_days = 5
        if window_days not in {2, 5, 10}:
            window_days = 5
        try:
            speed = float(payload.get("speed", 100.0))
        except (TypeError, ValueError):
            speed = 100.0
        if speed not in {20.0, 50.0, 100.0}:
            speed = 100.0

        with self._bg_lock:
            if self._replay_running:
                return {"started": False, "message": "Historical Replay is already running."}
            self._replay_running = True

        def _worker() -> None:
            try:
                engine = HistoricalReplayEngine()
                cb = self.web_cockpit.update_replay_state if self.web_cockpit is not None else None
                engine.run_replay(
                    feed_router=self.feed_router,
                    symbols=symbols_arg,
                    window_days=window_days,
                    speed=speed,
                    on_update=cb,
                )
            except Exception as exc:
                if self.web_cockpit is not None:
                    self.web_cockpit.update_replay_state({"status": "ERROR", "message": str(exc)})
            finally:
                with self._bg_lock:
                    self._replay_running = False

        threading.Thread(target=_worker, name="ApexWebReplayThread", daemon=True).start()
        return {"started": True, "message": f"Started {int(speed)}x ({window_days}d) Replay on {target_label}."}

    def _handle_web_refresh_ai(self) -> dict[str, object]:
        with self._bg_lock:
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
            entry_px = float(getattr(pos, "entry_price", 0.0) or traj.get("entry_price", 0.0) or 0.0)
            curr_px = float(getattr(pos, "current_price", 0.0) or traj.get("mark_price", entry_px) or entry_px)
            sl_px = float(getattr(pos, "stop_loss", 0.0) or traj.get("current_sl", 0.0) or 0.0)
            tp_px = float(getattr(pos, "take_profit", 0.0) or traj.get("current_tp", 0.0) or 0.0)
            lots_val = float(getattr(pos, "lots", 0.0) or traj.get("lots", 0.0) or 0.0)
            profit_val = float(getattr(pos, "profit_usd", 0.0) or traj.get("profit_usd", 0.0) or 0.0)
            serialized_positions.append(
                {
                    "symbol": str(getattr(pos, "symbol", "")),
                    "side": str(getattr(getattr(pos, "side", ""), "value", getattr(pos, "side", ""))),
                    "lots": lots_val,
                    "entry_price": entry_px,
                    "current_price": curr_px,
                    "stop_loss": sl_px,
                    "take_profit": tp_px,
                    "profit_usd": profit_val,
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

        catalog_grouped: dict[str, list[str]] = {}
        try:
            catalog_grouped = self.universe_scanner.get_grouped_catalog()
        except Exception as exc:
            logger.warning("Failed to build grouped catalog for cockpit: %s", exc)
            catalog_grouped = {}

        self.web_cockpit.push_state(
            {
                "version": __version__,
                "display_version": DISPLAY_VERSION,
                "run_mode": self.mode.upper(),
                "mind_state_badge": mind_badge,
                "universe_scanned_count": self.last_scan_summary.open_scanned if self.last_scan_summary else 0,
                "universe_total_count": self.last_scan_summary.total_scanned if self.last_scan_summary else 0,
                "catalog_symbols": catalog_grouped,
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
                "compute_fabric": get_hyperion_fabric().to_cockpit_dict(),
                "paused": self.paused,
                "open_positions": serialized_positions,
                "pending_confirmations": self._serialize_pending_confirmations(),
                "rows": serialized_rows,
                "broker_events": serialized_events,
            }
        )

    def _build_approved_order_request(
        self,
        row: SymbolCockpitRow,
        open_symbols: set[str],
        min_net_r: float,
        risk_pct: float,
        risk_amount: float,
    ) -> Optional[tuple[ExecutionOrderRequest, str]]:
        """Validate one approved row through the Subtract-Only Diode & build ExecutionOrderRequest."""
        is_approved = "APPROVE" in row.ai_verdict.upper()
        quant_ok = is_approved and (row.net_avg_r is not None and row.net_avg_r >= min_net_r)
        ai_risk_mult = float((row.pipeline_trace or {}).get("ai_risk_multiplier", 1.0) or 1.0)
        diode = AegisVerifier.apply_subtract_only_diode(
            quant_approved=quant_ok,
            ai_decision="APPROVE" if is_approved else "VETO",
            requested_risk_multiplier=ai_risk_mult,
            recommended_risk_pct=row.rec_risk_pct if row.rec_risk_pct is not None else risk_pct,
            sl_price=row.stop_loss,
            tp_price=row.take_profit,
            max_risk_pct=risk_pct,
        )
        if not diode.final_approved:
            return None

        can_sym = self.feed_router.mt5.alias_mapper.canonicalize(row.symbol)
        if can_sym.upper() in open_symbols or row.symbol.upper() in open_symbols:
            return None

        if row.entry_price <= 0.0 or diode.stop_loss <= 0.0 or diode.take_profit <= 0.0:
            return None

        effective_risk_amount = (
            (self.execution_router.account_equity_usd * (diode.final_risk_pct / 100.0))
            if self.execution_router.account_equity_usd > 0
            else risk_amount
        )
        lots = self.execution_router.compute_compliant_lot_size(
            symbol=row.symbol,
            entry_price=row.entry_price,
            stop_loss=diode.stop_loss,
            risk_usd=effective_risk_amount,
        )
        if lots <= 0.0:
            return None
        side = row.side if row.side is not None else Side.BUY
        trace_dict = row.pipeline_trace or {}
        req = ExecutionOrderRequest(
            symbol=row.symbol,
            side=side,
            lots=lots,
            entry_price=row.entry_price,
            stop_loss=diode.stop_loss,
            take_profit=diode.take_profit,
            risk_pct=diode.final_risk_pct,
            risk_usd=effective_risk_amount,
            force_paper_shadow=(self.mode == "shadow-ab"),
            comment="ApexConfirm" if self.mode == "confirm" else "ApexAuto",
            predicted_net_r=float(row.net_avg_r or 0.0),
            regime=str(row.regime or "UNKNOWN"),
            ai_decision=str(row.ai_verdict or "APPROVE"),
            ai_confidence=float(trace_dict.get("ai_confidence", 1.0) or 1.0),
            ai_reason=str(trace_dict.get("ai_reason", "") or ""),
            dom_obi=row.obi,
        )
        return req, can_sym.upper()

    def _record_receipt_open(self, order_req: ExecutionOrderRequest, receipt: Any) -> None:
        """Persist confirmed fill or paper-shadow open entry to `trade_ledger` (`is_closed = 0`)."""
        if getattr(receipt, "status", "") not in ("FILLED", "PAPER_SHADOW_FILLED"):
            return
        targets = getattr(receipt, "child_receipts", ()) or (receipt,)
        for rec in targets:
            if getattr(rec, "status", "") not in ("FILLED", "PAPER_SHADOW_FILLED"):
                continue
            exec_px = float(getattr(rec, "executed_price", 0.0) or order_req.entry_price)
            sl_px = float(getattr(rec, "stop_loss", order_req.stop_loss) or order_req.stop_loss)
            exec_mode = (
                SymbolExecutionMode.PAPER_SHADOW
                if (getattr(rec, "status", "") == "PAPER_SHADOW_FILLED" or getattr(rec, "venue", "") == "PAPER_SHADOW" or order_req.force_paper_shadow)
                else SymbolExecutionMode.ACTIVE
            )
            try:
                self.ledger.record_trade_open(
                    order_id=str(getattr(rec, "order_id", "")),
                    symbol=str(getattr(rec, "symbol", order_req.symbol)),
                    side=order_req.side.value,
                    lots=float(getattr(rec, "lots", order_req.lots)),
                    entry_price=exec_px,
                    stop_loss=sl_px,
                    take_profit=float(getattr(rec, "take_profit", order_req.take_profit)),
                    initial_one_r=abs(exec_px - sl_px) or abs(order_req.entry_price - order_req.stop_loss),
                    predicted_net_r=order_req.predicted_net_r,
                    regime=order_req.regime,
                    venue=str(getattr(rec, "winning_venue", "") or getattr(rec, "venue", "mt5")),
                    execution_mode=exec_mode,
                    dom_obi=order_req.dom_obi,
                    ai_decision=order_req.ai_decision,
                    ai_confidence=order_req.ai_confidence,
                    ai_reason=order_req.ai_reason,
                    realized_slippage_r=float(getattr(rec, "realized_slippage_r", 0.0)),
                    mesh_topology=str(getattr(rec, "mesh_topology", "MT5_SOLO")),
                    sor_pips_saved=float(getattr(rec, "sor_pips_saved", 0.0)),
                    sor_usd_saved=float(getattr(rec, "sor_usd_saved", 0.0)),
                )
            except Exception as exc:
                logger.debug("Orchestrator record_trade_open skipped: %s", exc)

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

            built = self._build_approved_order_request(row, open_symbols, min_net_r, risk_pct, risk_amount)
            if built is None:
                continue
            order_req, can_sym_up = built

            t_now, _ = self.feed_router.get_tick(row.symbol)
            curr_spread = float(t_now.spread) if t_now is not None else 0.0
            dom_now = self.feed_router.get_dom_analytics(row.symbol)
            receipt = self.execution_router.execute_order(
                order=order_req,
                closed_pnl_today_usd=float(getattr(self, "_last_closed_pnl_today_usd", 0.0)),
                floating_pnl_usd=float(getattr(self, "_last_floating_pnl_usd", 0.0)),
                open_positions=open_positions,
                current_spread_price=curr_spread,
                dom_analytics=dom_now,
            )
            logger.info("Dispatched order for %s: %s on %s", row.symbol, receipt.status, receipt.venue)
            if receipt.status in ("FILLED", "PAPER_SHADOW_FILLED"):
                self._record_receipt_open(order_req, receipt)
                open_symbols.add(can_sym_up)
                open_symbols.add(row.symbol.upper())
                open_positions = self.execution_router.get_open_positions()

    def _process_confirm_mode_candidates(
        self,
        rows: Sequence[SymbolCockpitRow],
        open_positions: Sequence[PositionRiskSpec],
    ) -> None:
        """Handle `--mode confirm`: interactive `[y/N]` prompt or Web Cockpit ticket queue."""
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

        interactive_prompt = self.confirm_input_fn
        if interactive_prompt is None and sys.stdin is not None and sys.stdin.isatty() and not os.environ.get("PYTEST_CURRENT_TEST"):
            interactive_prompt = input

        for row in rows:
            if len(open_positions) >= self.execution_router.max_concurrent_positions:
                break
            built = self._build_approved_order_request(row, open_symbols, min_net_r, risk_pct, risk_amount)
            if built is None:
                continue
            order_req, can_sym_up = built
            confirm_id = f"{can_sym_up}:{order_req.side.value}"

            if interactive_prompt is not None:
                prompt_msg = (
                    f"[CONFIRM] Execute {order_req.side.value} {row.symbol} {order_req.lots:.2f} lots @ "
                    f"{order_req.entry_price:.5f} (SL={order_req.stop_loss:.5f}, TP={order_req.take_profit:.5f}, "
                    f"net_avg_R={float(row.net_avg_r or 0.0):+.2f}R, risk={order_req.risk_pct:.2f}%)? [y/N]: "
                )
                try:
                    ans = str(interactive_prompt(prompt_msg) or "").strip().lower()
                except (EOFError, OSError):
                    ans = ""
                if ans in {"y", "yes"}:
                    t_now, _ = self.feed_router.get_tick(row.symbol)
                    curr_spread = float(t_now.spread) if t_now is not None else 0.0
                    dom_now = self.feed_router.get_dom_analytics(row.symbol)
                    receipt = self.execution_router.execute_order(
                        order=order_req,
                        closed_pnl_today_usd=float(getattr(self, "_last_closed_pnl_today_usd", 0.0)),
                        floating_pnl_usd=float(getattr(self, "_last_floating_pnl_usd", 0.0)),
                        open_positions=open_positions,
                        current_spread_price=curr_spread,
                        dom_analytics=dom_now,
                    )
                    logger.info("Confirmed order executed for %s: %s (%s)", row.symbol, receipt.status, receipt.venue)
                    if receipt.status in ("FILLED", "PAPER_SHADOW_FILLED"):
                        self._record_receipt_open(order_req, receipt)
                        open_symbols.add(can_sym_up)
                        open_symbols.add(row.symbol.upper())
                        open_positions = self.execution_router.get_open_positions()
                else:
                    logger.info("User declined candidate trade %s in confirm mode.", confirm_id)
                continue

            with self._bg_lock:
                self._pending_confirmations[confirm_id] = {
                    "confirm_id": confirm_id,
                    "symbol": row.symbol,
                    "side": order_req.side.value,
                    "lots": order_req.lots,
                    "entry_price": order_req.entry_price,
                    "stop_loss": order_req.stop_loss,
                    "take_profit": order_req.take_profit,
                    "risk_pct": order_req.risk_pct,
                    "net_avg_r": float(row.net_avg_r or 0.0),
                    "created_at_utc": time.time(),
                    "order_request": order_req,
                }
        self._push_pending_confirmations()

    def _serialize_pending_confirmations(self) -> list[dict[str, Any]]:
        """Serialize queued `--mode confirm` tickets without non-JSON `ExecutionOrderRequest`."""
        with self._bg_lock:
            return [
                {k: v for k, v in item.items() if k != "order_request"}
                for item in self._pending_confirmations.values()
            ]

    def _push_pending_confirmations(self) -> None:
        """Broadcast current `--mode confirm` queue to attached WebCockpit if present."""
        if self.web_cockpit is not None:
            self.web_cockpit.push_state({"pending_confirmations": self._serialize_pending_confirmations()})

    def _handle_web_confirm(self, payload: dict[str, Any] | None = None) -> dict[str, object]:
        """Execute or reject a queued `--mode confirm` candidate ticket via WebCockpit `/api/control/confirm`."""
        payload = payload or {}
        confirm_id = str(payload.get("confirm_id", "") or "").strip().upper()
        decision = str(payload.get("decision", "REJECT") or "REJECT").strip().upper()
        with self._bg_lock:
            ticket = self._pending_confirmations.pop(confirm_id, None)
        if ticket is None:
            return {"executed": False, "message": f"Unknown or expired confirmation ticket: {confirm_id}"}
        if decision != "APPROVE":
            self._push_pending_confirmations()
            return {"executed": False, "rejected": True, "confirm_id": confirm_id, "message": f"Rejected {confirm_id}."}

        order_req = ticket["order_request"]
        open_positions = self.execution_router.get_open_positions()
        t_now, _ = self.feed_router.get_tick(order_req.symbol)
        curr_spread = float(t_now.spread) if t_now is not None else 0.0
        dom_now = self.feed_router.get_dom_analytics(order_req.symbol)
        receipt = self.execution_router.execute_order(
            order=order_req,
            closed_pnl_today_usd=float(getattr(self, "_last_closed_pnl_today_usd", 0.0)),
            floating_pnl_usd=float(getattr(self, "_last_floating_pnl_usd", 0.0)),
            open_positions=open_positions,
            current_spread_price=curr_spread,
            dom_analytics=dom_now,
        )
        if receipt.status in ("FILLED", "PAPER_SHADOW_FILLED"):
            self._record_receipt_open(order_req, receipt)
        if self.web_cockpit is not None:
            latest_positions = self.execution_router.get_open_positions()
            serialized_pos = [
                {
                    "symbol": str(getattr(p, "symbol", "")),
                    "side": str(getattr(getattr(p, "side", ""), "value", getattr(p, "side", ""))),
                    "lots": float(getattr(p, "lots", 0.0) or 0.0),
                    "entry_price": float(getattr(p, "entry_price", 0.0) or 0.0),
                    "current_price": float(getattr(p, "current_price", 0.0) or getattr(p, "entry_price", 0.0) or 0.0),
                    "stop_loss": float(getattr(p, "stop_loss", 0.0) or 0.0),
                    "take_profit": float(getattr(p, "take_profit", 0.0) or 0.0),
                    "profit_usd": float(getattr(p, "profit_usd", 0.0) or 0.0),
                }
                for p in latest_positions
            ]
            self.web_cockpit.push_state(
                {
                    "pending_confirmations": self._serialize_pending_confirmations(),
                    "open_positions": serialized_pos,
                    "open_positions_count": len(serialized_pos),
                }
            )
        else:
            self._push_pending_confirmations()
        return {
            "executed": True,
            "confirm_id": confirm_id,
            "status": receipt.status,
            "venue": receipt.venue,
            "message": f"Executed {confirm_id}: {receipt.status} on {receipt.venue}",
        }

    def start_telegram_polling(self) -> bool:
        """Start supervised background Telegram `getUpdates` poller when configured."""
        if not self.telegram.is_configured:
            return False
        if self._telegram_thread is not None and self._telegram_thread.is_alive():
            return True
        self._telegram_stop_event.clear()

        def _poll_worker() -> None:
            async def _runner() -> None:
                async_stop = asyncio.Event()
                poll_task = asyncio.create_task(self.telegram.poll_updates_loop(stop_event=async_stop))
                try:
                    while not poll_task.done():
                        if self._telegram_stop_event.is_set():
                            async_stop.set()
                            break
                        await asyncio.sleep(0.5)
                    await poll_task
                except asyncio.CancelledError:
                    async_stop.set()
                    poll_task.cancel()

            try:
                asyncio.run(_runner())
            except Exception as exc:
                logger.warning("Telegram polling thread exited: %s", exc)

        self._telegram_thread = threading.Thread(target=_poll_worker, name="ApexTelegramPoller", daemon=True)
        self._telegram_thread.start()
        return True

    def stop_telegram_polling(self) -> None:
        """Signal the supervised Telegram poller to exit cleanly."""
        self._telegram_stop_event.set()
        if self._telegram_thread is not None and self._telegram_thread.is_alive():
            self._telegram_thread.join(timeout=2.0)
        self._telegram_thread = None

    def shutdown(self) -> None:
        self.stop_telegram_polling()
        self.feed_router.shutdown()
        if hasattr(self.ledger, "close"):
            self.ledger.close()


def _is_pid_running(pid: int) -> bool:
    """Return True if `pid` corresponds to a live Python process."""
    if pid <= 0 or pid == os.getpid():
        return False
    if os.name == "nt":
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        SYNCHRONIZE = 0x00100000
        STILL_ACTIVE = 259
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE, False, int(pid))
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong(0)
            if ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)) == 0:
                return False
            if exit_code.value != STILL_ACTIVE:
                return False
            buf = ctypes.create_unicode_buffer(512)
            size = ctypes.c_ulong(512)
            if ctypes.windll.kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
                return "python" in buf.value.lower()
            return True
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _terminate_pid(pid: int) -> None:
    """Cleanly terminate an existing `apex_runner.py` PID before taking over the single-instance lock."""
    if pid <= 0 or pid == os.getpid():
        return
    try:
        if os.name == "nt":
            PROCESS_TERMINATE = 0x0001
            SYNCHRONIZE = 0x00100000
            handle = ctypes.windll.kernel32.OpenProcess(PROCESS_TERMINATE | SYNCHRONIZE, False, int(pid))
            if handle:
                try:
                    ctypes.windll.kernel32.TerminateProcess(handle, 0)
                    ctypes.windll.kernel32.WaitForSingleObject(handle, 3000)
                finally:
                    ctypes.windll.kernel32.CloseHandle(handle)
        else:
            os.kill(pid, signal.SIGTERM)
            time.sleep(0.5)
    except OSError as exc:
        logger.debug("Existing apex_runner PID %d termination skipped: %s", pid, exc)


def _acquire_single_instance_lock(lock_path: Path = Path("trader/apex_runner.lock")) -> Path:
    """Ensure only one daemon `apex_runner.py` instance holds SQLite WAL & port 8765."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        try:
            raw_pid = int(lock_path.read_text(encoding="utf-8").strip() or "0")
        except (OSError, ValueError):
            raw_pid = 0
        if _is_pid_running(raw_pid):
            logger.warning("Replacing existing live apex_runner PID %d holding %s", raw_pid, lock_path)
            _terminate_pid(raw_pid)
    lock_path.write_text(str(os.getpid()), encoding="utf-8")

    def _cleanup() -> None:
        try:
            if lock_path.exists() and lock_path.read_text(encoding="utf-8").strip() == str(os.getpid()):
                lock_path.unlink(missing_ok=True)
        except OSError as exc:
            logger.debug("Lockfile cleanup skipped: %s", exc)

    atexit.register(_cleanup)
    return lock_path


def format_web_cockpit_banner(cockpit_url: str, mode: str) -> str:
    """Return a compact 5-line startup banner for the Web Cockpit engine console."""
    return (
        "============================================================\n"
        f"  POCKETBOT APEX {DISPLAY_VERSION}  |  Mode: {mode.upper()}\n"
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
    line = f"[{ts}] APEX {DISPLAY_VERSION} | {state} | {scanned}/{total} syms | Eq: {eq:,.2f}"
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

    is_daemon = (bool(args.dashboard) or bool(args.tui_only)) and not bool(args.once)
    lock_path: Path | None = None
    if is_daemon and not os.environ.get("PYTEST_CURRENT_TEST"):
        lock_path = _acquire_single_instance_lock(Path("trader/apex_runner.lock"))

    orchestrator = ApexOrchestrator(
        mode=args.mode,
        config_path=args.config,
        is_live=bool(args.unlock_live),
        cli_args=sys.argv if argv is None else argv,
    )
    web_server: WebCockpitServer | None = None
    wrote_heartbeat = False
    try:
        if args.once or (not args.dashboard and not args.tui_only):
            orchestrator.run_cycle()
            return 0

        orchestrator.start_telegram_polling()

        if args.tui_only:
            with Live(
                console=console,
                refresh_per_second=4,
                screen=True,
                vertical_overflow="crop",
            ) as live:
                while True:
                    is_paused = (
                        orchestrator.paused
                        or orchestrator.telegram.scanning_paused
                        or orchestrator.telegram.scanning_halted
                    )
                    if not is_paused:
                        try:
                            orchestrator.run_cycle(live=live, render_console_tui=True)
                        except Exception as exc:
                            logger.exception("Cycle error recovered: %s", exc)
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
            is_paused = (
                orchestrator.paused
                or orchestrator.telegram.scanning_paused
                or orchestrator.telegram.scanning_halted
            )
            if not is_paused:
                try:
                    orchestrator.run_cycle(render_console_tui=False)
                except Exception as exc:
                    logger.exception("Cycle error recovered: %s", exc)
            now = time.monotonic()
            if now - last_heartbeat >= 5.0:
                sys.stdout.write("\r" + format_single_line_heartbeat(orchestrator))
                sys.stdout.flush()
                wrote_heartbeat = True
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
                    wrote_heartbeat = True
                    last_heartbeat = now_tick
    except KeyboardInterrupt:
        if wrote_heartbeat:
            sys.stdout.write("\n")
            wrote_heartbeat = False
        sys.stdout.write("PocketBot Apex stopped by user.\n")
        sys.stdout.flush()
        return 0
    finally:
        if wrote_heartbeat:
            try:
                sys.stdout.write("\n")
                sys.stdout.flush()
            except OSError:
                pass
        if web_server is not None:
            web_server.stop()
        orchestrator.shutdown()
        if lock_path is not None:
            try:
                if lock_path.exists() and lock_path.read_text(encoding="utf-8").strip() == str(os.getpid()):
                    lock_path.unlink(missing_ok=True)
            except OSError as exc:
                logger.debug("Final lockfile unlink skipped: %s", exc)


if __name__ == "__main__":
    raise SystemExit(main())
