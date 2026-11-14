"""Apex Hydra Active-Active Dual-Venue Smart Order Router (SOR), Synthetic NBBO Mesh & Cross-Feed Guard."""

from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from dotenv import dotenv_values

from apex.feeds.ctrader_feed import CTraderOpenApiFeed
from apex.feeds.hydra_nbbo import (
    HydraNBBOEngine,
    MeshOperatingMode,
    ResolvedMeshTopology,
    SingleBrokerSpikeRejector,
    SyntheticNBBOResult,
    VenueFeeProfile,
)
from apex.feeds.models import (
    AssetClass,
    Candle,
    ContractSpec,
    DOMAnalytics,
    DiscoveredSymbol,
    Side,
    Tick,
    TickDiscrepancyAlert,
    VWAPExecutionEstimate,
    verify_live_unlock,
)
from apex.feeds.mt5_feed import MT5Feed

logger = logging.getLogger(__name__)


class CrossFeedTickGuard:
    """Statistical Cross-Feed Tick Discrepancy Guard (`dual_feed_guard`).

    Maintains a rolling distribution of price diffs ``|mid_primary - mid_standby|`` per symbol
    and flags single-broker bad ticks when the observed divergence exceeds ``z_score_threshold``
    standard deviations (> 3.0z by default) while both feeds are online.
    """

    def __init__(
        self,
        enabled: bool = True,
        z_score_threshold: float = 3.0,
        rolling_window_ticks: int = 100,
        min_samples: int = 10,
        max_tick_age_seconds: float = 5.0,
    ) -> None:
        if z_score_threshold <= 0.0:
            raise ValueError(f"z_score_threshold must be > 0, got {z_score_threshold}")
        if rolling_window_ticks < 5:
            raise ValueError(f"rolling_window_ticks must be >= 5, got {rolling_window_ticks}")

        self.enabled: bool = bool(enabled)
        self.z_score_threshold: float = float(z_score_threshold)
        self.rolling_window_ticks: int = int(rolling_window_ticks)
        self.min_samples: int = int(min_samples)
        self.max_tick_age_seconds: float = float(max_tick_age_seconds)

        self._diff_history: dict[str, deque[float]] = {}

    def reset(self, symbol: str | None = None) -> None:
        """Clear rolling divergence history for ``symbol`` or all symbols."""
        if symbol is None:
            self._diff_history.clear()
        else:
            self._diff_history.pop(symbol.upper(), None)

    def evaluate(
        self,
        primary_tick: Tick,
        standby_tick: Tick,
    ) -> TickDiscrepancyAlert:
        """Compare concurrent ticks from cTrader and MT5 feeds."""
        sym = primary_tick.symbol.upper()
        abs_diff = abs(primary_tick.mid - standby_tick.mid)
        ts_utc = max(primary_tick.timestamp_utc, standby_tick.timestamp_utc)

        if not self.enabled:
            return TickDiscrepancyAlert(
                symbol=sym,
                timestamp_utc=ts_utc,
                primary_mid=primary_tick.mid,
                standby_mid=standby_tick.mid,
                abs_diff=abs_diff,
                rolling_mean_diff=0.0,
                rolling_std_diff=0.0,
                z_score=0.0,
                threshold_z=self.z_score_threshold,
                is_bad_tick=False,
                reason="dual_feed_guard_disabled",
            )

        tick_skew = abs(primary_tick.timestamp_utc - standby_tick.timestamp_utc)
        if tick_skew > self.max_tick_age_seconds:
            return TickDiscrepancyAlert(
                symbol=sym,
                timestamp_utc=ts_utc,
                primary_mid=primary_tick.mid,
                standby_mid=standby_tick.mid,
                abs_diff=abs_diff,
                rolling_mean_diff=0.0,
                rolling_std_diff=0.0,
                z_score=0.0,
                threshold_z=self.z_score_threshold,
                is_bad_tick=False,
                reason=f"stale_comparison_skew_{tick_skew:.2f}s",
            )

        history = self._diff_history.setdefault(
            sym, deque(maxlen=self.rolling_window_ticks)
        )

        if len(history) < self.min_samples:
            history.append(abs_diff)
            arr = np.asarray(history, dtype=np.float64)
            mean_val = float(np.mean(arr))
            std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            return TickDiscrepancyAlert(
                symbol=sym,
                timestamp_utc=ts_utc,
                primary_mid=primary_tick.mid,
                standby_mid=standby_tick.mid,
                abs_diff=abs_diff,
                rolling_mean_diff=mean_val,
                rolling_std_diff=std_val,
                z_score=0.0,
                threshold_z=self.z_score_threshold,
                is_bad_tick=False,
                reason=f"warming_up_{len(history)}/{self.min_samples}",
            )

        arr = np.asarray(history, dtype=np.float64)
        mean_diff = float(np.mean(arr))
        std_diff = float(np.std(arr, ddof=1))
        safe_std = max(std_diff, 1e-8)
        z_score = float((abs_diff - mean_diff) / safe_std)
        is_bad = z_score > self.z_score_threshold

        if not is_bad:
            history.append(abs_diff)
            reason = "ok"
        else:
            reason = (
                f"CROSS_FEED_DIVERGENCE: |{primary_tick.mid:.5f} - {standby_tick.mid:.5f}| "
                f"= {abs_diff:.5f} ({z_score:.2f}z > {self.z_score_threshold:.2f}z)"
            )
            logger.warning("[%s] %s", sym, reason)

        return TickDiscrepancyAlert(
            symbol=sym,
            timestamp_utc=ts_utc,
            primary_mid=primary_tick.mid,
            standby_mid=standby_tick.mid,
            abs_diff=abs_diff,
            rolling_mean_diff=mean_diff,
            rolling_std_diff=std_diff,
            z_score=z_score,
            threshold_z=self.z_score_threshold,
            is_bad_tick=is_bad,
            reason=reason,
        )


class MarketFeedRouter:
    """Apex Hydra Active-Active Dual-Venue Market Feed Router & Synthetic NBBO Mesh.

    Self-detects all 4 topologies (`AUTO_HYDRA_MESH` -> `ACTIVE_ACTIVE_SOR`, `MT5_SOLO`,
    `CTRADER_SOLO`, plus `CTRADER_EYES_MT5_HANDS`, `MT5_EYES_CTRADER_HANDS`, and `SPLIT_EXECUTION`)
    with automatic hot-reload of `.env`, `config/apex_config.json`, and live broker heartbeats.
    """

    def __init__(
        self,
        config_path: str | Path = "config/apex_config.json",
        env_path: str | Path = ".env",
        ctrader_feed: CTraderOpenApiFeed | None = None,
        mt5_feed: MT5Feed | None = None,
        is_live: bool = False,
        cli_args: Sequence[str] | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.env_path = Path(env_path)
        self._config_mtime: float = -1.0
        self._env_mtime: float = -1.0
        self.config: dict[str, Any] = self._load_config(self.config_path)

        cfg_mode = str(self.config.get("account_mode", "DEMO")).upper()
        effective_live = bool(is_live or cfg_mode == "LIVE")
        verify_live_unlock(effective_live, cli_args=cli_args)
        self.is_live: bool = effective_live

        feeds_cfg = self.config.get("feeds", {})
        filters_cfg = self.config.get("filters", {})
        guard_cfg = filters_cfg.get("dual_feed_guard", {})

        self.mesh_mode: str = str(feeds_cfg.get("mesh_mode", MeshOperatingMode.AUTO_HYDRA_MESH.value)).upper()

        self.ctrader: CTraderOpenApiFeed = ctrader_feed or CTraderOpenApiFeed(
            is_live=self.is_live,
            top_n_dom_levels=int(feeds_cfg.get("dom_top_n_levels", 5)),
            wall_multiplier=float(feeds_cfg.get("liquidity_wall_multiplier", 3.0)),
            cli_args=cli_args,
        )
        self.mt5: MT5Feed = mt5_feed or MT5Feed(
            allow_live=self.is_live,
            cli_args=cli_args,
        )
        self.tick_guard: CrossFeedTickGuard = CrossFeedTickGuard(
            enabled=bool(guard_cfg.get("enabled", True)),
            z_score_threshold=float(guard_cfg.get("z_score_threshold", 3.0)),
            rolling_window_ticks=int(guard_cfg.get("rolling_window_ticks", 100)),
            min_samples=int(guard_cfg.get("min_samples", 10)),
            max_tick_age_seconds=float(guard_cfg.get("max_tick_age_seconds", 5.0)),
        )

        self.nbbo_engine = HydraNBBOEngine(
            ctrader_profile=VenueFeeProfile(
                venue="ctrader",
                commission_per_lot_round_turn_usd=float(feeds_cfg.get("ctrader_commission_per_lot_usd", 4.50)),
                pip_value_per_lot_usd=10.0,
                ping_ms=float(feeds_cfg.get("ctrader_ping_ms", 12.0)),
                latency_penalty_pips_per_100ms=float(feeds_cfg.get("latency_penalty_pips_per_100ms", 0.05)),
                margin_and_autotrading_ok=True,
            ),
            mt5_profile=VenueFeeProfile(
                venue="mt5",
                commission_per_lot_round_turn_usd=float(feeds_cfg.get("mt5_commission_per_lot_usd", 4.50)),
                pip_value_per_lot_usd=10.0,
                ping_ms=float(feeds_cfg.get("mt5_ping_ms", 18.0)),
                latency_penalty_pips_per_100ms=float(feeds_cfg.get("latency_penalty_pips_per_100ms", 0.05)),
                margin_and_autotrading_ok=True,
            ),
            spike_rejector=SingleBrokerSpikeRejector(
                max_basis_multiple=float(feeds_cfg.get("spike_max_basis_multiple", 2.5)),
            ),
        )

        self._active_source: str = "uninitialized"
        self.hot_reload_mesh_state(force=True)

    @staticmethod
    def _load_config(path: Path) -> dict[str, Any]:
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                data: dict[str, Any] = json.load(fh)
                return data
        return {
            "account_mode": "DEMO",
            "feeds": {
                "mesh_mode": "AUTO_HYDRA_MESH",
                "primary": "ctrader",
                "standby": "mt5",
                "dom_top_n_levels": 5,
                "liquidity_wall_multiplier": 3.0,
            },
            "filters": {
                "regime": {"enabled": True},
                "dom_obi": {"enabled": True},
                "correlation_cap": {"enabled": True},
                "ai_veto": {"enabled": True},
                "dual_feed_guard": {"enabled": True, "z_score_threshold": 3.0, "rolling_window_ticks": 100, "min_samples": 10},
            },
        }

    def hot_reload_mesh_state(self, force: bool = False) -> ResolvedMeshTopology:
        """Hot-reload `.env` credentials, `config/apex_config.json`, and live MT5/cTrader heartbeats."""
        try:
            cfg_mtime = self.config_path.stat().st_mtime if self.config_path.exists() else 0.0
        except OSError:
            cfg_mtime = 0.0

        try:
            env_mtime = self.env_path.stat().st_mtime if self.env_path.exists() else 0.0
        except OSError:
            env_mtime = 0.0

        if force or cfg_mtime != self._config_mtime:
            self._config_mtime = cfg_mtime
            self.config = self._load_config(self.config_path)
            feeds_cfg = self.config.get("feeds", {})
            guard_cfg = self.config.get("filters", {}).get("dual_feed_guard", {})
            raw_mesh = guard_cfg.get("mesh_mode") or feeds_cfg.get("mesh_mode") or MeshOperatingMode.AUTO_HYDRA_MESH.value
            self.mesh_mode = str(raw_mesh).upper()
            self.nbbo_engine.ctrader_profile = VenueFeeProfile(
                venue="ctrader",
                commission_per_lot_round_turn_usd=float(feeds_cfg.get("ctrader_commission_per_lot_usd", 4.50)),
                pip_value_per_lot_usd=10.0,
                ping_ms=float(feeds_cfg.get("ctrader_ping_ms", 12.0)),
                latency_penalty_pips_per_100ms=float(feeds_cfg.get("latency_penalty_pips_per_100ms", 0.05)),
                margin_and_autotrading_ok=True,
            )
            self.nbbo_engine.mt5_profile = VenueFeeProfile(
                venue="mt5",
                commission_per_lot_round_turn_usd=float(feeds_cfg.get("mt5_commission_per_lot_usd", 4.50)),
                pip_value_per_lot_usd=10.0,
                ping_ms=float(feeds_cfg.get("mt5_ping_ms", 18.0)),
                latency_penalty_pips_per_100ms=float(feeds_cfg.get("latency_penalty_pips_per_100ms", 0.05)),
                margin_and_autotrading_ok=True,
            )
            self.nbbo_engine.spike_rejector.max_basis_multiple = float(
                feeds_cfg.get("spike_max_basis_multiple", 2.5)
            )

        if force or env_mtime != self._env_mtime:
            self._env_mtime = env_mtime
            if self.env_path.exists():
                vals = dotenv_values(self.env_path)
                if vals.get("CTRADER_CLIENT_ID"):
                    self.ctrader.client_id = str(vals["CTRADER_CLIENT_ID"]).strip()
                if vals.get("CTRADER_CLIENT_SECRET"):
                    self.ctrader.client_secret = str(vals["CTRADER_CLIENT_SECRET"]).strip()
                if vals.get("CTRADER_ACCESS_TOKEN"):
                    self.ctrader.access_token = str(vals["CTRADER_ACCESS_TOKEN"]).strip()
                if vals.get("CTRADER_REFRESH_TOKEN"):
                    self.ctrader.refresh_token = str(vals["CTRADER_REFRESH_TOKEN"]).strip()
                raw_acct = vals.get("CTRADER_ACCOUNT_ID", "")
                if str(raw_acct).strip().isdigit():
                    self.ctrader.account_id = int(str(raw_acct).strip())

        return HydraNBBOEngine.resolve_topology(
            configured_mode=self.mesh_mode,
            ctrader_online=self.ctrader.is_connected,
            mt5_online=self.mt5.is_connected,
        )

    @property
    def hydra_nbbo(self) -> HydraNBBOEngine:
        """Alias for `nbbo_engine`."""
        return self.nbbo_engine

    @property
    def configured_mesh_mode(self) -> MeshOperatingMode:
        """Return the hot-reloaded `MeshOperatingMode`."""
        self.hot_reload_mesh_state()
        try:
            return MeshOperatingMode(self.mesh_mode)
        except ValueError:
            return MeshOperatingMode.AUTO_HYDRA_MESH

    @property
    def resolved_topology(self) -> ResolvedMeshTopology:
        """Return the live resolved 4-Topology Mesh state based on current venue heartbeats."""
        self.hot_reload_mesh_state()
        return HydraNBBOEngine.resolve_topology(
            configured_mode=self.mesh_mode,
            ctrader_online=self.ctrader.is_connected,
            mt5_online=self.mt5.is_connected,
        )

    @property
    def dashboard_mesh_badge(self) -> str:
        """Format live Mesh Topology badge for TUI header, e.g., `Mesh: HYDRA_AUTO [MT5 Active | cTrader Ready-On-Key]`."""
        topo = self.resolved_topology
        ct_on = self.ctrader.is_connected
        mt5_on = self.mt5.is_connected

        if topo == ResolvedMeshTopology.ACTIVE_ACTIVE_SOR:
            return "Mesh: ACTIVE_ACTIVE_SOR [cTrader + MT5 NBBO Fusion]"
        if topo == ResolvedMeshTopology.CTRADER_EYES_MT5_HANDS:
            return "Mesh: CTRADER_EYES_MT5_HANDS [cTrader L2 DOM -> MT5 Execution]"
        if topo == ResolvedMeshTopology.MT5_EYES_CTRADER_HANDS:
            return "Mesh: MT5_EYES_CTRADER_HANDS [MT5 Price -> cTrader Execution]"
        if topo == ResolvedMeshTopology.SPLIT_EXECUTION:
            return "Mesh: SPLIT_EXECUTION [50/50 Parallel cTrader + MT5]"
        if topo == ResolvedMeshTopology.MT5_SOLO:
            ct_state = "cTrader Connecting" if self.ctrader.has_credentials else "cTrader Ready-On-Key"
            return f"Mesh: HYDRA_AUTO [MT5 Active | {ct_state}]"
        if topo == ResolvedMeshTopology.CTRADER_SOLO:
            return "Mesh: HYDRA_AUTO [cTrader Solo Active | MT5 Standby]"
        return f"Mesh: {self.mesh_mode} [mt5={mt5_on}, ctrader={ct_on}]"

    @property
    def active_source(self) -> str:
        """Return currently active routing source (`'hydra_sor'`, `'ctrader'`, `'mt5'`, or `'offline'`)."""
        topo = self.resolved_topology
        if topo in (
            ResolvedMeshTopology.ACTIVE_ACTIVE_SOR,
            ResolvedMeshTopology.SPLIT_EXECUTION,
        ):
            return "hydra_sor"
        if topo == ResolvedMeshTopology.CTRADER_EYES_MT5_HANDS:
            return "mt5"
        if topo == ResolvedMeshTopology.MT5_EYES_CTRADER_HANDS:
            return "ctrader"
        if self.ctrader.is_connected:
            return "ctrader"
        if self.mt5.is_connected:
            return "mt5"
        return self._active_source

    def connect(self) -> str:
        """Connect feeds: bring up MT5 and cTrader if credentials exist, then resolve Hydra topology."""
        verify_live_unlock(self.is_live)
        self.hot_reload_mesh_state()

        mt5_ok = self.mt5.connect()

        if self.ctrader.has_credentials and not self.ctrader.is_connected:
            try:
                self.ctrader.start()
                logger.info("MarketFeedRouter: cTrader Open API started (MT5 online=%s).", mt5_ok)
            except Exception as exc:
                logger.warning("cTrader start failed (%s); operating on MT5 solo.", exc)
        elif not self.ctrader.has_credentials:
            logger.info("cTrader credentials not yet in .env; running MT5_SOLO until key is added.")

        if self.ctrader.is_connected and mt5_ok:
            self._active_source = "hydra_sor"
        elif self.ctrader.is_connected:
            self._active_source = "ctrader"
        elif mt5_ok:
            self._active_source = "mt5"
        else:
            self._active_source = "offline"
        return self.active_source

    def discover_catalog(self, auto_enable_market_watch: bool = True) -> list[DiscoveredSymbol]:
        """Discover the combined universe catalog across active broker venues."""
        combined: dict[str, DiscoveredSymbol] = {}
        if self.mt5.is_connected or self._active_source in ("mt5", "hydra_sor"):
            for s in self.mt5.discover_catalog(auto_enable_market_watch=auto_enable_market_watch):
                combined[s.name] = s

        if self.ctrader.is_connected or self._active_source in ("ctrader", "hydra_sor"):
            for s in self.ctrader.discover_catalog():
                if s.name not in combined:
                    combined[s.name] = s

        return list(combined.values())

    def shutdown(self) -> None:
        """Cleanly stop both cTrader and MT5 feeds."""
        self.ctrader.stop()
        self.mt5.disconnect()
        self._active_source = "offline"

    def evaluate_nbbo(
        self,
        symbol: str,
        one_r_price_distance: float = 0.0010,
        target_volume_units: float = 100_000.0,
        side: Side = Side.BUY,
        ctrader_slippage_price: float | None = None,
        mt5_slippage_price: float | None = None,
    ) -> SyntheticNBBOResult:
        """Evaluate Synthetic Consolidated NBBO, All-In Effective Cost per venue, and Cross-Pollinated DOM."""
        self.hot_reload_mesh_state()
        norm_sym = symbol.upper()
        spec = self.get_contract_spec(norm_sym)
        pip_size = spec.pip_size if spec else (0.01 if "JPY" in norm_sym or "XAU" in norm_sym else 0.0001)
        safe_pip = max(pip_size, 1e-9)

        ct_tick = self.ctrader.get_latest_tick(norm_sym) if self.ctrader.is_connected else None
        mt5_tick = self.mt5.get_tick(norm_sym) if self.mt5.is_connected else None
        ct_dom = self.get_dom_analytics(norm_sym)

        ct_slip_pips = (float(ctrader_slippage_price) / safe_pip) if ctrader_slippage_price is not None else 0.10
        mt5_slip_pips = (float(mt5_slippage_price) / safe_pip) if mt5_slippage_price is not None else 0.15
        if norm_sym in self.ctrader._dom_engines:
            engine = self.ctrader._dom_engines[norm_sym]
            snap = engine.snapshot()
            if (side == Side.BUY and snap.asks) or (side == Side.SELL and snap.bids):
                try:
                    vwap_est = engine.compute_vwap_fill(side=side, target_volume_units=target_volume_units)
                    ct_slip_pips = max(0.0, vwap_est.slippage_pips)
                except Exception as exc:
                    logger.debug("VWAP slippage estimate skipped for %s: %s", norm_sym, exc)

        return self.nbbo_engine.evaluate_symbol(
            symbol=norm_sym,
            configured_mode=self.mesh_mode,
            ctrader_tick=ct_tick,
            mt5_tick=mt5_tick,
            pip_size=pip_size,
            one_r_price_distance=one_r_price_distance,
            ctrader_vwap_slippage_pips=ct_slip_pips,
            mt5_vwap_slippage_pips=mt5_slip_pips,
            ctrader_dom=ct_dom,
        )

    def get_tick(self, symbol: str) -> tuple[Tick | None, TickDiscrepancyAlert | None]:
        """Return the latest validated ``Tick`` (or consolidated NBBO tick when both feeds are online)."""
        self.hot_reload_mesh_state()
        norm_sym = symbol.upper()
        ct_tick = self.ctrader.get_latest_tick(norm_sym) if self.ctrader.is_connected else None
        mt5_tick = self.mt5.get_tick(norm_sym) if self.mt5.is_connected else None

        if ct_tick is not None and mt5_tick is not None:
            alert = self.tick_guard.evaluate(ct_tick, mt5_tick)
            if alert.is_bad_tick:
                return None, alert
            # Return Synthetic Consolidated NBBO tick (best bid, best ask)
            best_bid = max(ct_tick.bid, mt5_tick.bid)
            best_ask = min(ct_tick.ask, mt5_tick.ask)
            if best_ask < best_bid:
                # Crossed market across venues: use winning venue's top-of-book
                nbbo = self.evaluate_nbbo(norm_sym)
                chosen = ct_tick if nbbo.winning_venue == "ctrader" else mt5_tick
                return chosen, alert
            syn_tick = Tick(
                symbol=norm_sym,
                bid=best_bid,
                ask=best_ask,
                timestamp_utc=max(ct_tick.timestamp_utc, mt5_tick.timestamp_utc),
                source="hydra_nbbo",
                volume=ct_tick.volume + mt5_tick.volume,
            )
            return syn_tick, alert

        if ct_tick is not None:
            return ct_tick, None

        if mt5_tick is not None:
            return mt5_tick, None

        return None, None

    def get_candles(self, symbol: str, timeframe: str = "M5", count: int = 200) -> list[Candle]:
        """Fetch historical M5/M15 candles from active feed."""
        if self.mt5.is_connected:
            return self.mt5.get_candles(symbol=symbol, timeframe=timeframe, count=count)
        return []

    def get_contract_spec(self, symbol: str) -> ContractSpec | None:
        """Fetch instrument contract specification from active feed."""
        norm = symbol.upper()
        if norm in self.ctrader._contract_specs:
            return self.ctrader._contract_specs[norm]
        if self.mt5.is_connected:
            return self.mt5.get_contract_spec(norm)
        return None

    def get_dom_analytics(self, symbol: str) -> DOMAnalytics | None:
        """Return Cross-Pollinated Level 2 DOM analytics for ``symbol`` whenever cTrader DOM has quotes."""
        norm = symbol.upper()
        if norm not in self.ctrader._dom_engines:
            canon = self.ctrader.alias_mapper.canonicalize(norm)
            if canon in self.ctrader._dom_engines:
                norm = canon
            else:
                return None
        snap = self.ctrader._dom_engines[norm].snapshot()
        if not snap.bids and not snap.asks:
            return None
        return self.ctrader._dom_engines[norm].compute_analytics()

    def estimate_vwap_fill(
        self,
        symbol: str,
        side: Side,
        target_volume_units: float,
    ) -> VWAPExecutionEstimate:
        """Compute expected Level 2 DOM VWAP fill price and slippage for ``symbol``."""
        norm = symbol.upper()
        if norm not in self.ctrader._dom_engines:
            canon = self.ctrader.alias_mapper.canonicalize(norm)
            if canon in self.ctrader._dom_engines:
                norm = canon
        engine = self.ctrader.get_or_create_dom_engine(norm)
        return engine.compute_vwap_fill(side=side, target_volume_units=target_volume_units)
