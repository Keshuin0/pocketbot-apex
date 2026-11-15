"""Clean MetaTrader 5 IPC feed normalizing server UTC offsets, live ticks, M5/M15 candles, and contract specs."""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Sequence

import MetaTrader5 as mt5

from apex.feeds.alias_mapper import SmartAliasMapper
from apex.feeds.models import (
    Candle,
    ContractSpec,
    DiscoveredSymbol,
    Tick,
    verify_live_unlock,
)

logger = logging.getLogger(__name__)

MT5_TIMEFRAME_MAP: dict[str, int] = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}


class MT5Feed:
    """MetaTrader 5 local IPC feed with automatic broker server-to-UTC clock offset normalization.

    Enforces the non-negotiable DEMO account lock (`verify_live_unlock`) when connected to an
    MT5 terminal: if the terminal's active account has ``trade_mode == ACCOUNT_TRADE_MODE_REAL``,
    initialization immediately aborts unless both ``--unlock-live`` and
    ``APEX_UNLOCK_LIVE=I_ACCEPT_REAL_RISK`` are present.
    """

    def __init__(
        self,
        terminal_path: str | None = None,
        allow_live: bool = False,
        cli_args: Sequence[str] | None = None,
    ) -> None:
        self.terminal_path: str | None = terminal_path
        self.allow_live: bool = bool(allow_live)
        self._cli_args: Sequence[str] | None = cli_args
        self._initialized: bool = False
        self._server_utc_offset_seconds: int = 0
        self._symbol_map: dict[str, str] = {}
        self.alias_mapper: SmartAliasMapper = SmartAliasMapper()
        self._catalog: dict[str, DiscoveredSymbol] = {}

    @property
    def is_connected(self) -> bool:
        """Return True if the MT5 IPC connection is active."""
        override = getattr(self, "_connected", None)
        if override is not None:
            return bool(override)
        if not self._initialized:
            return False
        term_info = mt5.terminal_info()
        return term_info is not None

    @property
    def server_utc_offset_seconds(self) -> int:
        """Detected broker server time minus real UTC time in seconds (rounded to nearest 900s / 15min)."""
        return self._server_utc_offset_seconds

    def connect(self) -> bool:
        """Initialize the MT5 IPC session, enforce the DEMO safety invariant, and calibrate UTC offset."""
        if self.allow_live:
            verify_live_unlock(True, cli_args=self._cli_args)

        ok = mt5.initialize(self.terminal_path) if self.terminal_path else mt5.initialize()
        if not ok:
            err = mt5.last_error()
            logger.warning("MT5 IPC initialize() did not connect (terminal may be closed): %s", err)
            self._initialized = False
            return False

        acct = mt5.account_info()
        if acct is not None:
            is_real_account = int(getattr(acct, "trade_mode", 0)) == int(mt5.ACCOUNT_TRADE_MODE_REAL)
            if is_real_account:
                try:
                    verify_live_unlock(True, cli_args=self._cli_args)
                except Exception:
                    mt5.shutdown()
                    self._initialized = False
                    raise

        self._initialized = True
        self._calibrate_server_utc_offset()
        return True

    def disconnect(self) -> None:
        """Shut down the MT5 IPC connection."""
        if self._initialized:
            mt5.shutdown()
            self._initialized = False

    @staticmethod
    def compute_utc_offset_seconds(server_timestamp: float, local_utc_timestamp: float) -> int:
        """Compute broker server timezone offset from true UTC in seconds, snapped to 15-minute increments.

        MT5 ``tick.time`` and ``copy_rates_from_pos`` timestamps are stamped in the broker's
        trade server timezone (e.g., UTC+2 / UTC+3) rather than true UTC. Comparing a fresh
        server timestamp against ``time.time()`` (true UTC) yields the exact timezone shift
        when rounded to the nearest 15 minutes (900 seconds) within ``[-43200, +50400]``.
        """
        raw_diff = float(server_timestamp) - float(local_utc_timestamp)
        if abs(raw_diff) < 900:
            return 0
        snapped = int(round(raw_diff / 900.0) * 900)
        if snapped < -43200 or snapped > 50400:
            return 0
        return snapped

    def normalize_server_timestamp_to_utc(self, server_timestamp: float) -> float:
        """Convert an MT5 broker-timezone timestamp into a true UTC epoch timestamp."""
        return float(server_timestamp) - float(self._server_utc_offset_seconds)

    def _calibrate_server_utc_offset(self) -> int:
        """Detect the broker's server UTC offset via max(tick.time) consensus across active symbols."""
        if not self._initialized:
            return 0

        now_utc = time.time()
        probe_symbols = ("BTCUSD", "ETHUSD", "SOLUSD", "EURUSD", "USDJPY", "XAUUSD")
        tick_times: list[float] = []
        seen_names: set[str] = set()

        for sym in probe_symbols:
            resolved = self.resolve_symbol(sym)
            if not resolved or resolved in seen_names:
                continue
            seen_names.add(resolved)
            tick = mt5.symbol_info_tick(resolved)
            if tick is not None and float(getattr(tick, "time", 0) or 0) > 0.0:
                tick_times.append(float(tick.time))

        # Fallback consensus scan across mt5.symbols_get() so 24/7 crypto/active feeds override stale FX
        try:
            all_syms = mt5.symbols_get() or ()
            for s in all_syms:
                sname = str(getattr(s, "name", "") or "")
                if not sname or sname in seen_names:
                    continue
                seen_names.add(sname)
                tick = mt5.symbol_info_tick(sname)
                if tick is not None and float(getattr(tick, "time", 0) or 0) > 0.0:
                    tick_times.append(float(tick.time))
        except Exception as exc:
            logger.debug("Fallback mt5.symbols_get() clock scan skipped: %s", exc)

        if not tick_times:
            return self._server_utc_offset_seconds

        max_tick_time = max(tick_times)
        snapped_offset = self.compute_utc_offset_seconds(max_tick_time, now_utc)
        if -43200 <= snapped_offset <= 50400 and abs((max_tick_time - snapped_offset) - now_utc) <= 180.0:
            self._server_utc_offset_seconds = snapped_offset
            return snapped_offset

        logger.debug(
            "MT5 clock calibration skipped: max_tick=%.0f, snapped_offset=%d, residual=%.1fs",
            max_tick_time,
            snapped_offset,
            (max_tick_time - snapped_offset) - now_utc,
        )
        return self._server_utc_offset_seconds

    def resolve_symbol(self, symbol: str) -> str | None:
        """Resolve a canonical symbol (e.g. ``EURUSD``, ``NVDA``) to the broker's exact symbol name."""
        norm = symbol.upper().strip()
        if norm in self._symbol_map:
            return self._symbol_map[norm]

        if not self._initialized:
            return None

        # Check SmartAliasMapper first (e.g. NVDA -> NVIDIA, SPX500 -> US500, USOUSD -> XTIUSD)
        candidate = self.alias_mapper.resolve_to_broker(norm, broker="mt5")
        if candidate in self._symbol_map:
            return self._symbol_map[candidate]

        info = mt5.symbol_info(candidate)
        if info is not None:
            if not info.visible:
                mt5.symbol_select(candidate, True)
            self._symbol_map[norm] = candidate
            self._symbol_map[candidate] = candidate
            self.alias_mapper.register_symbol(candidate, canonical_name=norm, broker="mt5")
            return candidate

        # Try raw symbol
        info = mt5.symbol_info(norm)
        if info is not None:
            if not info.visible:
                mt5.symbol_select(norm, True)
            self._symbol_map[norm] = norm
            return norm

        # Search for suffixed broker symbols (e.g., EURUSD.a, EURUSDm, EURUSD_i)
        matches = mt5.symbols_get(f"{norm}*")
        if matches:
            best = matches[0].name
            mt5.symbol_select(best, True)
            self._symbol_map[norm] = best
            return best

        return None

    def discover_catalog(self, auto_enable_market_watch: bool = True) -> list[DiscoveredSymbol]:
        """Discover and categorize ALL symbols available in the MT5 broker catalog.

        Scans the full broker catalog via ``mt5.symbols_get()``, categorizes each into an
        exact ``AssetClass`` (FOREX, METALS, INDICES, ENERGY, CRYPTO, US_EQUITIES, COMMODITIES),
        and optionally auto-enables all tradable instruments in Market Watch.
        """
        if not self._initialized:
            if not self.connect():
                return list(self._catalog.values())

        symbols_raw = mt5.symbols_get()
        if not symbols_raw:
            return list(self._catalog.values())

        discovered: list[DiscoveredSymbol] = []
        for s in symbols_raw:
            sym_name = str(s.name)
            sym_path = str(getattr(s, "path", ""))
            sym_desc = str(getattr(s, "description", ""))
            trade_mode = int(getattr(s, "trade_mode", 0))
            is_vis = bool(getattr(s, "visible", False))

            asset_class = self.alias_mapper.classify_asset_class(
                symbol=sym_name,
                path=sym_path,
                description=sym_desc,
            )
            canonical_name = self.alias_mapper.canonicalize(sym_name)

            if auto_enable_market_watch and not is_vis:
                try:
                    mt5.symbol_select(sym_name, True)
                    is_vis = True
                except Exception as exc:
                    logger.debug("Failed to select symbol %s in Market Watch: %s", sym_name, exc)

            self.alias_mapper.register_symbol(
                broker_symbol=sym_name,
                canonical_name=canonical_name,
                broker="mt5",
            )
            self._symbol_map[canonical_name] = sym_name
            self._symbol_map[sym_name.upper()] = sym_name

            item = DiscoveredSymbol(
                name=sym_name,
                canonical_name=canonical_name,
                asset_class=asset_class,
                path=sym_path,
                digits=int(s.digits),
                point=float(s.point),
                trade_mode=trade_mode,
                is_visible=is_vis,
                source="mt5",
                description=sym_desc,
            )
            self._catalog[sym_name] = item
            discovered.append(item)

        logger.info(
            "MT5 catalog discovery complete: %d symbols found across %d asset classes.",
            len(discovered),
            len(set(d.asset_class for d in discovered)),
        )
        return discovered

    def get_discovered_symbols(self) -> list[DiscoveredSymbol]:
        """Return cached discovered catalog symbols."""
        return list(self._catalog.values())

    @staticmethod
    def compute_pip_size(digits: int, point: float) -> float:
        """Compute standard FX/CFD pip size from MT5 ``digits`` and ``point``."""
        if digits in (3, 5):
            return round(point * 10.0, 10)
        if digits == 2:
            return round(point * 10.0, 10)
        return float(point)

    def get_contract_spec(self, symbol: str) -> ContractSpec | None:
        """Fetch normalized ``ContractSpec`` for ``symbol`` from MT5."""
        resolved = self.resolve_symbol(symbol)
        if not resolved:
            return None

        info = mt5.symbol_info(resolved)
        if info is None:
            return None

        digits = int(info.digits)
        point = float(info.point) if info.point > 0 else float(10 ** (-digits))
        pip_size = self.compute_pip_size(digits, point)

        return ContractSpec(
            symbol=symbol.upper(),
            digits=digits,
            pip_size=pip_size,
            point_size=point,
            lot_size=float(info.trade_contract_size),
            min_volume_lots=float(info.volume_min),
            max_volume_lots=float(info.volume_max),
            step_volume_lots=float(info.volume_step),
            source="mt5",
            trade_tick_value=float(getattr(info, "trade_tick_value", 0.0) or 0.0),
            trade_tick_size=float(getattr(info, "trade_tick_size", 0.0) or 0.0),
        )

    def is_symbol_trade_mode_full(self, symbol: str) -> bool:
        """Return True strictly if MT5 symbol_info.trade_mode == SYMBOL_TRADE_MODE_FULL (4)."""
        if not self._initialized:
            return False
        resolved = self.resolve_symbol(symbol)
        if not resolved:
            return False
        info = mt5.symbol_info(resolved)
        if info is None:
            return False
        full_mode = int(getattr(mt5, "SYMBOL_TRADE_MODE_FULL", 4))
        return int(getattr(info, "trade_mode", 0)) == full_mode

    def get_tick(self, symbol: str) -> Tick | None:
        """Fetch the latest live tick for ``symbol`` with UTC-normalized timestamp."""
        resolved = self.resolve_symbol(symbol)
        if not resolved:
            return None

        raw_tick = mt5.symbol_info_tick(resolved)
        if raw_tick is None:
            return None

        bid = float(raw_tick.bid)
        ask = float(raw_tick.ask)
        if bid <= 0.0 or ask <= 0.0 or math.isnan(bid) or math.isnan(ask):
            return None

        raw_ms = getattr(raw_tick, "time_msc", 0)
        if raw_ms and raw_ms > 0:
            server_ts = float(raw_ms) / 1000.0
        else:
            server_ts = float(raw_tick.time)

        now_utc = time.time()
        # Only update broker server UTC offset when not yet calibrated or on an active FX/crypto tick
        if self._server_utc_offset_seconds == 0 or abs(server_ts - now_utc - self._server_utc_offset_seconds) <= 180.0:
            offset = self.compute_utc_offset_seconds(server_ts, now_utc)
            if offset != 0 and abs(offset) <= 43200:
                self._server_utc_offset_seconds = offset
        utc_ts = server_ts - float(self._server_utc_offset_seconds)

        vol = float(getattr(raw_tick, "volume_real", 0.0) or getattr(raw_tick, "volume", 0.0))
        return Tick(
            symbol=symbol.upper(),
            bid=bid,
            ask=ask,
            timestamp_utc=utc_ts,
            source="mt5",
            volume=vol,
        )

    def get_candles(
        self,
        symbol: str,
        timeframe: str = "M5",
        count: int = 200,
    ) -> list[Candle]:
        """Fetch historical M5/M15 candles from MT5 with UTC-normalized timestamps."""
        norm_tf = timeframe.upper()
        if norm_tf not in MT5_TIMEFRAME_MAP:
            raise ValueError(f"Unsupported MT5 timeframe '{timeframe}'. Supported: {list(MT5_TIMEFRAME_MAP)}")

        resolved = self.resolve_symbol(symbol)
        if not resolved:
            return []

        rates = mt5.copy_rates_from_pos(resolved, MT5_TIMEFRAME_MAP[norm_tf], 0, int(count))
        if rates is None or len(rates) == 0:
            return []

        candles: list[Candle] = []
        for row in rates:
            utc_ts = self.normalize_server_timestamp_to_utc(float(row["time"]))
            vol = float(row["real_volume"]) if row["real_volume"] > 0 else float(row["tick_volume"])
            candles.append(
                Candle(
                    symbol=symbol.upper(),
                    timeframe=norm_tf,
                    timestamp_utc=utc_ts,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=vol,
                    source="mt5",
                )
            )
        candles.sort(key=lambda c: c.timestamp_utc)
        return candles

    def get_account_info(self) -> dict[str, Any]:
        """Fetch MT5 account status, balance, equity, and currency."""
        if not self._initialized:
            return {}
        info = mt5.account_info()
        if info is None:
            return {}
        return {
            "login": int(getattr(info, "login", 0)),
            "server": str(getattr(info, "server", "")),
            "currency": str(getattr(info, "currency", "USD")),
            "balance": float(getattr(info, "balance", 0.0)),
            "equity": float(getattr(info, "equity", 0.0)),
            "margin": float(getattr(info, "margin", 0.0)),
            "margin_free": float(getattr(info, "margin_free", 0.0)),
            "trade_allowed": bool(getattr(info, "trade_allowed", False)),
            "trade_mode": int(getattr(info, "trade_mode", 0)),
        }

    def get_account_currency(self) -> str:
        """Return the MT5 account base currency (e.g., 'CAD', 'USD')."""
        info = self.get_account_info()
        return info.get("currency", "USD")

    def get_open_positions(self) -> list[dict[str, Any]]:
        """Return list of open MT5 positions."""
        if not self._initialized:
            return []
        positions = mt5.positions_get()
        if not positions:
            return []
        res: list[dict[str, Any]] = []
        for p in positions:
            res.append(
                {
                    "ticket": int(getattr(p, "ticket", 0)),
                    "symbol": str(getattr(p, "symbol", "")),
                    "type": int(getattr(p, "type", 0)),
                    "volume": float(getattr(p, "volume", 0.0)),
                    "price_open": float(getattr(p, "price_open", 0.0)),
                    "sl": float(getattr(p, "sl", 0.0)),
                    "tp": float(getattr(p, "tp", 0.0)),
                    "price_current": float(getattr(p, "price_current", 0.0)),
                    "profit": float(getattr(p, "profit", 0.0)),
                    "comment": str(getattr(p, "comment", "")),
                }
            )
        return res

    def compute_compliant_lot_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        risk_amount: float,
    ) -> float:
        """Compute broker-compliant lot size based on stop loss distance, tick value, and volume constraints.

        Uses the exact broker tick value (reported in account currency) and snaps to
        volume_min, volume_max, and volume_step.
        """
        resolved = self.resolve_symbol(symbol)
        if not resolved or not self._initialized:
            return 0.01

        info = mt5.symbol_info(resolved)
        if info is None:
            return 0.01

        vol_min = float(getattr(info, "volume_min", 0.01) or 0.01)
        vol_max = float(getattr(info, "volume_max", 100.0) or 100.0)
        vol_step = float(getattr(info, "volume_step", 0.01) or 0.01)

        sl_dist = abs(float(entry_price) - float(stop_loss))
        if sl_dist <= 1e-12:
            return vol_min

        tick_size = float(getattr(info, "trade_tick_size", 0.0) or getattr(info, "point", 0.0) or 1e-5)
        tick_val = float(
            getattr(info, "trade_tick_value_loss", 0.0)
            or getattr(info, "trade_tick_value", 0.0)
            or 0.0
        )

        # In MT5, trade_tick_value is expressed in account currency per 1.0 lot per tick_size
        if tick_size > 0.0 and tick_val > 0.0:
            loss_per_lot = (sl_dist / tick_size) * tick_val
        else:
            contract_sz = float(getattr(info, "trade_contract_size", 100_000.0) or 100_000.0)
            loss_per_lot = sl_dist * contract_sz

        if loss_per_lot <= 1e-9:
            return vol_min

        lots_raw = float(risk_amount) / loss_per_lot

        if vol_step > 0.0:
            steps = round(lots_raw / vol_step)
            lots_snapped = steps * vol_step
        else:
            lots_snapped = lots_raw

        lots_final = max(vol_min, min(vol_max, lots_snapped))
        decimals = max(0, -int(math.floor(math.log10(vol_step)))) if vol_step < 1.0 else 0
        return round(lots_final, decimals)

