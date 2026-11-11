"""Clean MetaTrader 5 IPC feed normalizing server UTC offsets, live ticks, M5/M15 candles, and contract specs."""

from __future__ import annotations

import logging
import math
import time
from typing import Sequence

import MetaTrader5 as mt5

from apex.feeds.models import Candle, ContractSpec, Tick, verify_live_unlock

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
        when rounded to the nearest 15 minutes (900 seconds). Only applies shifts >= 30 minutes
        (1800s) so quiet symbols with slightly older last ticks are not mis-shifted.
        """
        raw_diff = float(server_timestamp) - float(local_utc_timestamp)
        # Broker server offsets are multiples of 30 or 60 minutes (typically +7200s or +10800s)
        # Guard against stale weekend ticks (> 24 hours old)
        if abs(raw_diff) > 14 * 3600 or abs(raw_diff) < 1800:
            return 0
        return int(round(raw_diff / 1800.0) * 1800)

    def normalize_server_timestamp_to_utc(self, server_timestamp: float) -> float:
        """Convert an MT5 broker-timezone timestamp into a true UTC epoch timestamp."""
        return float(server_timestamp) - float(self._server_utc_offset_seconds)

    def _calibrate_server_utc_offset(self) -> int:
        """Detect the broker's server UTC offset from active symbols."""
        if not self._initialized:
            return 0

        now_utc = time.time()
        probe_symbols = ("EURUSD", "USDJPY", "GBPUSD", "XAUUSD", "BTCUSD")
        for sym in probe_symbols:
            resolved = self.resolve_symbol(sym)
            if not resolved:
                continue
            tick = mt5.symbol_info_tick(resolved)
            if tick is not None and getattr(tick, "time", 0) > 0:
                offset = self.compute_utc_offset_seconds(float(tick.time), now_utc)
                self._server_utc_offset_seconds = offset
                return offset
        return self._server_utc_offset_seconds

    def resolve_symbol(self, symbol: str) -> str | None:
        """Resolve a canonical symbol (e.g. ``EURUSD``) to the broker's exact symbol name (e.g. ``EURUSDm``, ``EURUSD.pro``)."""
        norm = symbol.upper().strip()
        if norm in self._symbol_map:
            return self._symbol_map[norm]

        if not self._initialized:
            return None

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
        )

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
        offset = self.compute_utc_offset_seconds(server_ts, now_utc)
        if offset != 0:
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
