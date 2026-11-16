"""Official Spotware cTrader Open API binary Protobuf over TLS TCP feed & Level 2 DOM Engine."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from typing import Callable, Iterable, Sequence

import numpy as np
from ctrader_open_api import Auth, Client, EndPoints, Protobuf, TcpProtocol
from ctrader_open_api.messages import (
    OpenApiMessages_pb2 as oa_msg,
    OpenApiModelMessages_pb2 as oa_model,
)
from dotenv import load_dotenv

from apex.feeds.alias_mapper import SmartAliasMapper
from apex.feeds.models import (
    AssetClass,
    Candle,
    ContractSpec,
    DOMAnalytics,
    DepthLevel,
    DiscoveredSymbol,
    LiquidityWall,
    OrderBookSnapshot,
    Side,
    Tick,
    VWAPExecutionEstimate,
    verify_live_unlock,
)

logger = logging.getLogger(__name__)

# Spotware Open API encodes raw integer prices scaled by 1e-5 (100,000 units = 1.0)
# and volumes in cents of a unit (100 = 1 unit; 10,000,000 = 1 standard lot of 100,000 units).
CTRADER_PRICE_SCALE: float = 100_000.0
CTRADER_VOLUME_CENTS_PER_UNIT: float = 100.0


class OrderBookEngine:
    """High-speed deterministic Level 2 Depth of Market (DOM) engine for a single instrument.

    Maintains an incremental quote-ID-addressed order book updated via ``ProtoOADepthEvent``
    (or direct ``DepthLevel`` updates) and computes:
      1. Order-Book Imbalance: ``OBI = (bid_vol - ask_vol) / (bid_vol + ask_vol)`` across top N levels.
      2. Liquidity Wall Detection: price levels where volume >= ``wall_multiplier * median(book_volumes)``.
      3. True VWAP DOM expected fill price + slippage (in price & pips) for any target volume.
    """

    def __init__(
        self,
        symbol: str,
        pip_size: float = 0.0001,
        digits: int = 5,
        top_n_levels: int = 5,
        wall_multiplier: float = 3.0,
    ) -> None:
        if pip_size <= 0.0:
            raise ValueError(f"pip_size must be positive, got {pip_size}")
        if top_n_levels < 1:
            raise ValueError(f"top_n_levels must be >= 1, got {top_n_levels}")
        if wall_multiplier <= 1.0:
            raise ValueError(f"wall_multiplier must be > 1.0, got {wall_multiplier}")

        self.symbol: str = symbol
        self.pip_size: float = float(pip_size)
        self.digits: int = int(digits)
        self.top_n_levels: int = int(top_n_levels)
        self.wall_multiplier: float = float(wall_multiplier)

        self._quotes: dict[int, DepthLevel] = {}
        self._last_update_utc: float = 0.0
        self._lock = threading.RLock()
        self._prev_bids: list[tuple[float, float]] = []
        self._prev_asks: list[tuple[float, float]] = []
        self._prev_mid: float = 0.0
        self._ofi_history: deque[float] = deque(maxlen=25)
        self._vpin_buckets: deque[float] = deque(maxlen=50)
        self._vpin_bucket_buy: float = 0.0
        self._vpin_bucket_sell: float = 0.0
        self._vpin_bucket_capacity: float = 0.0

    def clear(self) -> None:
        """Reset all active depth quotes."""
        with self._lock:
            self._quotes.clear()
            self._last_update_utc = time.time()
            self._prev_bids.clear()
            self._prev_asks.clear()
            self._prev_mid = 0.0
            self._ofi_history.clear()
            self._vpin_buckets.clear()
            self._vpin_bucket_buy = 0.0
            self._vpin_bucket_sell = 0.0
            self._vpin_bucket_capacity = 0.0

    def apply_protobuf_depth_event(
        self,
        new_quotes: Iterable[oa_model.ProtoOADepthQuote],
        deleted_quote_ids: Iterable[int],
        timestamp_utc: float | None = None,
    ) -> OrderBookSnapshot:
        """Apply a Spotware ``ProtoOADepthEvent`` delta update to the Level 2 book.

        In Spotware's ``ProtoOADepthQuote``:
          - ``id`` is the unique quote identifier.
          - ``size`` is volume in cents of a unit (divide by 100.0 for units).
          - ``bid`` > 0 indicates a Bid quote at ``bid / 100_000.0`` (rounded to ``digits``).
          - ``ask`` > 0 indicates an Ask quote at ``ask / 100_000.0`` (rounded to ``digits``).
        """
        with self._lock:
            for qid in deleted_quote_ids:
                self._quotes.pop(int(qid), None)

            for q in new_quotes:
                qid = int(q.id)
                vol_units = float(q.size) / CTRADER_VOLUME_CENTS_PER_UNIT
                if vol_units <= 0.0:
                    self._quotes.pop(qid, None)
                    continue

                raw_bid = int(getattr(q, "bid", 0) or 0)
                raw_ask = int(getattr(q, "ask", 0) or 0)

                if raw_bid > 0:
                    price = round(raw_bid / CTRADER_PRICE_SCALE, self.digits)
                    self._quotes[qid] = DepthLevel(
                        quote_id=qid,
                        price=price,
                        volume_units=vol_units,
                        side=Side.BUY,
                    )
                elif raw_ask > 0:
                    price = round(raw_ask / CTRADER_PRICE_SCALE, self.digits)
                    self._quotes[qid] = DepthLevel(
                        quote_id=qid,
                        price=price,
                        volume_units=vol_units,
                        side=Side.SELL,
                    )

            self._last_update_utc = timestamp_utc if timestamp_utc is not None else time.time()
            snap = self.snapshot()
            self._update_order_flow_and_vpin(snap)
            return snap

    def set_levels(
        self,
        bids: Sequence[tuple[float, float]],
        asks: Sequence[tuple[float, float]],
        timestamp_utc: float | None = None,
    ) -> OrderBookSnapshot:
        """Directly populate the book from ``(price, volume_units)`` pairs."""
        with self._lock:
            self._quotes.clear()
            next_id = 1
            for price, vol in bids:
                if vol > 0.0:
                    self._quotes[next_id] = DepthLevel(
                        quote_id=next_id,
                        price=round(float(price), self.digits),
                        volume_units=float(vol),
                        side=Side.BUY,
                    )
                    next_id += 1
            for price, vol in asks:
                if vol > 0.0:
                    self._quotes[next_id] = DepthLevel(
                        quote_id=next_id,
                        price=round(float(price), self.digits),
                        volume_units=float(vol),
                        side=Side.SELL,
                    )
                    next_id += 1
            self._last_update_utc = timestamp_utc if timestamp_utc is not None else time.time()
            snap = self.snapshot()
            self._update_order_flow_and_vpin(snap)
            return snap

    def _update_order_flow_and_vpin(self, snap: OrderBookSnapshot) -> None:
        """Track rolling Cont-Kukanov-Stoikov OFI across top-N levels and volume-bucketed VPIN."""
        n = min(max(1, self.top_n_levels), 5)
        cur_bids = [(lvl.price, lvl.volume_units) for lvl in snap.bids[:n]]
        cur_asks = [(lvl.price, lvl.volume_units) for lvl in snap.asks[:n]]
        cur_mid = (
            0.5 * (cur_bids[0][0] + cur_asks[0][0])
            if (cur_bids and cur_asks and cur_bids[0][0] > 0.0 and cur_asks[0][0] > 0.0)
            else 0.0
        )

        if self._prev_bids or self._prev_asks:
            raw_ofi = 0.0
            depth_scale = 0.0
            for idx in range(n):
                w = 1.0 / float(idx + 1)
                pb_p, pb_v = self._prev_bids[idx] if idx < len(self._prev_bids) else (0.0, 0.0)
                cb_p, cb_v = cur_bids[idx] if idx < len(cur_bids) else (0.0, 0.0)
                if cb_p > 0.0 and pb_p > 0.0:
                    if cb_p > pb_p:
                        e_b = cb_v
                    elif cb_p == pb_p:
                        e_b = cb_v - pb_v
                    else:
                        e_b = -pb_v
                elif cb_p > 0.0:
                    e_b = cb_v
                else:
                    e_b = -pb_v

                pa_p, pa_v = self._prev_asks[idx] if idx < len(self._prev_asks) else (0.0, 0.0)
                ca_p, ca_v = cur_asks[idx] if idx < len(cur_asks) else (0.0, 0.0)
                if ca_p > 0.0 and pa_p > 0.0:
                    if ca_p < pa_p:
                        e_a = ca_v
                    elif ca_p == pa_p:
                        e_a = ca_v - pa_v
                    else:
                        e_a = -pa_v
                elif ca_p > 0.0:
                    e_a = ca_v
                else:
                    e_a = -pa_v

                raw_ofi += w * (e_b - e_a)
                depth_scale += w * 0.5 * (cb_v + pb_v + ca_v + pa_v)

            norm_ofi = max(-1.0, min(1.0, raw_ofi / max(depth_scale, abs(raw_ofi), 1e-9)))
            self._ofi_history.append(float(norm_ofi))

            total_cur_depth = sum(v for _, v in cur_bids) + sum(v for _, v in cur_asks)
            if self._vpin_bucket_capacity <= 0.0 and total_cur_depth > 0.0:
                self._vpin_bucket_capacity = max(0.5 * total_cur_depth, 1.0)
            elif total_cur_depth > 0.0:
                self._vpin_bucket_capacity = max(0.8 * self._vpin_bucket_capacity + 0.2 * (0.5 * total_cur_depth), 1.0)

            flow_vol = abs(raw_ofi)
            if flow_vol > 1e-9:
                if raw_ofi > 0.0 or (cur_mid > 0.0 and self._prev_mid > 0.0 and cur_mid > self._prev_mid):
                    self._ingest_vpin_flow(flow_vol, is_buy=True)
                else:
                    self._ingest_vpin_flow(flow_vol, is_buy=False)

        self._prev_bids = cur_bids
        self._prev_asks = cur_asks
        if cur_mid > 0.0:
            self._prev_mid = cur_mid

    def _ingest_vpin_flow(self, volume: float, is_buy: bool) -> None:
        """Accumulate signed volume into fixed-capacity VPIN buckets."""
        vol = max(0.0, float(volume))
        if vol <= 0.0:
            return
        cap = self._vpin_bucket_capacity if self._vpin_bucket_capacity > 0.0 else max(vol, 1.0)
        if is_buy:
            self._vpin_bucket_buy += vol
        else:
            self._vpin_bucket_sell += vol
        total = self._vpin_bucket_buy + self._vpin_bucket_sell
        while total >= cap and cap > 1e-9:
            ratio_buy = self._vpin_bucket_buy / total
            ratio_sell = self._vpin_bucket_sell / total
            b_slice = cap * ratio_buy
            s_slice = cap * ratio_sell
            self._vpin_buckets.append(min(1.0, abs(b_slice - s_slice) / cap))
            self._vpin_bucket_buy = max(0.0, self._vpin_bucket_buy - b_slice)
            self._vpin_bucket_sell = max(0.0, self._vpin_bucket_sell - s_slice)
            total = self._vpin_bucket_buy + self._vpin_bucket_sell

    def record_trade_flow(self, volume: float, side: Side) -> None:
        """Record an explicit signed trade fill into the VPIN bucket estimator."""
        with self._lock:
            self._ingest_vpin_flow(volume, is_buy=(side == Side.BUY))

    def compute_micro_price(self, snap: OrderBookSnapshot | None = None) -> float:
        """Compute Stoikov Micro-Price: (V_ask * P_bid + V_bid * P_ask) / (V_bid + V_ask)."""
        book = snap if snap is not None else self.snapshot()
        if not book.bids or not book.asks:
            return 0.0
        best_bid_lvl = book.bids[0]
        best_ask_lvl = book.asks[0]
        denom = best_bid_lvl.volume_units + best_ask_lvl.volume_units
        if denom <= 1e-12:
            return float(0.5 * (best_bid_lvl.price + best_ask_lvl.price))
        return float(
            (best_ask_lvl.volume_units * best_bid_lvl.price + best_bid_lvl.volume_units * best_ask_lvl.price)
            / denom
        )

    def compute_ofi(self) -> float:
        """Return rolling exponentially smoothed Cont-Kukanov-Stoikov OFI in [-1.0, +1.0]."""
        with self._lock:
            if not self._ofi_history:
                return 0.0
            vals = np.asarray(self._ofi_history, dtype=np.float64)
            weights = np.exp(np.linspace(-1.2, 0.0, len(vals)))
            weights /= np.sum(weights)
            return float(np.clip(np.sum(vals * weights), -1.0, 1.0))

    def compute_vpin(self) -> float:
        """Return Volume-Synchronized Probability of Informed Trading in [0.0, 1.0]."""
        with self._lock:
            if self._vpin_buckets:
                return float(np.clip(np.mean(self._vpin_buckets), 0.0, 1.0))
            partial = self._vpin_bucket_buy + self._vpin_bucket_sell
            if partial > 1e-9 and self._vpin_bucket_capacity > 1e-9:
                fill_frac = min(1.0, partial / self._vpin_bucket_capacity)
                if fill_frac >= 0.25:
                    return float(np.clip(abs(self._vpin_bucket_buy - self._vpin_bucket_sell) / partial, 0.0, 1.0))
            return 0.0

    def snapshot(self) -> OrderBookSnapshot:
        """Return a thread-safe sorted snapshot (bids descending by price, asks ascending by price).

        Aggregates multiple quotes sitting at the exact same price level so that each level
        represents the consolidated liquidity at that price.
        """
        with self._lock:
            bid_map: dict[float, tuple[int, float]] = {}
            ask_map: dict[float, tuple[int, float]] = {}

            for q in self._quotes.values():
                if q.side == Side.BUY:
                    if q.price in bid_map:
                        first_id, cur_vol = bid_map[q.price]
                        bid_map[q.price] = (first_id, cur_vol + q.volume_units)
                    else:
                        bid_map[q.price] = (q.quote_id, q.volume_units)
                else:
                    if q.price in ask_map:
                        first_id, cur_vol = ask_map[q.price]
                        ask_map[q.price] = (first_id, cur_vol + q.volume_units)
                    else:
                        ask_map[q.price] = (q.quote_id, q.volume_units)

            sorted_bids = tuple(
                DepthLevel(quote_id=qid, price=price, volume_units=vol, side=Side.BUY)
                for price, (qid, vol) in sorted(bid_map.items(), key=lambda item: item[0], reverse=True)
            )
            sorted_asks = tuple(
                DepthLevel(quote_id=qid, price=price, volume_units=vol, side=Side.SELL)
                for price, (qid, vol) in sorted(ask_map.items(), key=lambda item: item[0])
            )

            return OrderBookSnapshot(
                symbol=self.symbol,
                timestamp_utc=self._last_update_utc,
                bids=sorted_bids,
                asks=sorted_asks,
                pip_size=self.pip_size,
            )

    def compute_obi(self, top_n: int | None = None) -> float:
        """Compute Order-Book Imbalance across the top N levels:
        ``OBI = (bid_vol - ask_vol) / (bid_vol + ask_vol)`` in ``[-1.0, +1.0]``.
        """
        snap = self.snapshot()
        n = top_n if top_n is not None else self.top_n_levels
        bid_vol = sum(level.volume_units for level in snap.bids[:n])
        ask_vol = sum(level.volume_units for level in snap.asks[:n])
        total = bid_vol + ask_vol
        if total <= 0.0:
            return 0.0
        return float((bid_vol - ask_vol) / total)

    def detect_liquidity_walls(
        self,
        wall_multiplier: float | None = None,
    ) -> tuple[LiquidityWall, ...]:
        """Detect abnormal liquidity walls across the Level 2 order book.

        A level is flagged as a liquidity wall if its volume is at least
        ``wall_multiplier`` times the median level volume across the book.
        """
        snap = self.snapshot()
        mult = wall_multiplier if wall_multiplier is not None else self.wall_multiplier
        all_levels = list(snap.bids) + list(snap.asks)
        if len(all_levels) < 2:
            return ()

        volumes = np.array([lvl.volume_units for lvl in all_levels], dtype=np.float64)
        median_vol = float(np.median(volumes))
        if median_vol <= 0.0:
            return ()

        best_bid = snap.bids[0].price if snap.bids else 0.0
        best_ask = snap.asks[0].price if snap.asks else 0.0
        mid_price = (
            0.5 * (best_bid + best_ask)
            if (best_bid > 0.0 and best_ask > 0.0)
            else (best_bid or best_ask)
        )

        walls: list[LiquidityWall] = []
        for lvl in all_levels:
            ratio = lvl.volume_units / median_vol
            if ratio >= mult:
                dist_pips = (
                    abs(lvl.price - mid_price) / self.pip_size if mid_price > 0.0 else 0.0
                )
                walls.append(
                    LiquidityWall(
                        side=lvl.side,
                        price=lvl.price,
                        volume_units=lvl.volume_units,
                        ratio_to_median=float(ratio),
                        distance_from_mid_pips=float(dist_pips),
                    )
                )
        return tuple(walls)

    def compute_vwap_fill(
        self,
        side: Side,
        target_volume_units: float,
    ) -> VWAPExecutionEstimate:
        """Compute true Level 2 DOM VWAP expected fill price and slippage for ``target_volume_units``.

        - A ``Side.BUY`` market order sweeps the ``asks`` book from lowest ask upward.
        - A ``Side.SELL`` market order sweeps the ``bids`` book from highest bid downward.
        - If ``target_volume_units`` exceeds total available book depth, the unfilled remainder
          is priced at the worst available book level and ``is_fully_fillable`` is set to ``False``.
        """
        if target_volume_units <= 0.0:
            raise ValueError(f"target_volume_units must be > 0, got {target_volume_units}")

        snap = self.snapshot()
        ladder = snap.asks if side == Side.BUY else snap.bids
        if not ladder:
            raise RuntimeError(f"Cannot compute VWAP fill for {self.symbol}: {side.value} book is empty.")

        top_price = ladder[0].price
        remaining = float(target_volume_units)
        notional_sum = 0.0
        filled = 0.0
        levels_used = 0

        for lvl in ladder:
            if remaining <= 0.0:
                break
            take = min(remaining, lvl.volume_units)
            notional_sum += take * lvl.price
            filled += take
            remaining -= take
            levels_used += 1

        is_fully_fillable = remaining <= 1e-9
        unfilled = max(0.0, remaining)

        if not is_fully_fillable:
            # Price any residual unfilled volume at the worst level in the ladder as a conservative bound
            worst_price = ladder[-1].price
            effective_vwap = (notional_sum + unfilled * worst_price) / float(target_volume_units)
        else:
            effective_vwap = notional_sum / filled

        # Slippage is positive when execution is worse than top-of-book
        if side == Side.BUY:
            slippage_price = effective_vwap - top_price
        else:
            slippage_price = top_price - effective_vwap

        slippage_pips = slippage_price / self.pip_size

        return VWAPExecutionEstimate(
            symbol=self.symbol,
            side=side,
            target_volume_units=float(target_volume_units),
            filled_volume_units=float(filled),
            unfilled_volume_units=float(unfilled),
            top_of_book_price=float(top_price),
            vwap_fill_price=float(effective_vwap),
            slippage_price=float(slippage_price),
            slippage_pips=float(slippage_pips),
            levels_consumed=levels_used,
            is_fully_fillable=is_fully_fillable,
        )

    def compute_analytics(self, top_n: int | None = None) -> DOMAnalytics:
        """Compute a unified ``DOMAnalytics`` microstructure summary."""
        snap = self.snapshot()
        n = top_n if top_n is not None else self.top_n_levels
        bid_vol = sum(level.volume_units for level in snap.bids[:n])
        ask_vol = sum(level.volume_units for level in snap.asks[:n])
        total = bid_vol + ask_vol
        obi = (bid_vol - ask_vol) / total if total > 0.0 else 0.0

        best_bid = snap.bids[0].price if snap.bids else 0.0
        best_ask = snap.asks[0].price if snap.asks else 0.0
        spread_pips = (
            (best_ask - best_bid) / self.pip_size
            if (best_bid > 0.0 and best_ask > 0.0)
            else 0.0
        )

        return DOMAnalytics(
            symbol=self.symbol,
            timestamp_utc=snap.timestamp_utc,
            top_n_levels=n,
            bid_volume_top_n=float(bid_vol),
            ask_volume_top_n=float(ask_vol),
            obi=float(obi),
            best_bid=float(best_bid),
            best_ask=float(best_ask),
            spread_pips=float(spread_pips),
            liquidity_walls=self.detect_liquidity_walls(),
            micro_price=self.compute_micro_price(snap),
            ofi=self.compute_ofi(),
            vpin=self.compute_vpin(),
        )


TIMEFRAME_TO_CTRADER_PERIOD: dict[str, int] = {
    "M1": oa_model.ProtoOATrendbarPeriod.M1,
    "M5": oa_model.ProtoOATrendbarPeriod.M5,
    "M15": oa_model.ProtoOATrendbarPeriod.M15,
    "M30": oa_model.ProtoOATrendbarPeriod.M30,
    "H1": oa_model.ProtoOATrendbarPeriod.H1,
    "H4": oa_model.ProtoOATrendbarPeriod.H4,
    "D1": oa_model.ProtoOATrendbarPeriod.D1,
}

CTRADER_PERIOD_TO_TIMEFRAME: dict[int, str] = {
    v: k for k, v in TIMEFRAME_TO_CTRADER_PERIOD.items()
}


class CTraderOpenApiFeed:
    """Official Spotware Open API binary Protobuf over TLS TCP client.

    Supports:
      - Demo (`demo.ctraderapi.com:5035`) and Live (`live.ctraderapi.com:5035`) endpoints
        with strict DEMO lock safety enforcement (`verify_live_unlock`).
      - Automatic OAuth2 token refresh (`ProtoOARefreshTokenReq` & HTTP token refresh).
      - Historical M5/M15 trendbars (`ProtoOAGetTrendbarsReq`).
      - Live spot tick streaming (`ProtoOASubscribeSpotsReq` / `ProtoOASpotEvent`).
      - Real-time Level 2 Depth of Market streaming (`ProtoOASubscribeDepthQuotesReq` / `ProtoOADepthEvent`).
    """

    DEMO_HOST: str = EndPoints.PROTOBUF_DEMO_HOST  # "demo.ctraderapi.com"
    LIVE_HOST: str = EndPoints.PROTOBUF_LIVE_HOST  # "live.ctraderapi.com"
    PORT: int = EndPoints.PROTOBUF_PORT  # 5035

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        access_token: str | None = None,
        refresh_token: str | None = None,
        account_id: int | None = None,
        is_live: bool = False,
        top_n_dom_levels: int = 5,
        wall_multiplier: float = 3.0,
        cli_args: Sequence[str] | None = None,
    ) -> None:
        load_dotenv()

        self.is_live: bool = bool(is_live)
        verify_live_unlock(self.is_live, cli_args=cli_args)

        self.client_id: str = (client_id or os.environ.get("CTRADER_CLIENT_ID", "")).strip()
        self.client_secret: str = (client_secret or os.environ.get("CTRADER_CLIENT_SECRET", "")).strip()
        self.access_token: str = (access_token or os.environ.get("CTRADER_ACCESS_TOKEN", "")).strip()
        self.refresh_token: str = (refresh_token or os.environ.get("CTRADER_REFRESH_TOKEN", "")).strip()

        raw_acct = account_id if account_id is not None else os.environ.get("CTRADER_ACCOUNT_ID", "")
        self.account_id: int = int(raw_acct) if str(raw_acct).strip().isdigit() else 0

        self.host: str = self.LIVE_HOST if self.is_live else self.DEMO_HOST
        self.port: int = self.PORT
        self.top_n_dom_levels: int = top_n_dom_levels
        self.wall_multiplier: float = wall_multiplier

        self._client: Client | None = None
        self._is_authenticated: bool = False
        self._symbol_name_to_id: dict[str, int] = {}
        self._symbol_id_to_name: dict[int, str] = {}
        self._contract_specs: dict[str, ContractSpec] = {}
        self._dom_engines: dict[str, OrderBookEngine] = {}
        self._latest_ticks: dict[str, Tick] = {}
        self._tick_callbacks: list[Callable[[Tick], None]] = []
        self._dom_callbacks: list[Callable[[DOMAnalytics], None]] = []
        self.alias_mapper: SmartAliasMapper = SmartAliasMapper()
        self._catalog: dict[str, DiscoveredSymbol] = {}
        self._lock = threading.RLock()

    @property
    def has_credentials(self) -> bool:
        """Return True if all required cTrader Open API credentials are configured."""
        return bool(
            self.client_id
            and self.client_secret
            and self.access_token
            and self.account_id > 0
        )

    @property
    def is_connected(self) -> bool:
        """Return True if the TCP TLS session is connected and account-authenticated."""
        override = getattr(self, "_connected", None)
        if override is not None:
            return bool(override)
        return bool(self._client is not None and self._client.isConnected and self._is_authenticated)

    def register_tick_callback(self, callback: Callable[[Tick], None]) -> None:
        """Subscribe a listener to normalized incoming spot ticks."""
        with self._lock:
            self._tick_callbacks.append(callback)

    def register_dom_callback(self, callback: Callable[[DOMAnalytics], None]) -> None:
        """Subscribe a listener to real-time Level 2 DOM analytics updates."""
        with self._lock:
            self._dom_callbacks.append(callback)

    def get_or_create_dom_engine(
        self,
        symbol: str,
        pip_size: float = 0.0001,
        digits: int = 5,
    ) -> OrderBookEngine:
        """Retrieve or initialize the ``OrderBookEngine`` for ``symbol``."""
        with self._lock:
            if symbol not in self._dom_engines:
                spec = self._contract_specs.get(symbol)
                eff_pip = spec.pip_size if spec else pip_size
                eff_digits = spec.digits if spec else digits
                self._dom_engines[symbol] = OrderBookEngine(
                    symbol=symbol,
                    pip_size=eff_pip,
                    digits=eff_digits,
                    top_n_levels=self.top_n_dom_levels,
                    wall_multiplier=self.wall_multiplier,
                )
            return self._dom_engines[symbol]

    def get_or_create_dom(
        self,
        symbol: str,
        pip_size: float = 0.0001,
        digits: int = 5,
    ) -> OrderBookEngine:
        """Alias for ``get_or_create_dom_engine``."""
        return self.get_or_create_dom_engine(symbol=symbol, pip_size=pip_size, digits=digits)

    def get_tick(self, symbol: str) -> Tick | None:
        """Alias for ``get_latest_tick``."""
        return self.get_latest_tick(symbol=symbol)

    def get_latest_tick(self, symbol: str) -> Tick | None:
        """Return the latest spot tick received for ``symbol`` (resolving aliases automatically)."""
        norm = symbol.upper().strip()
        with self._lock:
            tick = self._latest_ticks.get(norm)
            if tick is not None:
                return tick
            resolved = self.resolve_symbol(norm)
            if resolved:
                return self._latest_ticks.get(resolved)
            return None

    def get_contract_spec(self, symbol: str) -> ContractSpec | None:
        """Return cached ``ContractSpec`` for ``symbol`` (resolving aliases automatically)."""
        norm = symbol.upper().strip()
        with self._lock:
            spec = self._contract_specs.get(norm)
            if spec is not None:
                return spec
            resolved = self.resolve_symbol(norm)
            if resolved:
                return self._contract_specs.get(resolved)
            return None

    def refresh_oauth_token_http(self, redirect_uri: str = "http://127.0.0.1:8787/callback") -> dict[str, str]:
        """Refresh the OAuth2 access token via Spotware's HTTPS token endpoint."""
        if not (self.client_id and self.client_secret and self.refresh_token):
            raise RuntimeError("Cannot refresh cTrader token: missing client_id, client_secret, or refresh_token.")

        auth = Auth(self.client_id, self.client_secret, redirect_uri)
        token_data = auth.refreshToken(self.refresh_token)
        if "accessToken" in token_data and token_data["accessToken"]:
            self.access_token = str(token_data["accessToken"])
            if token_data.get("refreshToken"):
                self.refresh_token = str(token_data["refreshToken"])
            return {
                "access_token": self.access_token,
                "refresh_token": self.refresh_token,
            }
        raise RuntimeError(f"cTrader OAuth token refresh failed: {token_data}")

    def send_protobuf_refresh_token(self) -> object:
        """Send an in-band ``ProtoOARefreshTokenReq`` over the active Protobuf TCP session."""
        if self._client is None or not self._client.isConnected:
            raise RuntimeError("cTrader TCP client is not connected.")
        req = oa_msg.ProtoOARefreshTokenReq()
        req.refreshToken = self.refresh_token
        d = self._client.send(req)
        d.addCallback(self._on_refresh_token_response)
        return d

    def _on_refresh_token_response(self, message: object) -> None:
        extracted = Protobuf.extract(message)
        if isinstance(extracted, oa_msg.ProtoOARefreshTokenRes):
            self.access_token = str(extracted.accessToken)
            if getattr(extracted, "refreshToken", ""):
                self.refresh_token = str(extracted.refreshToken)
            logger.info("cTrader Open API access token refreshed over Protobuf TCP.")

    def _start_reactor_thread(self) -> None:
        """Start the Twisted reactor in a background daemon thread if not already running."""
        try:
            from twisted.internet import reactor

            if not getattr(reactor, "running", False):
                t = threading.Thread(
                    target=lambda: reactor.run(installSignalHandlers=False),
                    daemon=True,
                    name="cTrader-TwistedReactor",
                )
                t.start()
        except Exception as exc:
            logger.debug("Twisted reactor thread start skipped: %s", exc)

    def start(self) -> None:
        """Initialize the Twisted Protobuf TLS TCP client and authenticate."""
        verify_live_unlock(self.is_live)
        if not self.has_credentials:
            raise RuntimeError(
                "cTrader credentials missing in environment (.env). "
                "Run tools/setup_ctrader_oauth.py or fallback to MT5."
            )
        self._client = Client(self.host, self.port, TcpProtocol)
        self._client.setConnectedCallback(self._on_connected)
        self._client.setDisconnectedCallback(self._on_disconnected)
        self._client.setMessageReceivedCallback(self._on_message_received)
        self._client.startService()
        self._start_reactor_thread()

    def stop(self) -> None:
        """Stop the cTrader TCP client service."""
        self._is_authenticated = False
        if self._client is not None:
            self._client.stopService()

    def _on_connected(self, client: Client) -> None:
        logger.info("Connected to cTrader Open API at %s:%d. Sending ApplicationAuthReq...", self.host, self.port)
        app_req = oa_msg.ProtoOAApplicationAuthReq()
        app_req.clientId = self.client_id
        app_req.clientSecret = self.client_secret
        d = client.send(app_req)
        d.addErrback(lambda f: logger.error("cTrader ApplicationAuth failed: %s", f))

    def _on_disconnected(self, client: Client, reason: object) -> None:
        self._is_authenticated = False
        logger.warning("Disconnected from cTrader Open API: %s", reason)

    def _on_message_received(self, client: Client, message: object) -> None:
        try:
            extracted = Protobuf.extract(message)
        except Exception as exc:
            logger.debug("Failed to extract Protobuf message: %s", exc)
            return

        if isinstance(extracted, oa_msg.ProtoOAApplicationAuthRes):
            acct_req = oa_msg.ProtoOAAccountAuthReq()
            acct_req.ctidTraderAccountId = self.account_id
            acct_req.accessToken = self.access_token
            client.send(acct_req)

        elif isinstance(extracted, oa_msg.ProtoOAAccountAuthRes):
            self._is_authenticated = True
            logger.info("cTrader account %d authenticated. Requesting symbol catalog & reconciliation...", self.account_id)
            sym_req = oa_msg.ProtoOASymbolsListReq()
            sym_req.ctidTraderAccountId = self.account_id
            client.send(sym_req)
            try:
                rec_req = oa_msg.ProtoOAReconcileReq()
                rec_req.ctidTraderAccountId = self.account_id
                client.send(rec_req)
            except Exception as exc:
                logger.debug("Initial cTrader ProtoOAReconcileReq failed: %s", exc)

        elif isinstance(extracted, oa_msg.ProtoOASymbolsListRes):
            with self._lock:
                for light_sym in extracted.symbol:
                    if light_sym.enabled and light_sym.symbolName:
                        name = str(light_sym.symbolName).upper()
                        sid = int(light_sym.symbolId)
                        self._symbol_name_to_id[name] = sid
                        self._symbol_id_to_name[sid] = name
                        canon = self.alias_mapper.canonicalize(name)
                        desc = str(getattr(light_sym, "description", ""))
                        ac = self.alias_mapper.classify_asset_class(name, description=desc)
                        self.alias_mapper.register_symbol(name, canonical_name=canon, broker="ctrader")
                        self._catalog[name] = DiscoveredSymbol(
                            name=name,
                            canonical_name=canon,
                            asset_class=ac,
                            path="",
                            digits=5,
                            point=1e-5,
                            trade_mode=4 if light_sym.enabled else 0,
                            is_visible=True,
                            source="ctrader",
                            description=desc,
                        )

        elif isinstance(extracted, oa_msg.ProtoOASymbolByIdRes):
            with self._lock:
                for sym in extracted.symbol:
                    sid = int(sym.symbolId)
                    name = self._symbol_id_to_name.get(sid, str(sid))
                    digits = int(sym.digits)
                    pip_pos = int(sym.pipPosition)
                    pip_size = float(10 ** (-pip_pos))
                    point_size = float(10 ** (-digits))
                    lot_units = float(getattr(sym, "lotSize", 10_000_000)) / CTRADER_VOLUME_CENTS_PER_UNIT
                    min_lots = (float(sym.minVolume) / CTRADER_VOLUME_CENTS_PER_UNIT) / max(lot_units, 1.0)
                    max_lots = (float(sym.maxVolume) / CTRADER_VOLUME_CENTS_PER_UNIT) / max(lot_units, 1.0)
                    step_lots = (float(sym.stepVolume) / CTRADER_VOLUME_CENTS_PER_UNIT) / max(lot_units, 1.0)
                    self._contract_specs[name] = ContractSpec(
                        symbol=name,
                        digits=digits,
                        pip_size=pip_size,
                        point_size=point_size,
                        lot_size=lot_units,
                        min_volume_lots=min_lots,
                        max_volume_lots=max_lots,
                        step_volume_lots=step_lots,
                        source="ctrader",
                    )
                    if name in self._catalog:
                        old = self._catalog[name]
                        self._catalog[name] = DiscoveredSymbol(
                            name=old.name,
                            canonical_name=old.canonical_name,
                            asset_class=old.asset_class,
                            path=old.path,
                            digits=digits,
                            point=point_size,
                            trade_mode=old.trade_mode,
                            is_visible=old.is_visible,
                            source="ctrader",
                            description=old.description,
                        )
                    self.get_or_create_dom_engine(name, pip_size=pip_size, digits=digits)

        elif isinstance(extracted, oa_msg.ProtoOASpotEvent):
            self.handle_spot_event(extracted)

        elif isinstance(extracted, oa_msg.ProtoOADepthEvent):
            self.handle_depth_event(extracted)

        elif isinstance(extracted, oa_msg.ProtoOAGetTrendbarsRes):
            self._handle_trendbars_res(extracted)

        elif isinstance(extracted, oa_msg.ProtoOAExecutionEvent):
            self._handle_execution_event(extracted)

        elif isinstance(extracted, oa_msg.ProtoOAReconcileRes):
            self._handle_reconcile_res(extracted)

        elif isinstance(extracted, oa_msg.ProtoOAAccountsTokenInvalidatedEvent):
            logger.warning("cTrader token invalidated event received; triggering token refresh...")
            if self.refresh_token:
                self.send_protobuf_refresh_token()

    def register_symbol_mapping(
        self,
        symbol: str,
        symbol_id: int,
        pip_size: float = 0.0001,
        digits: int = 5,
    ) -> None:
        """Register a symbol name <-> Spotware symbolId mapping and initialize its DOM engine."""
        with self._lock:
            norm = symbol.upper()
            self._symbol_name_to_id[norm] = int(symbol_id)
            self._symbol_id_to_name[int(symbol_id)] = norm
            self.alias_mapper.register_symbol(norm, broker="ctrader")
            canon = self.alias_mapper.canonicalize(norm)
            ac = self.alias_mapper.classify_asset_class(norm)
            self._catalog[norm] = DiscoveredSymbol(
                name=norm,
                canonical_name=canon,
                asset_class=ac,
                path="",
                digits=int(digits),
                point=float(10 ** (-digits)),
                trade_mode=4,
                is_visible=True,
                source="ctrader",
            )
            self.get_or_create_dom_engine(norm, pip_size=pip_size, digits=digits)

    def resolve_symbol(self, symbol: str) -> str | None:
        """Resolve a canonical or broker symbol to the cTrader symbol name."""
        norm = symbol.upper().strip()
        with self._lock:
            if norm in self._symbol_name_to_id:
                return norm
            mapped = self.alias_mapper.resolve_to_broker(norm, broker="ctrader")
            if mapped in self._symbol_name_to_id:
                return mapped
            canon = self.alias_mapper.canonicalize(norm)
            if canon in self._symbol_name_to_id:
                return canon
            return norm if norm in self._symbol_name_to_id else None

    def discover_catalog(self) -> list[DiscoveredSymbol]:
        """Return discovered catalog symbols registered in cTrader, requesting from server if empty."""
        with self._lock:
            if not self._catalog and self._symbol_name_to_id:
                for name, sid in self._symbol_name_to_id.items():
                    spec = self._contract_specs.get(name)
                    dig = spec.digits if spec else 5
                    pt = spec.point_size if spec else 1e-5
                    canon = self.alias_mapper.canonicalize(name)
                    ac = self.alias_mapper.classify_asset_class(name)
                    self._catalog[name] = DiscoveredSymbol(
                        name=name,
                        canonical_name=canon,
                        asset_class=ac,
                        path="",
                        digits=dig,
                        point=pt,
                        trade_mode=4,
                        is_visible=True,
                        source="ctrader",
                    )
            elif not self._symbol_name_to_id and self.is_connected and self._client is not None:
                sym_req = oa_msg.ProtoOASymbolsListReq()
                sym_req.ctidTraderAccountId = self.account_id
                self._client.send(sym_req)

            return list(self._catalog.values())

    def get_discovered_symbols(self) -> list[DiscoveredSymbol]:
        """Return cached discovered cTrader symbols."""
        return self.discover_catalog()

    def subscribe_spots(self, symbols: Sequence[str], subscribe_to_spot_timestamp: bool = True) -> object:
        """Send ``ProtoOASubscribeSpotsReq`` for the requested symbols."""
        if self._client is None or not self._client.isConnected:
            raise RuntimeError("cTrader client is not connected.")
        req = oa_msg.ProtoOASubscribeSpotsReq()
        req.ctidTraderAccountId = self.account_id
        req.subscribeToSpotTimestamp = subscribe_to_spot_timestamp
        for sym in symbols:
            sid = self._symbol_name_to_id.get(sym.upper())
            if sid is not None:
                req.symbolId.append(sid)
        return self._client.send(req)

    def subscribe_depth_quotes(self, symbols: Sequence[str]) -> object:
        """Send ``ProtoOASubscribeDepthQuotesReq`` to stream Level 2 DOM updates."""
        if self._client is None or not self._client.isConnected:
            raise RuntimeError("cTrader client is not connected.")
        req = oa_msg.ProtoOASubscribeDepthQuotesReq()
        req.ctidTraderAccountId = self.account_id
        for sym in symbols:
            sid = self._symbol_name_to_id.get(sym.upper())
            if sid is not None:
                req.symbolId.append(sid)
        return self._client.send(req)

    def request_trendbars(
        self,
        symbol: str,
        timeframe: str = "M5",
        from_timestamp_utc: float | None = None,
        to_timestamp_utc: float | None = None,
        count: int = 200,
    ) -> object:
        """Build and send ``ProtoOAGetTrendbarsReq`` for historical M1/M5/M15/M30/H1/H4/D1 candles."""
        self._ensure_runtime_buffers()
        if not self.is_connected or self._client is None:
            raise RuntimeError("cTrader client is not connected.")

        norm_tf = timeframe.upper()
        if norm_tf not in TIMEFRAME_TO_CTRADER_PERIOD:
            raise ValueError(f"Unsupported timeframe '{timeframe}'. Supported: {list(TIMEFRAME_TO_CTRADER_PERIOD)}")

        resolved = self.resolve_symbol(symbol) or symbol.upper().strip()
        sid = self._symbol_name_to_id.get(resolved)
        if sid is None:
            raise KeyError(f"Symbol '{symbol}' not found in cTrader symbol mapping.")

        period_minutes_map = {
            "M1": 1,
            "M5": 5,
            "M15": 15,
            "M30": 30,
            "H1": 60,
            "H4": 240,
            "D1": 1440,
        }
        now_ms = int((to_timestamp_utc if to_timestamp_utc is not None else time.time()) * 1000)
        period_minutes = period_minutes_map.get(norm_tf, 5)
        default_from_ms = now_ms - int(max(count, 10) * period_minutes * 60 * 1000 * 1.5)
        from_ms = int(from_timestamp_utc * 1000) if from_timestamp_utc is not None else default_from_ms

        req = oa_msg.ProtoOAGetTrendbarsReq()
        req.ctidTraderAccountId = self.account_id
        req.symbolId = sid
        req.period = TIMEFRAME_TO_CTRADER_PERIOD[norm_tf]
        req.fromTimestamp = from_ms
        req.toTimestamp = now_ms
        req.count = int(count)
        return self._send_proto(req)

    @staticmethod
    def decode_trendbars(
        symbol: str,
        timeframe: str,
        trendbars: Iterable[oa_model.ProtoOATrendbar],
        digits: int = 5,
    ) -> list[Candle]:
        """Decode Spotware ``ProtoOATrendbar`` relative delta-encoded bars into ``Candle`` objects.

        In Spotware Open API:
          - ``low`` is the raw integer low price (scaled by 100,000).
          - ``open = (low + deltaOpen) / 100,000``
          - ``high = (low + deltaHigh) / 100,000``
          - ``close = (low + deltaClose) / 100,000``
          - ``utcTimestampInMinutes * 60`` is the UTC epoch timestamp in seconds.
        """
        candles: list[Candle] = []
        for tb in trendbars:
            raw_low = int(tb.low)
            low_p = round(raw_low / CTRADER_PRICE_SCALE, digits)
            open_p = round((raw_low + int(getattr(tb, "deltaOpen", 0))) / CTRADER_PRICE_SCALE, digits)
            high_p = round((raw_low + int(getattr(tb, "deltaHigh", 0))) / CTRADER_PRICE_SCALE, digits)
            close_p = round((raw_low + int(getattr(tb, "deltaClose", 0))) / CTRADER_PRICE_SCALE, digits)
            ts_utc = float(int(tb.utcTimestampInMinutes) * 60)
            vol = float(getattr(tb, "volume", 0))
            candles.append(
                Candle(
                    symbol=symbol.upper(),
                    timeframe=timeframe.upper(),
                    timestamp_utc=ts_utc,
                    open=open_p,
                    high=high_p,
                    low=low_p,
                    close=close_p,
                    volume=vol,
                    source="ctrader",
                )
            )
        candles.sort(key=lambda c: c.timestamp_utc)
        return candles

    def handle_spot_event(self, event: oa_msg.ProtoOASpotEvent) -> Tick | None:
        """Process a ``ProtoOASpotEvent`` into a normalized ``Tick``."""
        with self._lock:
            sid = int(event.symbolId)
            symbol = self._symbol_id_to_name.get(sid, str(sid))
            spec = self._contract_specs.get(symbol)
            digits = spec.digits if spec else 5
            prev_tick = self._latest_ticks.get(symbol)

            raw_bid = int(getattr(event, "bid", 0) or 0)
            raw_ask = int(getattr(event, "ask", 0) or 0)

            bid = round(raw_bid / CTRADER_PRICE_SCALE, digits) if raw_bid > 0 else (prev_tick.bid if prev_tick else 0.0)
            ask = round(raw_ask / CTRADER_PRICE_SCALE, digits) if raw_ask > 0 else (prev_tick.ask if prev_tick else 0.0)
            if bid <= 0.0 or ask <= 0.0:
                return None

            raw_ts = int(getattr(event, "timestamp", 0) or 0)
            ts_utc = (raw_ts / 1000.0) if raw_ts > 0 else time.time()

            tick = Tick(
                symbol=symbol,
                bid=bid,
                ask=ask,
                timestamp_utc=ts_utc,
                source="ctrader",
            )
            self._latest_ticks[symbol] = tick
            callbacks = list(self._tick_callbacks)

        for cb in callbacks:
            cb(tick)
        return tick

    def handle_depth_event(self, event: oa_msg.ProtoOADepthEvent) -> DOMAnalytics:
        """Process a ``ProtoOADepthEvent`` into the symbol's ``OrderBookEngine`` and emit ``DOMAnalytics``."""
        with self._lock:
            sid = int(event.symbolId)
            symbol = self._symbol_id_to_name.get(sid, str(sid))
            engine = self.get_or_create_dom_engine(symbol)
            engine.apply_protobuf_depth_event(
                new_quotes=event.newQuotes,
                deleted_quote_ids=event.deletedQuotes,
            )
            analytics = engine.compute_analytics()
            callbacks = list(self._dom_callbacks)

        for cb in callbacks:
            cb(analytics)
        return analytics

    def _ensure_runtime_buffers(self) -> None:
        """Lazily initialize async cache & waiter structures cleanly (mock-safe)."""
        if not hasattr(self, "_trendbars_cache"):
            self._trendbars_cache: dict[tuple[str, str], tuple[float, list[Candle]]] = {}
        if not hasattr(self, "_trendbar_waiters"):
            self._trendbar_waiters: dict[tuple[str, str], list[threading.Event]] = {}
        if not hasattr(self, "_execution_waiters"):
            self._execution_waiters: dict[str, tuple[threading.Event, dict[str, object]]] = {}
        if not hasattr(self, "_recent_execution_events"):
            self._recent_execution_events: list[dict[str, object]] = []
        if not hasattr(self, "_open_positions"):
            self._open_positions: dict[int, dict[str, object]] = {}

    @staticmethod
    def _is_reactor_running() -> bool:
        try:
            from twisted.internet import reactor
            return bool(getattr(reactor, "running", False))
        except Exception:
            return False

    def _send_proto(self, req: object) -> object:
        """Thread-safe dispatch to Spotware Twisted TCP client."""
        if self._client is None:
            raise RuntimeError("cTrader client is not initialized.")
        try:
            from twisted.internet import reactor
            from twisted.python.threadable import isInIOThread
            if getattr(reactor, "running", False) and not isInIOThread():
                reactor.callFromThread(self._client.send, req)
                return None
        except Exception as exc:
            logger.debug("Twisted reactor dispatch unavailable, falling back to direct send: %s", exc)
        return self._client.send(req)

    def _handle_trendbars_res(self, extracted: oa_msg.ProtoOAGetTrendbarsRes) -> list[Candle]:
        self._ensure_runtime_buffers()
        sid = int(getattr(extracted, "symbolId", 0))
        period = int(getattr(extracted, "period", 0))
        tf = CTRADER_PERIOD_TO_TIMEFRAME.get(period, "M5")
        with self._lock:
            sym_name = self._symbol_id_to_name.get(sid, str(sid))
            spec = self._contract_specs.get(sym_name)
            digits = spec.digits if spec else 5
            candles = self.decode_trendbars(
                symbol=sym_name,
                timeframe=tf,
                trendbars=extracted.trendbar,
                digits=digits,
            )
            now_utc = time.time()
            self._trendbars_cache[(sym_name, tf)] = (now_utc, candles)
            canon = self.alias_mapper.canonicalize(sym_name)
            if canon and canon != sym_name:
                self._trendbars_cache[(canon, tf)] = (now_utc, candles)
            waiters = list(self._trendbar_waiters.pop((sym_name, tf), ()))
            if canon and canon != sym_name:
                waiters.extend(self._trendbar_waiters.pop((canon, tf), ()))
        for ev in waiters:
            ev.set()
        return candles

    def _handle_reconcile_res(self, extracted: oa_msg.ProtoOAReconcileRes) -> None:
        self._ensure_runtime_buffers()
        with self._lock:
            new_positions: dict[int, dict[str, object]] = {}
            for pos in getattr(extracted, "position", ()):
                pid = int(getattr(pos, "positionId", 0))
                if pid <= 0:
                    continue
                td = getattr(pos, "tradeData", None)
                sid = int(getattr(td, "symbolId", 0)) if td is not None else 0
                sym = self._symbol_id_to_name.get(sid, str(sid))
                spec = self._contract_specs.get(sym)
                lot_units = spec.lot_size if spec else 100_000.0
                vol_cents = int(getattr(td, "volume", 0)) if td is not None else 0
                lots = (vol_cents / CTRADER_VOLUME_CENTS_PER_UNIT) / max(lot_units, 1.0)
                tside = int(getattr(td, "tradeSide", 1)) if td is not None else 1
                side = Side.BUY if tside == int(oa_model.ProtoOATradeSide.BUY) else Side.SELL
                new_positions[pid] = {
                    "position_id": pid,
                    "symbol_id": sid,
                    "symbol": sym,
                    "canonical_symbol": self.alias_mapper.canonicalize(sym),
                    "side": side.value,
                    "lots": round(lots, 4),
                    "volume_cents": vol_cents,
                    "entry_price": float(getattr(pos, "price", 0.0) or 0.0),
                    "stop_loss": float(getattr(pos, "stopLoss", 0.0) or 0.0),
                    "take_profit": float(getattr(pos, "takeProfit", 0.0) or 0.0),
                    "open_timestamp_utc": float(getattr(td, "openTimestamp", 0) or 0) / 1000.0,
                }
            self._open_positions = new_positions

    def _handle_execution_event(self, extracted: oa_msg.ProtoOAExecutionEvent) -> dict[str, object]:
        self._ensure_runtime_buffers()
        exec_type = int(getattr(extracted, "executionType", 0))
        err_code = str(getattr(extracted, "errorCode", "") or "")
        pos = getattr(extracted, "position", None)
        order_pb = getattr(extracted, "order", None)
        deal = getattr(extracted, "deal", None)

        pos_id = int(getattr(pos, "positionId", 0) or 0) if pos is not None else 0
        order_id = int(getattr(order_pb, "orderId", 0) or 0) if order_pb is not None else 0
        client_oid = str(getattr(order_pb, "clientOrderId", "") or "") if order_pb is not None else ""
        exec_px = float(
            (getattr(deal, "executionPrice", 0.0) if deal is not None else 0.0)
            or (getattr(order_pb, "executionPrice", 0.0) if order_pb is not None else 0.0)
            or (getattr(pos, "price", 0.0) if pos is not None else 0.0)
            or 0.0
        )

        rejected = bool(err_code) or exec_type in (
            int(oa_model.ProtoOAExecutionType.ORDER_REJECTED),
            int(oa_model.ProtoOAExecutionType.ORDER_CANCEL_REJECTED),
        )
        payload: dict[str, object] = {
            "ok": not rejected,
            "status": "REJECTED" if rejected else "FILLED",
            "execution_type": exec_type,
            "error_code": err_code,
            "position_id": pos_id,
            "order_id": str(pos_id or order_id or client_oid),
            "client_order_id": client_oid,
            "executed_price": exec_px,
            "timestamp_utc": time.time(),
        }

        with self._lock:
            self._recent_execution_events.append(payload)
            if len(self._recent_execution_events) > 100:
                self._recent_execution_events.pop(0)

            if pos is not None and pos_id > 0:
                pos_status = int(getattr(pos, "positionStatus", 1) or 1)
                # PositionStatus: OPEN=1, CLOSED=2
                if pos_status == 2:
                    self._open_positions.pop(pos_id, None)
                else:
                    td = getattr(pos, "tradeData", None)
                    sid = int(getattr(td, "symbolId", 0)) if td is not None else 0
                    sym = self._symbol_id_to_name.get(sid, str(sid))
                    spec = self._contract_specs.get(sym)
                    lot_units = spec.lot_size if spec else 100_000.0
                    vol_cents = int(getattr(td, "volume", 0)) if td is not None else 0
                    lots = (vol_cents / CTRADER_VOLUME_CENTS_PER_UNIT) / max(lot_units, 1.0)
                    tside = int(getattr(td, "tradeSide", 1)) if td is not None else 1
                    side = Side.BUY if tside == int(oa_model.ProtoOATradeSide.BUY) else Side.SELL
                    self._open_positions[pos_id] = {
                        "position_id": pos_id,
                        "symbol_id": sid,
                        "symbol": sym,
                        "canonical_symbol": self.alias_mapper.canonicalize(sym),
                        "side": side.value,
                        "lots": round(lots, 4),
                        "volume_cents": vol_cents,
                        "entry_price": float(getattr(pos, "price", exec_px) or exec_px),
                        "stop_loss": float(getattr(pos, "stopLoss", 0.0) or 0.0),
                        "take_profit": float(getattr(pos, "takeProfit", 0.0) or 0.0),
                        "open_timestamp_utc": float(getattr(td, "openTimestamp", 0) or 0) / 1000.0,
                    }

            for key in (client_oid, f"pos_{pos_id}" if pos_id > 0 else ""):
                if key and key in self._execution_waiters:
                    ev, res_box = self._execution_waiters.pop(key)
                    res_box.update(payload)
                    ev.set()

        if rejected:
            logger.warning("cTrader execution event rejected: %s (client_oid=%s)", err_code, client_oid)
        return payload

    def get_candles(
        self,
        symbol: str,
        timeframe: str = "M5",
        count: int = 200,
        timeout: float = 2.0,
        cache_ttl_seconds: float = 30.0,
    ) -> list[Candle]:
        """Synchronously retrieve historical candles from cTrader cache or Protobuf request."""
        self._ensure_runtime_buffers()
        norm_tf = timeframe.upper()
        resolved = self.resolve_symbol(symbol) or symbol.upper().strip()
        canon = self.alias_mapper.canonicalize(symbol.upper().strip())

        with self._lock:
            for key_sym in (resolved, symbol.upper().strip(), canon):
                entry = self._trendbars_cache.get((key_sym, norm_tf))
                if entry is not None:
                    ts, bars = entry
                    if len(bars) >= count and (time.time() - ts) <= cache_ttl_seconds:
                        return list(bars[-count:])

        if not self.is_connected or self._client is None:
            return []
        if resolved not in self._symbol_name_to_id:
            return []

        ev = threading.Event()
        with self._lock:
            self._trendbar_waiters.setdefault((resolved, norm_tf), []).append(ev)

        try:
            self.request_trendbars(symbol=resolved, timeframe=norm_tf, count=count)
        except Exception as exc:
            logger.debug("cTrader request_trendbars(%s, %s) failed: %s", resolved, norm_tf, exc)
            return []

        wait_budget = float(timeout) if self._is_reactor_running() else min(float(timeout), 0.02)
        if wait_budget > 0.0:
            ev.wait(wait_budget)

        with self._lock:
            entry = self._trendbars_cache.get((resolved, norm_tf)) or self._trendbars_cache.get((canon, norm_tf))
            if entry is not None:
                return list(entry[1][-count:])
            waiter_list = self._trendbar_waiters.get((resolved, norm_tf), [])
            if ev in waiter_list:
                waiter_list.remove(ev)
        return []

    def get_open_positions(self) -> list[dict[str, object]]:
        """Return currently cached/reconciled open cTrader positions."""
        self._ensure_runtime_buffers()
        with self._lock:
            return [dict(v) for v in self._open_positions.values()]

    def send_new_order(
        self,
        req: oa_msg.ProtoOANewOrderReq | None = None,
        timeout: float = 3.0,
        *,
        symbol: str = "",
        side: Side = Side.BUY,
        volume_lots: float = 0.01,
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
    ) -> dict[str, object]:
        """Dispatch ``ProtoOANewOrderReq`` thread-safely and await execution confirmation when live."""
        self._ensure_runtime_buffers()
        if not self.is_connected or self._client is None:
            return {"ok": False, "status": "TIMEOUT_OR_OFFLINE", "error_code": "CTRADER_OFFLINE"}
        if req is None:
            if not symbol:
                return {"ok": False, "status": "REJECTED", "error_code": "MISSING_ORDER_REQUEST"}
            resolved = self.resolve_symbol(symbol) or symbol.upper()
            sid = self._symbol_name_to_id.get(resolved) or self._symbol_name_to_id.get(symbol.upper())
            if sid is None:
                return {"ok": False, "status": "REJECTED", "error_code": f"UNRESOLVED_SYMBOL:{symbol}"}
            spec = self._contract_specs.get(resolved) or self._contract_specs.get(symbol.upper())
            lot_units = spec.lot_size if spec else 100_000.0
            req = oa_msg.ProtoOANewOrderReq()
            req.ctidTraderAccountId = int(self.account_id)
            req.symbolId = int(sid)
            req.orderType = oa_model.ProtoOAOrderType.MARKET
            req.tradeSide = oa_model.ProtoOATradeSide.BUY if side == Side.BUY else oa_model.ProtoOATradeSide.SELL
            req.volume = max(1, int(round(float(volume_lots) * float(lot_units) * CTRADER_VOLUME_CENTS_PER_UNIT)))
            if stop_loss > 0.0:
                req.stopLoss = float(stop_loss)
            if take_profit > 0.0:
                req.takeProfit = float(take_profit)
            req.clientOrderId = f"apex_{int(time.time() * 1000)}"

        wire_req = oa_msg.ProtoOANewOrderReq()
        wire_req.CopyFrom(req)
        # Spotware MARKET orders allow relativeStopLoss/relativeTakeProfit ONLY (absolute fields only on LIMIT/STOP)
        if wire_req.orderType == oa_model.ProtoOAOrderType.MARKET and type(self._client).__name__ == "Client":
            wire_req.ClearField("stopLoss")
            wire_req.ClearField("takeProfit")

        client_oid = str(getattr(wire_req, "clientOrderId", "") or f"apex_{int(time.time() * 1000)}")
        ev = threading.Event()
        res_box: dict[str, object] = {}
        with self._lock:
            self._execution_waiters[client_oid] = (ev, res_box)

        try:
            self._send_proto(wire_req)
        except Exception as exc:
            with self._lock:
                self._execution_waiters.pop(client_oid, None)
            return {"ok": False, "status": "REJECTED", "error_code": str(exc)}

        reactor_alive = self._is_reactor_running()
        wait_budget = float(timeout) if reactor_alive else min(float(timeout), 0.02)
        if wait_budget > 0.0 and ev.wait(wait_budget):
            return res_box

        with self._lock:
            self._execution_waiters.pop(client_oid, None)

        # Fail closed unless Twisted execution event explicitly confirmed fill/rejection
        return {
            "ok": False,
            "status": "TIMEOUT_OR_OFFLINE",
            "error_code": "CTRADER_EXECUTION_TIMEOUT" if reactor_alive else "CTRADER_REACTOR_OFFLINE",
            "order_id": client_oid,
        }

    def amend_position_sltp(
        self,
        position_id: int,
        stop_loss: float,
        take_profit: float,
        symbol: str = "",
        timeout: float = 3.0,
    ) -> dict[str, object]:
        """Send ``ProtoOAAmendPositionSLTPReq`` for an open cTrader position."""
        self._ensure_runtime_buffers()
        if not self.is_connected or self._client is None:
            return {"ok": False, "status": "REJECTED", "error_code": "CTRADER_OFFLINE"}

        resolved = self.resolve_symbol(symbol) if symbol else ""
        spec = self._contract_specs.get(resolved or symbol.upper()) if symbol else None
        digits = spec.digits if spec else 5

        req = oa_msg.ProtoOAAmendPositionSLTPReq()
        req.ctidTraderAccountId = int(self.account_id)
        req.positionId = int(position_id)
        req.stopLoss = round(float(stop_loss), digits)
        req.takeProfit = round(float(take_profit), digits)

        key = f"pos_{int(position_id)}"
        ev = threading.Event()
        res_box: dict[str, object] = {}
        with self._lock:
            self._execution_waiters[key] = (ev, res_box)

        try:
            self._send_proto(req)
        except Exception as exc:
            with self._lock:
                self._execution_waiters.pop(key, None)
            return {"ok": False, "status": "REJECTED", "error_code": str(exc)}

        reactor_alive = self._is_reactor_running()
        wait_budget = float(timeout) if reactor_alive else min(float(timeout), 0.02)
        if wait_budget > 0.0 and ev.wait(wait_budget):
            return res_box

        with self._lock:
            self._execution_waiters.pop(key, None)
        return {
            "ok": False,
            "status": "TIMEOUT_OR_OFFLINE",
            "error_code": "CTRADER_AMEND_TIMEOUT" if reactor_alive else "CTRADER_REACTOR_OFFLINE",
            "position_id": int(position_id),
        }

    def close_position(
        self,
        position_id: int,
        volume_lots: float,
        symbol: str = "",
        timeout: float = 3.0,
    ) -> dict[str, object]:
        """Send ``ProtoOAClosePositionReq`` for an open cTrader position."""
        self._ensure_runtime_buffers()
        if not self.is_connected or self._client is None:
            return {"ok": False, "status": "REJECTED", "error_code": "CTRADER_OFFLINE"}

        resolved = self.resolve_symbol(symbol) if symbol else ""
        spec = self._contract_specs.get(resolved or symbol.upper()) if symbol else None
        lot_units = spec.lot_size if spec else 100_000.0
        with self._lock:
            cached = self._open_positions.get(int(position_id))
            vol_cents = (
                int(cached["volume_cents"])
                if cached and cached.get("volume_cents")
                else int(round(float(volume_lots) * float(lot_units) * CTRADER_VOLUME_CENTS_PER_UNIT))
            )

        req = oa_msg.ProtoOAClosePositionReq()
        req.ctidTraderAccountId = int(self.account_id)
        req.positionId = int(position_id)
        req.volume = max(1, int(vol_cents))

        key = f"pos_{int(position_id)}"
        ev = threading.Event()
        res_box: dict[str, object] = {}
        with self._lock:
            self._execution_waiters[key] = (ev, res_box)

        try:
            self._send_proto(req)
        except Exception as exc:
            with self._lock:
                self._execution_waiters.pop(key, None)
            return {"ok": False, "status": "REJECTED", "error_code": str(exc)}

        reactor_alive = self._is_reactor_running()
        wait_budget = float(timeout) if reactor_alive else min(float(timeout), 0.02)
        if wait_budget > 0.0 and ev.wait(wait_budget):
            return res_box

        with self._lock:
            self._execution_waiters.pop(key, None)
        return {
            "ok": False,
            "status": "TIMEOUT_OR_OFFLINE",
            "error_code": "CTRADER_CLOSE_TIMEOUT" if reactor_alive else "CTRADER_REACTOR_OFFLINE",
            "position_id": int(position_id),
        }
