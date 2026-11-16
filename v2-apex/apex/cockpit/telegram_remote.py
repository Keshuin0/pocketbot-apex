"""Async Telegram Remote Controller (`apex/cockpit/telegram_remote.py`) for PocketBot Apex.

Activated automatically when `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are configured in `.env`.
Supports:
  - Formatted HTML/Markdown trade execution & risk cards.
  - Interactive remote commands:
      * `/status`: Returns live feed status, daily P/L, and open positions count.
      * `/quarantine`: Lists per-symbol rolling 20-trade PF and `ACTIVE` vs `PAPER_SHADOW` states.
      * `/pause`: Toggles scanning pause state.
      * `/panic_flatten`: Immediately closes all open Demo/Live positions across cTrader & MT5
        and halts scanning!
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import TYPE_CHECKING, Any, Callable

import aiohttp
from dotenv import load_dotenv

from apex.ledger.walkforward_db import WalkForwardLedgerDB

if TYPE_CHECKING:
    from apex.execution.router import DualBrokerExecutionRouter

logger = logging.getLogger(__name__)


class TelegramRemoteController:
    """Optional Async Telegram Bot Controller (`apex/cockpit/telegram_remote.py`)."""

    API_BASE: str = "https://api.telegram.org/bot{token}/{method}"
    _MD2_RE = re.compile(r"([_\*\[\]\(\)~`>#\+\-=\|\{\}\.!\\])")

    def __init__(
        self,
        execution_router: DualBrokerExecutionRouter,
        ledger_db: WalkForwardLedgerDB,
        bot_token: str | None = None,
        chat_id: str | None = None,
        mode_change_callback: Callable[[str], Any] | None = None,
    ) -> None:
        load_dotenv()
        self.execution_router: DualBrokerExecutionRouter = execution_router
        self.ledger_db: WalkForwardLedgerDB = ledger_db
        self.bot_token: str = (bot_token if bot_token is not None else os.environ.get("TELEGRAM_BOT_TOKEN", "")).strip()
        self.chat_id: str = (chat_id if chat_id is not None else os.environ.get("TELEGRAM_CHAT_ID", "")).strip()
        self.scanning_paused: bool = False
        self.scanning_halted: bool = False
        self.execution_mode: str = "demo-auto"
        self.mode_change_callback = mode_change_callback
        self._last_update_id: int = 0

    @classmethod
    def escape_markdown_v2(cls, text: str) -> str:
        """Escape all Telegram MarkdownV2 reserved characters cleanly."""
        return cls._MD2_RE.sub(r"\\\1", str(text or ""))

    @property
    def is_configured(self) -> bool:
        """Return True if both TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are present."""
        return bool(self.bot_token and self.chat_id)

    @staticmethod
    def format_trade_card(
        symbol: str,
        side: str,
        lots: float,
        entry: float,
        sl: float,
        tp: float,
        net_avg_r: float,
        cvar_95_r: float,
        ai_decision: str,
        venue: str,
    ) -> str:
        """Format a structured trade execution notification card."""
        return (
            f"⚡ *POCKETBOT APEX — TRADE CARD*\n"
            f"• *Symbol:* `{symbol}` ({side.upper()})\n"
            f"• *Size:* `{lots:.2f} lots` | *Venue:* `{venue}`\n"
            f"• *Entry:* `{entry:.5f}` | *SL:* `{sl:.5f}` | *TP:* `{tp:.5f}`\n"
            f"• *TimeMachine v2:* `net_avg_R = {net_avg_r:+.3f}R` | `CVaR95 = {cvar_95_r:+.2f}R`\n"
            f"• *AI Veto Committee:* `{ai_decision}`"
        )

    def handle_command(self, text: str) -> str:
        """Process `/status`, `/positions`, `/daily`, `/quarantine`, `/pause`, `/resume`, `/mode`, `/panic`, `/help`."""
        parts = text.strip().split()
        raw_cmd = parts[0].lower() if parts else ""
        cmd = raw_cmd.split("@")[0]

        if cmd == "/status":
            src = self.execution_router.feed_router.active_source
            state_lbl = (
                "HALTED (PANIC)"
                if self.scanning_halted
                else ("PAUSED" if self.scanning_paused else "RUNNING")
            )
            shadow_cnt = len(self.execution_router._shadow_positions)
            return (
                f"📡 *PocketBot Apex Status*\n"
                f"• Engine State: `{state_lbl}` (Mode: `{self.execution_mode}`)\n"
                f"• Active Feed: `{src.upper()}`\n"
                f"• Equity: `${self.execution_router.account_equity_usd:,.2f}`\n"
                f"• Shadow Positions: `{shadow_cnt}`"
            )

        if cmd == "/positions":
            lines: list[str] = ["📊 *Open Positions*"]
            shadows = getattr(self.execution_router, "_shadow_positions", []) or []
            for p in shadows:
                sym = getattr(p, "symbol", None) or (p.get("symbol") if isinstance(p, dict) else "UNKNOWN")
                side = getattr(p, "side", None) or (p.get("side") if isinstance(p, dict) else "")
                side_str = side.value if hasattr(side, "value") else str(side).upper()
                lots = float(getattr(p, "lots", 0.0) or (p.get("lots", 0.0) if isinstance(p, dict) else 0.0))
                entry = float(getattr(p, "entry_price", 0.0) or (p.get("entry_price", 0.0) if isinstance(p, dict) else 0.0))
                lines.append(f"• `{sym}` {side_str} `{lots:.2f}` @ `{entry:.5f}` [SHADOW]")
            get_live = getattr(self.execution_router, "list_open_positions", None)
            if callable(get_live):
                try:
                    for lp in get_live() or []:
                        lsym = getattr(lp, "symbol", None) or (lp.get("symbol") if isinstance(lp, dict) else "UNKNOWN")
                        llots = float(getattr(lp, "lots", 0.0) or (lp.get("lots", 0.0) if isinstance(lp, dict) else 0.0))
                        lprofit = float(getattr(lp, "profit", 0.0) or (lp.get("profit", 0.0) if isinstance(lp, dict) else 0.0))
                        lines.append(f"• `{lsym}` `{llots:.2f}` PnL `{lprofit:+.2f}`")
                except Exception:
                    pass
            if len(lines) == 1:
                lines.append("• No open positions.")
            return "\n".join(lines)

        if cmd == "/daily":
            eq = float(getattr(self.execution_router, "account_equity_usd", 0.0) or 0.0)
            daily_pnl = float(getattr(self.execution_router, "daily_pnl_usd", 0.0) or 0.0)
            dd_pct = (daily_pnl / eq * 100.0) if eq > 0.0 else 0.0
            closed_cnt = 0
            if hasattr(self.ledger_db, "compute_mfe_mae_capture"):
                try:
                    mfe_stats = self.ledger_db.compute_mfe_mae_capture()
                    closed_cnt = int(mfe_stats.get("closed_count", 0))
                except Exception:
                    pass
            return (
                f"📈 *Daily Risk & Performance*\n"
                f"• Equity: `${eq:,.2f}`\n"
                f"• Daily PnL: `${daily_pnl:+,.2f}` (`{dd_pct:+.2f}%`)\n"
                f"• Closed Trades Recorded: `{closed_cnt}`"
            )

        if cmd == "/quarantine":
            states = self.ledger_db.list_all_quarantine_states()
            if not states:
                return "🛡️ *Quarantine Ledger:* All symbols `ACTIVE` (no demotions recorded)."
            lines = ["🛡️ *Symbol Quarantine Ledger:*"]
            for st in states:
                lines.append(
                    f"• `{st.symbol}`: *{st.mode.value}* (Trades={st.window_trades}, "
                    f"PF={st.profit_factor:.2f}, RealEV={st.mean_realized_r:+.2f}R vs Pred={st.mean_predicted_r:+.2f}R)"
                )
            return "\n".join(lines)

        if cmd == "/pause":
            self.scanning_paused = not self.scanning_paused
            return (
                "⏸️ *Scanning PAUSED.* Send `/resume` or `/pause` to resume."
                if self.scanning_paused
                else "▶️ *Scanning RESUMED.*"
            )

        if cmd == "/resume":
            self.scanning_paused = False
            self.scanning_halted = False
            if self.mode_change_callback is not None:
                try:
                    self.mode_change_callback("resume")
                except Exception:
                    pass
            return "▶️ *Scanning RESUMED.* Execution checks re-armed."

        if cmd == "/mode":
            arg = parts[1].lower().strip() if len(parts) > 1 else ""
            if arg not in {"auto", "demo-auto", "confirm", "halt"}:
                return "⚙️ Usage: `/mode <auto|confirm|halt>`"
            if arg == "halt":
                self.scanning_paused = True
                self.scanning_halted = True
                self.execution_mode = "halt"
                if self.mode_change_callback is not None:
                    self.mode_change_callback("halt")
                return "🛑 Mode set to *HALT*. Scanning paused."
            mapped = "demo-auto" if arg == "auto" else arg
            self.execution_mode = mapped
            self.scanning_paused = False
            self.scanning_halted = False
            if self.mode_change_callback is not None:
                self.mode_change_callback(mapped)
            return f"⚙️ Execution mode updated to `{mapped}`."

        if cmd in ("/panic_flatten", "/panic"):
            self.scanning_paused = True
            self.scanning_halted = True
            summary = self.execution_router.panic_flatten_all()
            return (
                "🚨 *PANIC FLATTEN EXECUTED* 🚨\n"
                f"• MT5 Positions Closed: `{summary['mt5_closed']}`\n"
                f"• cTrader Positions Closed: `{summary['ctrader_closed']}`\n"
                f"• Paper-Shadow Positions Cleared: `{summary['shadow_closed']}`\n"
                "• Scanning is now *HALTED*."
            )

        if cmd == "/help":
            return (
                "🤖 *PocketBot Apex Remote Commands*\n"
                "• `/status` — Feed, mode, equity & shadow count\n"
                "• `/positions` — Active open positions & PnL\n"
                "• `/daily` — Daily PnL, equity & drawdown %\n"
                "• `/quarantine` — Symbol PF & quarantine state\n"
                "• `/pause` / `/resume` — Toggle or resume scanning\n"
                "• `/mode <auto|confirm|halt>` — Switch execution mode\n"
                "• `/panic` (`/panic_flatten`) — Emergency flatten all & halt"
            )

        return (
            "Available commands: `/status`, `/positions`, `/daily`, `/quarantine`, "
            "`/pause`, `/resume`, `/mode <auto|confirm|halt>`, `/panic`, `/help`"
        )

    def process_update(self, update: dict[str, Any]) -> str | None:
        """Authorize incoming Bot API update strictly by configured `chat_id` and dispatch command."""
        try:
            upd_id = int(update.get("update_id", 0))
            if upd_id >= self._last_update_id:
                self._last_update_id = upd_id + 1
        except (TypeError, ValueError):
            pass

        msg = update.get("message") or update.get("edited_message")
        if not isinstance(msg, dict):
            return None
        chat = msg.get("chat") or {}
        incoming_chat_id = str(chat.get("id", "") or "").strip()
        if not self.chat_id or incoming_chat_id != self.chat_id:
            logger.warning(
                "Rejected unauthorized Telegram command from chat_id=%r (expected=%r)",
                incoming_chat_id,
                self.chat_id,
            )
            return None
        text = str(msg.get("text", "") or "").strip()
        if not text.startswith("/"):
            return None
        return self.handle_command(text)

    async def send_message_async(
        self,
        text: str,
        parse_mode: str = "MarkdownV2",
        pre_escaped: bool = False,
    ) -> bool:
        """Send a Telegram message safely using MarkdownV2 escaping when enabled."""
        if not self.is_configured:
            return False
        url = self.API_BASE.format(token=self.bot_token, method="sendMessage")
        body_text = (
            self.escape_markdown_v2(text)
            if (parse_mode == "MarkdownV2" and not pre_escaped)
            else text
        )
        payload: dict[str, Any] = {"chat_id": self.chat_id, "text": body_text}
        if parse_mode:
            payload["parse_mode"] = parse_mode
        try:
            timeout = aiohttp.ClientTimeout(total=5.0)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as resp:
                    return resp.status == 200
        except Exception as exc:
            logger.debug("Telegram sendMessage failed: %s", exc)
            return False

    async def poll_updates_loop(
        self,
        stop_event: asyncio.Event | None = None,
        poll_timeout: int = 20,
        max_iterations: int | None = None,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Long-poll Telegram `getUpdates`, enforce strict `chat_id` auth, and reply."""
        if not self.is_configured:
            return
        owns_session = session is None
        client_timeout = aiohttp.ClientTimeout(total=float(poll_timeout) + 10.0)
        active_session = session or aiohttp.ClientSession(timeout=client_timeout)
        iterations = 0
        try:
            while True:
                if stop_event is not None and stop_event.is_set():
                    break
                if max_iterations is not None and iterations >= max_iterations:
                    break
                iterations += 1
                url = self.API_BASE.format(token=self.bot_token, method="getUpdates")
                params = {
                    "offset": self._last_update_id,
                    "timeout": int(poll_timeout),
                }
                try:
                    async with active_session.get(url, params=params) as resp:
                        if resp.status != 200:
                            await asyncio.sleep(1.0)
                            continue
                        data = await resp.json()
                    for upd in data.get("result", []) or []:
                        reply = self.process_update(upd)
                        if reply:
                            await self.send_message_async(reply)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.debug("Telegram getUpdates poll error: %s", exc)
                    await asyncio.sleep(1.0)
        finally:
            if owns_session:
                await active_session.close()
