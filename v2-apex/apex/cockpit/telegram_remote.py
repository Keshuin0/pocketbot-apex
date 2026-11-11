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

import logging
import os
from typing import Callable

import aiohttp
from dotenv import load_dotenv

from apex.execution.router import DualBrokerExecutionRouter
from apex.ledger.walkforward_db import WalkForwardLedgerDB

logger = logging.getLogger(__name__)


class TelegramRemoteController:
    """Optional Async Telegram Bot Controller (`apex/cockpit/telegram_remote.py`)."""

    API_BASE: str = "https://api.telegram.org/bot{token}/{method}"

    def __init__(
        self,
        execution_router: DualBrokerExecutionRouter,
        ledger_db: WalkForwardLedgerDB,
        bot_token: str | None = None,
        chat_id: str | None = None,
    ) -> None:
        load_dotenv()
        self.execution_router: DualBrokerExecutionRouter = execution_router
        self.ledger_db: WalkForwardLedgerDB = ledger_db
        self.bot_token: str = (bot_token if bot_token is not None else os.environ.get("TELEGRAM_BOT_TOKEN", "")).strip()
        self.chat_id: str = (chat_id if chat_id is not None else os.environ.get("TELEGRAM_CHAT_ID", "")).strip()
        self.scanning_paused: bool = False
        self.scanning_halted: bool = False
        self._last_update_id: int = 0

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
        """Process `/status`, `/quarantine`, `/pause`, or `/panic_flatten` deterministically."""
        cmd = text.strip().split()[0].lower() if text.strip() else ""

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
                f"• Engine State: `{state_lbl}`\n"
                f"• Active Feed: `{src.upper()}`\n"
                f"• Equity: `${self.execution_router.account_equity_usd:,.2f}`\n"
                f"• Shadow Positions: `{shadow_cnt}`"
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
                "⏸️ *Scanning PAUSED.* Send `/pause` again to resume."
                if self.scanning_paused
                else "▶️ *Scanning RESUMED.*"
            )

        if cmd == "/panic_flatten":
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

        return "Available commands: `/status`, `/quarantine`, `/pause`, `/panic_flatten`"

    async def send_message_async(self, text: str) -> bool:
        """Send a Telegram message if `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are set."""
        if not self.is_configured:
            return False
        url = self.API_BASE.format(token=self.bot_token, method="sendMessage")
        payload = {"chat_id": self.chat_id, "text": text, "parse_mode": "Markdown"}
        try:
            timeout = aiohttp.ClientTimeout(total=5.0)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as resp:
                    return resp.status == 200
        except Exception as exc:
            logger.debug("Telegram sendMessage failed: %s", exc)
            return False
