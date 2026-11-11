"""Rich Live Terminal Dashboard (`apex/cockpit/tui_dashboard.py`) for PocketBot Apex.

Displays:
  - Active Dual-Broker Feed Status (cTrader Protobuf TCP + MT5 Hot-Standby)
  - Live 3-Tier AI Brain Status & Fallback Chain (`AI Brain: AGY_PRO [Auto -> Pure Math fallback]`)
  - Live Regime per symbol (`TRENDING_EXPANSION`, `MEAN_REVERTING_CHOP`, `VOLATILITY_SHOCK`)
  - Visual Level 2 DOM OBI meter (`[-1.0, +1.0]`)
  - Time Machine `net_avg_R` & Quarter-Kelly risk %
  - Open positions, Daily P/L ($ and %), and Auto-Quarantine states (`ACTIVE` vs `PAPER_SHADOW`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table


@dataclass(frozen=True, slots=True)
class SymbolCockpitRow:
    """Single symbol telemetry row for the Rich TUI Cockpit."""

    symbol: str
    tier: str
    timeframe: str
    bid: float
    ask: float
    spread_pips: float
    regime: str
    hurst: float
    kaufman_er: float
    obi: float | None
    net_avg_r: float | None
    rec_risk_pct: float | None
    ai_verdict: str
    quarantine_mode: str
    profit_factor: float
    best_venue: str = "MT5"


class ApexTUIDashboard:
    """Rich Terminal Dashboard for PocketBot Apex."""

    def __init__(self, console: Console | None = None) -> None:
        self.console: Console = console or Console(width=145)

    @staticmethod
    def format_obi_meter(obi: float | None) -> str:
        """Render an ASCII visual meter for Order-Book Imbalance in [-1.0, +1.0]."""
        if obi is None:
            return "[dim]N/A (MT5 Standby)[/dim]"
        clamped = max(-1.0, min(1.0, float(obi)))
        bars = int(round(abs(clamped) * 5))
        if clamped >= 0.15:
            return f"[bold green]{clamped:+.2f} {'█' * bars}{'·' * (5 - bars)}[/bold green]"
        if clamped <= -0.15:
            return f"[bold red]{clamped:+.2f} {'█' * bars}{'·' * (5 - bars)}[/bold red]"
        return f"[yellow]{clamped:+.2f} {'▪' * bars}{'·' * (5 - bars)}[/yellow]"

    @staticmethod
    def format_regime_badge(regime: str) -> str:
        if "TRENDING" in regime:
            return "[bold green]TRENDING_EXPANSION[/bold green]"
        if "SHOCK" in regime:
            return "[bold white on red]VOLATILITY_SHOCK[/bold white on red]"
        return "[yellow]MEAN_REVERTING_CHOP[/yellow]"

    @staticmethod
    def format_quarantine_badge(mode: str, pf: float) -> str:
        if mode == "PAPER_SHADOW":
            return f"[bold magenta]PAPER_SHADOW (PF {pf:.2f})[/bold magenta]"
        return f"[bold cyan]ACTIVE (PF {pf:.2f})[/bold cyan]"

    def build_renderable(
        self,
        run_mode: str,
        active_feed: str,
        ctrader_connected: bool,
        mt5_connected: bool,
        account_equity_usd: float,
        daily_pnl_usd: float,
        open_positions_count: int,
        rows: Sequence[SymbolCockpitRow],
        ai_brain_badge: str = "AI Brain: AGY_PRO [Auto -> Pure Math fallback]",
        mesh_badge: str = "Mesh: HYDRA_AUTO [MT5 Active | cTrader Ready-On-Key]",
        sor_saved_pips: float = 0.0,
        sor_saved_usd: float = 0.0,
        universe_scanned_count: int = 0,
        universe_total_count: int = 0,
        universe_badge: str | None = None,
    ) -> Group:
        """Build the composite Rich Dashboard layout."""
        dd_pct = (daily_pnl_usd / account_equity_usd * 100.0) if account_equity_usd > 0 else 0.0
        pnl_style = "bold green" if daily_pnl_usd >= 0 else "bold red"

        ct_badge = "[bold green]ONLINE[/bold green]" if ctrader_connected else "[yellow]STANDBY (.env unset)[/yellow]"
        mt5_badge = "[bold green]ONLINE[/bold green]" if mt5_connected else "[red]OFFLINE[/red]"

        universe_str = ""
        if universe_badge:
            universe_str = f"\n[bold bright_yellow]{universe_badge}[/bold bright_yellow]"
        elif universe_total_count > 0:
            closed = max(0, universe_total_count - universe_scanned_count)
            universe_str = (
                f"\n[bold bright_yellow]Universe: {universe_scanned_count}/{universe_total_count} Symbols Scanned "
                f"({closed} Closed) | Top Active Candidates Shown[/bold bright_yellow]"
            )

        header_text = (
            f"[bold cyan]POCKETBOT APEX v1.0[/bold cyan] | Mode: [bold]{run_mode.upper()}[/bold] | "
            f"[bold bright_cyan]{mesh_badge}[/bold bright_cyan] (cTrader: {ct_badge} | MT5: {mt5_badge}) | "
            f"SOR Saved: [bold green]+{sor_saved_pips:.1f}p (${sor_saved_usd:,.2f})[/bold green]\n"
            f"[bold magenta]{ai_brain_badge}[/bold magenta] | "
            f"Equity: [bold]${account_equity_usd:,.2f}[/bold] | "
            f"Daily P/L: [{pnl_style}]${daily_pnl_usd:+,.2f} ({dd_pct:+.2f}%)[/{pnl_style}] | "
            f"Open Positions: [bold]{open_positions_count}/2[/bold]"
            f"{universe_str}"
        )
        header_panel = Panel(header_text, border_style="cyan", padding=(0, 1))

        table = Table(
            title="Real-Time Hydra NBBO, Microstructure, Regime, Time Machine v2 & Auto-Quarantine Matrix",
            expand=True,
            header_style="bold cyan",
        )
        table.add_column("Symbol", style="bold white")
        table.add_column("Tier / TF")
        table.add_column("Bid / Ask", justify="right")
        table.add_column("Spread", justify="right")
        table.add_column("Best Venue")
        table.add_column("Market Regime")
        table.add_column("Hurst / ER", justify="right")
        table.add_column("L2 DOM OBI")
        table.add_column("TM net_avg_R", justify="right")
        table.add_column("Q-Kelly Risk", justify="right")
        table.add_column("AI Veto")
        table.add_column("Ledger State")

        for r in rows:
            price_fmt = f"{r.bid:.5f} / {r.ask:.5f}" if r.bid < 10.0 else f"{r.bid:.2f} / {r.ask:.2f}"
            net_r_str = (
                f"[green]{r.net_avg_r:+.3f}R[/green]"
                if (r.net_avg_r is not None and r.net_avg_r > 0)
                else (f"[red]{r.net_avg_r:+.3f}R[/red]" if r.net_avg_r is not None else "[dim]-[/dim]")
            )
            kelly_str = f"{r.rec_risk_pct:.2f}%" if r.rec_risk_pct is not None else "[dim]-[/dim]"
            ai_str = (
                f"[bold green]{r.ai_verdict}[/bold green]"
                if r.ai_verdict.startswith("APPROVE")
                else (
                    f"[bold red]{r.ai_verdict}[/bold red]"
                    if r.ai_verdict.startswith("VETO")
                    else f"[dim]{r.ai_verdict}[/dim]"
                )
            )
            venue_style = "bold green" if "CTRADER" in r.best_venue.upper() else "bold bright_blue"
            table.add_row(
                r.symbol,
                f"{r.tier.replace('_SPREAD_', ' ')} ({r.timeframe})",
                price_fmt,
                f"{r.spread_pips:.1f}p",
                f"[{venue_style}]{r.best_venue}[/{venue_style}]",
                self.format_regime_badge(r.regime),
                f"H={r.hurst:.2f} | ER={r.kaufman_er:.2f}",
                self.format_obi_meter(r.obi),
                net_r_str,
                kelly_str,
                ai_str,
                self.format_quarantine_badge(r.quarantine_mode, r.profit_factor),
            )

        return Group(header_panel, table)

    def print_snapshot(
        self,
        run_mode: str,
        active_feed: str,
        ctrader_connected: bool,
        mt5_connected: bool,
        account_equity_usd: float,
        daily_pnl_usd: float,
        open_positions_count: int,
        rows: Sequence[SymbolCockpitRow],
        ai_brain_badge: str = "AI Brain: AGY_PRO [Auto -> Pure Math fallback]",
        mesh_badge: str = "Mesh: HYDRA_AUTO [MT5 Active | cTrader Ready-On-Key]",
        sor_saved_pips: float = 0.0,
        sor_saved_usd: float = 0.0,
        universe_scanned_count: int = 0,
        universe_total_count: int = 0,
        universe_badge: str | None = None,
    ) -> None:
        """Render a static Rich snapshot table to the terminal."""
        group = self.build_renderable(
            run_mode=run_mode,
            active_feed=active_feed,
            ctrader_connected=ctrader_connected,
            mt5_connected=mt5_connected,
            account_equity_usd=account_equity_usd,
            daily_pnl_usd=daily_pnl_usd,
            open_positions_count=open_positions_count,
            rows=rows,
            ai_brain_badge=ai_brain_badge,
            mesh_badge=mesh_badge,
            sor_saved_pips=sor_saved_pips,
            sor_saved_usd=sor_saved_usd,
            universe_scanned_count=universe_scanned_count,
            universe_total_count=universe_total_count,
            universe_badge=universe_badge,
        )
        self.console.print(group)
