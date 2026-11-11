"""Interactive Zero-Code Broker Mesh Switcher (`tools/configure_broker_mesh.py`).

Displays live connection status, resolved Hydra Mesh Topology, and a numbered menu so the user can
switch broker mesh topologies (`AUTO_HYDRA_MESH`, `CTRADER_EYES_MT5_HANDS`, `SPLIT_EXECUTION`),
run a live Broker Spread/Commission/Ping Shootout (cTrader vs MT5), or launch the 60-second
cTrader OAuth Key Wizard without editing code or JSON files.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from apex.feeds.dual_router import MarketFeedRouter
from apex.feeds.models import Side
from apex.quant.momentum_dom import InstrumentTier, classify_instrument_tier

console = Console(width=125)

CHOICE_TO_MESH_CONFIG: dict[str, dict[str, Any]] = {
    "1": {
        "mesh_mode": "AUTO_HYDRA_MESH",
        "description": "Auto Hydra Mesh (Active-Active SOR when both online; seamless solo fallback)",
    },
    "2": {
        "mesh_mode": "CTRADER_EYES_MT5_HANDS",
        "description": "cTrader Eyes + MT5 Hands (Use free cTrader L2 Order Book to filter MT5 trades)",
    },
    "3": {
        "mesh_mode": "SPLIT_EXECUTION",
        "description": "Split Execution (50/50 parallel risk split across cTrader + MT5)",
    },
}


def apply_mesh_choice(
    choice: str,
    config_path: Path = WORKSPACE_ROOT / "config" / "apex_config.json",
) -> dict[str, Any]:
    """Update `config/apex_config.json` with the selected `mesh_mode`."""
    preset = CHOICE_TO_MESH_CONFIG[choice]
    data = json.loads(config_path.read_text(encoding="utf-8"))
    feeds_cfg = data.setdefault("feeds", {})
    feeds_cfg["mesh_mode"] = preset["mesh_mode"]
    dual_cfg = data.setdefault("filters", {}).setdefault("dual_feed_guard", {})
    dual_cfg["enabled"] = True
    dual_cfg["mesh_mode"] = preset["mesh_mode"]
    config_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return dual_cfg


def render_mesh_status(router: MarketFeedRouter) -> None:
    """Render a Rich status table of both broker venues and the active Hydra Mesh Topology."""
    router.hot_reload_mesh_state()
    ct_online = router.ctrader.is_connected
    mt5_online = router.mt5.is_connected
    ct_creds = bool(router.ctrader.has_credentials)

    table = Table(title="Apex Hydra Dual-Venue Smart Order Router — Live Status", header_style="bold cyan")
    table.add_column("Broker Venue", style="bold white")
    table.add_column("Protocol & Feed Tier")
    table.add_column("Connection State")
    table.add_column("Fee & Latency Profile")

    ct_state = (
        "[bold green]ONLINE (L2 Depth Active)[/bold green]"
        if ct_online
        else (
            "[yellow]CREDENTIALS SET (Connect Ready)[/yellow]"
            if ct_creds
            else "[yellow]READY-ON-KEY (.env unset)[/yellow]"
        )
    )
    mt5_state = "[bold green]ONLINE (IPC Connected)[/bold green]" if mt5_online else "[red]OFFLINE[/red]"

    ct_prof = router.hydra_nbbo.ctrader_profile
    mt5_prof = router.hydra_nbbo.mt5_profile

    table.add_row(
        "cTrader Open API",
        "Spotware Protobuf over TLS TCP (Level 2 DOM + Raw ECN)",
        ct_state,
        f"${ct_prof.commission_per_lot_round_turn_usd:.2f}/lot RT | ~{ct_prof.ping_ms:.0f}ms ping",
    )
    table.add_row(
        "MetaTrader 5 IPC",
        "Local Terminal IPC Bridge (L1 Quotes + Instant Fallback)",
        mt5_state,
        f"${mt5_prof.commission_per_lot_round_turn_usd:.2f}/lot RT | ~{mt5_prof.ping_ms:.0f}ms ping",
    )
    console.print(table)
    console.print(
        Panel(
            f"Configured Mode: [bold cyan]{router.configured_mesh_mode.value}[/bold cyan] | "
            f"Resolved Topology: [bold green]{router.resolved_topology.value}[/bold green]\n"
            f"Dashboard Header Badge: [bold bright_cyan]{router.dashboard_mesh_badge}[/bold bright_cyan]",
            border_style="cyan",
        )
    )


def run_broker_shootout(router: MarketFeedRouter, symbols: Sequence[str] | None = None) -> Table:
    """Run a Live Broker Spread, Commission & Ping Shootout across the symbol universe."""
    router.connect()
    target_symbols = list(symbols or ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "BTCUSD"])

    shootout = Table(
        title="Live Broker Spread, Commission & Ping Shootout (cTrader Raw+$4.50/lot vs MT5 Zero-Comm)",
        header_style="bold magenta",
    )
    shootout.add_column("Symbol", style="bold white")
    shootout.add_column("Tier")
    shootout.add_column("Synthetic NBBO (Bid / Ask)", justify="right")
    shootout.add_column("cTrader All-In (pips)", justify="right")
    shootout.add_column("MT5 All-In (pips)", justify="right")
    shootout.add_column("Winning Venue", style="bold green")
    shootout.add_column("SOR Savings / Lot", justify="right")

    for sym in target_symbols:
        norm = sym.upper()
        tier = classify_instrument_tier(norm)
        spec = router.get_contract_spec(norm)
        pip_size = spec.pip_size if spec else (0.01 if "JPY" in norm or "XAU" in norm else 0.0001)
        one_r_dist = 15.0 * pip_size

        # Evaluate NBBO (if cTrader is offline on this machine, also compute a synthetic Fusion Raw ECN
        # reference quote alongside live MT5 so the user can compare both fee structures side by side)
        mt5_tick = router.mt5.get_tick(norm) if router.mt5.is_connected else None
        ct_tick = router.ctrader.get_tick(norm) if router.ctrader.is_connected else None

        if ct_tick is None and mt5_tick is not None:
            # Build a realistic tight raw-spread cTrader ECN comparison quote around the live MT5 mid
            mid = (mt5_tick.bid + mt5_tick.ask) / 2.0
            raw_half_spread = 0.10 * pip_size
            from apex.feeds.models import Tick

            synthetic_ct_tick = Tick(
                symbol=norm,
                bid=mid - raw_half_spread,
                ask=mid + raw_half_spread,
                timestamp_utc=mt5_tick.timestamp_utc,
                source="ctrader_sim",
            )
            nbbo = router.hydra_nbbo.evaluate_symbol(
                symbol=norm,
                configured_mode=router.mesh_mode,
                ctrader_tick=synthetic_ct_tick,
                mt5_tick=mt5_tick,
                pip_size=pip_size,
                one_r_price_distance=one_r_dist,
            )
            ct_suffix = " (Raw ECN Est.)"
        else:
            nbbo = router.evaluate_nbbo(norm, one_r_price_distance=one_r_dist, side=Side.BUY)
            ct_suffix = ""

        nbbo_px = (
            f"{nbbo.best_bid:.5f} / {nbbo.best_ask:.5f}"
            if nbbo.best_bid < 10.0
            else f"{nbbo.best_bid:.2f} / {nbbo.best_ask:.2f}"
        )
        ct_str = f"{nbbo.ctrader_quote.effective_cost_pips:.2f}p{ct_suffix}" if nbbo.ctrader_quote else "OFFLINE"
        mt5_str = f"{nbbo.mt5_quote.effective_cost_pips:.2f}p" if nbbo.mt5_quote else "OFFLINE"
        sav_str = f"+{nbbo.pips_saved:.2f}p (${nbbo.usd_saved_per_lot:.2f}/lot)"

        shootout.add_row(
            norm,
            tier.value.replace("_SPREAD_", " "),
            nbbo_px,
            ct_str,
            mt5_str,
            nbbo.winning_venue.upper(),
            sav_str,
        )

    console.print(shootout)
    return shootout


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PocketBot Apex — Interactive Zero-Code Broker Mesh Switcher")
    parser.add_argument("--config", default=str(WORKSPACE_ROOT / "config" / "apex_config.json"))
    parser.add_argument("--choice", choices=["1", "2", "3", "4", "5"], help="Non-interactive menu choice")
    args = parser.parse_args(argv)

    cfg_path = Path(args.config)
    router = MarketFeedRouter(config_path=cfg_path)
    router.connect()
    try:
        render_mesh_status(router)

        console.print("\n[bold cyan]Select Broker Mesh Operating Mode or Diagnostic Action:[/bold cyan]")
        console.print("  [bold][1][/bold] Auto Hydra Mesh (Active-Active SOR when both online; seamless solo fallback)")
        console.print("  [bold][2][/bold] cTrader Eyes + MT5 Hands (Use free cTrader L2 Order Book to filter MT5 trades)")
        console.print("  [bold][3][/bold] Split Execution (50/50 parallel risk split across cTrader + MT5)")
        console.print("  [bold][4][/bold] Run Live Broker Spread, Commission & Ping Shootout (cTrader vs MT5)")
        console.print("  [bold][5][/bold] Launch cTrader OAuth Key Wizard (connect Fusion Markets cTrader in 60s)")
        console.print("  [bold][Q][/bold] Exit without changes\n")

        choice = args.choice
        if not choice:
            choice = input("Enter choice [1-5 or Q]: ").strip().upper()

        if choice in ("Q", ""):
            console.print("[dim]No changes made.[/dim]")
            return 0

        if choice in CHOICE_TO_MESH_CONFIG:
            updated = apply_mesh_choice(choice, config_path=cfg_path)
            console.print(
                f"\n[bold green]Updated config/apex_config.json -> mesh_mode={updated['mesh_mode']}[/bold green]"
            )
            render_mesh_status(router)
            return 0

        if choice == "4":
            run_broker_shootout(router)
            return 0

        if choice == "5":
            py_exe = WORKSPACE_ROOT / ".venv" / "Scripts" / "python.exe"
            wizard = WORKSPACE_ROOT / "tools" / "setup_ctrader_oauth.py"
            return subprocess.call([str(py_exe), str(wizard)])

        console.print(f"[red]Unrecognized option: {choice}[/red]")
        return 1
    finally:
        router.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
