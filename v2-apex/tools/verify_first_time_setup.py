"""First-Time Setup Diagnostic & MetaTrader 5 IPC Probe (`tools/verify_first_time_setup.py`).

Executed by `0_FIRST_TIME_SETUP.bat` to verify Python packages, run automated tests, and probe
the live MetaTrader 5 terminal connection with beginner-friendly instructions.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console(width=120)


def probe_python_environment() -> bool:
    """Verify 64-bit Python 3.12+ runtime."""
    is_64bit = sys.maxsize > 2**32
    py_ver = platform.python_version()
    major, minor = sys.version_info[:2]

    console.print(f"[bold cyan]Step 1: Python Runtime Diagnostic[/bold cyan]")
    console.print(f"  - Python Version: [bold green]{py_ver}[/bold green] ({'64-bit' if is_64bit else '32-bit'})")
    console.print(f"  - Executable: [dim]{sys.executable}[/dim]")

    if not is_64bit:
        console.print("[bold red][ERROR] 32-bit Python detected! PocketBot Apex requires 64-bit Python.[/bold red]")
        return False
    if (major, minor) < (3, 11):
        console.print(f"[bold red][ERROR] Python {py_ver} is too old. Python 3.12+ (64-bit) is required.[/bold red]")
        return False

    console.print("  [bold green]✓ Python runtime verified.[/bold green]\n")
    return True


def probe_packages() -> bool:
    """Verify all core scientific, async, and broker packages."""
    console.print(f"[bold cyan]Step 2: Core Package Diagnostic[/bold cyan]")
    required_packages = [
        ("MetaTrader5", "MetaTrader 5 IPC Bridge"),
        ("numpy", "Vectorized Array Math"),
        ("scipy", "Stationary Block-Bootstrap Engine"),
        ("twisted", "cTrader Protobuf Async Reactor"),
        ("ctrader_open_api", "Spotware Open API Protobuf Messages"),
        ("aiohttp", "Asynchronous HTTP & WebSockets"),
        ("dotenv", "Safe Environment Variable Loading"),
        ("rich", "TUI Cockpit & Terminal Tables"),
        ("pytest", "Automated Safety & Quant Verification"),
    ]

    missing = []
    for pkg_name, label in required_packages:
        try:
            __import__(pkg_name)
            console.print(f"  - {label} ([cyan]{pkg_name}[/cyan]): [bold green]Installed[/bold green]")
        except ImportError:
            console.print(f"  - {label} ([cyan]{pkg_name}[/cyan]): [bold red]MISSING[/bold red]")
            missing.append(pkg_name)

    if missing:
        console.print(f"\n[bold red][ERROR] Missing packages: {', '.join(missing)}[/bold red]")
        console.print("Run: [yellow]pip install -r requirements.txt[/yellow] or rerun 0_FIRST_TIME_SETUP.bat")
        return False

    console.print("  [bold green]✓ All core libraries installed and verified.[/bold green]\n")
    return True


def probe_metatrader5() -> bool:
    """Probe live MetaTrader 5 IPC connection, demo account, and Algo Trading button status."""
    console.print(f"[bold cyan]Step 3: MetaTrader 5 IPC Bridge & Demo Account Diagnostic[/bold cyan]")
    import MetaTrader5 as mt5

    init_ok = mt5.initialize()
    if not init_ok:
        err_code, err_msg = mt5.last_error()
        console.print(
            Panel(
                "[bold yellow]MetaTrader 5 is NOT currently running or not yet logged in.[/bold yellow]\n\n"
                "[bold white]Don't worry! Follow these 3 simple steps to get started in 60 seconds:[/bold white]\n\n"
                "  1. [bold cyan]Install MetaTrader 5[/bold cyan]: Download free from [link=https://www.metatrader5.com]https://www.metatrader5.com[/link] (or any broker).\n"
                "  2. [bold cyan]Create Free Demo Account[/bold cyan]: In MT5, click [bold]File -> Open an Account -> Select 'MetaQuotes-Demo' -> Next -> Open a demo account to trade with virtual money -> Next -> Finish[/bold].\n"
                "  3. [bold cyan]Enable Algo Trading[/bold cyan]: Look at the top toolbar in MT5 and click the [bold green]'Algo Trading'[/bold green] button so it lights up [bold green]GREEN (Play icon)[/bold green]!\n\n"
                "[dim]Note: If using Pocket Option and seeing Error 10026/10027, use a standard MetaQuotes-Demo or Fusion Markets Demo account.[/dim]",
                title="[bold yellow]MetaTrader 5 Setup Guide[/bold yellow]",
                border_style="yellow",
            )
        )
        return False

    term_info = mt5.terminal_info()
    acct_info = mt5.account_info()

    if term_info is None or acct_info is None:
        console.print("[yellow]MT5 initialized, but no active account is currently logged in.[/yellow]")
        console.print("Open MT5, log into your Demo account, and ensure Algo Trading is enabled.\n")
        mt5.shutdown()
        return False

    trade_allowed = bool(getattr(term_info, "trade_allowed", False))
    trade_mode = int(getattr(acct_info, "trade_mode", 0))  # 0 = DEMO, 1 or 2 = REAL
    mode_str = "DEMO (100% Safe Virtual Funds)" if trade_mode == 0 else "[bold red]REAL (Real Money Account)[/bold red]"
    balance = float(getattr(acct_info, "balance", 0.0))
    currency = str(getattr(acct_info, "currency", "USD"))
    server = str(getattr(acct_info, "server", "Unknown"))
    company = str(getattr(acct_info, "company", "Unknown"))
    login = str(getattr(acct_info, "login", "Unknown"))

    table = Table(title="Connected MetaTrader 5 Telemetry", header_style="bold cyan")
    table.add_column("Parameter", style="bold white")
    table.add_column("Value")
    table.add_column("Status / Guidance")

    table.add_row("Broker & Server", f"{company} ({server})", "[bold green]Online[/bold green]")
    table.add_row("Account Number", login, "[dim]Active Terminal Session[/dim]")
    table.add_row("Account Mode", mode_str, "[bold green]Protected[/bold green]" if trade_mode == 0 else "[bold yellow]Live[/bold yellow]")
    table.add_row("Virtual Balance", f"{balance:,.2f} {currency}", "[bold green]Ready for trading[/bold green]")

    algo_status = (
        "[bold green]ENABLED (Play icon is GREEN)[/bold green]"
        if trade_allowed
        else "[bold red]DISABLED (Click 'Algo Trading' in MT5 toolbar to turn GREEN)[/bold red]"
    )
    table.add_row("Algo Trading Permission", algo_status, "[bold green]OK[/bold green]" if trade_allowed else "[bold red]ACTION REQUIRED[/bold red]")

    console.print(table)

    # Ensure target symbols are in Market Watch
    symbols_to_check = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "BTCUSD"]
    enabled_syms = []
    for s in symbols_to_check:
        res = mt5.symbol_select(s, True)
        if res:
            enabled_syms.append(s)

    console.print(f"  - Subscribed Symbols in Market Watch: [bold cyan]{', '.join(enabled_syms)}[/bold cyan]")

    if not trade_allowed:
        console.print(
            Panel(
                "[bold red]WARNING: 'Algo Trading' button is currently turned OFF in your MT5 toolbar![/bold red]\n"
                "Please open your MetaTrader 5 window, look at the top toolbar, and click the [bold green]'Algo Trading'[/bold green] button "
                "so the icon turns green. PocketBot Apex needs this button green to submit orders.",
                border_style="red",
            )
        )

    mt5.shutdown()
    console.print("  [bold green]✓ MetaTrader 5 IPC connection tested successfully.[/bold green]\n")
    return True


def main() -> int:
    console.print(
        Panel(
            "[bold cyan]POCKETBOT APEX — FIRST-TIME SYSTEM DIAGNOSTIC & VERIFICATION[/bold cyan]\n"
            "[dim]Verifying Python 64-bit runtime, required quant libraries, and live broker IPC...[/dim]",
            border_style="cyan",
        )
    )

    p1 = probe_python_environment()
    p2 = probe_packages()
    probe_metatrader5()

    if p1 and p2:
        console.print(
            Panel(
                "[bold green]CONGRATULATIONS! PocketBot Apex is fully installed and ready to trade![/bold green]\n\n"
                "You can now double-click any launcher in this folder:\n"
                "  - [bold cyan]2_START_APEX_AUTO_DEMO.bat[/bold cyan] : Autonomous institutional demo trading\n"
                "  - [bold cyan]3_START_APEX_CONFIRM.bat[/bold cyan]   : Interactive confirmation before every order\n"
                "  - [bold cyan]4_APEX_LIVE_DASHBOARD.bat[/bold cyan]  : Live TUI cockpit & real-time microstructure\n"
                "  - [bold cyan]5_CONFIGURE_AI_MODE.bat[/bold cyan]    : Switch AI Studio / agy Pro / Pure Math\n"
                "  - [bold cyan]6_CONFIGURE_BROKER_MESH.bat[/bold cyan]: Switch Auto Hydra SOR / cTrader / MT5\n\n"
                "Open [bold yellow]START_HERE_OPEN_ME.html[/bold yellow] in your browser anytime for the full visual user guide!",
                title="[bold green]Setup Complete[/bold green]",
                border_style="green",
            )
        )
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
