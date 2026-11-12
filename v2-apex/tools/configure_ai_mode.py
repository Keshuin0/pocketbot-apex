"""Interactive Zero-Code AI Router Switcher (`tools/configure_ai_mode.py`).

Displays live detection status of:
  1. Google AI Studio API Key (`studio_api`) in `.env`
  2. Google AI Pro CLI (`agy_pro` via `agy.exe`)
  3. Deterministic Pure Math Engine (`pure_math`)

Provides a clean numbered menu:
  [1] Auto Priority: AI Studio API -> Google AI Pro -> Pure Math (Default)
  [2] Auto Priority: Google AI Pro -> AI Studio API -> Pure Math
  [3] Force Pure Math Only (Disable AI Veto completely)
  [4] Add / Update / Clear Google AI Studio API Key in .env
  [5] Run Live Test of Current AI Waterfall Right Now
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Sequence

from dotenv import dotenv_values, set_key
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from apex.ai.veto_committee import (
    AIProviderName,
    AIVetoCommittee,
    AIVetoMode,
    CandidateSetupAuditContext,
    MODE_DEFAULT_PRIORITY,
)
from apex.feeds.models import Side

console = Console(width=120)


def update_ai_veto_config(
    config_path: Path,
    mode: str,
    enabled: bool = True,
    priority_order: Sequence[str] | None = None,
) -> dict[str, object]:
    """Update `filters.ai_veto` in `config/apex_config.json` atomically for live hot-reload."""
    cfg = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    filters = cfg.setdefault("filters", {})
    ai_cfg = filters.setdefault("ai_veto", {})

    resolved_order = list(
        priority_order
        if priority_order is not None
        else MODE_DEFAULT_PRIORITY.get(mode, MODE_DEFAULT_PRIORITY[AIVetoMode.AUTO_STUDIO_FIRST.value])
    )
    ai_cfg["enabled"] = bool(enabled)
    ai_cfg["mode"] = str(mode)
    ai_cfg["priority_order"] = resolved_order
    ai_cfg.setdefault("studio_timeout_seconds", 4.0)
    ai_cfg.setdefault("agy_pro_timeout_seconds", 15.0)

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return ai_cfg


def update_studio_api_key_in_env(env_path: Path, new_api_key: str) -> None:
    """Safely add, update, or clear `GEMINI_API_KEY` in `.env`."""
    if not env_path.exists():
        env_path.write_text("", encoding="utf-8")
    _ = dotenv_values(env_path)
    set_key(str(env_path), "GEMINI_API_KEY", new_api_key.strip())


def render_status_panel(
    committee: AIVetoCommittee,
    probe_cli: bool = False,
) -> None:
    """Render live detection status of Google AI Studio, Google AI Pro (`agy.exe`), and Pure Math."""
    status = committee.get_router_status(probe_agy_auth=probe_cli)

    table = Table(
        title="PocketBot Apex — Hybrid 3-Tier AI Router Live Discovery",
        header_style="bold cyan",
        expand=True,
    )
    table.add_column("Tier", style="bold white")
    table.add_column("Provider ID")
    table.add_column("Live Status")
    table.add_column("Details & Endpoint / Binary")

    for idx, st in enumerate((status.studio_status, status.agy_pro_status, status.pure_math_status), start=1):
        badge = "[bold green]READY[/bold green]" if st.ready else "[yellow]UNAVAILABLE (Auto-Skip)[/yellow]"
        if st.provider == status.active_primary:
            badge += " [bold cyan]<-- ACTIVE PRIMARY[/bold cyan]"
        elif st.provider in status.fallback_chain:
            badge += " [dim](Fallback)[/dim]"
        table.add_row(f"Tier {idx}", st.provider, badge, st.detail)

    console.print(table)
    console.print(
        Panel(
            f"[bold]Configured Mode:[/bold] `{status.mode}` | "
            f"[bold]Priority Order:[/bold] `{' -> '.join(status.priority_order)}`\n"
            f"[bold magenta]{status.dashboard_badge}[/bold magenta]",
            border_style="cyan",
        )
    )


def run_live_waterfall_test(committee: AIVetoCommittee) -> int:
    """Run an immediate live audit through the active AI waterfall and display the result."""
    sample_ctx = CandidateSetupAuditContext(
        symbol="USDJPY",
        side=Side.BUY,
        timeframe="M5",
        entry_price=159.740,
        stop_loss=159.540,
        take_profit=160.140,
        regime="TRENDING_EXPANSION",
        hurst_exponent=0.58,
        kaufman_er=0.54,
        volatility_percentile=64.0,
        dom_obi=0.24,
        spread_pct_of_sl=0.05,
        net_avg_R=0.871,
        prob_profit=0.61,
        cvar_95_R=-1.08,
        recommended_risk_pct=0.50,
        session_utc_iso=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
    console.print("\n[bold cyan]Running live test of current AI waterfall on sample USDJPY BUY setup...[/bold cyan]")
    decision = committee.evaluate_sync(sample_ctx)
    console.print(
        Panel(
            f"[bold]Decision:[/bold] `{decision.decision}` (Confidence: `{decision.confidence:.2f}`)\n"
            f"[bold]Provider Used:[/bold] `{decision.provider_used}`"
            + (f" (`{decision.model_used}`)" if decision.model_used else "")
            + "\n"
            f"[bold]Cascade Path:[/bold] `{' -> '.join(decision.cascade_path)}`\n"
            f"[bold]Latency:[/bold] `{decision.latency_ms:.1f} ms`\n"
            f"[bold]Reason:[/bold] {decision.reason}",
            title="Live AI Waterfall Test Verdict",
            border_style="green" if decision.decision == "APPROVE" else "red",
        )
    )
    return 0


def inspect_discovered_models(committee: AIVetoCommittee) -> None:
    """Display all dynamically discovered Gemini models, quota health, and rolling EWMA latency."""
    import time

    committee.hot_reload_if_needed(force=True)
    if committee.api_key:
        committee.model_selector.discover_studio_models(committee.api_key, force=True)
    if committee._cached_agy_path:
        committee.model_selector.discover_agy_models(committee._cached_agy_path, probe_cli=False, force=True)

    models = committee.model_selector.get_models()
    table = Table(
        title="Discovered Gemini Models — Quota Health & Rolling Latency",
        header_style="bold cyan",
        expand=True,
    )
    table.add_column("Provider", style="bold white")
    table.add_column("Model ID", style="cyan")
    table.add_column("Tier")
    table.add_column("Daily Calls / RPD", justify="right")
    table.add_column("Avg Latency", justify="right")
    table.add_column("Status")

    now_utc = time.time()
    if not models:
        table.add_row("--", "No models discovered yet", "--", "0/0", "--", "[yellow]UNCONFIGURED[/yellow]")
    else:
        for m in models:
            if now_utc < m.cooldown_until_utc:
                rem = int(m.cooldown_until_utc - now_utc)
                st_str = f"[red]COOLDOWN ({rem}s)[/red]"
            elif m.daily_calls >= m.rpd_limit:
                st_str = "[red]RPD EXHAUSTED[/red]"
            else:
                st_str = "[bold green]READY[/bold green]"
            table.add_row(
                m.provider,
                m.model_id,
                m.tier.value,
                f"{m.daily_calls}/{m.rpd_limit}",
                f"{m.ewma_latency_ms / 1000.0:.2f}s",
                st_str,
            )
    console.print(table)
    status = committee.get_router_status()
    console.print(f"[bold magenta]{status.model_quota_badge}[/bold magenta]")


def apply_menu_choice(
    choice: str,
    committee: AIVetoCommittee,
    config_path: Path,
    env_path: Path,
    new_key_input: str | None = None,
) -> bool:
    """Apply menu option [1]..[6]. Returns False if user chose to exit."""
    c = choice.strip().lower()
    if c in ("q", "0", "exit"):
        return False

    if c == "1":
        update_ai_veto_config(
            config_path=config_path,
            mode=AIVetoMode.AUTO_STUDIO_FIRST.value,
            enabled=True,
            priority_order=["studio_api", "agy_pro", "pure_math"],
        )
        console.print("[bold green]✓ Set Mode [1]: Auto Priority (AI Studio API -> Google AI Pro -> Pure Math)[/bold green]")
    elif c == "2":
        update_ai_veto_config(
            config_path=config_path,
            mode=AIVetoMode.AUTO_PRO_FIRST.value,
            enabled=True,
            priority_order=["agy_pro", "studio_api", "pure_math"],
        )
        console.print("[bold green]✓ Set Mode [2]: Auto Priority (Google AI Pro -> AI Studio API -> Pure Math)[/bold green]")
    elif c == "3":
        update_ai_veto_config(
            config_path=config_path,
            mode=AIVetoMode.FORCE_PURE_MATH.value,
            enabled=False,
            priority_order=["pure_math"],
        )
        console.print("[bold green]✓ Set Mode [3]: Force Pure Math Only (AI Veto Disabled, 0ms Deterministic Quant)[/bold green]")
    elif c == "4":
        key_val = (
            new_key_input
            if new_key_input is not None
            else console.input("[bold cyan]Enter GEMINI_API_KEY (leave blank to clear):[/bold cyan] ")
        )
        update_studio_api_key_in_env(env_path, key_val)
        action_lbl = "Updated" if key_val.strip() else "Cleared"
        console.print(f"[bold green]✓ {action_lbl} GEMINI_API_KEY in {env_path}[/bold green]")
    elif c == "5":
        run_live_waterfall_test(committee)
    elif c == "6":
        inspect_discovered_models(committee)
    else:
        console.print(f"[red]Unknown option: {choice}[/red]")

    committee.hot_reload_if_needed(force=True)
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PocketBot Apex — Interactive Zero-Code AI Mode Switcher")
    parser.add_argument("--config", default=str(ROOT_DIR / "config" / "apex_config.json"))
    parser.add_argument("--env", default=str(ROOT_DIR / ".env"))
    parser.add_argument("--choice", choices=["1", "2", "3", "4", "5", "6"], help="Non-interactive menu selection")
    parser.add_argument("--api-key", default=None, help="API key value when using --choice 4")
    parser.add_argument("--status-only", action="store_true", help="Print live discovery status and exit")
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    env_path = Path(args.env).resolve()
    committee = AIVetoCommittee(env_path=env_path, config_path=config_path)

    if args.status_only:
        render_status_panel(committee, probe_cli=False)
        return 0

    if args.choice:
        apply_menu_choice(args.choice, committee, config_path, env_path, new_key_input=args.api_key)
        render_status_panel(committee, probe_cli=False)
        return 0

    while True:
        console.print()
        render_status_panel(committee, probe_cli=False)
        console.print(
            "\n[bold cyan]Select AI Routing Mode (Hot-Reloads Live Without Restarting Bot):[/bold cyan]\n"
            "  [bold][1][/bold] Auto Priority: AI Studio API -> Google AI Pro -> Pure Math (Default)\n"
            "  [bold][2][/bold] Auto Priority: Google AI Pro -> AI Studio API -> Pure Math\n"
            "  [bold][3][/bold] Force Pure Math Only (Disable AI Veto completely)\n"
            "  [bold][4][/bold] Add / Update / Clear Google AI Studio API Key in .env\n"
            "  [bold][5][/bold] Run Live Test of Current AI Waterfall Right Now\n"
            "  [bold][6][/bold] Inspect Discovered Gemini Models, Quota Health & Latency\n"
            "  [bold][0][/bold] Exit\n"
        )
        ans = console.input("[bold green]Enter choice [0-6]:[/bold green] ").strip()
        if not apply_menu_choice(ans, committee, config_path, env_path):
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
