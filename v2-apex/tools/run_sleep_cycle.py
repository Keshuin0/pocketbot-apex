"""Manual 1-Click Circadian Sleep-Wake Consolidation CLI (`tools/run_sleep_cycle.py`)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from apex.ai.sleep_consolidation import SleepWakeConsolidator
from apex.ai.veto_committee import AIVetoCommittee
from apex.ledger.walkforward_db import WalkForwardLedgerDB


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PocketBot Apex — Manual Sleep-Wake Memory Consolidation")
    parser.add_argument("--db", default="trader/apex_ledger.db", help="Path to SQLite ledger DB")
    parser.add_argument("--memory", default="trader/semantic_memory.json", help="Path to semantic_memory.json")
    parser.add_argument("--config", default="config/apex_config.json", help="Path to apex_config.json")
    args = parser.parse_args(argv)

    console = Console()
    ledger = WalkForwardLedgerDB(db_path=args.db)
    committee = AIVetoCommittee(config_path=args.config, ledger_db=ledger)
    consolidator = SleepWakeConsolidator(ledger_db=ledger, memory_path=args.memory)

    result = consolidator.run_full_cycle(
        trigger_reason="MANUAL_1CLICK_REFRESH",
        ai_committee=committee,
    )
    console.print(f"[bold green]{result.status_message}[/bold green]")
    console.print(
        f"Cycle #{result.cycle_number} | Purged Entries: {result.purged_entries} | "
        f"GC Collected: {result.gc_collected} | Canary Passed: {result.canary_passed}"
    )
    console.print(consolidator.get_mind_state_badge())
    return 0 if result.canary_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
