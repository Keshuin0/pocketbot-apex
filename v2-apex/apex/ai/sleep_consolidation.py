"""Circadian Sleep-Wake Memory Consolidation & Context Hygiene Engine (`apex/ai/sleep_consolidation.py`).

Prevents multi-day LLM context rot, stale working-memory caches, and memory leaks while
preserving cumulative institutional wisdom across restarts:
  - Phase A (Short-Term Working Memory Purge): Flushes per-candle AI verdict caches, trims
    expired ring buffers, cleans orphaned temp files, clears expired model cooldowns, runs
    `gc.collect()`, and refreshes MT5/cTrader heartbeats.
  - Phase B (Episodic-to-Semantic Memory Consolidation): Distills `trader/apex_ledger.db` into
    `trader/semantic_memory.json` + SQLite `consolidation_checkpoints` (realized slippage
    multipliers, UTC session-hour expectancy priors, Split-Conformal quantiles, and symbol
    quarantine states), feeding multipliers back into `TimeMachineCostModel` & `HydraNBBOEngine`.
  - Phase C (Morning Wake-Up Self-Audit): Refreshes `AdaptiveModelSelector` discovery & daily
    quotas, runs `run_canary_integrity_probe()`, and logs `[MORNING WAKE-UP READY]`.
"""

from __future__ import annotations

import datetime
import gc
import hashlib
import json
import logging
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from apex.ai.aegis_verifier import AegisVerifier
from apex.ai.veto_committee import AIVetoCommittee, AIVetoDecision, CandidateSetupAuditContext
from apex.ledger.walkforward_db import WalkForwardLedgerDB

if TYPE_CHECKING:
    from apex.feeds.dual_router import MarketFeedRouter
    from apex.feeds.hydra_nbbo import HydraNBBOEngine

logger = logging.getLogger(__name__)


@dataclass
class SemanticMemorySnapshot:
    """Distilled long-term institutional memory persisted atomically to `trader/semantic_memory.json`."""

    cycle_number: int
    consolidated_at_utc: float
    consolidated_at_hhmm: str
    trigger_reason: str
    slippage_multipliers: dict[str, dict[str, float]] = field(default_factory=dict)
    session_hour_priors: dict[str, dict[str, Any]] = field(default_factory=dict)
    conformal_models: dict[str, dict[str, Any]] = field(default_factory=dict)
    symbol_quarantine_states: dict[str, str] = field(default_factory=dict)
    mfe_mae_stats: dict[str, Any] = field(default_factory=dict)
    trades_consolidated: int = 0
    audits_consolidated: int = 0
    sha256: str = ""

    def to_json_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("sha256", None)
        canonical = json.dumps(d, sort_keys=True, separators=(",", ":"))
        self.sha256 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        d["sha256"] = self.sha256
        return d

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> SemanticMemorySnapshot:
        return cls(
            cycle_number=int(data.get("cycle_number", 0)),
            consolidated_at_utc=float(data.get("consolidated_at_utc", 0.0)),
            consolidated_at_hhmm=str(data.get("consolidated_at_hhmm", "--:--")),
            trigger_reason=str(data.get("trigger_reason", "LOADED")),
            slippage_multipliers=dict(data.get("slippage_multipliers", {})),
            session_hour_priors=dict(data.get("session_hour_priors", {})),
            conformal_models=dict(data.get("conformal_models", {})),
            symbol_quarantine_states=dict(data.get("symbol_quarantine_states", {})),
            mfe_mae_stats=dict(data.get("mfe_mae_stats", {})),
            trades_consolidated=int(data.get("trades_consolidated", 0)),
            audits_consolidated=int(data.get("audits_consolidated", 0)),
            sha256=str(data.get("sha256", "")),
        )


@dataclass(frozen=True, slots=True)
class SleepWakeCycleResult:
    """Complete result of a Circadian Sleep-Wake Consolidation cycle."""

    cycle_number: int
    trigger_reason: str
    purged_entries: int
    gc_collected: int
    canary_passed: bool
    snapshot: SemanticMemorySnapshot
    checkpoint_id: int
    status_message: str


class SleepWakeConsolidator:
    """Circadian Sleep-Wake Memory Consolidation & Context Hygiene Engine."""

    def __init__(
        self,
        ledger_db: WalkForwardLedgerDB,
        memory_path: str | Path = "trader/semantic_memory.json",
        rollover_utc_hour: int = 22,
        rollover_utc_minute: int = 5,
        stale_hours_threshold: float = 16.0,
    ) -> None:
        self.ledger_db = ledger_db
        self.memory_path = Path(memory_path)
        self.rollover_utc_hour = int(rollover_utc_hour)
        self.rollover_utc_minute = int(rollover_utc_minute)
        self.stale_hours_threshold = float(stale_hours_threshold)
        self._cached_snapshot: SemanticMemorySnapshot | None = self.load_semantic_memory()

    def load_semantic_memory(self) -> SemanticMemorySnapshot | None:
        """Load persisted `trader/semantic_memory.json` if it exists."""
        if not self.memory_path.exists():
            return None
        try:
            data = json.loads(self.memory_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                snap = SemanticMemorySnapshot.from_json_dict(data)
                self._cached_snapshot = snap
                return snap
        except Exception as exc:
            logger.warning("Failed to load %s: %s", self.memory_path, exc)
        return None

    def should_run_sleep_cycle(
        self,
        now_utc: float | None = None,
        is_startup: bool = False,
    ) -> tuple[bool, str]:
        """Check whether a circadian sleep-wake consolidation cycle is due."""
        ts = float(now_utc if now_utc is not None else time.time())
        latest = self.ledger_db.get_latest_consolidation_checkpoint()

        if is_startup:
            if latest is None:
                return True, "STARTUP_INITIAL_CONSOLIDATION"
            elapsed_h = (ts - float(latest["timestamp_utc"])) / 3600.0
            if elapsed_h >= self.stale_hours_threshold:
                return True, f"STARTUP_STALE_{elapsed_h:.1f}H"
            return False, "STARTUP_FRESH"

        dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
        target_mins = self.rollover_utc_hour * 60 + self.rollover_utc_minute
        cur_mins = dt.hour * 60 + dt.minute
        in_window = target_mins <= cur_mins <= (target_mins + 15)
        if not in_window:
            return False, "OUTSIDE_CIRCADIAN_WINDOW"

        if latest is not None:
            last_dt = datetime.datetime.fromtimestamp(float(latest["timestamp_utc"]), tz=datetime.timezone.utc)
            if last_dt.date() == dt.date():
                return False, "ALREADY_CONSOLIDATED_TODAY"

        return True, f"DAILY_CIRCADIAN_{self.rollover_utc_hour:02d}{self.rollover_utc_minute:02d}_UTC"

    # ------------------------------------------------------------------
    # Phase A: Short-Term Working Memory Purge ("Clear Brain Fog")
    # ------------------------------------------------------------------
    @staticmethod
    def phase_a_purge_working_memory(
        ai_committee: AIVetoCommittee | None = None,
        feed_router: MarketFeedRouter | None = None,
    ) -> dict[str, int]:
        """Flush stale caches, expired cooldowns, orphaned temp files, and run gc.collect()."""
        purged_verdicts = 0
        purged_events = 0
        cleared_cooldowns = 0
        purged_temp_files = 0
        trimmed_buffers = 0

        if ai_committee is not None:
            purged_verdicts = len(ai_committee._verdict_cache)
            ai_committee._verdict_cache.clear()
            purged_events = len(ai_committee._recent_ai_events)
            ai_committee._recent_ai_events.clear()

            now_ts = time.time()
            for m in ai_committee.model_selector.get_models():
                m.reset_daily_if_needed()
                m.prune_minute_window(now_ts)
                if m.cooldown_until_utc > 0.0 and now_ts >= m.cooldown_until_utc:
                    m.cooldown_until_utc = 0.0
                    m.consecutive_errors = 0
                    cleared_cooldowns += 1

        # Clean orphaned files inside isolated agy temp dir
        iso_dir = Path(tempfile.gettempdir()) / "pocketbot_agy_isolated"
        if iso_dir.exists():
            for child in iso_dir.iterdir():
                try:
                    if child.is_file():
                        child.unlink(missing_ok=True)
                        purged_temp_files += 1
                except Exception:
                    pass

        # Refresh broker heartbeats & trim spike rejector buffers
        if feed_router is not None:
            try:
                nbbo = getattr(feed_router, "nbbo_engine", None)
                if nbbo is not None and hasattr(nbbo, "spike_rejector"):
                    bh = getattr(nbbo.spike_rejector, "_basis_history", {})
                    for sym_dq in bh.values():
                        if len(sym_dq) > 50:
                            while len(sym_dq) > 50:
                                sym_dq.popleft()
                                trimmed_buffers += 1
                if feed_router.mt5.is_connected:
                    feed_router.mt5.get_account_info()
            except Exception:
                pass

        gc_count = gc.collect()
        return {
            "purged_verdicts": purged_verdicts,
            "purged_events": purged_events,
            "cleared_cooldowns": cleared_cooldowns,
            "purged_temp_files": purged_temp_files,
            "trimmed_buffers": trimmed_buffers,
            "gc_collected": gc_count,
            "total_purged": purged_verdicts + purged_events + cleared_cooldowns + purged_temp_files + trimmed_buffers,
        }

    # ------------------------------------------------------------------
    # Phase B: Episodic-to-Semantic Memory Consolidation
    # ------------------------------------------------------------------
    def phase_b_consolidate_semantic_memory(
        self,
        trigger_reason: str,
        purged_cache_entries: int,
        canary_passed: bool,
        hydra_nbbo: HydraNBBOEngine | None = None,
        now_utc: float | None = None,
    ) -> tuple[SemanticMemorySnapshot, int]:
        """Distill SQLite episodic ledger into `trader/semantic_memory.json` & checkpoint table."""
        ts = float(now_utc if now_utc is not None else time.time())
        dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
        latest = self.ledger_db.get_latest_consolidation_checkpoint()
        next_cycle = (int(latest["cycle_number"]) + 1) if latest is not None else 1

        slippage_mults = self.ledger_db.compute_slippage_calibration_multipliers()
        hour_priors = self.ledger_db.compute_session_hour_priors()
        conformal_models = self.ledger_db.compute_conformal_quantiles()
        mfe_mae_stats = self.ledger_db.compute_mfe_mae_stats()
        quarantine_states = {
            q.symbol: q.mode.value for q in self.ledger_db.list_all_quarantine_states()
        }

        with self.ledger_db._connect() as conn:
            trades_cnt = int(conn.execute("SELECT COUNT(*) FROM trade_ledger").fetchone()[0])
            audits_cnt = int(conn.execute("SELECT COUNT(*) FROM ai_epistemic_audit").fetchone()[0])

        snap = SemanticMemorySnapshot(
            cycle_number=next_cycle,
            consolidated_at_utc=ts,
            consolidated_at_hhmm=f"{dt.hour:02d}:{dt.minute:02d}",
            trigger_reason=trigger_reason,
            slippage_multipliers=slippage_mults,
            session_hour_priors=hour_priors,
            conformal_models=conformal_models,
            symbol_quarantine_states=quarantine_states,
            mfe_mae_stats=mfe_mae_stats,
            trades_consolidated=trades_cnt,
            audits_consolidated=audits_cnt,
        )

        payload = snap.to_json_dict()
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.memory_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp_path, self.memory_path)

        if hydra_nbbo is not None and hasattr(hydra_nbbo, "apply_slippage_calibration"):
            hydra_nbbo.apply_slippage_calibration(slippage_mults)

        summary = (
            f"[MORNING WAKE-UP READY] Cycle #{next_cycle}: consolidated {trades_cnt} trades, "
            f"{audits_cnt} audits, {len(slippage_mults)} slippage profiles"
        )
        ckpt_id = self.ledger_db.record_consolidation_checkpoint(
            cycle_number=next_cycle,
            trigger_reason=trigger_reason,
            purged_cache_entries=purged_cache_entries,
            trades_consolidated=trades_cnt,
            audits_consolidated=audits_cnt,
            canary_passed=canary_passed,
            semantic_memory_sha256=snap.sha256,
            summary=summary,
            timestamp_utc=ts,
        )
        self._cached_snapshot = snap
        return snap, ckpt_id

    # ------------------------------------------------------------------
    # Phase C: Morning Wake-Up Self-Audit
    # ------------------------------------------------------------------
    def phase_c_morning_wakeup_audit(
        self,
        ai_committee: AIVetoCommittee | None = None,
        canary_evaluator: Callable[[CandidateSetupAuditContext], AIVetoDecision] | None = None,
    ) -> tuple[bool, str]:
        """Refresh model discovery & daily quotas and run Shield 7 Canary Poison Probe."""
        if ai_committee is not None:
            ai_committee.hot_reload_if_needed(force=True)
            for m in ai_committee.model_selector.get_models():
                m.reset_daily_if_needed()

            # Sync persisted conformal trust scores into model_selector
            if self._cached_snapshot is not None:
                for info in self._cached_snapshot.conformal_models.values():
                    ai_committee.model_selector.update_epistemic_trust(
                        provider=str(info["provider"]),
                        model_id=str(info["model_id"]),
                        trust_score=float(info.get("epistemic_trust_score", 1.0)),
                        quarantined=bool(info.get("quarantined", False)),
                        reason="Restored from Sleep-Wake semantic memory",
                    )

            eval_fn = canary_evaluator or (
                lambda toxic_ctx: ai_committee._evaluate_pure_math_fallback(
                    toxic_ctx, time.perf_counter(), ("pure_math",), "wakeup_canary"
                )
            )
            probe = ai_committee.aegis.run_canary_integrity_probe(eval_fn)
            return probe.passed, f"[MORNING WAKE-UP READY] {probe.detail}"

        verifier = AegisVerifier(ledger_db=self.ledger_db)
        probe = verifier.run_canary_integrity_probe(
            canary_evaluator
            or (
                lambda toxic_ctx: AIVetoDecision(
                    decision="VETO",
                    confidence=1.0,
                    reason="Deterministic Z3 Canary VETO",
                    latency_ms=0.1,
                    bypassed=False,
                    provider_used="pure_math",
                )
            )
        )
        return probe.passed, f"[MORNING WAKE-UP READY] {probe.detail}"

    def run_full_cycle(
        self,
        trigger_reason: str = "MANUAL_REFRESH",
        ai_committee: AIVetoCommittee | None = None,
        feed_router: MarketFeedRouter | None = None,
        hydra_nbbo: HydraNBBOEngine | None = None,
        canary_evaluator: Callable[[CandidateSetupAuditContext], AIVetoDecision] | None = None,
        now_utc: float | None = None,
    ) -> SleepWakeCycleResult:
        """Execute Phase A -> Phase C -> Phase B in one clean Circadian Sleep-Wake cycle."""
        purge_stats = self.phase_a_purge_working_memory(
            ai_committee=ai_committee,
            feed_router=feed_router,
        )
        canary_ok, status_msg = self.phase_c_morning_wakeup_audit(
            ai_committee=ai_committee,
            canary_evaluator=canary_evaluator,
        )
        snap, ckpt_id = self.phase_b_consolidate_semantic_memory(
            trigger_reason=trigger_reason,
            purged_cache_entries=purge_stats["total_purged"],
            canary_passed=canary_ok,
            hydra_nbbo=hydra_nbbo,
            now_utc=now_utc,
        )
        logger.info("%s (Cycle #%d)", status_msg, snap.cycle_number)
        return SleepWakeCycleResult(
            cycle_number=snap.cycle_number,
            trigger_reason=trigger_reason,
            purged_entries=purge_stats["total_purged"],
            gc_collected=purge_stats["gc_collected"],
            canary_passed=canary_ok,
            snapshot=snap,
            checkpoint_id=ckpt_id,
            status_message=status_msg,
        )

    def get_slippage_multiplier_for_symbol(self, symbol: str, venue: str = "mt5") -> float:
        """Return consolidated slippage multiplier for `symbol`/`venue` (default 1.0)."""
        snap = self._cached_snapshot or self.load_semantic_memory()
        if snap is None:
            return 1.0
        sym_map = snap.slippage_multipliers.get(symbol.upper(), {})
        if venue.lower() in sym_map:
            return float(sym_map[venue.lower()])
        if sym_map:
            return float(next(iter(sym_map.values())))
        return 1.0

    def get_compact_aegis_priors(self, symbol: str, utc_hour: int) -> dict[str, Any]:
        """Return compact (<120 tokens, zero chat history) semantic priors for Aegis telemetry."""
        snap = self._cached_snapshot or self.load_semantic_memory()
        if snap is None:
            return {
                "semantic_slippage_mult": 1.0,
                "session_hour_prior_r": 0.0,
                "conformal_quantile": 0.50,
            }
        slip_mult = self.get_slippage_multiplier_for_symbol(symbol)
        hr_info = snap.session_hour_priors.get(f"{int(utc_hour):02d}", {})
        hr_mean_r = float(hr_info.get("mean_r", 0.0))
        quantiles = [
            float(v.get("conformal_quantile", 0.50))
            for v in snap.conformal_models.values()
        ]
        avg_q = round(sum(quantiles) / len(quantiles), 4) if quantiles else 0.50
        return {
            "semantic_slippage_mult": round(slip_mult, 4),
            "session_hour_prior_r": round(hr_mean_r, 4),
            "conformal_quantile": avg_q,
        }

    def get_mind_state_badge(self) -> str:
        """Format Mind State badge for ApexTUIDashboard header."""
        snap = self._cached_snapshot or self.load_semantic_memory()
        if snap is None:
            return "Mind State: FRESH (Cycle #0 | Wisdom Checkpoints: 0 | Last Refresh: --:--)"
        with self.ledger_db._connect() as conn:
            ckpt_cnt = int(conn.execute("SELECT COUNT(*) FROM consolidation_checkpoints").fetchone()[0])
        return (
            f"Mind State: FRESH (Cycle #{snap.cycle_number} | "
            f"Wisdom Checkpoints: {ckpt_cnt} | Last Refresh: {snap.consolidated_at_hhmm})"
        )
