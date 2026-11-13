"""Self-Healing SQLite Ledger & Auto-Quarantine Engine (`apex/ledger/walkforward_db.py`).

Logs every scan, regime state, DOM snapshot, predicted `net_avg_R`, AI verdict, and realized fill
slippage + net R into `trader/apex_ledger.db` (SQLite with WAL concurrency).

Auto-Quarantine Engine:
  - Evaluates a rolling 20-trade window per symbol.
  - If a symbol is currently in `ACTIVE_DEMO_LIVE` mode (`PAPER_SHADOW == False`):
      * If after >= 15 trades its rolling Profit Factor < `0.90`, OR
      * Its realized expectancy (`mean(realized_net_R)`) drops more than `1.5` standard errors below
        the Time Machine predicted EV (`mean(predicted_net_R) - 1.5 * SE(realized_net_R)`),
      the symbol is automatically demoted to `PAPER_SHADOW` mode!
  - While a symbol is in `PAPER_SHADOW` mode:
      * It remains quarantined in `PAPER_SHADOW` until its rolling 15-trade paper Profit Factor
        recovers strictly above `1.25` (`PF > 1.25`).
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


class SymbolExecutionMode(str, Enum):
    """Per-symbol execution routing state controlled by the Auto-Quarantine Engine."""

    ACTIVE = "ACTIVE"
    PAPER_SHADOW = "PAPER_SHADOW"


@dataclass(frozen=True, slots=True)
class ModelEpistemicStats:
    """Rolling Aegis Shield 6 epistemic integrity, Brier calibration, and veto value-added metrics."""

    provider: str
    model_id: str
    total_audits: int
    hallucination_count: int
    hallucination_rate: float
    verified_proofs: int
    graded_vetoes: int
    veto_value_added_r: float
    avg_brier_score: float
    epistemic_trust_score: float
    should_demote_to_pure_math: bool
    demote_reason: str


@dataclass(frozen=True, slots=True)
class SymbolQuarantineState:
    """Current quarantine telemetry and rolling performance metrics for a symbol."""

    symbol: str
    mode: SymbolExecutionMode
    window_trades: int
    mean_predicted_r: float
    mean_realized_r: float
    std_error_r: float
    z_deficit: float
    profit_factor: float
    reason: str
    updated_at_utc: float


class WalkForwardLedgerDB:
    """Self-Healing SQLite Ledger & Auto-Quarantine Engine (`trader/apex_ledger.db`)."""

    ROLLING_WINDOW: int = 20
    MIN_TRADES_FOR_DEMOTION: int = 15
    DEMOTION_SE_THRESHOLD: float = 1.5
    DEMOTION_PF_THRESHOLD: float = 0.90
    RECOVERY_WINDOW: int = 15
    RECOVERY_PF_THRESHOLD: float = 1.25

    def __init__(self, db_path: str | Path = "trader/apex_ledger.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS scan_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp_utc REAL NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    regime TEXT NOT NULL,
                    hurst REAL NOT NULL,
                    kaufman_er REAL NOT NULL,
                    vol_pct REAL NOT NULL,
                    dom_obi REAL,
                    spread_pct_of_sl REAL,
                    passed_quant INTEGER NOT NULL,
                    detail TEXT
                );

                CREATE TABLE IF NOT EXISTS trade_ledger (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id TEXT NOT NULL,
                    timestamp_utc REAL NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    execution_mode TEXT NOT NULL,
                    regime TEXT NOT NULL,
                    dom_obi REAL,
                    predicted_net_r REAL NOT NULL,
                    ai_decision TEXT NOT NULL,
                    ai_confidence REAL NOT NULL,
                    ai_reason TEXT,
                    entry_price REAL NOT NULL,
                    executed_price REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    take_profit REAL NOT NULL,
                    realized_slippage_r REAL NOT NULL,
                    realized_net_r REAL,
                    is_closed INTEGER NOT NULL DEFAULT 0,
                    winning_venue TEXT NOT NULL DEFAULT 'mt5',
                    mesh_topology TEXT NOT NULL DEFAULT 'MT5_SOLO',
                    sor_pips_saved REAL NOT NULL DEFAULT 0.0,
                    sor_usd_saved REAL NOT NULL DEFAULT 0.0,
                    is_drill INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS sor_savings_ledger (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp_utc REAL NOT NULL,
                    order_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    lots REAL NOT NULL,
                    mesh_topology TEXT NOT NULL,
                    winning_venue TEXT NOT NULL,
                    ctrader_effective_pips REAL,
                    mt5_effective_pips REAL,
                    pips_saved REAL NOT NULL,
                    usd_saved REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS symbol_quarantine (
                    symbol TEXT PRIMARY KEY,
                    mode TEXT NOT NULL,
                    window_trades INTEGER NOT NULL,
                    mean_predicted_r REAL NOT NULL,
                    mean_realized_r REAL NOT NULL,
                    std_error_r REAL NOT NULL,
                    z_deficit REAL NOT NULL,
                    profit_factor REAL NOT NULL,
                    reason TEXT NOT NULL,
                    updated_at_utc REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS ai_epistemic_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp_utc REAL NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    input_sha256 TEXT NOT NULL,
                    raw_decision TEXT NOT NULL,
                    final_decision TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    risk_multiplier REAL NOT NULL,
                    hallucination_detected INTEGER NOT NULL DEFAULT 0,
                    violation_codes TEXT NOT NULL DEFAULT '',
                    reason TEXT NOT NULL DEFAULT '',
                    is_canary INTEGER NOT NULL DEFAULT 0,
                    predicted_net_r REAL NOT NULL DEFAULT 0.0,
                    shadow_realized_r REAL,
                    brier_score REAL,
                    veto_value_added_r REAL,
                    is_graded INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS consolidation_checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cycle_number INTEGER NOT NULL,
                    timestamp_utc REAL NOT NULL,
                    trigger_reason TEXT NOT NULL,
                    purged_cache_entries INTEGER NOT NULL DEFAULT 0,
                    trades_consolidated INTEGER NOT NULL DEFAULT 0,
                    audits_consolidated INTEGER NOT NULL DEFAULT 0,
                    canary_passed INTEGER NOT NULL DEFAULT 1,
                    semantic_memory_sha256 TEXT NOT NULL DEFAULT '',
                    summary TEXT NOT NULL DEFAULT ''
                );
                """
            )
            # Safe idempotent schema migrations for existing SQLite files
            existing_cols = {
                str(r["name"])
                for r in conn.execute("PRAGMA table_info(trade_ledger)").fetchall()
            }
            migrations = [
                ("winning_venue", "TEXT NOT NULL DEFAULT 'mt5'"),
                ("mesh_topology", "TEXT NOT NULL DEFAULT 'MT5_SOLO'"),
                ("sor_pips_saved", "REAL NOT NULL DEFAULT 0.0"),
                ("sor_usd_saved", "REAL NOT NULL DEFAULT 0.0"),
                ("is_drill", "INTEGER NOT NULL DEFAULT 0"),
            ]
            for col_name, col_def in migrations:
                if col_name not in existing_cols:
                    conn.execute(f"ALTER TABLE trade_ledger ADD COLUMN {col_name} {col_def}")

    def log_scan_event(
        self,
        symbol: str,
        timeframe: str,
        regime: str,
        hurst: float,
        kaufman_er: float,
        vol_pct: float,
        dom_obi: float | None,
        spread_pct_of_sl: float | None,
        passed_quant: bool,
        detail: str = "",
        timestamp_utc: float | None = None,
    ) -> int:
        """Record a scan / regime / DOM evaluation in `scan_events`."""
        ts = float(timestamp_utc if timestamp_utc is not None else time.time())
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO scan_events (
                    timestamp_utc, symbol, timeframe, regime, hurst, kaufman_er,
                    vol_pct, dom_obi, spread_pct_of_sl, passed_quant, detail
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    symbol.upper(),
                    timeframe.upper(),
                    regime,
                    float(hurst),
                    float(kaufman_er),
                    float(vol_pct),
                    float(dom_obi) if dom_obi is not None else None,
                    float(spread_pct_of_sl) if spread_pct_of_sl is not None else None,
                    1 if passed_quant else 0,
                    detail,
                ),
            )
            return int(cur.lastrowid or 0)

    def log_trade_outcome(
        self,
        order_id: str,
        symbol: str,
        side: str,
        execution_mode: SymbolExecutionMode | str,
        regime: str,
        dom_obi: float | None,
        predicted_net_r: float,
        ai_decision: str,
        ai_confidence: float,
        ai_reason: str,
        entry_price: float,
        executed_price: float,
        stop_loss: float,
        take_profit: float,
        realized_slippage_r: float,
        realized_net_r: float | None = None,
        timestamp_utc: float | None = None,
        winning_venue: str = "mt5",
        mesh_topology: str = "MT5_SOLO",
        sor_pips_saved: float = 0.0,
        sor_usd_saved: float = 0.0,
        is_drill: bool = False,
    ) -> SymbolQuarantineState:
        """Log a trade execution (and optional closed realized_net_r) and update Auto-Quarantine state."""
        ts = float(timestamp_utc if timestamp_utc is not None else time.time())
        mode_str = execution_mode.value if isinstance(execution_mode, SymbolExecutionMode) else str(execution_mode)
        is_closed = 1 if realized_net_r is not None else 0
        norm_sym = symbol.upper()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO trade_ledger (
                    order_id, timestamp_utc, symbol, side, execution_mode, regime, dom_obi,
                    predicted_net_r, ai_decision, ai_confidence, ai_reason, entry_price,
                    executed_price, stop_loss, take_profit, realized_slippage_r,
                    realized_net_r, is_closed, winning_venue, mesh_topology,
                    sor_pips_saved, sor_usd_saved, is_drill
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    order_id,
                    ts,
                    norm_sym,
                    side.upper(),
                    mode_str,
                    regime,
                    float(dom_obi) if dom_obi is not None else None,
                    float(predicted_net_r),
                    ai_decision,
                    float(ai_confidence),
                    ai_reason,
                    float(entry_price),
                    float(executed_price),
                    float(stop_loss),
                    float(take_profit),
                    float(realized_slippage_r),
                    float(realized_net_r) if realized_net_r is not None else None,
                    is_closed,
                    str(winning_venue),
                    str(mesh_topology),
                    float(sor_pips_saved),
                    float(sor_usd_saved),
                    1 if is_drill else 0,
                ),
            )

        if is_drill:
            return self.get_quarantine_state(norm_sym)
        return self.evaluate_quarantine(norm_sym)

    def log_sor_execution_savings(
        self,
        order_id: str,
        symbol: str,
        lots: float,
        mesh_topology: str,
        winning_venue: str,
        pips_saved: float,
        usd_saved: float,
        ctrader_effective_pips: float | None = None,
        mt5_effective_pips: float | None = None,
        timestamp_utc: float | None = None,
    ) -> int:
        """Record exact pips and USD saved by Active-Active SOR vs the slower/wider venue."""
        ts = float(timestamp_utc if timestamp_utc is not None else time.time())
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO sor_savings_ledger (
                    timestamp_utc, order_id, symbol, lots, mesh_topology, winning_venue,
                    ctrader_effective_pips, mt5_effective_pips, pips_saved, usd_saved
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    str(order_id),
                    symbol.upper(),
                    float(lots),
                    str(mesh_topology),
                    str(winning_venue),
                    float(ctrader_effective_pips) if ctrader_effective_pips is not None else None,
                    float(mt5_effective_pips) if mt5_effective_pips is not None else None,
                    float(pips_saved),
                    float(usd_saved),
                ),
            )
            return int(cur.lastrowid or 0)

    def get_cumulative_sor_savings(self) -> tuple[float, float]:
        """Return `(total_pips_saved, total_usd_saved)` accumulated across `sor_savings_ledger` + `trade_ledger`."""
        with self._connect() as conn:
            r1 = conn.execute(
                "SELECT COALESCE(SUM(pips_saved), 0.0) AS p, COALESCE(SUM(usd_saved), 0.0) AS u FROM sor_savings_ledger"
            ).fetchone()
            p1 = float(r1["p"]) if r1 else 0.0
            u1 = float(r1["u"]) if r1 else 0.0
            if p1 > 0.0 or u1 > 0.0:
                return p1, u1
            r2 = conn.execute(
                "SELECT COALESCE(SUM(sor_pips_saved), 0.0) AS p, COALESCE(SUM(sor_usd_saved), 0.0) AS u FROM trade_ledger WHERE is_drill = 0"
            ).fetchone()
            return (float(r2["p"]) if r2 else 0.0, float(r2["u"]) if r2 else 0.0)

    @staticmethod
    def compute_profit_factor(r_values: Sequence[float] | np.ndarray) -> float:
        """Compute Profit Factor = sum(positive R) / |sum(negative R)|."""
        arr = np.asarray(r_values, dtype=np.float64)
        if len(arr) == 0:
            return 1.0
        gross_win = float(np.sum(arr[arr > 0.0]))
        gross_loss = float(np.abs(np.sum(arr[arr < 0.0])))
        if gross_loss <= 1e-9:
            return 99.0 if gross_win > 0.0 else 1.0
        return float(gross_win / gross_loss)

    def get_quarantine_state(self, symbol: str) -> SymbolQuarantineState:
        """Return the current `SymbolQuarantineState` for `symbol` (defaults to `ACTIVE`)."""
        norm_sym = symbol.upper()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM symbol_quarantine WHERE symbol = ?",
                (norm_sym,),
            ).fetchone()
            if row is not None:
                return SymbolQuarantineState(
                    symbol=str(row["symbol"]),
                    mode=SymbolExecutionMode(str(row["mode"])),
                    window_trades=int(row["window_trades"]),
                    mean_predicted_r=float(row["mean_predicted_r"]),
                    mean_realized_r=float(row["mean_realized_r"]),
                    std_error_r=float(row["std_error_r"]),
                    z_deficit=float(row["z_deficit"]),
                    profit_factor=float(row["profit_factor"]),
                    reason=str(row["reason"]),
                    updated_at_utc=float(row["updated_at_utc"]),
                )
        return SymbolQuarantineState(
            symbol=norm_sym,
            mode=SymbolExecutionMode.ACTIVE,
            window_trades=0,
            mean_predicted_r=0.0,
            mean_realized_r=0.0,
            std_error_r=0.0,
            z_deficit=0.0,
            profit_factor=1.0,
            reason="Active (fewer than 15 closed trades)",
            updated_at_utc=time.time(),
        )

    def evaluate_quarantine(self, symbol: str) -> SymbolQuarantineState:
        """Evaluate rolling 20-trade window and auto-demote or auto-promote `symbol`."""
        norm_sym = symbol.upper()
        current_state = self.get_quarantine_state(norm_sym)

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT predicted_net_r, realized_net_r, execution_mode
                FROM trade_ledger
                WHERE symbol = ? AND is_closed = 1 AND realized_net_r IS NOT NULL AND is_drill = 0
                ORDER BY id DESC
                LIMIT ?
                """,
                (norm_sym, self.ROLLING_WINDOW),
            ).fetchall()

        # Chronological order (oldest to newest within the rolling 20-trade window)
        rows = list(reversed(rows))
        n = len(rows)
        now_utc = time.time()

        if n == 0:
            return current_state

        pred_arr = np.array([float(r["predicted_net_r"]) for r in rows], dtype=np.float64)
        real_arr = np.array([float(r["realized_net_r"]) for r in rows], dtype=np.float64)

        mean_pred = float(np.mean(pred_arr))
        mean_real = float(np.mean(real_arr))
        std_real = float(np.std(real_arr, ddof=1)) if n > 1 else 0.0
        se_real = std_real / math.sqrt(n) if n > 1 else 0.0
        safe_se = max(se_real, 1e-6)
        # Positive z_deficit means realized expectancy is BELOW predicted expectancy
        z_deficit = (mean_pred - mean_real) / safe_se
        pf_20 = self.compute_profit_factor(real_arr)

        new_mode = current_state.mode
        reason = current_state.reason

        if current_state.mode == SymbolExecutionMode.ACTIVE:
            if n >= self.MIN_TRADES_FOR_DEMOTION:
                if z_deficit > self.DEMOTION_SE_THRESHOLD:
                    new_mode = SymbolExecutionMode.PAPER_SHADOW
                    reason = (
                        f"AUTO_QUARANTINE: realized EV ({mean_real:+.3f}R) is {z_deficit:.2f} SE below "
                        f"predicted EV ({mean_pred:+.3f}R) [> {self.DEMOTION_SE_THRESHOLD:.1f} SE limit]"
                    )
                elif pf_20 < self.DEMOTION_PF_THRESHOLD:
                    new_mode = SymbolExecutionMode.PAPER_SHADOW
                    reason = (
                        f"AUTO_QUARANTINE: rolling {n}-trade Profit Factor ({pf_20:.2f}) < "
                        f"{self.DEMOTION_PF_THRESHOLD:.2f} minimum"
                    )
                else:
                    reason = (
                        f"ACTIVE: rolling {n}-trade PF={pf_20:.2f}, realized={mean_real:+.3f}R "
                        f"vs pred={mean_pred:+.3f}R"
                    )
            else:
                reason = f"ACTIVE: warming up ({n}/{self.MIN_TRADES_FOR_DEMOTION} trades)"
        else:
            # Currently in PAPER_SHADOW mode -> check last 15 paper trades for PF > 1.25 recovery
            paper_rows = [r for r in rows if str(r["execution_mode"]) == SymbolExecutionMode.PAPER_SHADOW.value]
            recent_15 = real_arr[-self.RECOVERY_WINDOW :] if len(real_arr) >= self.RECOVERY_WINDOW else real_arr
            paper_15 = (
                np.array([float(r["realized_net_r"]) for r in paper_rows[-self.RECOVERY_WINDOW :]], dtype=np.float64)
                if len(paper_rows) >= self.RECOVERY_WINDOW
                else recent_15
            )
            pf_recovery = self.compute_profit_factor(paper_15)
            if len(paper_rows) >= self.RECOVERY_WINDOW and pf_recovery > self.RECOVERY_PF_THRESHOLD:
                new_mode = SymbolExecutionMode.ACTIVE
                reason = (
                    f"RECOVERED_TO_ACTIVE: rolling {self.RECOVERY_WINDOW}-trade Paper Profit Factor "
                    f"({pf_recovery:.2f}) > {self.RECOVERY_PF_THRESHOLD:.2f}"
                )
            else:
                reason = (
                    f"QUARANTINED (PAPER_SHADOW): rolling 15-trade PF={pf_recovery:.2f} "
                    f"(needs > {self.RECOVERY_PF_THRESHOLD:.2f} across {self.RECOVERY_WINDOW} paper trades)"
                )

        updated = SymbolQuarantineState(
            symbol=norm_sym,
            mode=new_mode,
            window_trades=n,
            mean_predicted_r=mean_pred,
            mean_realized_r=mean_real,
            std_error_r=se_real,
            z_deficit=z_deficit,
            profit_factor=pf_20,
            reason=reason,
            updated_at_utc=now_utc,
        )

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO symbol_quarantine (
                    symbol, mode, window_trades, mean_predicted_r, mean_realized_r,
                    std_error_r, z_deficit, profit_factor, reason, updated_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    mode=excluded.mode,
                    window_trades=excluded.window_trades,
                    mean_predicted_r=excluded.mean_predicted_r,
                    mean_realized_r=excluded.mean_realized_r,
                    std_error_r=excluded.std_error_r,
                    z_deficit=excluded.z_deficit,
                    profit_factor=excluded.profit_factor,
                    reason=excluded.reason,
                    updated_at_utc=excluded.updated_at_utc
                """,
                (
                    updated.symbol,
                    updated.mode.value,
                    updated.window_trades,
                    updated.mean_predicted_r,
                    updated.mean_realized_r,
                    updated.std_error_r,
                    updated.z_deficit,
                    updated.profit_factor,
                    updated.reason,
                    updated.updated_at_utc,
                ),
            )

        return updated

    def list_all_quarantine_states(self) -> list[SymbolQuarantineState]:
        """Return quarantine states for all tracked symbols."""
        with self._connect() as conn:
            rows = conn.execute("SELECT symbol FROM symbol_quarantine ORDER BY symbol ASC").fetchall()
        return [self.get_quarantine_state(str(r["symbol"])) for r in rows]

    # ------------------------------------------------------------------
    # Shield 6: Shadow Counterfactual Grading & Brier Calibration
    # ------------------------------------------------------------------

    def log_ai_epistemic_audit(
        self,
        symbol: str,
        side: str,
        timeframe: str,
        provider: str,
        model_id: str,
        input_sha256: str,
        raw_decision: str,
        final_decision: str,
        confidence: float,
        risk_multiplier: float = 1.0,
        hallucination_detected: bool = False,
        violation_codes: str = "",
        reason: str = "",
        is_canary: bool = False,
        predicted_net_r: float = 0.0,
        timestamp_utc: float | None = None,
    ) -> int:
        """Log an AI Risk Veto & Aegis symbolic verification audit to `ai_epistemic_audit`."""
        ts = float(timestamp_utc if timestamp_utc is not None else time.time())
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO ai_epistemic_audit (
                    timestamp_utc, symbol, side, timeframe, provider, model_id,
                    input_sha256, raw_decision, final_decision, confidence,
                    risk_multiplier, hallucination_detected, violation_codes,
                    reason, is_canary, predicted_net_r
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    symbol.upper(),
                    side.upper(),
                    timeframe.upper(),
                    str(provider),
                    str(model_id or "unknown"),
                    str(input_sha256),
                    raw_decision.upper(),
                    final_decision.upper(),
                    float(max(0.0, min(1.0, confidence))),
                    float(max(0.0, min(1.0, risk_multiplier))),
                    1 if hallucination_detected else 0,
                    str(violation_codes),
                    str(reason)[:500],
                    1 if is_canary else 0,
                    float(predicted_net_r),
                ),
            )
            return int(cur.lastrowid or 0)

    def grade_ai_shadow_outcome(self, audit_id: int, realized_r: float) -> None:
        """Grade a logged AI audit with realized (or virtual shadow) outcome R-multiple.

        Computes Brier calibration score and counterfactual `veto_value_added_r`:
          - For VETO: `veto_value_added_r = -realized_r` (positive = True Save, negative = False Veto).
          - For APPROVE: `veto_value_added_r = 0.0`.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT raw_decision, final_decision, confidence FROM ai_epistemic_audit WHERE id = ?",
                (int(audit_id),),
            ).fetchone()
            if row is None:
                return
            raw_dec = str(row["raw_decision"]).upper()
            conf = float(row["confidence"])
            actual_win = 1.0 if float(realized_r) > 0.0 else 0.0
            # For VETO, AI predicted trade failure (prob_win = 1 - conf) unless calibrated as win prob
            pred_win_prob = (1.0 - conf) if raw_dec == "VETO" else conf
            brier = (pred_win_prob - actual_win) ** 2
            vva_r = -float(realized_r) if raw_dec == "VETO" else 0.0

            conn.execute(
                """
                UPDATE ai_epistemic_audit
                SET shadow_realized_r = ?, brier_score = ?, veto_value_added_r = ?, is_graded = 1
                WHERE id = ?
                """,
                (float(realized_r), round(brier, 6), round(vva_r, 6), int(audit_id)),
            )

    def get_model_epistemic_stats(
        self,
        provider: str | None = None,
        model_id: str | None = None,
        window: int = 20,
    ) -> ModelEpistemicStats:
        """Compute rolling epistemic trust, hallucination rate, Brier score, and demotion flag."""
        clauses: list[str] = []
        params: list[Any] = []
        if provider is not None:
            clauses.append("provider = ?")
            params.append(provider)
        if model_id is not None:
            clauses.append("model_id = ?")
            params.append(model_id)

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(int(max(1, window)))

        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT hallucination_detected, is_canary, raw_decision, final_decision,
                       is_graded, brier_score, veto_value_added_r
                FROM ai_epistemic_audit
                {where_sql}
                ORDER BY id DESC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()

        total = len(rows)
        if total == 0:
            return ModelEpistemicStats(
                provider=provider or "all",
                model_id=model_id or "all",
                total_audits=0,
                hallucination_count=0,
                hallucination_rate=0.0,
                verified_proofs=0,
                graded_vetoes=0,
                veto_value_added_r=0.0,
                avg_brier_score=0.0,
                epistemic_trust_score=1.0,
                should_demote_to_pure_math=False,
                demote_reason="",
            )

        hall_count = sum(int(r["hallucination_detected"]) for r in rows)
        hall_rate = hall_count / total
        verified_proofs = total - hall_count

        briers = [float(r["brier_score"]) for r in rows if int(r["is_graded"]) == 1 and r["brier_score"] is not None]
        avg_brier = sum(briers) / len(briers) if briers else 0.0

        veto_vals = [
            float(r["veto_value_added_r"])
            for r in rows
            if int(r["is_graded"]) == 1 and str(r["raw_decision"]).upper() == "VETO" and r["veto_value_added_r"] is not None
        ]
        graded_vetoes = len(veto_vals)
        avg_vva_r = (sum(veto_vals) / graded_vetoes) if graded_vetoes > 0 else 0.0

        canary_failed = any(
            int(r["is_canary"]) == 1 and str(r["raw_decision"]).upper() == "APPROVE"
            for r in rows
        )

        should_demote = False
        demote_reason = ""
        if canary_failed:
            should_demote = True
            demote_reason = "EPISTEMICALLY_COMPROMISED: Failed Canary Trap audit"
        elif total >= 20 and hall_rate > 0.08:
            should_demote = True
            demote_reason = f"Rolling hallucination_rate {hall_rate * 100:.1f}% > 8.0% over {total} audits"
        elif graded_vetoes >= 20 and avg_vva_r < -0.15:
            should_demote = True
            demote_reason = f"Rolling veto_value_added_R {avg_vva_r:+.3f}R < -0.15R over {graded_vetoes} vetoes"

        # Compute trust score in [0.0, 1.0]
        trust = 1.0 - min(0.80, hall_rate * 2.5) - min(0.20, avg_brier * 0.4)
        if avg_vva_r < 0.0 and graded_vetoes >= 5:
            trust -= min(0.30, abs(avg_vva_r) * 0.5)
        if should_demote:
            trust = 0.0
        trust = max(0.0, min(1.0, round(trust, 4)))

        return ModelEpistemicStats(
            provider=provider or "all",
            model_id=model_id or "all",
            total_audits=total,
            hallucination_count=hall_count,
            hallucination_rate=round(hall_rate, 4),
            verified_proofs=verified_proofs,
            graded_vetoes=graded_vetoes,
            veto_value_added_r=round(avg_vva_r, 4),
            avg_brier_score=round(avg_brier, 4),
            epistemic_trust_score=trust,
            should_demote_to_pure_math=should_demote,
            demote_reason=demote_reason,
        )

    def get_conformal_p_value(
        self,
        provider: str,
        model_id: str,
        raw_confidence: float,
        window: int = 50,
    ) -> float:
        """Compute Split Conformal Prediction p-value against rolling graded calibration scores.

        Non-conformity score `alpha_i = |y_i - p_i|` where `y_i in {0, 1}` is realized win
        and `p_i` is predicted win probability. Higher p-value means the candidate confidence
        conforms well to historical calibrated accuracy.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT raw_decision, confidence, shadow_realized_r
                FROM ai_epistemic_audit
                WHERE provider = ? AND model_id = ? AND is_graded = 1 AND shadow_realized_r IS NOT NULL
                ORDER BY id DESC
                LIMIT ?
                """,
                (provider, model_id, int(max(1, window))),
            ).fetchall()

        if len(rows) < 5:
            # Insufficient calibration history -> conservative neutral conformal p-value
            return round(min(0.65, max(0.20, float(raw_confidence) * 0.75)), 4)

        alphas: list[float] = []
        for r in rows:
            y_i = 1.0 if float(r["shadow_realized_r"]) > 0.0 else 0.0
            c_i = float(r["confidence"])
            p_i = (1.0 - c_i) if str(r["raw_decision"]).upper() == "VETO" else c_i
            alphas.append(abs(y_i - p_i))

        cand_alpha = 1.0 - max(0.0, min(1.0, float(raw_confidence)))
        ge_count = sum(1 for a in alphas if a >= cand_alpha - 1e-9)
        p_val = (ge_count + 1) / (len(alphas) + 1)
        return round(float(p_val), 4)

    def get_aegis_integrity_summary(self, window: int = 100) -> dict[str, Any]:
        """Return compact Aegis Z3 Epistemic Integrity summary for the TUI cockpit header."""
        stats = self.get_model_epistemic_stats(window=window)
        hall_pct = stats.hallucination_rate * 100.0
        return {
            "total_audits": stats.total_audits,
            "hallucination_count": stats.hallucination_count,
            "hallucination_rate": stats.hallucination_rate,
            "verified_proofs": stats.verified_proofs,
            "veto_value_added_r": stats.veto_value_added_r,
            "avg_brier_score": stats.avg_brier_score,
            "badge": f"Aegis Z3: {stats.verified_proofs} Verified Proofs | {hall_pct:.0f}% Uncaught Hallucinations",
        }

    # ------------------------------------------------------------------
    # Phase 8: Circadian Sleep-Wake Consolidation Queries & Checkpoints
    # ------------------------------------------------------------------

    def record_consolidation_checkpoint(
        self,
        cycle_number: int,
        trigger_reason: str,
        purged_cache_entries: int,
        trades_consolidated: int,
        audits_consolidated: int,
        canary_passed: bool,
        semantic_memory_sha256: str,
        summary: str,
        timestamp_utc: float | None = None,
    ) -> int:
        """Persist a versioned Sleep-Wake consolidation checkpoint."""
        ts = float(timestamp_utc if timestamp_utc is not None else time.time())
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO consolidation_checkpoints (
                    cycle_number, timestamp_utc, trigger_reason, purged_cache_entries,
                    trades_consolidated, audits_consolidated, canary_passed,
                    semantic_memory_sha256, summary
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(cycle_number),
                    ts,
                    str(trigger_reason),
                    int(purged_cache_entries),
                    int(trades_consolidated),
                    int(audits_consolidated),
                    1 if canary_passed else 0,
                    str(semantic_memory_sha256),
                    str(summary)[:500],
                ),
            )
            return int(cur.lastrowid or 0)

    def get_latest_consolidation_checkpoint(self) -> dict[str, Any] | None:
        """Return the latest Sleep-Wake consolidation checkpoint or None."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM consolidation_checkpoints ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row is not None else None

    def compute_slippage_calibration_multipliers(self) -> dict[str, dict[str, float]]:
        """Compute realized-vs-baseline slippage multipliers per symbol and venue, clamped to [0.5, 3.0]."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT symbol, winning_venue, realized_slippage_r
                FROM trade_ledger
                WHERE is_drill = 0
                ORDER BY id DESC
                LIMIT 500
                """
            ).fetchall()

        grouped: dict[tuple[str, str], list[float]] = {}
        for r in rows:
            sym = str(r["symbol"]).upper()
            venue = str(r["winning_venue"] or "mt5").lower()
            slip_r = abs(float(r["realized_slippage_r"] or 0.0))
            grouped.setdefault((sym, venue), []).append(slip_r)

        # Baseline expected slippage in R is ~0.04R; multiplier scales proportionally
        baseline_slip_r = 0.04
        result: dict[str, dict[str, float]] = {}
        for (sym, venue), vals in grouped.items():
            mean_slip = sum(vals) / len(vals)
            raw_mult = mean_slip / baseline_slip_r if mean_slip > 0.0 else 1.0
            clamped = max(0.5, min(3.0, round(raw_mult, 4)))
            result.setdefault(sym, {})[venue] = clamped
        return result

    def compute_session_hour_priors(self) -> dict[str, dict[str, Any]]:
        """Compute closed-trade expectancy priors grouped by UTC hour (`00`..`23`)."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT timestamp_utc, realized_net_r
                FROM trade_ledger
                WHERE is_closed = 1 AND realized_net_r IS NOT NULL AND is_drill = 0
                ORDER BY id DESC
                LIMIT 1000
                """
            ).fetchall()

        import datetime

        by_hour: dict[str, list[float]] = {}
        for r in rows:
            dt = datetime.datetime.fromtimestamp(float(r["timestamp_utc"]), tz=datetime.timezone.utc)
            hr_key = f"{dt.hour:02d}"
            by_hour.setdefault(hr_key, []).append(float(r["realized_net_r"]))

        priors: dict[str, dict[str, Any]] = {}
        for hr_key, vals in sorted(by_hour.items()):
            mean_r = sum(vals) / len(vals)
            priors[hr_key] = {
                "count": len(vals),
                "mean_r": round(mean_r, 4),
                "regime_tag": "TOXIC_HOUR" if (len(vals) >= 5 and mean_r < -0.10) else ("HIGH_EDGE" if (len(vals) >= 5 and mean_r > 0.15) else "NEUTRAL"),
            }
        return priors

    def compute_conformal_quantiles(self, alpha: float = 0.10) -> dict[str, dict[str, Any]]:
        """Compute Split-Conformal non-conformity quantiles and Brier trust per provider/model."""
        with self._connect() as conn:
            pairs = conn.execute(
                "SELECT DISTINCT provider, model_id FROM ai_epistemic_audit"
            ).fetchall()

        summary: dict[str, dict[str, Any]] = {}
        for p in pairs:
            prov = str(p["provider"])
            mid = str(p["model_id"])
            stats = self.get_model_epistemic_stats(provider=prov, model_id=mid, window=50)
            with self._connect() as conn:
                graded = conn.execute(
                    """
                    SELECT raw_decision, confidence, shadow_realized_r
                    FROM ai_epistemic_audit
                    WHERE provider = ? AND model_id = ? AND is_graded = 1 AND shadow_realized_r IS NOT NULL
                    ORDER BY id DESC
                    LIMIT 50
                    """,
                    (prov, mid),
                ).fetchall()

            q_val = 0.50
            if len(graded) >= 5:
                alphas = sorted(
                    abs(
                        (1.0 if float(r["shadow_realized_r"]) > 0.0 else 0.0)
                        - ((1.0 - float(r["confidence"])) if str(r["raw_decision"]).upper() == "VETO" else float(r["confidence"]))
                    )
                    for r in graded
                )
                idx = min(len(alphas) - 1, max(0, int(math.ceil((1.0 - alpha) * (len(alphas) + 1))) - 1))
                q_val = round(float(alphas[idx]), 4)

            summary[f"{prov}:{mid}"] = {
                "provider": prov,
                "model_id": mid,
                "conformal_quantile": q_val,
                "epistemic_trust_score": stats.epistemic_trust_score,
                "hallucination_rate": stats.hallucination_rate,
                "veto_value_added_r": stats.veto_value_added_r,
                "quarantined": stats.should_demote_to_pure_math,
            }
        return summary
