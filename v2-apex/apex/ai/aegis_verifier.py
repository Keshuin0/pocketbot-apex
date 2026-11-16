"""Apex Aegis — 7-Shield Formal Neuro-Symbolic Anti-Hallucination Firewall (`apex/ai/aegis_verifier.py`).

Implements 7 formal verification shields around `AIVetoCommittee`:
  - Shield 1 (Subtract-Only Authority Diode): `APPROVE`, `VETO`, or `ABSTAIN` with `risk_scale` in `[0.0, 1.0]`;
    physically incapable of initiating quant-rejected trades, widening stops, or exceeding 0.50% risk.
  - Shield 2 (Ticker-Blinded Canonicalization + Closed-World Contract): Strips symbol names and raw
    prices (`100.00` base + `R`-multiples + UTC HH:MM + Contagion Oracle telemetry); bans external news/dates.
  - Shield 3 (Proof-Carrying Atomic Claim AST Grounding): Verifies `bull_claims` / `bear_claims` (and
    `bull_case_claim` / `bear_trap_claim` / `evidence_claims`) against SHA-256-fingerprinted input telemetry.
  - Shield 4 (Microsoft `z3-solver` Formal SMT Consistency & Axiom Prover): Encodes telemetry, cited
    inequalities, and hard domain safety axioms into `z3.Solver()`. Catches "true numbers + contradictory
    logic" hallucinations (`z3.unsat`) in <2ms and falls back to deterministic `pure_math` + Z3 axioms.
  - Shield 5 (Borderline Order-Swap Semantic Invariance): On borderline high-stakes setups
    (`0.10 <= net_avg_R <= 0.20` and `vol_pct >= 80`), swaps bull/bear prompt order; verdict flip ->
    `HIGH_SEMANTIC_ENTROPY` -> force `ABSTAIN`.
  - Shield 6 (Split Conformal Prediction Calibration & Shadow Counterfactual Grading): Computes calibrated
    conformal p-values and tracks `veto_value_added_R` / `hallucination_rate` in `WalkForwardLedgerDB`.
  - Shield 7 (Canary Poison Probe): `run_canary_integrity_probe()` injects a Z3-violating toxic setup
    (`net_avg_R=-0.45`, `spread_pct_of_sl=27.5`) and immediately quarantines any model returning `APPROVE`.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import re
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import z3

_Z3_LOCK = threading.RLock()

from apex.feeds.alias_mapper import SmartAliasMapper
from apex.feeds.models import Side

if TYPE_CHECKING:
    from apex.ai.model_selector import AdaptiveModelSelector
    from apex.ai.veto_committee import AIVetoDecision, CandidateSetupAuditContext
    from apex.ledger.walkforward_db import WalkForwardLedgerDB

_VALID_OPS = {"<", "<=", "==", ">=", ">"}
_MAX_DIODE_RISK_PCT = 0.50

_EXTERNAL_NEWS_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:fomc|cpi|ppi|nfp|non-?farm|payroll|powell|yellen|lagarde|ecb|boe|boj|rba|boc)\b", re.IGNORECASE),
    re.compile(r"\b(?:fed|federal\s+reserve|central\s+bank|rate\s+cut|rate\s+hike|interest\s+rate\s+decision)\b", re.IGNORECASE),
    re.compile(r"\b(?:earnings|eps\s+beat|eps\s+miss|guidance|10-k|10-q|8-k|sec\s+filing|dividend)\b", re.IGNORECASE),
    re.compile(r"\b(?:tariff|sanction|geopolitical|war|bankruptcy|merger|acquisition|headline|breaking\s+news|rumor|tweet)\b", re.IGNORECASE),
    re.compile(r"\b(?:gdp\s+report|inflation\s+report|jobs\s+report|ism\s+pmi)\b", re.IGNORECASE),
    re.compile(r"\b20\d{2}-\d{2}-\d{2}\b"),
    re.compile(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?\b", re.IGNORECASE),
)

_NUMBER_TOKEN_RE = re.compile(r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?(?![A-Za-z_])")
_STRUCTURAL_COUNT_RE = re.compile(
    r"\b\d+(?:\s+(?:verified|atomic|bull|bear|grounded|valid|active|open|closed))*\s+"
    r"(?:claims?|shields?|axioms?|signals?|checks?|factors?|rules?|tiers?|steps?|bars?|positions?)\b",
    re.IGNORECASE,
)

_APPROVE_CONTRADICTION_RE = re.compile(
    r"\b(?:must\s+veto|should\s+veto|veto\s+this|reject\s+this|do\s+not\s+enter|hallucination\s+detected)\b",
    re.IGNORECASE,
)
_VETO_CONTRADICTION_RE = re.compile(
    r"\b(?:approve\s+this|clear\s+to\s+(?:buy|sell|enter)|strong\s+buy|strong\s+sell)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class BlindedTelemetry:
    """Shield 2 ticker-blinded, SHA-256-fingerprinted telemetry payload."""

    telemetry: dict[str, Any]
    input_sha256: str
    raw_symbol: str
    raw_entry_price: float
    raw_stop_loss: float
    raw_take_profit: float


@dataclass(frozen=True, slots=True)
class AtomicClaimResult:
    """Result of verifying a single atomic proof-carrying claim."""

    field: str
    op: str
    cited_value: float
    limit: float
    actual_value: float | None
    valid: bool
    error: str = ""


@dataclass(frozen=True, slots=True)
class AegisVerificationResult:
    """Full 7-Shield symbolic + Z3 SMT verification outcome."""

    passed: bool
    hallucination_detected: bool
    fallback_to_pure_math: bool
    raw_decision: str
    final_decision: str
    confidence: float
    risk_multiplier: float
    reason: str
    input_sha256: str
    violation_codes: tuple[str, ...] = field(default_factory=tuple)
    verified_claims: tuple[AtomicClaimResult, ...] = field(default_factory=tuple)
    z3_checked: bool = False
    z3_sat: bool = True
    z3_latency_ms: float = 0.0
    conformal_p_value: float = 0.5


@dataclass(frozen=True, slots=True)
class DiodeEnforcementResult:
    """Shield 1 Subtract-Only Authority Diode enforcement output."""

    final_approved: bool
    final_decision: str
    final_risk_pct: float
    risk_multiplier: float
    stop_loss: float
    take_profit: float
    note: str


@dataclass(frozen=True, slots=True)
class CanaryProbeResult:
    """Shield 7 Canary Poison Probe result."""

    passed: bool
    quarantined: bool
    provider: str
    model_id: str
    decision_returned: str
    detail: str


def _parse_utc_hhmm(session_utc_iso: str, bar_timestamp: float) -> tuple[int, int, str]:
    """Extract exact UTC hour, minute, and HH:MM string without leaking calendar dates."""
    dt: datetime.datetime | None = None
    if session_utc_iso:
        try:
            iso_clean = session_utc_iso.strip().replace("Z", "+00:00")
            dt = datetime.datetime.fromisoformat(iso_clean).astimezone(datetime.timezone.utc)
        except Exception:
            dt = None
    if dt is None and bar_timestamp > 0:
        dt = datetime.datetime.fromtimestamp(float(bar_timestamp), tz=datetime.timezone.utc)
    if dt is None:
        dt = datetime.datetime.now(datetime.timezone.utc)
    return dt.hour, dt.minute, f"{dt.hour:02d}:{dt.minute:02d}"


def _eval_op(lhs: float, op: str, rhs: float) -> bool:
    eps = 1e-9
    if op == "<":
        return lhs < rhs
    if op == "<=":
        return lhs <= rhs + eps
    if op == "==":
        return abs(lhs - rhs) <= 1e-4
    if op == ">=":
        return lhs >= rhs - eps
    if op == ">":
        return lhs > rhs
    return False


def _safe_float(val: Any) -> float | None:
    """Convert non-bool numeric input to finite float; return None on invalid/NaN/Inf."""
    if isinstance(val, bool) or val is None:
        return None
    try:
        out = float(val)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _z3_real(val: float, ctx: z3.Context) -> z3.RatNumRef:
    """Convert a Python float cleanly to an exact Z3 Real constant in `ctx`."""
    return z3.RealVal(f"{float(val):.6f}", ctx=ctx)


class AegisVerifier:
    """7-Shield Formal Neuro-Symbolic Anti-Hallucination Firewall."""

    NUMERIC_TOLERANCE: float = 1e-4
    MIN_ABSTAIN_CONFIDENCE: float = 0.35
    CONFORMAL_ALPHA_TARGET: float = 0.20

    def __init__(
        self,
        ledger_db: WalkForwardLedgerDB | None = None,
        model_selector: AdaptiveModelSelector | None = None,
        min_abstain_confidence: float = 0.35,
    ) -> None:
        self.ledger_db = ledger_db
        self.model_selector = model_selector
        self.min_abstain_confidence = float(min_abstain_confidence)

    # ------------------------------------------------------------------
    # Shield 1: Subtract-Only Authority Diode
    # ------------------------------------------------------------------
    @staticmethod
    def apply_subtract_only_diode(
        quant_approved: bool,
        ai_decision: str,
        requested_risk_multiplier: float,
        recommended_risk_pct: float,
        sl_price: float,
        tp_price: float,
        max_risk_pct: float = _MAX_DIODE_RISK_PCT,
        conformal_p_value: float | None = None,
        conformal_alpha_target: float = 0.20,
    ) -> DiodeEnforcementResult:
        """Enforce subtract-only authority diode invariants in code."""
        dec = str(ai_decision or "ABSTAIN").upper().strip()
        if dec not in ("APPROVE", "VETO", "ABSTAIN"):
            dec = "VETO"

        req_mult = _safe_float(requested_risk_multiplier)
        mult = max(0.0, min(1.0, req_mult if req_mult is not None else 0.0))
        if conformal_p_value is not None:
            conf_p = _safe_float(conformal_p_value)
            if conf_p is None:
                mult *= 0.25
            else:
                alpha_t = max(1e-6, float(conformal_alpha_target))
                conformal_scale = max(0.25, min(1.0, conf_p / alpha_t))
                mult = round(mult * conformal_scale, 4)

        rec_r = _safe_float(recommended_risk_pct)
        base_risk = max(0.0, min(rec_r if rec_r is not None else 0.0, float(max_risk_pct)))

        if not quant_approved:
            return DiodeEnforcementResult(
                final_approved=False,
                final_decision="VETO" if dec == "VETO" else dec,
                final_risk_pct=0.0,
                risk_multiplier=0.0,
                stop_loss=float(sl_price),
                take_profit=float(tp_price),
                note="Shield 1 Diode: Quant rejected setup; AI cannot initiate or override.",
            )

        if dec == "VETO":
            return DiodeEnforcementResult(
                final_approved=False,
                final_decision="VETO",
                final_risk_pct=0.0,
                risk_multiplier=0.0,
                stop_loss=float(sl_price),
                take_profit=float(tp_price),
                note="Shield 1 Diode: AI VETO enforced (0.0% risk).",
            )

        if dec == "ABSTAIN":
            return DiodeEnforcementResult(
                final_approved=True,
                final_decision="ABSTAIN",
                final_risk_pct=round(base_risk, 4),
                risk_multiplier=1.0,
                stop_loss=float(sl_price),
                take_profit=float(tp_price),
                note="Shield 1 Diode: AI ABSTAIN -> deterministic quant + Z3 pass-through.",
            )

        final_risk = round(base_risk * mult, 4)
        return DiodeEnforcementResult(
            final_approved=True,
            final_decision="APPROVE",
            final_risk_pct=final_risk,
            risk_multiplier=round(mult, 4),
            stop_loss=float(sl_price),
            take_profit=float(tp_price),
            note=f"Shield 1 Diode: Approved with {mult:.2f}x risk_scale ({final_risk:.4f}% <= {max_risk_pct:.2f}%).",
        )

    # ------------------------------------------------------------------
    # Shield 2: Ticker-Blinded Canonicalization + Closed-World Contract
    # ------------------------------------------------------------------
    @staticmethod
    def normalize_ticker_blinded_telemetry(
        ctx: CandidateSetupAuditContext,
        asset_class: str | None = None,
    ) -> BlindedTelemetry:
        """Strip symbol name & raw price levels; replace with asset class, 100.00 base & R-units."""
        inferred_class = asset_class or SmartAliasMapper.classify_asset_class(ctx.symbol).value
        sl_dist = abs(float(ctx.entry_price) - float(ctx.stop_loss))
        if sl_dist <= 1e-12:
            sl_dist = 1e-5
        tp_r = abs(float(ctx.take_profit) - float(ctx.entry_price)) / sl_dist
        is_buy = (ctx.side == Side.BUY) if hasattr(ctx.side, "value") else (str(ctx.side).upper() == "BUY")

        norm_entry = 100.00
        norm_sl = 99.00 if is_buy else 101.00
        norm_tp = round(100.00 + tp_r, 4) if is_buy else round(100.00 - tp_r, 4)
        utc_hr, utc_min, utc_hhmm = _parse_utc_hhmm(ctx.session_utc_iso, ctx.bar_timestamp)

        telemetry: dict[str, Any] = {
            "asset_class": str(inferred_class),
            "side": ctx.side.value if hasattr(ctx.side, "value") else str(ctx.side).upper(),
            "timeframe": str(ctx.timeframe).upper().strip(),
            "normalized_entry": norm_entry,
            "normalized_stop_loss": norm_sl,
            "normalized_take_profit": norm_tp,
            "reward_to_risk_R": round(tp_r, 4),
            "regime": str(ctx.regime),
            "hurst": round(float(ctx.hurst_exponent), 4),
            "kaufman_er": round(float(ctx.kaufman_er), 4),
            "vol_pct": round(float(ctx.volatility_percentile), 2),
            "dom_obi": round(float(ctx.dom_obi), 4) if ctx.dom_obi is not None else None,
            "spread_pct_of_sl": round(float(ctx.spread_pct_of_sl) * 100.0, 2),
            "timemachine_net_avg_R": round(float(ctx.net_avg_R), 4),
            "timemachine_prob_profit": round(float(ctx.prob_profit), 4),
            "timemachine_cvar_95_R": round(float(ctx.cvar_95_R), 4),
            "recommended_risk_pct": round(float(ctx.recommended_risk_pct), 4),
            "rollover_illiquidity_flag": bool(getattr(ctx, "rollover_illiquidity_flag", False)),
            "cross_asset_shock_z": round(float(getattr(ctx, "cross_asset_shock_z", 0.0)), 3),
            "session_utc_hour": utc_hr,
            "session_utc_minute": utc_min,
            "session_utc_hhmm": utc_hhmm,
            "closed_world_contract": True,
        }
        canonical = json.dumps(telemetry, sort_keys=True, separators=(",", ":"))
        sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return BlindedTelemetry(
            telemetry=telemetry,
            input_sha256=sha,
            raw_symbol=ctx.symbol.upper().strip(),
            raw_entry_price=float(ctx.entry_price),
            raw_stop_loss=float(ctx.stop_loss),
            raw_take_profit=float(ctx.take_profit),
        )

    @staticmethod
    def build_aegis_prompt(blinded: BlindedTelemetry, swap_order: bool = False) -> str:
        """Build the Shield 2, 3 & 5 Closed-World Proof-Carrying Prompt."""
        schema_example = (
            '{"bull_claims":[{"field":"timemachine_net_avg_R","op":">=","cited_value":0.35,"threshold":0.10}],'
            '"bear_claims":[{"field":"spread_pct_of_sl","op":"<=","cited_value":4.0,"threshold":15.0}],'
            '"decision":"APPROVE","confidence":0.85,"risk_scale":1.0,'
            '"reason":"timemachine_net_avg_R 0.35 >= 0.10 and spread_pct_of_sl 4.0 <= 15.0"}'
        )
        role_order = (
            "Order: Evaluate BEAR TRAP risks first, then BULL CASE merits."
            if swap_order
            else "Order: Evaluate BULL CASE merits first, then BEAR TRAP risks."
        )
        return (
            "Do NOT call any tools or read any files. Respond immediately.\n"
            "CLOSED-WORLD CONTRACT (STRICT):\n"
            "1. Cite ONLY exact field names and numbers present in the Telemetry JSON below.\n"
            "2. FORBIDDEN: Do NOT invent numbers, dates, macro news, FOMC, CPI, NFP, Fed, or earnings.\n"
            "3. Provide both `bull_claims` and `bear_claims` as lists of "
            '{"field":"<exact_key>","op":"<|<=|==|>=|>","cited_value":<exact_number>,"threshold":<limit>}.\n'
            f"4. {role_order}\n"
            "5. `decision` MUST be `APPROVE`, `VETO`, or `ABSTAIN`; `risk_scale` in [0.0, 1.0].\n"
            "6. In `reason`, cite ONLY telemetry numbers and your claim thresholds.\n"
            f"Telemetry JSON (SHA256={blinded.input_sha256[:12]}): {json.dumps(blinded.telemetry, separators=(',', ':'))}\n"
            f"Respond ONLY with a single-line JSON object matching this schema:\n{schema_example}"
        )

    # ------------------------------------------------------------------
    # Shield 3: Proof-Carrying Atomic Claim AST Grounding
    # ------------------------------------------------------------------
    def _verify_single_claim(
        self,
        claim_obj: Any,
        telemetry: dict[str, Any],
        role_name: str,
    ) -> AtomicClaimResult:
        if not isinstance(claim_obj, dict):
            return AtomicClaimResult(
                field="", op="", cited_value=0.0, limit=0.0, actual_value=None, valid=False,
                error=f"{role_name}: claim must be a JSON object",
            )
        fld = str(claim_obj.get("field", "")).strip()
        op = str(claim_obj.get("op", "")).strip()
        if fld not in telemetry:
            return AtomicClaimResult(
                field=fld, op=op, cited_value=0.0, limit=0.0, actual_value=None, valid=False,
                error=f"{role_name}: unknown field '{fld}'",
            )
        actual_raw = telemetry[fld]
        if isinstance(actual_raw, bool) or not isinstance(actual_raw, (int, float)):
            return AtomicClaimResult(
                field=fld, op=op, cited_value=0.0, limit=0.0, actual_value=None, valid=False,
                error=f"{role_name}: field '{fld}' is non-numeric ({actual_raw!r})",
            )
        actual_val = float(actual_raw)

        if op not in _VALID_OPS:
            return AtomicClaimResult(
                field=fld, op=op, cited_value=0.0, limit=0.0, actual_value=actual_val, valid=False,
                error=f"{role_name}: invalid operator '{op}'",
            )
        raw_thresh = claim_obj.get("threshold", claim_obj.get("limit"))
        try:
            cited_val = float(claim_obj["cited_value"])
            limit_val = float(raw_thresh)
        except (KeyError, TypeError, ValueError):
            return AtomicClaimResult(
                field=fld, op=op, cited_value=0.0, limit=0.0, actual_value=actual_val, valid=False,
                error=f"{role_name}: cited_value and threshold must be numeric",
            )

        if math.isnan(cited_val) or math.isnan(limit_val) or math.isinf(cited_val) or math.isinf(limit_val):
            return AtomicClaimResult(
                field=fld, op=op, cited_value=0.0, limit=0.0, actual_value=actual_val, valid=False,
                error=f"{role_name}: NaN/Inf numeric value",
            )

        if abs(cited_val - actual_val) > self.NUMERIC_TOLERANCE:
            return AtomicClaimResult(
                field=fld, op=op, cited_value=cited_val, limit=limit_val, actual_value=actual_val, valid=False,
                error=f"{role_name}: NUMERIC_HALLUCINATION on '{fld}' (cited {cited_val} != actual {actual_val})",
            )

        if not _eval_op(cited_val, op, limit_val):
            return AtomicClaimResult(
                field=fld, op=op, cited_value=cited_val, limit=limit_val, actual_value=actual_val, valid=False,
                error=f"{role_name}: MATH_CONTRADICTION ({cited_val} {op} {limit_val} is False)",
            )

        return AtomicClaimResult(
            field=fld, op=op, cited_value=cited_val, limit=limit_val, actual_value=actual_val, valid=True,
        )

    # ------------------------------------------------------------------
    # Shield 4: Microsoft z3-solver Formal SMT Consistency & Axiom Prover
    # ------------------------------------------------------------------
    @staticmethod
    def verify_z3_smt_axioms(
        telemetry: dict[str, Any],
        raw_decision: str,
        verified_claims: list[AtomicClaimResult],
        risk_scale: float = 1.0,
    ) -> tuple[bool, float, str]:
        """Encode telemetry, cited inequalities, and symbolic counterexamples into Z3.

        Returns `(is_sat, latency_ms, detail)`. If `z3.unsat`, `z3.unknown`, invalid
        numeric telemetry, or an existential worst-case drawdown counterexample
        `Exists([d_shock, eps_slip], Loss > remaining_daily_budget)` is satisfiable,
        fails closed (`False`).
        """
        t0 = time.perf_counter()

        net_r_val = _safe_float(telemetry.get("timemachine_net_avg_R", 0.0))
        spread_val = _safe_float(telemetry.get("spread_pct_of_sl", 0.0))
        vol_val = _safe_float(telemetry.get("vol_pct", 0.0))
        shock_val = _safe_float(telemetry.get("cross_asset_shock_z", 0.0))
        risk_val = _safe_float(telemetry.get("recommended_risk_pct", 0.50))
        budget_val = _safe_float(telemetry.get("remaining_daily_budget_pct", 3.0))
        scale_val = _safe_float(telemetry.get("risk_scale", risk_scale))

        for name, val in (
            ("timemachine_net_avg_R", net_r_val),
            ("spread_pct_of_sl", spread_val),
            ("vol_pct", vol_val),
            ("cross_asset_shock_z", shock_val),
            ("recommended_risk_pct", risk_val),
            ("remaining_daily_budget_pct", budget_val),
            ("risk_scale", scale_val),
        ):
            if val is None:
                lat_ms = (time.perf_counter() - t0) * 1000.0
                return False, round(lat_ms, 3), f"Z3_INVALID_TELEMETRY: non-finite {name}"

        assert (
            net_r_val is not None
            and spread_val is not None
            and vol_val is not None
            and shock_val is not None
            and risk_val is not None
            and budget_val is not None
            and scale_val is not None
        )
        rollover_val = bool(telemetry.get("rollover_illiquidity_flag", False))

        dec_map = {"APPROVE": 1, "ABSTAIN": 0, "VETO": -1}
        dec_int = dec_map.get(str(raw_decision).upper().strip(), 0)

        with _Z3_LOCK:
            ctx = z3.Context()
            solver = z3.Solver(ctx=ctx)
            solver.set("timeout", 50)

            z_dec = z3.Int("decision", ctx=ctx)
            solver.add(z_dec == dec_int)

            z_net_r = z3.Real("timemachine_net_avg_R", ctx=ctx)
            z_spread = z3.Real("spread_pct_of_sl", ctx=ctx)
            z_vol = z3.Real("vol_pct", ctx=ctx)
            z_shock = z3.Real("cross_asset_shock_z", ctx=ctx)
            z_rollover = z3.Bool("rollover_illiquidity_flag", ctx=ctx)

            solver.add(z_net_r == _z3_real(net_r_val, ctx))
            solver.add(z_spread == _z3_real(spread_val, ctx))
            solver.add(z_vol == _z3_real(vol_val, ctx))
            solver.add(z_shock == _z3_real(shock_val, ctx))
            solver.add(z_rollover == z3.BoolVal(rollover_val, ctx=ctx))

            # Hard domain safety axioms: toxic setups CANNOT be APPROVED
            solver.add(
                z3.Implies(
                    z3.Or(
                        z_net_r < z3.RealVal("0.10", ctx=ctx),
                        z_spread > z3.RealVal("15.0", ctx=ctx),
                        z3.And(z_rollover, z_vol > z3.RealVal("85.0", ctx=ctx)),
                        z_shock > z3.RealVal("3.5", ctx=ctx),
                    ),
                    z_dec != 1,
                )
            )

            # Encode every valid cited inequality from the AI's proof-carrying AST
            z_fields: dict[str, z3.ArithRef] = {
                "timemachine_net_avg_R": z_net_r,
                "spread_pct_of_sl": z_spread,
                "vol_pct": z_vol,
                "cross_asset_shock_z": z_shock,
            }
            for vc in verified_claims:
                if not vc.valid:
                    continue
                zf = z_fields.get(vc.field)
                if zf is None:
                    extra_val = _safe_float(telemetry.get(vc.field))
                    if extra_val is None:
                        lat_ms = (time.perf_counter() - t0) * 1000.0
                        return False, round(lat_ms, 3), f"Z3_INVALID_TELEMETRY: non-finite {vc.field}"
                    zf = z3.Real(vc.field, ctx=ctx)
                    z_fields[vc.field] = zf
                    solver.add(zf == _z3_real(extra_val, ctx))
                lim_val = _safe_float(vc.limit)
                if lim_val is None:
                    lat_ms = (time.perf_counter() - t0) * 1000.0
                    return False, round(lat_ms, 3), f"Z3_INVALID_CLAIM_LIMIT: {vc.field}"
                z_lim = _z3_real(lim_val, ctx)
                if vc.op == "<":
                    solver.add(zf < z_lim)
                elif vc.op == "<=":
                    solver.add(zf <= z_lim)
                elif vc.op == "==":
                    solver.add(zf == z_lim)
                elif vc.op == ">=":
                    solver.add(zf >= z_lim)
                elif vc.op == ">":
                    solver.add(zf > z_lim)

            chk = solver.check()
            if chk != z3.sat:
                lat_ms = (time.perf_counter() - t0) * 1000.0
                if chk == z3.unknown:
                    return False, round(lat_ms, 3), "Z3_UNKNOWN_TIMEOUT: solver timed out failing closed"
                return (
                    False,
                    round(lat_ms, 3),
                    "Z3_UNSAT_HALLUCINATION: AI decision violates formal SMT safety axioms",
                )

            # Existential counterexample search over unbounded symbolic shock & slippage disturbances
            if dec_int == 1:
                effective_risk = max(0.0, risk_val * max(0.0, scale_val))
                if budget_val <= 0.0 and effective_risk > 0.0:
                    lat_ms = (time.perf_counter() - t0) * 1000.0
                    return False, round(lat_ms, 3), "Z3_UNSAT_BUDGET_DEPLETED: remaining daily budget <= 0"

                shock_cap = max(0.05, 0.25 + 0.15 * max(0.0, shock_val) + (0.35 if rollover_val else 0.0))
                slip_cap = max(0.05, max(0.0, spread_val) / 100.0 * 2.0 + 0.05)

                ce_solver = z3.Solver(ctx=ctx)
                ce_solver.set("timeout", 50)
                d_shock = z3.Real("d_shock", ctx=ctx)
                eps_slip = z3.Real("eps_slip", ctx=ctx)
                loss_expr = _z3_real(effective_risk, ctx) * (z3.RealVal("1.0", ctx=ctx) + d_shock + eps_slip)
                ce_solver.add(
                    z3.Exists(
                        [d_shock, eps_slip],
                        z3.And(
                            d_shock >= z3.RealVal("0", ctx=ctx),
                            d_shock <= _z3_real(shock_cap, ctx),
                            eps_slip >= z3.RealVal("0", ctx=ctx),
                            eps_slip <= _z3_real(slip_cap, ctx),
                            loss_expr > _z3_real(budget_val, ctx),
                        ),
                    )
                )
                ce_chk = ce_solver.check()
                if ce_chk != z3.unsat:
                    lat_ms = (time.perf_counter() - t0) * 1000.0
                    if ce_chk == z3.unknown:
                        return False, round(lat_ms, 3), "Z3_UNKNOWN_TIMEOUT: counterexample proof timed out"
                    return (
                        False,
                        round(lat_ms, 3),
                        "Z3_COUNTEREXAMPLE_BUDGET_BREACH: Exists(d_shock, eps_slip) where Loss > remaining_daily_budget",
                    )

        lat_ms = (time.perf_counter() - t0) * 1000.0
        return True, round(lat_ms, 3), f"Z3 SAT ({lat_ms:.2f}ms)"

    @staticmethod
    def z3_pure_math_axiom_gate(telemetry: dict[str, Any]) -> tuple[str, str]:
        """Deterministic Z3-backed safety gate for pure_math fallback."""
        is_sat, lat_ms, detail = AegisVerifier.verify_z3_smt_axioms(telemetry, "APPROVE", [])
        if not is_sat:
            return (
                "VETO",
                f"Z3 Axiom Gate VETO ({lat_ms:.2f}ms, {detail}): net_avg_R={telemetry.get('timemachine_net_avg_R')}, "
                f"spread={telemetry.get('spread_pct_of_sl')}%, shock_z={telemetry.get('cross_asset_shock_z')}",
            )
        return "APPROVE", f"Z3 Axiom Gate SAT ({lat_ms:.2f}ms)"

    @staticmethod
    def verify_z3_inflight_trajectory(
        *,
        proposed_action: str,
        bars_held: int,
        min_hold_bars: int,
        abs_move_atr: float,
        cross_asset_shock_z: float,
        forward_ev_r: float,
        current_r: float,
        invalidation_signals: int,
        is_zombie_decay: bool = False,
        is_weekend_close: bool = False,
        side: str = "BUY",
        entry_price: float | None = None,
        sl_old: float | None = None,
        sl_new: float | None = None,
        mark_price: float | None = None,
        tp_old: float | None = None,
        tp_new: float | None = None,
    ) -> tuple[bool, float, str]:
        """Shield 8: Formal Z3 SMT proof for in-flight actions (HOLD vs EARLY_EXIT vs RATCHET)."""
        t0 = time.perf_counter()

        move_val = _safe_float(abs_move_atr)
        shock_val = _safe_float(cross_asset_shock_z)
        ev_val = _safe_float(forward_ev_r)
        curr_val = _safe_float(current_r)
        if any(v is None for v in (move_val, shock_val, ev_val, curr_val)) or any(
            isinstance(iv, bool) for iv in (bars_held, min_hold_bars, invalidation_signals)
        ):
            lat_ms = (time.perf_counter() - t0) * 1000.0
            return False, round(lat_ms, 3), "Z3_SHIELD8_INVALID_INPUT: non-finite trajectory telemetry"

        assert move_val is not None and shock_val is not None and ev_val is not None and curr_val is not None
        try:
            bars_i = int(bars_held)
            min_bars_i = int(min_hold_bars)
            sigs_i = int(invalidation_signals)
        except (TypeError, ValueError):
            lat_ms = (time.perf_counter() - t0) * 1000.0
            return False, round(lat_ms, 3), "Z3_SHIELD8_INVALID_INT: integer field invalid"

        action_norm = str(proposed_action).upper().strip()
        is_early_exit = action_norm == "EARLY_EXIT"
        is_hold_or_ratchet = action_norm in ("HOLD", "RATCHET", "RATCHET_SLTP")

        with _Z3_LOCK:
            ctx = z3.Context()
            solver = z3.Solver(ctx=ctx)
            solver.set("timeout", 50)

            z_exit = z3.Bool("is_early_exit", ctx=ctx)
            z_hold_or_ratchet = z3.Bool("is_hold_or_ratchet", ctx=ctx)
            z_bars = z3.Int("bars_held", ctx=ctx)
            z_min_bars = z3.Int("min_hold_bars", ctx=ctx)
            z_move_atr = z3.Real("abs_move_atr", ctx=ctx)
            z_shock = z3.Real("cross_asset_shock_z", ctx=ctx)
            z_fwd_ev = z3.Real("forward_ev_r", ctx=ctx)
            z_curr_r = z3.Real("current_r", ctx=ctx)
            z_signals = z3.Int("invalidation_signals", ctx=ctx)
            z_zombie = z3.Bool("is_zombie_decay", ctx=ctx)
            z_weekend = z3.Bool("is_weekend_close", ctx=ctx)

            solver.add(z_exit == z3.BoolVal(is_early_exit, ctx=ctx))
            solver.add(z_hold_or_ratchet == z3.BoolVal(is_hold_or_ratchet, ctx=ctx))
            solver.add(z_bars == bars_i)
            solver.add(z_min_bars == min_bars_i)
            solver.add(z_move_atr == _z3_real(move_val, ctx))
            solver.add(z_shock == _z3_real(shock_val, ctx))
            solver.add(z_fwd_ev == _z3_real(ev_val, ctx))
            solver.add(z_curr_r == _z3_real(curr_val, ctx))
            solver.add(z_signals == sigs_i)
            solver.add(z_zombie == z3.BoolVal(bool(is_zombie_decay), ctx=ctx))
            solver.add(z_weekend == z3.BoolVal(bool(is_weekend_close), ctx=ctx))

            # Shock/Weekend emergency mandate: HOLD and RATCHET are UNSAT under extreme shock or weekend close
            solver.add(
                z3.Implies(
                    z3.Or(z_shock > z3.RealVal("4.2", ctx=ctx), z_weekend),
                    z3.Not(z_hold_or_ratchet),
                )
            )

            # Grace lock invariant: cannot early-exit during grace window & noise cone unless flash-crash or weekend shield
            grace_locked = z3.And(
                z_bars < z_min_bars,
                z_move_atr <= z3.RealVal("0.65", ctx=ctx),
                z_shock <= z3.RealVal("4.2", ctx=ctx),
                z3.Not(z_weekend),
            )
            solver.add(z3.Implies(grace_locked, z3.Not(z_exit)))

            # Thesis-invalidation early exit invariant
            valid_exit_basis = z3.Or(
                z_shock > z3.RealVal("4.2", ctx=ctx),
                z_weekend,
                z3.And(z_zombie, z_fwd_ev <= z3.RealVal("0.0", ctx=ctx)),
                z3.And(z_fwd_ev < z3.RealVal("-0.25", ctx=ctx), z_signals >= 3),
            )
            solver.add(z3.Implies(z_exit, valid_exit_basis))

            chk = solver.check()
            if chk != z3.sat:
                lat_ms = (time.perf_counter() - t0) * 1000.0
                if chk == z3.unknown:
                    return False, round(lat_ms, 3), "Z3_SHIELD8_UNKNOWN_TIMEOUT"
                return (
                    False,
                    round(lat_ms, 3),
                    f"Z3_SHIELD8_UNSAT: {action_norm} blocked (bars={bars_i}/{min_bars_i}, "
                    f"move_atr={move_val:.2f}, fwd_ev={ev_val:+.2f}R, sigs={sigs_i}, shock={shock_val:.2f}, weekend={is_weekend_close})",
                )

            # Symbolic Ratchet Stop-Loss Dominance Proof when SL transition is supplied
            if sl_old is not None and sl_new is not None:
                s_old = _safe_float(sl_old)
                s_new = _safe_float(sl_new)
                if s_old is None or s_new is None or s_old <= 0.0 or s_new <= 0.0:
                    lat_ms = (time.perf_counter() - t0) * 1000.0
                    return False, round(lat_ms, 3), "Z3_SHIELD8_INVALID_SL: non-positive or non-finite SL"
                side_up = str(side).upper().strip()
                e_val = _safe_float(entry_price)
                if e_val is None:
                    e_val = (s_old + 1.0) if side_up == "BUY" else max(s_old - 1.0, 0.0001)

                dom_solver = z3.Solver(ctx=ctx)
                dom_solver.set("timeout", 50)
                p_t = z3.Real("P_T", ctx=ctx)
                z_entry = _z3_real(e_val, ctx)
                z_sl_old = _z3_real(s_old, ctx)
                z_sl_new = _z3_real(s_new, ctx)
                if side_up == "BUY":
                    pnl_new = z3.If(p_t <= z_sl_new, z_sl_new - z_entry, p_t - z_entry)
                    pnl_old = z3.If(p_t <= z_sl_old, z_sl_old - z_entry, p_t - z_entry)
                    dom_solver.add(
                        z3.Exists([p_t], z3.And(p_t > z3.RealVal("0", ctx=ctx), p_t <= z_sl_old, pnl_new < pnl_old))
                    )
                else:
                    pnl_new = z3.If(p_t >= z_sl_new, z_entry - z_sl_new, z_entry - p_t)
                    pnl_old = z3.If(p_t >= z_sl_old, z_entry - z_sl_old, z_entry - p_t)
                    dom_solver.add(
                        z3.Exists([p_t], z3.And(p_t > z3.RealVal("0", ctx=ctx), p_t >= z_sl_old, pnl_new < pnl_old))
                    )
                dom_chk = dom_solver.check()
                if dom_chk != z3.unsat:
                    lat_ms = (time.perf_counter() - t0) * 1000.0
                    return (
                        False,
                        round(lat_ms, 3),
                        f"Z3_SHIELD8_RATCHET_UNSAT: Exists P_T where widened SL ({s_old}->{s_new}) yields worse stop PnL",
                    )

                # Mark-price & TP boundary axioms: SL cannot cross mark_price and TP must stay validly bounded
                m_val = _safe_float(mark_price)
                if m_val is not None and m_val > 0.0 and action_norm in ("RATCHET", "RATCHET_SLTP"):
                    invalid_mark_sl = (s_new >= m_val) if side_up == "BUY" else (s_new <= m_val)
                    if invalid_mark_sl:
                        lat_ms = (time.perf_counter() - t0) * 1000.0
                        return (
                            False,
                            round(lat_ms, 3),
                            f"Z3_SHIELD8_MARK_UNSAT: new SL ({s_new}) crosses live mark ({m_val})",
                        )

            t_new = _safe_float(tp_new) if tp_new is not None else None
            if t_new is not None:
                side_up = str(side).upper().strip()
                e_val = _safe_float(entry_price)
                m_val = _safe_float(mark_price)
                t_old = _safe_float(tp_old) if tp_old is not None else None
                if t_new <= 0.0:
                    lat_ms = (time.perf_counter() - t0) * 1000.0
                    return False, round(lat_ms, 3), "Z3_SHIELD8_TP_UNSAT: non-positive TP"
                tp_solver = z3.Solver(ctx=ctx)
                tp_solver.set("timeout", 50)
                z_tp = _z3_real(t_new, ctx)
                if e_val is not None and e_val > 0.0:
                    z_e = _z3_real(e_val, ctx)
                    tp_solver.add(z_tp > z_e if side_up == "BUY" else z_tp < z_e)
                if m_val is not None and m_val > 0.0 and action_norm in ("RATCHET", "RATCHET_SLTP"):
                    z_m = _z3_real(m_val, ctx)
                    tp_solver.add(z_tp > z_m if side_up == "BUY" else z_tp < z_m)
                if t_old is not None and t_old > 0.0 and e_val is not None and e_val > 0.0:
                    max_tp_dist = max(abs(t_old - e_val) * 3.0, abs(float(sl_old or e_val) - e_val) * 6.0, 1e-6)
                    tp_solver.add(z3.If(z_tp >= _z3_real(e_val, ctx), z_tp - _z3_real(e_val, ctx), _z3_real(e_val, ctx) - z_tp) <= _z3_real(max_tp_dist, ctx))
                if tp_solver.check() != z3.sat:
                    lat_ms = (time.perf_counter() - t0) * 1000.0
                    return False, round(lat_ms, 3), f"Z3_SHIELD8_TP_UNSAT: invalid TP bound ({t_new})"

        lat_ms = (time.perf_counter() - t0) * 1000.0
        return (
            True,
            round(lat_ms, 3),
            f"Z3_SHIELD8_SAT ({lat_ms:.2f}ms): EV_hold={ev_val:+.2f}R vs curr={curr_val:+.2f}R",
        )

    # ------------------------------------------------------------------
    # Shield 5: Borderline Order-Swap Semantic Invariance
    # ------------------------------------------------------------------
    @staticmethod
    def is_borderline_high_stakes(telemetry: dict[str, Any]) -> bool:
        """True when 0.10 <= net_avg_R <= 0.20 and vol_pct >= 80.0."""
        net_r = _safe_float(telemetry.get("timemachine_net_avg_R", 0.0))
        vol_pct = _safe_float(telemetry.get("vol_pct", 0.0))
        if net_r is None or vol_pct is None:
            return False
        return (0.10 <= net_r <= 0.20) and (vol_pct >= 80.0)

    @staticmethod
    def check_semantic_invariance(decision_forward: str, decision_swapped: str) -> bool:
        """Return False if model flips decision under bull/bear prompt-order swap."""
        return str(decision_forward).upper().strip() == str(decision_swapped).upper().strip()

    # ------------------------------------------------------------------
    # Combined Verification Pipeline (Shields 1-6)
    # ------------------------------------------------------------------
    def verify_response(
        self,
        payload: dict[str, Any],
        blinded: BlindedTelemetry,
        provider: str = "studio_api",
        model_id: str = "unknown",
    ) -> AegisVerificationResult:
        """Verify parsed AI JSON payload against Shields 1, 2, 3, 4, and 6."""
        violations: list[str] = []
        telemetry = blinded.telemetry

        raw_dec = str(payload.get("decision", "")).upper().strip()
        if raw_dec not in ("APPROVE", "VETO", "ABSTAIN"):
            violations.append("INVALID_DECISION")
            raw_dec = "ABSTAIN"

        try:
            conf = float(payload.get("confidence", 0.5))
            if math.isnan(conf) or math.isinf(conf):
                raise ValueError("NaN confidence")
            conf = max(0.0, min(1.0, conf))
        except (TypeError, ValueError):
            violations.append("INVALID_CONFIDENCE")
            conf = 0.0

        default_mult = 0.0 if raw_dec == "VETO" else 1.0
        raw_scale = payload.get("risk_scale", payload.get("risk_multiplier", default_mult))
        try:
            mult = float(raw_scale)
            if math.isnan(mult) or math.isinf(mult):
                raise ValueError("NaN risk_scale")
            mult = max(0.0, min(1.0, mult))
        except (TypeError, ValueError):
            violations.append("INVALID_RISK_SCALE")
            mult = 1.0

        reason = str(payload.get("reason", "")).strip()

        # Collect claims from both new (`bull_claims`/`bear_claims`) & legacy (`bull_case_claim`/`bear_trap_claim`/`evidence_claims`)
        verified_claims: list[AtomicClaimResult] = []
        bull_valid = 0
        bear_valid = 0

        bull_list = payload.get("bull_claims")
        if isinstance(bull_list, list):
            for idx, item in enumerate(bull_list):
                res = self._verify_single_claim(item, telemetry, f"bull_claims[{idx}]")
                verified_claims.append(res)
                if res.valid:
                    bull_valid += 1
                else:
                    self._record_claim_violation(res.error, violations)
        elif "bull_case_claim" in payload and payload["bull_case_claim"] is not None:
            res = self._verify_single_claim(payload["bull_case_claim"], telemetry, "bull_case_claim")
            verified_claims.append(res)
            if res.valid:
                bull_valid += 1
            else:
                self._record_claim_violation(res.error, violations)

        bear_list = payload.get("bear_claims")
        if isinstance(bear_list, list):
            for idx, item in enumerate(bear_list):
                res = self._verify_single_claim(item, telemetry, f"bear_claims[{idx}]")
                verified_claims.append(res)
                if res.valid:
                    bear_valid += 1
                else:
                    self._record_claim_violation(res.error, violations)
        elif "bear_trap_claim" in payload and payload["bear_trap_claim"] is not None:
            res = self._verify_single_claim(payload["bear_trap_claim"], telemetry, "bear_trap_claim")
            verified_claims.append(res)
            if res.valid:
                bear_valid += 1
            else:
                self._record_claim_violation(res.error, violations)

        raw_ev_claims = payload.get("evidence_claims")
        if isinstance(raw_ev_claims, list):
            for idx, item in enumerate(raw_ev_claims):
                res = self._verify_single_claim(item, telemetry, f"evidence_claims[{idx}]")
                verified_claims.append(res)
                if not res.valid:
                    self._record_claim_violation(res.error, violations)

        if raw_dec in ("APPROVE", "VETO") and (bull_valid == 0 or bear_valid == 0):
            violations.append("MISSING_COUNTER_ABDUCTIVE_SIDE")

        # Shield 2 & 3 lexical scans on `reason`
        for pat in _EXTERNAL_NEWS_PATTERNS:
            if pat.search(reason):
                violations.append("EXTERNAL_NEWS_OR_DATE_HALLUCINATION")
                break

        if len(blinded.raw_symbol) >= 3 and re.search(rf"\b{re.escape(blinded.raw_symbol)}\b", reason, re.IGNORECASE):
            violations.append("TICKER_LEAK_DETECTED")

        allowed_nums: set[float] = {0.0, 1.0, 100.0, 15.0, 0.10, 3.5, 85.0, round(conf, 4), round(mult, 4)}
        for k, v in telemetry.items():
            if not isinstance(v, bool) and isinstance(v, (int, float)):
                allowed_nums.add(float(v))
                if k == "timemachine_prob_profit":
                    allowed_nums.add(round(float(v) * 100.0, 2))
        for vc in verified_claims:
            if vc.valid:
                allowed_nums.add(vc.cited_value)
                allowed_nums.add(vc.limit)

        structural_spans = [m.span() for m in _STRUCTURAL_COUNT_RE.finditer(reason)]
        for match in _NUMBER_TOKEN_RE.finditer(reason):
            m_start, m_end = match.span()
            if any(s_start <= m_start and m_end <= s_end for s_start, s_end in structural_spans):
                continue
            try:
                num = float(match.group(0))
            except ValueError:
                continue
            if not any(abs(num - allowed) <= 1e-3 for allowed in allowed_nums):
                violations.append("UNPROVIDED_NUMBER_HALLUCINATION")
                break

        if raw_dec == "APPROVE" and _APPROVE_CONTRADICTION_RE.search(reason):
            violations.append("DECISION_REASON_CONTRADICTION")
        elif raw_dec == "VETO" and _VETO_CONTRADICTION_RE.search(reason):
            violations.append("DECISION_REASON_CONTRADICTION")

        # Shield 6: Split Conformal Prediction calibration p-value & Conformalized Kelly Sizing
        conformal_p = 0.5
        if self.ledger_db is not None:
            conformal_p = self.ledger_db.get_conformal_p_value(provider, model_id, conf)
        alpha_target = max(1e-6, float(self.CONFORMAL_ALPHA_TARGET))
        conformal_kelly_scale = max(0.25, min(1.0, float(conformal_p) / alpha_target))
        scaled_mult = round(mult * conformal_kelly_scale, 4)

        # Shield 4: Microsoft Z3 Formal SMT Consistency & Axiom Prover
        z3_sat, z3_ms, z3_msg = self.verify_z3_smt_axioms(
            telemetry,
            raw_dec,
            verified_claims,
            risk_scale=scaled_mult,
        )
        if not z3_sat:
            violations.append("Z3_UNSAT_HALLUCINATION")

        dedup_violations = tuple(dict.fromkeys(violations))
        hallucination_detected = len(dedup_violations) > 0

        if hallucination_detected:
            return AegisVerificationResult(
                passed=False,
                hallucination_detected=True,
                fallback_to_pure_math=True,
                raw_decision=raw_dec,
                final_decision="ABSTAIN",
                confidence=0.0,
                risk_multiplier=1.0,
                reason=f"HALLUCINATION_DETECTED ({', '.join(dedup_violations)}) [{z3_msg}]: {reason[:140]}",
                input_sha256=blinded.input_sha256,
                violation_codes=dedup_violations,
                verified_claims=tuple(verified_claims),
                z3_checked=True,
                z3_sat=z3_sat,
                z3_latency_ms=z3_ms,
                conformal_p_value=conformal_p,
            )

        if raw_dec == "ABSTAIN" or conf < self.min_abstain_confidence:
            return AegisVerificationResult(
                passed=True,
                hallucination_detected=False,
                fallback_to_pure_math=True,
                raw_decision=raw_dec,
                final_decision="ABSTAIN",
                confidence=conf,
                risk_multiplier=1.0,
                reason=f"Aegis ABSTAIN pass-through to pure_math (conf={conf:.2f}, conformal_p={conformal_p:.2f}): {reason}",
                input_sha256=blinded.input_sha256,
                violation_codes=(),
                verified_claims=tuple(verified_claims),
                z3_checked=True,
                z3_sat=z3_sat,
                z3_latency_ms=z3_ms,
                conformal_p_value=conformal_p,
            )

        return AegisVerificationResult(
            passed=True,
            hallucination_detected=False,
            fallback_to_pure_math=False,
            raw_decision=raw_dec,
            final_decision=raw_dec,
            confidence=conf,
            risk_multiplier=0.0 if raw_dec == "VETO" else scaled_mult,
            reason=reason,
            input_sha256=blinded.input_sha256,
            violation_codes=(),
            verified_claims=tuple(verified_claims),
            z3_checked=True,
            z3_sat=z3_sat,
            z3_latency_ms=z3_ms,
            conformal_p_value=conformal_p,
        )

    @staticmethod
    def _record_claim_violation(err: str, violations: list[str]) -> None:
        if "NUMERIC_HALLUCINATION" in err:
            violations.append("NUMERIC_HALLUCINATION")
        elif "MATH_CONTRADICTION" in err:
            violations.append("OPERATOR_MATH_CONTRADICTION")
        else:
            violations.append("INVALID_EVIDENCE_CLAIM")

    # ------------------------------------------------------------------
    # Shield 7: Canary Poison Probe
    # ------------------------------------------------------------------
    @staticmethod
    def build_canary_toxic_context() -> CandidateSetupAuditContext:
        """Build a randomized Z3-violating toxic setup for Shield 7 Canary Poison Probe."""
        import secrets
        from apex.ai.veto_committee import CandidateSetupAuditContext

        rand_alias = f"CNRY_{secrets.choice(['X', 'Q', 'K', 'Z'])}{secrets.randbelow(9000) + 1000}"
        entry = round(90.0 + secrets.randbelow(2000) / 100.0, 2)
        sl_dist = round(0.8 + secrets.randbelow(60) / 100.0, 2)
        spread_frac = round(0.24 + secrets.randbelow(120) / 1000.0, 4)  # 24%..36% of SL
        neg_r = round(-(0.35 + secrets.randbelow(250) / 1000.0), 4)
        hh = secrets.randbelow(24)
        mm = secrets.randbelow(60)
        dd = secrets.randbelow(28) + 1
        mo = secrets.randbelow(12) + 1
        iso_ts = f"2026-{mo:02d}-{dd:02d}T{hh:02d}:{mm:02d}:00Z"

        return CandidateSetupAuditContext(
            symbol=rand_alias,
            side=Side.BUY,
            timeframe=secrets.choice(["M5", "M15"]),
            entry_price=entry,
            stop_loss=round(entry - sl_dist, 4),
            take_profit=round(entry + 1.5 * sl_dist, 4),
            regime="MEAN_REVERTING_CHOP",
            hurst_exponent=0.41,
            kaufman_er=0.14,
            volatility_percentile=round(88.0 + secrets.randbelow(10), 1),
            dom_obi=-0.35,
            spread_pct_of_sl=spread_frac,
            net_avg_R=neg_r,
            prob_profit=0.32,
            cvar_95_R=-2.10,
            recommended_risk_pct=0.50,
            session_utc_iso=iso_ts,
            bar_timestamp=0.0,
        )

    def run_canary_integrity_probe(
        self,
        evaluator_fn: Callable[[CandidateSetupAuditContext], AIVetoDecision],
        provider: str = "agy_pro",
        model_id: str = "unknown",
    ) -> CanaryProbeResult:
        """Run Shield 7 Canary Poison Probe. Locks out any model returning `APPROVE`."""
        toxic_ctx = self.build_canary_toxic_context()
        decision = evaluator_fn(toxic_ctx)
        dec_str = str(decision.decision).upper().strip()
        used_provider = decision.provider_used or provider
        used_model = decision.model_used or model_id

        approved_toxic = dec_str == "APPROVE" and not decision.bypassed
        if approved_toxic:
            reason = "EPISTEMICALLY_COMPROMISED: Approved Canary Poison Probe (net_avg_R=-0.45, spread=27.5%)"
            if self.model_selector is not None:
                self.model_selector.quarantine_epistemically(used_provider, used_model, reason)
            if self.ledger_db is not None:
                blinded = self.normalize_ticker_blinded_telemetry(toxic_ctx)
                self.ledger_db.log_ai_epistemic_audit(
                    symbol=toxic_ctx.symbol,
                    side=toxic_ctx.side.value,
                    timeframe=toxic_ctx.timeframe,
                    provider=used_provider,
                    model_id=used_model,
                    input_sha256=blinded.input_sha256,
                    raw_decision="APPROVE",
                    final_decision="VETO",
                    confidence=decision.confidence,
                    risk_multiplier=0.0,
                    hallucination_detected=True,
                    violation_codes="CANARY_TRAP_FAILED",
                    reason=reason,
                    is_canary=True,
                    predicted_net_r=toxic_ctx.net_avg_R,
                )
            return CanaryProbeResult(
                passed=False,
                quarantined=True,
                provider=used_provider,
                model_id=used_model,
                decision_returned=dec_str,
                detail=reason,
            )

        if self.ledger_db is not None:
            blinded = self.normalize_ticker_blinded_telemetry(toxic_ctx)
            self.ledger_db.log_ai_epistemic_audit(
                symbol=toxic_ctx.symbol,
                side=toxic_ctx.side.value,
                timeframe=toxic_ctx.timeframe,
                provider=used_provider,
                model_id=used_model,
                input_sha256=blinded.input_sha256,
                raw_decision=dec_str,
                final_decision="VETO",
                confidence=decision.confidence,
                risk_multiplier=0.0,
                hallucination_detected=False,
                violation_codes="",
                reason=f"Canary Poison Probe passed ({dec_str})",
                is_canary=True,
                predicted_net_r=toxic_ctx.net_avg_R,
            )

        return CanaryProbeResult(
            passed=True,
            quarantined=False,
            provider=used_provider,
            model_id=used_model,
            decision_returned=dec_str,
            detail=f"Canary Poison Probe passed: returned {dec_str}",
        )
