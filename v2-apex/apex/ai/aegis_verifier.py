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
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import z3

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


def _z3_real(val: float) -> z3.RatNumRef:
    """Convert a Python float cleanly to an exact Z3 Real constant."""
    return z3.RealVal(f"{float(val):.6f}")


class AegisVerifier:
    """7-Shield Formal Neuro-Symbolic Anti-Hallucination Firewall."""

    NUMERIC_TOLERANCE: float = 1e-4
    MIN_ABSTAIN_CONFIDENCE: float = 0.35

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
    ) -> DiodeEnforcementResult:
        """Enforce subtract-only authority diode invariants in code."""
        dec = str(ai_decision or "ABSTAIN").upper().strip()
        if dec not in ("APPROVE", "VETO", "ABSTAIN"):
            dec = "VETO"

        mult = max(0.0, min(1.0, float(requested_risk_multiplier)))
        base_risk = max(0.0, min(float(recommended_risk_pct), float(max_risk_pct)))

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
    ) -> tuple[bool, float, str]:
        """Encode telemetry, cited inequalities, and hard safety axioms into z3.Solver().

        Returns `(is_sat, latency_ms, detail)`. If `z3.unsat`, the AI committed a
        logical/axiom hallucination (e.g., true numbers + contradictory APPROVE).
        """
        t0 = time.perf_counter()
        solver = z3.Solver()

        dec_map = {"APPROVE": 1, "ABSTAIN": 0, "VETO": -1}
        dec_int = dec_map.get(raw_decision.upper().strip(), 0)
        z_dec = z3.Int("decision")
        solver.add(z_dec == dec_int)

        z_net_r = z3.Real("timemachine_net_avg_R")
        z_spread = z3.Real("spread_pct_of_sl")
        z_vol = z3.Real("vol_pct")
        z_shock = z3.Real("cross_asset_shock_z")
        z_rollover = z3.Bool("rollover_illiquidity_flag")

        solver.add(z_net_r == _z3_real(float(telemetry.get("timemachine_net_avg_R", 0.0))))
        solver.add(z_spread == _z3_real(float(telemetry.get("spread_pct_of_sl", 0.0))))
        solver.add(z_vol == _z3_real(float(telemetry.get("vol_pct", 0.0))))
        solver.add(z_shock == _z3_real(float(telemetry.get("cross_asset_shock_z", 0.0))))
        solver.add(z_rollover == z3.BoolVal(bool(telemetry.get("rollover_illiquidity_flag", False))))

        # Hard domain safety axioms: toxic setups CANNOT be APPROVED
        solver.add(
            z3.Implies(
                z3.Or(
                    z_net_r < z3.RealVal("0.10"),
                    z_spread > z3.RealVal("15.0"),
                    z3.And(z_rollover, z_vol > z3.RealVal("85.0")),
                    z_shock > z3.RealVal("3.5"),
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
                zf = z3.Real(vc.field)
                z_fields[vc.field] = zf
                solver.add(zf == _z3_real(float(telemetry[vc.field])))
            z_lim = _z3_real(vc.limit)
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
        lat_ms = (time.perf_counter() - t0) * 1000.0
        if chk == z3.unsat:
            return (
                False,
                round(lat_ms, 3),
                "Z3_UNSAT_HALLUCINATION: AI decision violates formal SMT safety axioms",
            )
        return True, round(lat_ms, 3), f"Z3 SAT ({lat_ms:.2f}ms)"

    @staticmethod
    def z3_pure_math_axiom_gate(telemetry: dict[str, Any]) -> tuple[str, str]:
        """Deterministic Z3-backed safety gate for pure_math fallback."""
        is_sat, lat_ms, _ = AegisVerifier.verify_z3_smt_axioms(telemetry, "APPROVE", [])
        if not is_sat:
            return (
                "VETO",
                f"Z3 Axiom Gate VETO ({lat_ms:.2f}ms): net_avg_R={telemetry.get('timemachine_net_avg_R')}, "
                f"spread={telemetry.get('spread_pct_of_sl')}%, shock_z={telemetry.get('cross_asset_shock_z')}",
            )
        return "APPROVE", f"Z3 Axiom Gate SAT ({lat_ms:.2f}ms)"

    # ------------------------------------------------------------------
    # Shield 5: Borderline Order-Swap Semantic Invariance
    # ------------------------------------------------------------------
    @staticmethod
    def is_borderline_high_stakes(telemetry: dict[str, Any]) -> bool:
        """True when 0.10 <= net_avg_R <= 0.20 and vol_pct >= 80.0."""
        net_r = float(telemetry.get("timemachine_net_avg_R", 0.0))
        vol_pct = float(telemetry.get("vol_pct", 0.0))
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

        for match in _NUMBER_TOKEN_RE.finditer(reason):
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

        # Shield 4: Microsoft Z3 Formal SMT Consistency & Axiom Prover
        z3_sat, z3_ms, z3_msg = self.verify_z3_smt_axioms(telemetry, raw_dec, verified_claims)
        if not z3_sat:
            violations.append("Z3_UNSAT_HALLUCINATION")

        # Shield 6: Split Conformal Prediction calibration p-value
        conformal_p = 0.5
        if self.ledger_db is not None:
            conformal_p = self.ledger_db.get_conformal_p_value(provider, model_id, conf)

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
            risk_multiplier=0.0 if raw_dec == "VETO" else mult,
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
        """Build a known Z3-violating toxic setup for Shield 7 Canary Poison Probe."""
        from apex.ai.veto_committee import CandidateSetupAuditContext

        return CandidateSetupAuditContext(
            symbol="CANARY_TOXIC",
            side=Side.BUY,
            timeframe="M5",
            entry_price=100.0,
            stop_loss=99.0,
            take_profit=101.5,
            regime="MEAN_REVERTING_CHOP",
            hurst_exponent=0.41,
            kaufman_er=0.14,
            volatility_percentile=92.0,
            dom_obi=-0.35,
            spread_pct_of_sl=0.275,  # 27.5% of SL!
            net_avg_R=-0.45,         # Deeply negative EV!
            prob_profit=0.32,
            cvar_95_R=-2.10,
            recommended_risk_pct=0.50,
            session_utc_iso="2026-11-12T21:55:00Z",
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
