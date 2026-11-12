"""Apex Aegis — 6-Shield Neuro-Symbolic Anti-Hallucination Firewall (`apex/ai/aegis_verifier.py`).

Wraps `AIVetoCommittee` with deterministic symbolic & structural invariants:
  - Shield 1: One-Way Subtract-Only Diode (`APPROVE`/`VETO`/`ABSTAIN`, `risk_multiplier` in `[0.0, 1.0]`,
              never triggers quant-rejected setups, never alters SL/TP, never exceeds Quarter-Kelly / 0.50%).
  - Shield 2: Ticker-Blinded Telemetry Normalization (strips ticker & raw prices, normalizes to `100.00`
              base + `R`-multiples + UTC HH:MM, enforces `closed_world_contract`, SHA-256 input fingerprint).
  - Shield 3: Proof-Carrying Atomic Claim AST Verifier (deterministic Python verification of `field`,
              `cited_value` within `1e-4`, `<op> limit` truth, and regex/lexer scan of `reason` for
              invented numbers, dates, external news/FOMC/CPI/earnings, or decision contradictions).
  - Shield 4: Counter-Abductive Framing & `ABSTAIN` Handling (requires >=1 grounded bull & bear claim;
              clean pass-through to `pure_math` on `ABSTAIN` or low confidence).
  - Shield 5: Canary Trap Audit (`run_canary_integrity_probe` injects mathematically toxic setup
              `net_avg_R=-0.45, spread_pct_of_sl=27.5%` and quarantines any model that approves it).
  - Shield 6: Shadow Counterfactual Grading & Brier Calibration (wired to `WalkForwardLedgerDB`).
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

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
    """Full 6-Shield symbolic verification outcome."""

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


@dataclass(frozen=True, slots=True)
class DiodeEnforcementResult:
    """Shield 1 One-Way Subtract-Only Diode enforcement output."""

    final_approved: bool
    final_decision: str
    final_risk_pct: float
    risk_multiplier: float
    stop_loss: float
    take_profit: float
    note: str


@dataclass(frozen=True, slots=True)
class CanaryProbeResult:
    """Shield 5 Canary Trap Audit result."""

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


class AegisVerifier:
    """6-Shield Neuro-Symbolic Anti-Hallucination Firewall."""

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
    # Shield 1: One-Way Subtract-Only Diode
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
        """Enforce one-way subtract-only diode invariants in code.

        - AI can NEVER trigger an entry when `quant_approved` is False.
        - AI can ONLY return `APPROVE`, `VETO`, or `ABSTAIN`.
        - `risk_multiplier` is strictly clamped to `[0.0, 1.0]`.
        - `sl_price` and `tp_price` are returned unaltered.
        - Final risk % never exceeds `min(recommended_risk_pct, max_risk_pct)` (0.50% cap).
        """
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
                note="Shield 1 Diode: Quant rejected setup; AI cannot override or trigger entry.",
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
                note="Shield 1 Diode: AI ABSTAIN -> deterministic quant pass-through.",
            )

        final_risk = round(base_risk * mult, 4)
        return DiodeEnforcementResult(
            final_approved=True,
            final_decision="APPROVE",
            final_risk_pct=final_risk,
            risk_multiplier=round(mult, 4),
            stop_loss=float(sl_price),
            take_profit=float(tp_price),
            note=f"Shield 1 Diode: Approved with {mult:.2f}x multiplier ({final_risk:.4f}% <= {max_risk_pct:.2f}%).",
        )

    # ------------------------------------------------------------------
    # Shield 2: Ticker-Blinded Telemetry Normalization
    # ------------------------------------------------------------------
    @staticmethod
    def normalize_ticker_blinded_telemetry(
        ctx: CandidateSetupAuditContext,
        asset_class: str | None = None,
    ) -> BlindedTelemetry:
        """Strip ticker name & raw price levels; replace with asset class, 100.00 base & R-multiples."""
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
    def build_aegis_prompt(blinded: BlindedTelemetry) -> str:
        """Build the Shield 2 & 3 Closed-World Proof-Carrying Prompt."""
        schema_example = (
            '{"bull_case_claim":{"field":"timemachine_net_avg_R","op":">=","cited_value":0.35,"limit":0.10},'
            '"bear_trap_claim":{"field":"spread_pct_of_sl","op":"<=","cited_value":4.0,"limit":15.0},'
            '"evidence_claims":[{"field":"timemachine_net_avg_R","op":">=","cited_value":0.35,"limit":0.10}],'
            '"decision":"APPROVE","confidence":0.85,"risk_multiplier":1.0,'
            '"reason":"timemachine_net_avg_R is 0.35 >= 0.10 and spread_pct_of_sl is 4.0 <= 15.0"}'
        )
        return (
            "Do NOT call any tools or read any files. Respond immediately.\n"
            "CLOSED-WORLD CONTRACT (STRICT):\n"
            "1. You MUST cite ONLY exact field names and numbers present in the Telemetry JSON below.\n"
            "2. FORBIDDEN: Do NOT invent numbers, dates, macro news, FOMC, CPI, NFP, Fed, or earnings.\n"
            "3. Provide BOTH a `bull_case_claim` and a `bear_trap_claim`, plus `evidence_claims`, where every claim is "
            '{"field":"<exact_key>","op":"<|<=|==|>=|>","cited_value":<exact_number>,"limit":<threshold>}.\n'
            "4. `decision` MUST be `APPROVE`, `VETO`, or `ABSTAIN`; `risk_multiplier` in [0.0, 1.0].\n"
            "5. In `reason`, reference ONLY telemetry numbers and your claim limits.\n"
            f"Telemetry JSON (SHA256={blinded.input_sha256[:12]}): {json.dumps(blinded.telemetry, separators=(',', ':'))}\n"
            f"Respond ONLY with a single-line JSON object matching this schema:\n{schema_example}"
        )

    # ------------------------------------------------------------------
    # Shield 3 & 4: Proof-Carrying AST Verifier + Counter-Abductive Check
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
        try:
            cited_val = float(claim_obj["cited_value"])
            limit_val = float(claim_obj["limit"])
        except (KeyError, TypeError, ValueError):
            return AtomicClaimResult(
                field=fld, op=op, cited_value=0.0, limit=0.0, actual_value=actual_val, valid=False,
                error=f"{role_name}: cited_value and limit must be numeric",
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

    def verify_response(
        self,
        payload: dict[str, Any],
        blinded: BlindedTelemetry,
    ) -> AegisVerificationResult:
        """Verify parsed AI JSON payload against Shields 1, 2, 3, and 4."""
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
        try:
            mult = float(payload.get("risk_multiplier", default_mult))
            if math.isnan(mult) or math.isinf(mult):
                raise ValueError("NaN multiplier")
            mult = max(0.0, min(1.0, mult))
        except (TypeError, ValueError):
            violations.append("INVALID_RISK_MULTIPLIER")
            mult = 1.0

        reason = str(payload.get("reason", "")).strip()

        # Verify bull_case_claim, bear_trap_claim, and evidence_claims
        verified_claims: list[AtomicClaimResult] = []
        bull_res: AtomicClaimResult | None = None
        bear_res: AtomicClaimResult | None = None

        if "bull_case_claim" in payload and payload["bull_case_claim"] is not None:
            bull_res = self._verify_single_claim(payload["bull_case_claim"], telemetry, "bull_case_claim")
            verified_claims.append(bull_res)
            if not bull_res.valid:
                violations.append("INVALID_BULL_CLAIM")

        if "bear_trap_claim" in payload and payload["bear_trap_claim"] is not None:
            bear_res = self._verify_single_claim(payload["bear_trap_claim"], telemetry, "bear_trap_claim")
            verified_claims.append(bear_res)
            if not bear_res.valid:
                violations.append("INVALID_BEAR_CLAIM")

        raw_ev_claims = payload.get("evidence_claims")
        ev_valid_count = 0
        if isinstance(raw_ev_claims, list):
            for idx, item in enumerate(raw_ev_claims):
                res = self._verify_single_claim(item, telemetry, f"evidence_claims[{idx}]")
                verified_claims.append(res)
                if res.valid:
                    ev_valid_count += 1
                else:
                    if "NUMERIC_HALLUCINATION" in res.error:
                        violations.append("NUMERIC_HALLUCINATION")
                    elif "MATH_CONTRADICTION" in res.error:
                        violations.append("OPERATOR_MATH_CONTRADICTION")
                    else:
                        violations.append("INVALID_EVIDENCE_CLAIM")
        elif raw_dec != "ABSTAIN":
            violations.append("MISSING_EVIDENCE_CLAIMS")

        # Shield 4: Counter-abductive requirement for APPROVE / VETO
        if raw_dec in ("APPROVE", "VETO"):
            if bull_res is None or not bull_res.valid or bear_res is None or not bear_res.valid:
                violations.append("MISSING_COUNTER_ABDUCTIVE_SIDE")
            if ev_valid_count == 0 and not (bull_res and bull_res.valid and bear_res and bear_res.valid):
                violations.append("NO_VALID_PROOF_CLAIMS")

        # Shield 3: Reason lexical scans
        # (a) External news / macro / date hallucination
        for pat in _EXTERNAL_NEWS_PATTERNS:
            if pat.search(reason):
                violations.append("EXTERNAL_NEWS_OR_DATE_HALLUCINATION")
                break

        # (b) Ticker leak check (if raw_symbol >= 3 chars appears in reason, model leaked or echoed)
        if len(blinded.raw_symbol) >= 3 and re.search(rf"\b{re.escape(blinded.raw_symbol)}\b", reason, re.IGNORECASE):
            violations.append("TICKER_LEAK_DETECTED")

        # (c) Unprovided number scan in `reason`
        allowed_nums: set[float] = {0.0, 1.0, 100.0, round(conf, 4), round(mult, 4)}
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

        # (d) Decision-reason contradiction
        if raw_dec == "APPROVE" and _APPROVE_CONTRADICTION_RE.search(reason):
            violations.append("DECISION_REASON_CONTRADICTION")
        elif raw_dec == "VETO" and _VETO_CONTRADICTION_RE.search(reason):
            violations.append("DECISION_REASON_CONTRADICTION")

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
                reason=f"HALLUCINATION_DETECTED ({', '.join(dedup_violations)}): {reason[:160]}",
                input_sha256=blinded.input_sha256,
                violation_codes=dedup_violations,
                verified_claims=tuple(verified_claims),
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
                reason=f"Aegis ABSTAIN pass-through to pure_math (conf={conf:.2f}): {reason}",
                input_sha256=blinded.input_sha256,
                violation_codes=(),
                verified_claims=tuple(verified_claims),
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
        )

    # ------------------------------------------------------------------
    # Shield 5: Canary Trap Audit
    # ------------------------------------------------------------------
    @staticmethod
    def build_canary_toxic_context() -> CandidateSetupAuditContext:
        """Build a known mathematically toxic setup for Shield 5 Canary Trap Audits."""
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
        """Run Shield 5 Canary Trap Audit. Quarantines model if it returns `APPROVE`."""
        toxic_ctx = self.build_canary_toxic_context()
        decision = evaluator_fn(toxic_ctx)
        dec_str = str(decision.decision).upper().strip()
        used_provider = decision.provider_used or provider
        used_model = decision.model_used or model_id

        approved_toxic = dec_str == "APPROVE" and not decision.bypassed
        if approved_toxic:
            reason = "EPISTEMICALLY_COMPROMISED: Approved Canary Trap (net_avg_R=-0.45, spread=27.5%)"
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
                reason=f"Canary Trap passed ({dec_str})",
                is_canary=True,
                predicted_net_r=toxic_ctx.net_avg_R,
            )

        return CanaryProbeResult(
            passed=True,
            quarantined=False,
            provider=used_provider,
            model_id=used_model,
            decision_returned=dec_str,
            detail=f"Canary Trap passed: returned {dec_str}",
        )
