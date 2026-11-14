"""Adaptive Multi-Factor Gemini Model & Quota Selector (`apex/ai/model_selector.py`).

Dynamically discovers available Gemini models across:
  - Google AI Studio REST API (`/v1beta/models`)
  - Google AI Pro CLI (`agy.exe -p` in isolated temp dir)

Ranks candidate models before every AI Risk Veto audit using 4 real-time factors:
  - Factor A: Remaining Quota & Budget Pacing (RPM/RPD tracking, 429 exponential cooldown, Pro conservation)
  - Factor B: Setup Complexity & Stakes (volatility percentile, borderline net_avg_R, risk %)
  - Factor C: Rolling EWMA Latency vs Timeframe Urgency (<8s target on M5)
  - Factor D: Live Epistemic Trust Score (from Aegis Shield 6 Shadow Counterfactual Grading)
"""

from __future__ import annotations

import datetime
import json
import logging
import subprocess
import tempfile
import time
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ModelTier(str, Enum):
    """Capability & cost tier for discovered Gemini models."""

    PRO_REASONING = "PRO_REASONING"
    FLASH_FAST = "FLASH_FAST"
    FLASH_LITE = "FLASH_LITE"


TIER_DEFAULT_LIMITS: dict[ModelTier, tuple[int, int, float]] = {
    # (rpm_limit, rpd_limit, initial_ewma_latency_ms)
    ModelTier.PRO_REASONING: (5, 100, 4500.0),
    ModelTier.FLASH_FAST: (15, 1500, 1200.0),
    ModelTier.FLASH_LITE: (30, 1500, 650.0),
}


@dataclass
class DiscoveredModel:
    """Tracked runtime state, quota counters, and EWMA latency for a single model."""

    model_id: str
    provider: str  # "studio_api" or "agy_pro"
    tier: ModelTier
    family: str = ""
    display_name: str = ""
    rpm_limit: int = 15
    rpd_limit: int = 1500
    daily_calls: int = 0
    last_reset_date: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    )
    minute_timestamps: list[float] = field(default_factory=list)
    ewma_latency_ms: float = 1200.0
    sample_count: int = 0
    consecutive_errors: int = 0
    cooldown_until_utc: float = 0.0
    last_error: str = ""
    last_used_utc: float = 0.0
    epistemic_trust_score: float = 1.0
    epistemic_quarantined: bool = False
    epistemic_quarantine_reason: str = ""

    def reset_daily_if_needed(self) -> None:
        today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
        if today != self.last_reset_date:
            self.daily_calls = 0
            self.last_reset_date = today

    def prune_minute_window(self, now_utc: float) -> None:
        cutoff = now_utc - 60.0
        if self.minute_timestamps:
            self.minute_timestamps = [ts for ts in self.minute_timestamps if ts > cutoff]


@dataclass(frozen=True, slots=True)
class SetupStakes:
    """Evaluated trade setup complexity & urgency."""

    stakes_score: float
    is_high_stakes: bool
    is_urgent: bool


@dataclass(frozen=True, slots=True)
class RankedModelChoice:
    """Single scored candidate model for execution/cascade."""

    model: DiscoveredModel
    score: float
    reason: str


def classify_model_family(model_id: str) -> str:
    """Group a model ID into its quota/billing family (e.g., argon, hydrogen, flash, pro)."""
    low = model_id.lower().strip()
    if low == "default":
        return "default"
    if "argon" in low or "orchestrator" in low:
        return "argon"
    if "hydrogen" in low:
        return "hydrogen"
    if "flash" in low:
        return "flash"
    if "claude-opus" in low:
        return "claude-opus"
    if "claude-sonnet" in low:
        return "claude-sonnet"
    if "gpt-oss" in low:
        return "gpt-oss"
    if "2.5-pro" in low:
        return "gemini-2.5-pro"
    if "pro" in low:
        return "pro"
    return low.split("-")[0] if "-" in low else low


def classify_model_tier(model_id: str) -> ModelTier:
    """Classify a Gemini model ID into PRO_REASONING, FLASH_FAST, or FLASH_LITE."""
    low = model_id.lower()
    if "lite" in low or "flash-low" in low:
        return ModelTier.FLASH_LITE
    if "pro" in low or "ultra" in low or "thinking" in low or "deep" in low or "hydrogen" in low or "-high" in low:
        return ModelTier.PRO_REASONING
    return ModelTier.FLASH_FAST


def compute_setup_stakes(
    volatility_percentile: float,
    net_avg_R: float,
    recommended_risk_pct: float,
    timeframe: str,
) -> SetupStakes:
    """Compute setup stakes [0.0, 1.0] and urgency flag from quantitative telemetry."""
    stakes = 0.0
    if volatility_percentile >= 80.0:
        stakes += 0.40
    elif volatility_percentile >= 60.0:
        stakes += 0.20

    if 0.10 <= net_avg_R <= 0.25:
        stakes += 0.35
    elif net_avg_R > 0.25:
        stakes += 0.15

    if recommended_risk_pct >= 0.75:
        stakes += 0.25
    elif recommended_risk_pct >= 0.50:
        stakes += 0.10

    stakes = min(1.0, round(stakes, 4))
    is_high_stakes = stakes >= 0.50
    is_urgent = timeframe.upper().strip() in ("M1", "M5")
    return SetupStakes(
        stakes_score=stakes,
        is_high_stakes=is_high_stakes,
        is_urgent=is_urgent,
    )


class AdaptiveModelSelector:
    """Dynamic Gemini Model Discovery + Multi-Factor Quota, Latency & Stakes Optimizer."""

    def __init__(
        self,
        soft_pro_threshold_pct: float = 0.70,
        discovery_ttl_seconds: float = 3600.0,
        ledger_db: Any = None,
    ) -> None:
        self.soft_pro_threshold_pct: float = float(soft_pro_threshold_pct)
        self.discovery_ttl_seconds: float = float(discovery_ttl_seconds)
        self._models: dict[tuple[str, str], DiscoveredModel] = {}
        self._family_cooldowns: dict[tuple[str, str], tuple[float, str]] = {}
        self._last_verified_model: dict[str, str] = {}
        self._last_studio_discovery_utc: float = 0.0
        self._last_agy_discovery_utc: float = 0.0
        self._ledger_db: Any = None
        if ledger_db is not None:
            self.set_ledger_db(ledger_db)

    def set_ledger_db(self, ledger_db: Any, now_utc: float | None = None) -> None:
        """Attach WalkForwardLedgerDB and restore any active persisted family cooldowns."""
        self._ledger_db = ledger_db
        if ledger_db is None:
            return
        loader = getattr(ledger_db, "load_active_family_cooldowns", None)
        if not callable(loader):
            return
        try:
            active = loader(now_utc=now_utc)
        except Exception:
            return
        for key, (until_ts, reason) in active.items():
            if ":" in key:
                prov, fam = key.split(":", 1)
                self._family_cooldowns[(prov.strip(), fam.strip().lower())] = (float(until_ts), reason)
            else:
                self._family_cooldowns[("agy_pro", key.strip().lower())] = (float(until_ts), reason)

    def quarantine_family(
        self,
        provider: str,
        family: str,
        duration_seconds: float = 3600.0,
        reason: str = "",
        now_utc: float | None = None,
    ) -> None:
        """Quarantine an entire model family (e.g., all gemini-4-argon-* variants) on credit/quota exhaustion."""
        ts = now_utc if now_utc is not None else time.time()
        fam = (family or "").lower().strip()
        if not fam or fam == "default":
            return
        until_ts = ts + float(duration_seconds)
        prev_until, _ = self._family_cooldowns.get((provider, fam), (0.0, ""))
        if until_ts > prev_until:
            self._family_cooldowns[(provider, fam)] = (until_ts, reason[:120])
        if self._ledger_db is not None:
            saver = getattr(self._ledger_db, "save_family_cooldown", None)
            if callable(saver):
                try:
                    saver(f"{provider}:{fam}", until_ts, reason[:120], ts)
                except Exception:
                    pass
        for m in self.get_models(provider):
            m_fam = m.family or classify_model_family(m.model_id)
            if m_fam == fam:
                m.cooldown_until_utc = max(m.cooldown_until_utc, until_ts)
                if reason:
                    m.last_error = reason[:120]
        if self._last_verified_model.get(provider):
            if classify_model_family(self._last_verified_model[provider]) == fam:
                self._last_verified_model.pop(provider, None)

    def is_family_quarantined(
        self,
        provider: str,
        family: str,
        now_utc: float | None = None,
    ) -> tuple[bool, int, str]:
        """Return `(quarantined, remaining_seconds, reason)` for `(provider, family)`."""
        ts = now_utc if now_utc is not None else time.time()
        fam = (family or "").lower().strip()
        until_ts, reason = self._family_cooldowns.get((provider, fam), (0.0, ""))
        if ts < until_ts:
            return True, max(1, int(until_ts - ts)), reason
        return False, 0, ""

    def has_any_quarantined_family(self, provider: str, now_utc: float | None = None) -> bool:
        """Return True if at least one named family for `provider` is currently quarantined."""
        ts = now_utc if now_utc is not None else time.time()
        return any(
            prov == provider and ts < until_ts
            for (prov, _fam), (until_ts, _reason) in self._family_cooldowns.items()
        )

    def register_model(
        self,
        provider: str,
        model_id: str,
        tier: ModelTier | None = None,
        display_name: str = "",
        rpm_limit: int | None = None,
        rpd_limit: int | None = None,
        initial_latency_ms: float | None = None,
    ) -> DiscoveredModel:
        """Register or update a discovered model while preserving live quota/latency counters."""
        key = (provider, model_id)
        resolved_tier = tier or classify_model_tier(model_id)
        def_rpm, def_rpd, def_lat = TIER_DEFAULT_LIMITS[resolved_tier]

        if key in self._models:
            existing = self._models[key]
            if display_name:
                existing.display_name = display_name
            return existing

        model = DiscoveredModel(
            model_id=model_id,
            provider=provider,
            tier=resolved_tier,
            family=classify_model_family(model_id),
            display_name=display_name or model_id,
            rpm_limit=rpm_limit if rpm_limit is not None else def_rpm,
            rpd_limit=rpd_limit if rpd_limit is not None else def_rpd,
            ewma_latency_ms=initial_latency_ms if initial_latency_ms is not None else def_lat,
        )
        self._models[key] = model
        return model

    def get_models(self, provider: str | None = None) -> list[DiscoveredModel]:
        """Return all registered models (optionally filtered by provider)."""
        models = list(self._models.values())
        if provider is not None:
            models = [m for m in models if m.provider == provider]
        return models

    def discover_studio_models(
        self,
        api_key: str,
        timeout: float = 4.0,
        force: bool = False,
    ) -> list[DiscoveredModel]:
        """Query Google AI Studio `/v1beta/models` for live `generateContent`-capable Gemini models."""
        if not api_key:
            return []

        now_utc = time.time()
        existing = self.get_models("studio_api")
        if not force and existing and (now_utc - self._last_studio_discovery_utc < self.discovery_ttl_seconds):
            return existing

        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            logger.debug("AI Studio model discovery failed (%s); using fallback list if empty", exc)
            if not existing:
                # Seed safe baseline models so studio_api can still attempt calls if /models endpoint blocked
                self.register_model("studio_api", "gemini-2.5-flash", ModelTier.FLASH_FAST)
                self.register_model("studio_api", "gemini-2.5-pro", ModelTier.PRO_REASONING)
            return self.get_models("studio_api")

        raw_models = data.get("models", []) if isinstance(data, dict) else []
        found_any = False
        for item in raw_models:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", ""))
            methods = item.get("supportedGenerationMethods", [])
            if "generateContent" not in methods:
                continue
            model_id = name.removeprefix("models/").strip()
            low = model_id.lower()
            if "gemini" not in low:
                continue
            # Exclude non-text specialized models (tts, image-generation, robotics, computer-use)
            if any(skip in low for skip in ("embedding", "tts", "imagen", "veo", "aqa", "robotics", "computer-use")):
                continue
            tier = classify_model_tier(model_id)
            disp = str(item.get("displayName", model_id))
            self.register_model("studio_api", model_id, tier=tier, display_name=disp)
            found_any = True

        if not found_any and not existing:
            self.register_model("studio_api", "gemini-2.5-flash", ModelTier.FLASH_FAST)

        self._last_studio_discovery_utc = now_utc
        return self.get_models("studio_api")

    def discover_agy_models(
        self,
        agy_exe: str | None,
        probe_cli: bool = False,
        force: bool = False,
    ) -> list[DiscoveredModel]:
        """Discover live Gemini models from `agy.exe models` and optionally benchmark in isolated dir."""
        if not agy_exe:
            return []

        now_utc = time.time()
        existing = self.get_models("agy_pro")
        if not force and existing and not probe_cli and (now_utc - self._last_agy_discovery_utc < self.discovery_ttl_seconds):
            return existing

        iso_dir = Path(tempfile.gettempdir()) / "pocketbot_agy_isolated"
        iso_dir.mkdir(parents=True, exist_ok=True)

        # Dynamically query `agy.exe models` for live Antigravity Pro Gemini models
        discovered_any = False
        try:
            proc_m = subprocess.run(
                [agy_exe, "models"],
                cwd=str(iso_dir),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=10.0,
                encoding="utf-8",
                errors="replace",
            )
            if proc_m.returncode == 0 and proc_m.stdout:
                for line in proc_m.stdout.splitlines():
                    parts = line.strip().split("\t")
                    if not parts:
                        continue
                    mid = parts[0].strip()
                    disp = parts[1].strip() if len(parts) > 1 else mid
                    if not mid or "gemini" not in mid.lower():
                        continue
                    if "xhigh" in mid.lower():
                        continue  # Skip ultra-heavy xhigh tier for real-time trading vetoes
                    tier = classify_model_tier(mid)
                    self.register_model(
                        "agy_pro",
                        mid,
                        tier=tier,
                        display_name=disp,
                        rpm_limit=15,
                        rpd_limit=1000,
                    )
                    discovered_any = True
        except Exception as exc:
            logger.debug("agy models discovery failed (%s); falling back to default", exc)

        if not discovered_any and not existing:
            self.register_model(
                "agy_pro",
                "default",
                tier=ModelTier.FLASH_FAST,
                display_name="Antigravity Default",
                rpm_limit=15,
                rpd_limit=1000,
                initial_latency_ms=4000.0,
            )

        if probe_cli:
            ranked = self.rank_models(
                "agy_pro",
                SetupStakes(stakes_score=0.2, is_high_stakes=False, is_urgent=True),
                now_utc=now_utc,
            )
            probe_target = ranked[0].model if ranked else self.get_models("agy_pro")[0]
            cmd = [
                agy_exe,
                "-p",
                'Reply ONLY with JSON: {"ok": true}',
                "--output-format",
                "json",
                "--print-timeout",
                "28s",
                "--disable-slash-commands",
            ]
            if probe_target.model_id != "default":
                cmd.extend(["--model", probe_target.model_id])
            else:
                cmd.extend(["--effort", "low"])
            t0 = time.perf_counter()
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(iso_dir),
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    timeout=32.0,
                    encoding="utf-8",
                    errors="replace",
                )
                lat_ms = (time.perf_counter() - t0) * 1000.0
                if proc.returncode == 0 and (proc.stdout or "").strip():
                    self.record_success("agy_pro", probe_target.model_id, lat_ms, now_utc=now_utc)
                else:
                    self.record_failure("agy_pro", probe_target.model_id, f"probe rc={proc.returncode}", now_utc=now_utc)
            except Exception as exc:
                self.record_failure("agy_pro", probe_target.model_id, str(exc), now_utc=now_utc)

        self._last_agy_discovery_utc = now_utc
        return self.get_models("agy_pro")

    def score_model(
        self,
        model: DiscoveredModel,
        stakes: SetupStakes,
        now_utc: float | None = None,
    ) -> tuple[float, str]:
        """Compute multi-factor selection score [0.0, 1.0], or -1.0 if quarantined/exhausted."""
        ts = now_utc if now_utc is not None else time.time()
        model.reset_daily_if_needed()
        model.prune_minute_window(ts)

        fam = model.family or classify_model_family(model.model_id)
        fam_q, fam_rem, fam_reason = self.is_family_quarantined(model.provider, fam, now_utc=ts)
        if fam_q:
            return -1.0, f"family '{fam}' quarantined ({fam_rem}s: {fam_reason or 'quota'})"

        if ts < model.cooldown_until_utc:
            rem = int(model.cooldown_until_utc - ts)
            return -1.0, f"cooldown ({rem}s)"

        if model.epistemic_quarantined:
            return -1.0, f"epistemically quarantined ({model.epistemic_quarantine_reason or 'compromised'})"

        rpm_used = len(model.minute_timestamps)
        if rpm_used >= model.rpm_limit:
            return -1.0, f"RPM exhausted ({rpm_used}/{model.rpm_limit})"

        if model.daily_calls >= model.rpd_limit:
            return -1.0, f"RPD exhausted ({model.daily_calls}/{model.rpd_limit})"

        # Factor A: Remaining Quota & Budget Pacing
        rpm_rem_frac = 1.0 - (rpm_used / max(1, model.rpm_limit))
        rpd_used_frac = model.daily_calls / max(1, model.rpd_limit)
        rpd_rem_frac = 1.0 - rpd_used_frac
        quota_score = 0.5 * rpm_rem_frac + 0.5 * rpd_rem_frac
        if model.tier == ModelTier.PRO_REASONING and rpd_used_frac >= self.soft_pro_threshold_pct:
            quota_score *= 0.25  # Conserve remaining Pro quota by shifting to Flash

        # Factor B: Setup Complexity & Stakes
        if stakes.is_high_stakes:
            stakes_fit = {
                ModelTier.PRO_REASONING: 1.0,
                ModelTier.FLASH_FAST: 0.75,
                ModelTier.FLASH_LITE: 0.45,
            }[model.tier]
        else:
            stakes_fit = {
                ModelTier.FLASH_FAST: 1.0,
                ModelTier.FLASH_LITE: 0.90,
                ModelTier.PRO_REASONING: 0.55,
            }[model.tier]

        # Factor C: Rolling EWMA Latency vs Timeframe Urgency (<8s target on M5)
        lat_s = model.ewma_latency_ms / 1000.0
        if stakes.is_urgent:
            if lat_s > 8.0:
                latency_score = max(0.02, 0.22 - 0.03 * (lat_s - 8.0))
            elif lat_s > 6.0:
                latency_score = max(0.22, 0.45 - 0.115 * (lat_s - 6.0))
            else:
                latency_score = max(0.25, 1.0 - (lat_s / 8.0))
        else:
            latency_score = max(0.20, 1.0 - (lat_s / 15.0))

        # Factor D: Live Epistemic Trust Score (from Aegis Shield 6)
        trust_score = max(0.0, min(1.0, float(model.epistemic_trust_score)))

        # Slight preference for newer generation (2.5 / 3.x / 4.x) over older (1.5 / 2.0) on ties
        gen_bonus = 0.0
        low_id = model.model_id.lower()
        if "2.5" in low_id or "3." in low_id or "4-" in low_id or "4." in low_id:
            gen_bonus = 0.03
        elif "2.0" in low_id:
            gen_bonus = 0.01

        total = min(
            1.0,
            0.28 * quota_score
            + 0.27 * stakes_fit
            + 0.25 * latency_score
            + 0.20 * trust_score
            + gen_bonus,
        )
        reason = (
            f"score={total:.3f} (quota={quota_score:.2f}, stakes={stakes_fit:.2f}, "
            f"lat={lat_s:.1f}s->fit={latency_score:.2f}, trust={trust_score:.2f})"
        )
        return round(total, 4), reason

    def rank_models(
        self,
        provider: str,
        stakes: SetupStakes,
        now_utc: float | None = None,
    ) -> list[RankedModelChoice]:
        """Rank all available non-quarantined models for `provider` from highest to lowest score."""
        ts = now_utc if now_utc is not None else time.time()
        candidates: list[RankedModelChoice] = []
        for m in self.get_models(provider):
            sc, reason = self.score_model(m, stakes, now_utc=ts)
            if sc >= 0.0:
                candidates.append(RankedModelChoice(model=m, score=sc, reason=reason))
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    def rank_cross_family_candidates(
        self,
        provider: str,
        stakes: SetupStakes,
        now_utc: float | None = None,
    ) -> list[RankedModelChoice]:
        """Pick the highest-scoring model from EACH distinct non-quarantined model family."""
        ts = now_utc if now_utc is not None else time.time()
        ranked = self.rank_models(provider, stakes, now_utc=ts)
        seen_families: set[str] = set()
        cross_family: list[RankedModelChoice] = []
        has_named = any((c.model.family or classify_model_family(c.model.model_id)) != "default" for c in ranked)
        for choice in ranked:
            fam = choice.model.family or classify_model_family(choice.model.model_id)
            if fam == "default" and (has_named or self.has_any_quarantined_family(provider, now_utc=ts)):
                continue
            if fam in seen_families:
                continue
            seen_families.add(fam)
            cross_family.append(choice)
        return cross_family

    def record_success(
        self,
        provider: str,
        model_id: str,
        latency_ms: float,
        now_utc: float | None = None,
    ) -> None:
        """Record a successful call, updating quota counters and EWMA latency."""
        ts = now_utc if now_utc is not None else time.time()
        model = self.register_model(provider, model_id)
        model.reset_daily_if_needed()
        model.prune_minute_window(ts)
        model.daily_calls += 1
        model.minute_timestamps.append(ts)
        model.last_used_utc = ts
        model.consecutive_errors = 0
        model.cooldown_until_utc = 0.0
        model.last_error = ""
        self._last_verified_model[provider] = model_id

        if model.sample_count == 0:
            model.ewma_latency_ms = float(latency_ms)
        else:
            model.ewma_latency_ms = 0.7 * model.ewma_latency_ms + 0.3 * float(latency_ms)
        model.sample_count += 1

    def record_failure(
        self,
        provider: str,
        model_id: str,
        error_msg: str,
        is_quota_or_429: bool = False,
        now_utc: float | None = None,
    ) -> None:
        """Record a model failure and apply exponential or retirement cooldown."""
        ts = now_utc if now_utc is not None else time.time()
        model = self.register_model(provider, model_id)
        model.consecutive_errors += 1
        model.last_error = error_msg[:120]
        low_err = error_msg.lower()

        if any(tok in low_err for tok in ("404", "not found", "deprecated", "unsupported")):
            # Model retired or unsupported -> quarantine 1 hour
            model.cooldown_until_utc = ts + 3600.0
        elif is_quota_or_429 or any(
            tok in low_err
            for tok in ("429", "quota", "rate", "exhaust", "resource_exhausted", "credits", "balance", "too low")
        ):
            # Quarantine the entire model family for 60 minutes so we never waste retries on -low/-medium siblings
            fam = model.family or classify_model_family(model_id)
            self.quarantine_family(provider, fam, duration_seconds=3600.0, reason=error_msg[:120], now_utc=ts)
            model.cooldown_until_utc = max(model.cooldown_until_utc, ts + 3600.0)
        elif model.consecutive_errors >= 2:
            # Repeated timeout/error -> 10 min cooldown
            model.cooldown_until_utc = ts + 600.0

    def update_epistemic_trust(
        self,
        provider: str,
        model_id: str,
        trust_score: float,
        quarantined: bool = False,
        reason: str = "",
    ) -> None:
        """Update Factor D epistemic trust score (and optional quarantine) from Shield 6."""
        model = self.register_model(provider, model_id)
        model.epistemic_trust_score = max(0.0, min(1.0, float(trust_score)))
        if quarantined:
            model.epistemic_quarantined = True
            model.epistemic_quarantine_reason = reason or "EPISTEMICALLY_COMPROMISED"

    def quarantine_epistemically(self, provider: str, model_id: str, reason: str) -> None:
        """Immediately quarantine a model as epistemically compromised (e.g. Canary Trap failure)."""
        model = self.register_model(provider, model_id)
        model.epistemic_trust_score = 0.0
        model.epistemic_quarantined = True
        model.epistemic_quarantine_reason = reason or "EPISTEMICALLY_COMPROMISED"

    def get_quota_health_summary(
        self,
        provider: str,
        now_utc: float | None = None,
    ) -> dict[str, Any]:
        """Return compact quota health, total daily calls, avg latency, and top model for UI badges."""
        ts = now_utc if now_utc is not None else time.time()
        models = self.get_models(provider)
        if not models:
            return {
                "health": "UNCONFIGURED",
                "calls_today": 0,
                "avg_latency_s": 0.0,
                "selected_model": "none",
            }

        calls_today = 0
        active_models: list[DiscoveredModel] = []
        conserving = False
        for m in models:
            m.reset_daily_if_needed()
            calls_today += m.daily_calls
            if ts >= m.cooldown_until_utc and m.daily_calls < m.rpd_limit:
                active_models.append(m)
            if m.tier == ModelTier.PRO_REASONING and (m.daily_calls / max(1, m.rpd_limit)) >= self.soft_pro_threshold_pct:
                conserving = True

        if not active_models:
            return {
                "health": "COOLDOWN",
                "calls_today": calls_today,
                "avg_latency_s": 0.0,
                "selected_model": "all_families_exhausted",
            }

        # Rank with neutral stakes for display, preferring the last live-verified working model if still active
        ranked = self.rank_models(
            provider,
            SetupStakes(stakes_score=0.3, is_high_stakes=False, is_urgent=False),
            now_utc=ts,
        )
        verified_id = self._last_verified_model.get(provider)
        verified_model = next((c.model for c in ranked if c.model.model_id == verified_id), None)
        best = verified_model or (ranked[0].model if ranked else active_models[0])
        avg_lat_s = best.ewma_latency_ms / 1000.0
        health = "CONSERVING" if conserving else "HEALTHY"
        return {
            "health": health,
            "calls_today": calls_today,
            "avg_latency_s": round(avg_lat_s, 1),
            "selected_model": best.model_id,
        }
