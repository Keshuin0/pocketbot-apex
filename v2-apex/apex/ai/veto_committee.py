"""Hybrid Self-Detecting 3-Tier AI Router & Event-Driven Risk Veto Committee (`apex/ai/veto_committee.py`).

Supports three self-detecting execution tiers with automatic hot-reloading of `.env` and
`config/apex_config.json` (via file `mtime` inspection before every evaluation) and zero-crash
cascade fallback:

  1. ``studio_api``: Auto-detected READY when a non-empty `GEMINI_API_KEY` exists in `.env`
     (or process environment). Calls Gemini REST `generateContent` with a fast 4.0s timeout.
  2. ``agy_pro``: Auto-detected READY when `agy.exe` is found (`shutil.which("agy")` or
     `%LOCALAPPDATA%\\agy\\bin\\agy.exe`) and authenticated with Google AI Pro. Executes headless
     `agy -p "<prompt>" --output-format json --print-timeout 15s`, safely stripping any ANSI
     sequences or Windows non-TTY wrapper lines before parsing the JSON response.
  3. ``pure_math``: Always READY (0ms, $0 token cost) — deterministic quantitative pass-through
     relying strictly on `RegimeClassifier` + `MomentumDOMGate` + `10,000-Path Block-Bootstrap TimeMachine v2`
     + `6 Hard Pre-Trade Circuit Breakers`.

User-Configurable Priority Waterfall in `config/apex_config.json` under `filters.ai_veto`:
  - `"mode": "auto_studio_first"` -> `["studio_api", "agy_pro", "pure_math"]` (Default)
  - `"mode": "auto_pro_first"`    -> `["agy_pro", "studio_api", "pure_math"]`
  - `"mode": "force_studio"`      -> `["studio_api", "pure_math"]`
  - `"mode": "force_pro"`         -> `["agy_pro", "pure_math"]`
  - `"mode": "force_pure_math"`   -> `["pure_math"]`
  - Custom `"priority_order"` array override when `"mode": "custom"`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Sequence

import aiohttp
from dotenv import dotenv_values

from apex.ai.model_selector import AdaptiveModelSelector, compute_setup_stakes
from apex.feeds.models import Side

logger = logging.getLogger(__name__)

_ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


class AIProviderName(str, Enum):
    """Supported tiers in the Hybrid 3-Tier AI Router."""

    STUDIO_API = "studio_api"
    AGY_PRO = "agy_pro"
    PURE_MATH = "pure_math"


class AIVetoMode(str, Enum):
    """Configurable routing modes in `config/apex_config.json`."""

    AUTO_STUDIO_FIRST = "auto_studio_first"
    AUTO_PRO_FIRST = "auto_pro_first"
    FORCE_STUDIO = "force_studio"
    FORCE_PRO = "force_pro"
    FORCE_PURE_MATH = "force_pure_math"
    CUSTOM = "custom"


MODE_DEFAULT_PRIORITY: dict[str, tuple[str, ...]] = {
    AIVetoMode.AUTO_STUDIO_FIRST.value: ("studio_api", "agy_pro", "pure_math"),
    AIVetoMode.AUTO_PRO_FIRST.value: ("agy_pro", "studio_api", "pure_math"),
    AIVetoMode.FORCE_STUDIO.value: ("studio_api", "pure_math"),
    AIVetoMode.FORCE_PRO.value: ("agy_pro", "pure_math"),
    AIVetoMode.FORCE_PURE_MATH.value: ("pure_math",),
}


@dataclass(frozen=True, slots=True)
class AIProviderStatus:
    """Live discovery status for an individual AI provider tier."""

    provider: str
    ready: bool
    detail: str
    executable_or_endpoint: str = ""


@dataclass(frozen=True, slots=True)
class HybridAIRouterStatus:
    """Snapshot of the 3-tier AI router state for TUI Dashboard & CLI tools."""

    enabled: bool
    mode: str
    priority_order: tuple[str, ...]
    active_primary: str
    fallback_chain: tuple[str, ...]
    studio_status: AIProviderStatus
    agy_pro_status: AIProviderStatus
    pure_math_status: AIProviderStatus
    model_quota_summary: dict[str, Any] = field(default_factory=dict)

    @property
    def dashboard_badge(self) -> str:
        """Format compact header badge, e.g. `AI Brain: AGY_PRO [Auto -> Pure Math fallback]`."""
        primary_upper = self.active_primary.upper()
        if not self.enabled or self.active_primary == AIProviderName.PURE_MATH.value:
            return "AI Brain: PURE_MATH [0ms Deterministic Quant Only]"

        label_map = {
            "studio_api": " AI Studio",
            "agy_pro": "AGY_PRO",
            "pure_math": "Pure Math",
        }
        fallbacks = [label_map.get(p, p.upper()).strip() for p in self.fallback_chain]
        fb_str = " -> ".join(fallbacks) if fallbacks else "Pure Math"
        prefix = "Auto" if self.mode.startswith("auto_") else "Forced"
        return f"AI Brain: {primary_upper} [{prefix} -> {fb_str} fallback]"

    @property
    def model_quota_badge(self) -> str:
        """Format live selected model, quota health, and rolling latency for the TUI cockpit."""
        primary_upper = self.active_primary.upper()
        if not self.enabled or self.active_primary == AIProviderName.PURE_MATH.value:
            return "AI Brain: PURE_MATH [0ms Deterministic Quant Only]"
        q = self.model_quota_summary or {}
        sel = q.get("selected_model", "auto")
        health = q.get("health", "HEALTHY")
        calls = int(q.get("calls_today", 0))
        avg_s = float(q.get("avg_latency_s", 0.0))
        return f"AI Brain: {primary_upper} [Model: {sel} | Quota: {health} ({calls} calls today) | Avg {avg_s:.1f}s]"


@dataclass(frozen=True, slots=True)
class CandidateSetupAuditContext:
    """Quantitative context passed to the Event-Driven AI Risk Veto Committee."""

    symbol: str
    side: Side
    timeframe: str
    entry_price: float
    stop_loss: float
    take_profit: float
    regime: str
    hurst_exponent: float
    kaufman_er: float
    volatility_percentile: float
    dom_obi: float | None
    spread_pct_of_sl: float
    net_avg_R: float
    prob_profit: float
    cvar_95_R: float
    recommended_risk_pct: float
    session_utc_iso: str = ""
    bar_timestamp: float = 0.0


@dataclass(frozen=True, slots=True)
class AIVetoDecision:
    """Structured JSON decision returned by the Hybrid 3-Tier AI Router."""

    decision: str  # "APPROVE" or "VETO"
    confidence: float  # [0.0, 1.0]
    reason: str
    latency_ms: float
    bypassed: bool = False
    provider_used: str = "pure_math"
    cascade_path: tuple[str, ...] = field(default_factory=lambda: ("pure_math",))
    model_used: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
            "latency_ms": round(self.latency_ms, 2),
            "bypassed": self.bypassed,
            "provider_used": self.provider_used,
            "cascade_path": list(self.cascade_path),
            "model_used": self.model_used,
        }


class AIVetoCommittee:
    """Hybrid Self-Detecting 3-Tier AI Router (`apex/ai/veto_committee.py`).

    Monitors `.env` and `config/apex_config.json` modification times (`mtime`) before each
    evaluation so changing API keys, logging into `agy.exe`, or switching priority modes
    takes effect immediately in real time without restarting the bot.
    """

    GEMINI_ENDPOINT_TEMPLATE: str = (
        "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    )
    DEFAULT_STUDIO_MODEL: str = "gemini-2.5-flash"

    def __init__(
        self,
        enabled: bool = True,
        api_key: str | None = None,
        model: str = DEFAULT_STUDIO_MODEL,
        timeout_seconds: float = 4.0,
        agy_timeout_seconds: float = 15.0,
        mode: str = AIVetoMode.AUTO_STUDIO_FIRST.value,
        priority_order: Sequence[str] | None = None,
        env_path: str | Path = ".env",
        config_path: str | Path = "config/apex_config.json",
        agy_executable_override: str | None = None,
        probe_agy_auth: bool = False,
    ) -> None:
        self.env_path = Path(env_path)
        self.config_path = Path(config_path)
        self._explicit_api_key: str | None = api_key
        self._explicit_enabled: bool = bool(enabled)
        self._explicit_mode: str = mode
        self._explicit_priority: tuple[str, ...] | None = (
            tuple(priority_order) if priority_order is not None else None
        )
        self.model: str = model
        self.timeout_seconds: float = min(float(timeout_seconds), 4.0)
        self.agy_timeout_seconds: float = float(agy_timeout_seconds)
        self._agy_executable_override: str | None = agy_executable_override

        self._env_mtime: float = -1.0
        self._config_mtime: float = -1.0

        self.enabled: bool = bool(enabled)
        self.mode: str = str(mode)
        self.api_key: str = (api_key or "").strip()
        self._priority_order: tuple[str, ...] = self._resolve_priority_tuple(self.mode, priority_order)

        self._cached_agy_path: str | None = None
        self._cached_agy_auth_ok: bool | None = None
        self._verdict_cache: dict[tuple[str, str, str, int], AIVetoDecision] = {}
        self._agy_consecutive_failures: int = 0
        self._agy_cooldown_until_utc: float = 0.0
        self.min_net_avg_R: float = 0.10
        self.model_selector: AdaptiveModelSelector = AdaptiveModelSelector()
        self._recent_ai_events: list[dict[str, Any]] = []

        # Perform initial load of .env and config/apex_config.json
        self.hot_reload_if_needed(force=True, probe_agy_auth=probe_agy_auth)

    @classmethod
    def get_isolated_temp_dir(cls) -> str:
        """Return an isolated directory with zero .agents folder or repo configs to avoid MCP timeouts."""
        iso_dir = Path(tempfile.gettempdir()) / "pocketbot_agy_isolated"
        iso_dir.mkdir(parents=True, exist_ok=True)
        agents_sub = iso_dir / ".agents"
        if agents_sub.exists():
            try:
                shutil.rmtree(agents_sub, ignore_errors=True)
            except Exception:
                pass
        return str(iso_dir)

    @staticmethod
    def _resolve_priority_tuple(
        mode: str,
        custom_order: Sequence[str] | None = None,
    ) -> tuple[str, ...]:
        norm_mode = str(mode or AIVetoMode.AUTO_STUDIO_FIRST.value).lower().strip()
        if norm_mode == AIVetoMode.FORCE_PURE_MATH.value:
            return (AIProviderName.PURE_MATH.value,)
        if norm_mode in MODE_DEFAULT_PRIORITY:
            chain = list(MODE_DEFAULT_PRIORITY[norm_mode])
        elif custom_order:
            chain = [str(x).lower().strip() for x in custom_order if str(x).strip()]
        else:
            chain = list(MODE_DEFAULT_PRIORITY[AIVetoMode.AUTO_STUDIO_FIRST.value])

        # Guarantee pure_math is always at the bottom of the cascade so execution never crashes
        if AIProviderName.PURE_MATH.value not in chain:
            chain.append(AIProviderName.PURE_MATH.value)
        return tuple(chain)

    @staticmethod
    def locate_agy_executable(override: str | None = None) -> str | None:
        """Locate `agy.exe` via `shutil.which("agy")` or `%LOCALAPPDATA%\\agy\\bin\\agy.exe`."""
        if override is not None:
            if override == "":
                return None
            p = Path(override)
            return str(p) if p.exists() else override

        found = shutil.which("agy")
        if found and Path(found).exists():
            return str(Path(found).resolve())

        local_app_data = os.environ.get("LOCALAPPDATA", "")
        if local_app_data:
            candidate = Path(local_app_data) / "agy" / "bin" / "agy.exe"
            if candidate.exists():
                return str(candidate.resolve())

        user_profile = os.environ.get("USERPROFILE", "")
        if user_profile:
            candidate_alt = Path(user_profile) / "AppData" / "Local" / "agy" / "bin" / "agy.exe"
            if candidate_alt.exists():
                return str(candidate_alt.resolve())

        return None

    @classmethod
    def check_agy_pro_authenticated(
        cls,
        agy_path: str | None,
        probe_cli: bool = False,
        timeout_seconds: float = 8.0,
    ) -> tuple[bool, str]:
        """Verify `agy.exe` exists and is authenticated with Google AI Pro."""
        if not agy_path:
            return False, "agy.exe not found on PATH or %LOCALAPPDATA%\\agy\\bin\\agy.exe"

        exe_file = Path(agy_path)
        if not exe_file.exists():
            return False, f"agy.exe path does not exist: {agy_path}"

        if probe_cli:
            try:
                proc = subprocess.run(
                    [str(exe_file), "models"],
                    cwd=cls.get_isolated_temp_dir(),
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    encoding="utf-8",
                    errors="replace",
                )
                combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
                if proc.returncode == 0 and ("gemini-" in combined.lower() or "argon" in combined.lower()):
                    return True, f"Authenticated Google AI Pro CLI ({exe_file})"
                return False, f"agy.exe models exited with code {proc.returncode}"
            except Exception as exc:
                return False, f"agy.exe auth probe error: {exc}"

        # Fast non-blocking check: verify binary exists and user has `.gemini` profile directory
        home = Path(os.environ.get("USERPROFILE", str(Path.home())))
        gemini_dir = home / ".gemini"
        if gemini_dir.exists():
            return True, f"Ready via Google AI Pro login ({exe_file})"
        return True, f"Ready ({exe_file})"

    def hot_reload_if_needed(self, force: bool = False, probe_agy_auth: bool = False) -> bool:
        """Check `.env` and `config/apex_config.json` `mtime` and hot-reload settings if modified.

        Returns `True` if a reload occurred.
        """
        reloaded = False

        try:
            env_mtime = self.env_path.stat().st_mtime if self.env_path.exists() else 0.0
        except OSError:
            env_mtime = 0.0

        try:
            cfg_mtime = self.config_path.stat().st_mtime if self.config_path.exists() else 0.0
        except OSError:
            cfg_mtime = 0.0

        if force or env_mtime != self._env_mtime:
            self._env_mtime = env_mtime
            reloaded = True
            if self._explicit_api_key is not None:
                self.api_key = self._explicit_api_key.strip()
            else:
                file_vals = dotenv_values(self.env_path) if self.env_path.exists() else {}
                raw_key = file_vals.get("GEMINI_API_KEY")
                if raw_key is None:
                    raw_key = os.environ.get("GEMINI_API_KEY", "")
                self.api_key = str(raw_key or "").strip()

        if force or cfg_mtime != self._config_mtime:
            self._config_mtime = cfg_mtime
            reloaded = True
            if self.config_path.exists():
                try:
                    cfg_data = json.loads(self.config_path.read_text(encoding="utf-8"))
                    ai_cfg = (cfg_data.get("filters") or {}).get("ai_veto") or {}
                    self.enabled = bool(ai_cfg.get("enabled", self._explicit_enabled))
                    cfg_mode = str(ai_cfg.get("mode", self._explicit_mode or AIVetoMode.AUTO_STUDIO_FIRST.value))
                    cfg_order = ai_cfg.get("priority_order")
                    self.mode = cfg_mode
                    self._priority_order = self._resolve_priority_tuple(cfg_mode, cfg_order)
                    if "studio_timeout_seconds" in ai_cfg:
                        self.timeout_seconds = min(float(ai_cfg["studio_timeout_seconds"]), 4.0)
                    elif "timeout_seconds" in ai_cfg:
                        self.timeout_seconds = min(float(ai_cfg["timeout_seconds"]), 4.0)
                    if "agy_pro_timeout_seconds" in ai_cfg:
                        self.agy_timeout_seconds = float(ai_cfg["agy_pro_timeout_seconds"])
                except Exception as exc:
                    logger.debug("Failed to hot-reload %s: %s", self.config_path, exc)
            else:
                self.enabled = self._explicit_enabled
                self.mode = self._explicit_mode
                self._priority_order = self._resolve_priority_tuple(self.mode, self._explicit_priority)

        # Refresh agy.exe discovery & adaptive model catalog
        self._cached_agy_path = self.locate_agy_executable(self._agy_executable_override)
        if probe_agy_auth or self._cached_agy_auth_ok is None:
            ok, _ = self.check_agy_pro_authenticated(self._cached_agy_path, probe_cli=probe_agy_auth)
            self._cached_agy_auth_ok = ok

        if self.api_key:
            self.model_selector.discover_studio_models(self.api_key, timeout=self.timeout_seconds)
        if self._cached_agy_path and self._cached_agy_auth_ok:
            self.model_selector.discover_agy_models(self._cached_agy_path, probe_cli=False)

        return reloaded

    def get_router_status(self, probe_agy_auth: bool = False) -> HybridAIRouterStatus:
        """Return live discovery status for all 3 tiers and compute the active primary & fallback chain."""
        self.hot_reload_if_needed(force=probe_agy_auth, probe_agy_auth=probe_agy_auth)

        studio_ready = bool(self.api_key)
        masked_key = f"{self.api_key[:4]}...{self.api_key[-4:]}" if len(self.api_key) >= 8 else ("SET" if studio_ready else "UNSET")
        studio_status = AIProviderStatus(
            provider=AIProviderName.STUDIO_API.value,
            ready=studio_ready,
            detail=f"GEMINI_API_KEY={masked_key} (4.0s REST timeout)" if studio_ready else "GEMINI_API_KEY unset in .env",
            executable_or_endpoint="generativelanguage.googleapis.com",
        )

        now_utc = time.time()
        in_cooldown = now_utc < self._agy_cooldown_until_utc
        agy_ok, agy_detail = self.check_agy_pro_authenticated(
            self._cached_agy_path,
            probe_cli=probe_agy_auth,
        )
        if in_cooldown:
            rem_s = int(self._agy_cooldown_until_utc - now_utc)
            agy_ok = False
            agy_detail = f"In 10-min cooldown ({rem_s}s remaining after timeouts); pure_math active"

        agy_status = AIProviderStatus(
            provider=AIProviderName.AGY_PRO.value,
            ready=agy_ok,
            detail=agy_detail,
            executable_or_endpoint=self._cached_agy_path or "",
        )

        math_status = AIProviderStatus(
            provider=AIProviderName.PURE_MATH.value,
            ready=True,
            detail="Always READY (0ms, $0 deterministic quant + 6 circuit breakers)",
            executable_or_endpoint="apex/quant",
        )

        ready_map = {
            AIProviderName.STUDIO_API.value: studio_status.ready,
            AIProviderName.AGY_PRO.value: agy_status.ready,
            AIProviderName.PURE_MATH.value: True,
        }

        if not self.enabled or self.mode == AIVetoMode.FORCE_PURE_MATH.value:
            return HybridAIRouterStatus(
                enabled=self.enabled,
                mode=self.mode,
                priority_order=(AIProviderName.PURE_MATH.value,),
                active_primary=AIProviderName.PURE_MATH.value,
                fallback_chain=(),
                studio_status=studio_status,
                agy_pro_status=agy_status,
                pure_math_status=math_status,
            )

        ready_chain = [p for p in self._priority_order if ready_map.get(p, False)]
        if not ready_chain:
            ready_chain = [AIProviderName.PURE_MATH.value]

        active_primary = ready_chain[0]
        fallback_chain = tuple(ready_chain[1:])
        q_summary = self.model_selector.get_quota_health_summary(active_primary)

        return HybridAIRouterStatus(
            enabled=self.enabled,
            mode=self.mode,
            priority_order=self._priority_order,
            active_primary=active_primary,
            fallback_chain=fallback_chain,
            studio_status=studio_status,
            agy_pro_status=agy_status,
            pure_math_status=math_status,
            model_quota_summary=q_summary,
        )

    @staticmethod
    def build_adversarial_prompt(ctx: CandidateSetupAuditContext) -> str:
        """Build the compact adversarial prompt for the 2-role Risk Veto Committee."""
        payload = {
            "symbol": ctx.symbol,
            "side": ctx.side.value,
            "timeframe": ctx.timeframe,
            "entry": ctx.entry_price,
            "stop_loss": ctx.stop_loss,
            "take_profit": ctx.take_profit,
            "regime": ctx.regime,
            "hurst": round(ctx.hurst_exponent, 3),
            "kaufman_er": round(ctx.kaufman_er, 3),
            "vol_pct": round(ctx.volatility_percentile, 1),
            "dom_obi": round(ctx.dom_obi, 3) if ctx.dom_obi is not None else "N/A",
            "spread_pct_of_sl": round(ctx.spread_pct_of_sl * 100.0, 2),
            "timemachine_net_avg_R": round(ctx.net_avg_R, 3),
            "timemachine_prob_profit": round(ctx.prob_profit, 3),
            "timemachine_cvar_95_R": round(ctx.cvar_95_R, 3),
            "recommended_risk_pct": round(ctx.recommended_risk_pct, 3),
            "session_utc": ctx.session_utc_iso,
        }
        return (
            "Do NOT call any tools or read any files. Respond immediately.\n"
            "You are the PocketBot Apex Pre-Trade Risk Veto Committee consisting of two adversarial roles:\n"
            "1. Macro/Session Trap Auditor: Checks for illiquid session handoffs, late-session exhaustion, or macro traps.\n"
            "2. Structural Devil's Advocate: Checks if net_avg_R <= 0, excessive spread_pct_of_sl (>15%), or severe tail CVaR.\n"
            f"Candidate Setup Telemetry: {json.dumps(payload)}\n"
            "Respond ONLY with a valid single-line JSON object matching this exact schema:\n"
            '{"decision": "APPROVE" or "VETO", "confidence": 0.85, "reason": "concise synthesis of both roles"}'
        )

    @staticmethod
    def clean_and_extract_json_payload(raw_output: str) -> dict[str, Any]:
        """Strip ANSI escape sequences and non-TTY wrapper lines, extracting the inner JSON decision.

        Handles both:
          1. Direct model JSON: `{"decision": "APPROVE", "confidence": 0.9, "reason": "..."}`
          2. `agy.exe --output-format json` envelope:
             `{"conversation_id": "...", "status": "DONE", "response": "{\\"decision\\": \\"APPROVE\\", ...}"}`
        """
        no_ansi = _ANSI_ESCAPE_RE.sub("", raw_output or "").strip()
        if not no_ansi:
            raise ValueError("Empty stdout from AI provider")

        # Scan lines from bottom to top for valid JSON objects
        candidates: list[dict[str, Any]] = []
        for line in reversed(no_ansi.splitlines()):
            s = line.strip()
            if not s:
                continue
            start_idx = s.find("{")
            end_idx = s.rfind("}")
            if start_idx != -1 and end_idx > start_idx:
                snippet = s[start_idx : end_idx + 1]
                try:
                    parsed = json.loads(snippet)
                    if isinstance(parsed, dict):
                        candidates.append(parsed)
                except json.JSONDecodeError:
                    continue

        # Also try parsing the first `{ ... }` span across the entire string if multiline
        if not candidates:
            first_brace = no_ansi.find("{")
            last_brace = no_ansi.rfind("}")
            if first_brace != -1 and last_brace > first_brace:
                parsed = json.loads(no_ansi[first_brace : last_brace + 1])
                if isinstance(parsed, dict):
                    candidates.append(parsed)

        if not candidates:
            raise ValueError(f"Could not locate JSON object in output: {no_ansi[:200]!r}")

        for obj in candidates:
            # Check if this is an agy.exe JSON wrapper envelope
            if "status" in obj and ("response" in obj or "error" in obj):
                status_val = str(obj.get("status", "")).upper()
                if status_val == "ERROR" or obj.get("error"):
                    raise RuntimeError(f"agy_pro returned error envelope: {obj.get('error') or status_val}")
                inner_resp = str(obj.get("response") or "").strip()
                if not inner_resp:
                    raise RuntimeError("agy_pro returned empty response field")
                return AIVetoCommittee.clean_and_extract_json_payload(inner_resp)

            if "decision" in obj:
                return obj

        # If a JSON dict was found without 'decision', return the first candidate for normalization
        return candidates[0]

    @classmethod
    def parse_structured_decision(
        cls,
        raw_text: str,
        latency_ms: float,
        provider_used: str = AIProviderName.STUDIO_API.value,
        cascade_path: Sequence[str] = (),
        model_used: str = "",
    ) -> AIVetoDecision:
        """Parse and validate `{ "decision": "APPROVE" | "VETO", "confidence": 0.0-1.0, "reason": "..." }`."""
        data = cls.clean_and_extract_json_payload(raw_text)
        raw_dec = str(data.get("decision", "APPROVE")).upper().strip()
        if raw_dec not in ("APPROVE", "VETO"):
            raise ValueError(f"Invalid AI decision value: {raw_dec!r}")
        conf = float(max(0.0, min(1.0, float(data.get("confidence", 0.5)))))
        reason = str(data.get("reason", f"Audited by {provider_used}")).strip()
        path_tuple = tuple(cascade_path) if cascade_path else (provider_used,)
        return AIVetoDecision(
            decision=raw_dec,
            confidence=conf,
            reason=reason,
            latency_ms=latency_ms,
            bypassed=False,
            provider_used=provider_used,
            cascade_path=path_tuple,
            model_used=model_used,
        )

    def _get_ranked_model_ids(self, provider: str, ctx: CandidateSetupAuditContext, fallback_id: str) -> list[str]:
        """Score & rank available models for `provider` using stakes, quota, and latency."""
        stakes = compute_setup_stakes(
            volatility_percentile=ctx.volatility_percentile,
            net_avg_R=ctx.net_avg_R,
            recommended_risk_pct=ctx.recommended_risk_pct,
            timeframe=ctx.timeframe,
        )
        ranked = self.model_selector.rank_models(provider, stakes)
        if ranked:
            return [r.model.model_id for r in ranked[:2]]
        return [fallback_id]

    def _call_studio_api_sync(
        self,
        ctx: CandidateSetupAuditContext,
        cascade_path: Sequence[str],
    ) -> AIVetoDecision:
        """Call Google AI Studio REST API with adaptive model selection & intra-provider cascade."""
        if not self.api_key:
            raise RuntimeError("studio_api not ready: GEMINI_API_KEY is empty")

        prompt = self.build_adversarial_prompt(ctx)
        payload_bytes = json.dumps(
            {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 200,
                    "responseMimeType": "application/json",
                },
            }
        ).encode("utf-8")

        candidate_models = self._get_ranked_model_ids("studio_api", ctx, self.model)
        last_exc: Exception | None = None
        for model_id in candidate_models:
            t0 = time.perf_counter()
            url = self.GEMINI_ENDPOINT_TEMPLATE.format(model=model_id, api_key=self.api_key)
            req = urllib.request.Request(
                url,
                data=payload_bytes,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                    latency_ms = (time.perf_counter() - t0) * 1000.0
                    if resp.status != 200:
                        raise RuntimeError(f"studio_api HTTP {resp.status}")
                    resp_json = json.loads(resp.read().decode("utf-8"))
                    raw_text = resp_json["candidates"][0]["content"]["parts"][0]["text"]
                    dec = self.parse_structured_decision(
                        raw_text=raw_text,
                        latency_ms=latency_ms,
                        provider_used=AIProviderName.STUDIO_API.value,
                        cascade_path=cascade_path,
                        model_used=model_id,
                    )
                    self.model_selector.record_success("studio_api", model_id, latency_ms)
                    return dec
            except Exception as exc:
                last_exc = exc
                err_s = str(exc)
                self.model_selector.record_failure(
                    "studio_api",
                    model_id,
                    err_s,
                    is_quota_or_429=("429" in err_s or "quota" in err_s.lower()),
                )
                logger.warning("studio_api model %s failed (%s); cascading...", model_id, err_s)
        raise last_exc or RuntimeError("All studio_api candidate models failed")

    async def _call_studio_api_async(
        self,
        ctx: CandidateSetupAuditContext,
        cascade_path: Sequence[str],
    ) -> AIVetoDecision:
        """Async call to Google AI Studio REST API with adaptive model selection & cascade."""
        if not self.api_key:
            raise RuntimeError("studio_api not ready: GEMINI_API_KEY is empty")

        prompt = self.build_adversarial_prompt(ctx)
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 200,
                "responseMimeType": "application/json",
            },
        }
        candidate_models = self._get_ranked_model_ids("studio_api", ctx, self.model)
        last_exc: Exception | None = None
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for model_id in candidate_models:
                t0 = time.perf_counter()
                url = self.GEMINI_ENDPOINT_TEMPLATE.format(model=model_id, api_key=self.api_key)
                try:
                    async with session.post(url, json=body) as resp:
                        latency_ms = (time.perf_counter() - t0) * 1000.0
                        if resp.status != 200:
                            err_txt = await resp.text()
                            raise RuntimeError(f"studio_api HTTP {resp.status}: {err_txt[:120]}")
                        resp_json = await resp.json()
                        raw_text = resp_json["candidates"][0]["content"]["parts"][0]["text"]
                        dec = self.parse_structured_decision(
                            raw_text=raw_text,
                            latency_ms=latency_ms,
                            provider_used=AIProviderName.STUDIO_API.value,
                            cascade_path=cascade_path,
                            model_used=model_id,
                        )
                        self.model_selector.record_success("studio_api", model_id, latency_ms)
                        return dec
                except Exception as exc:
                    last_exc = exc
                    err_s = str(exc)
                    self.model_selector.record_failure(
                        "studio_api",
                        model_id,
                        err_s,
                        is_quota_or_429=("429" in err_s or "quota" in err_s.lower()),
                    )
                    logger.warning("studio_api async model %s failed (%s); cascading...", model_id, err_s)
        raise last_exc or RuntimeError("All studio_api async candidate models failed")

    def _call_agy_pro_sync(
        self,
        ctx: CandidateSetupAuditContext,
        cascade_path: Sequence[str],
    ) -> AIVetoDecision:
        """Call headless `agy -p "<prompt>" --output-format json --effort low` in isolated temp dir."""
        now_utc = time.time()
        if now_utc < self._agy_cooldown_until_utc:
            rem_s = int(self._agy_cooldown_until_utc - now_utc)
            raise RuntimeError(f"agy_pro in cooldown ({rem_s}s remaining); skipping to fallback")

        agy_exe = self._cached_agy_path
        if not agy_exe or not self._cached_agy_auth_ok:
            raise RuntimeError("agy_pro not ready or not found")

        prompt = self.build_adversarial_prompt(ctx)
        timeout_int = max(5, int(round(self.agy_timeout_seconds)))
        iso_dir = self.get_isolated_temp_dir()
        candidate_models = self._get_ranked_model_ids("agy_pro", ctx, "default")
        last_exc: Exception | None = None

        for model_id in candidate_models:
            t0 = time.perf_counter()
            cmd = [
                agy_exe,
                "-p",
                prompt,
                "--output-format",
                "json",
                "--print-timeout",
                f"{timeout_int}s",
                "--disable-slash-commands",
            ]
            if model_id != "default":
                cmd.extend(["--model", model_id])
            else:
                cmd.extend(["--effort", "low"])
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=iso_dir,
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    timeout=float(timeout_int) + 4.0,
                    encoding="utf-8",
                    errors="replace",
                )
                latency_ms = (time.perf_counter() - t0) * 1000.0
                stdout_txt = (proc.stdout or "").strip()
                if not stdout_txt:
                    raise RuntimeError(f"agy_pro exited {proc.returncode} with empty stdout: {(proc.stderr or '')[:160]}")

                decision = self.parse_structured_decision(
                    raw_text=stdout_txt,
                    latency_ms=latency_ms,
                    provider_used=AIProviderName.AGY_PRO.value,
                    cascade_path=cascade_path,
                    model_used=model_id,
                )
                self._agy_consecutive_failures = 0
                self.model_selector.record_success("agy_pro", model_id, latency_ms)
                return decision
            except Exception as exc:
                last_exc = exc
                err_s = str(exc)
                self.model_selector.record_failure(
                    "agy_pro",
                    model_id,
                    err_s,
                    is_quota_or_429=("quota" in err_s.lower() or "429" in err_s or "exhaust" in err_s.lower()),
                )
                logger.warning("agy_pro model %s failed (%s); cascading...", model_id, err_s)

        self._agy_consecutive_failures += 1
        if self._agy_consecutive_failures >= 2:
            self._agy_cooldown_until_utc = time.time() + 600.0
            logger.warning(
                "agy_pro failed %d times in a row (%s). Entering 10-minute cooldown to pure_math.",
                self._agy_consecutive_failures,
                last_exc,
            )
        raise last_exc or RuntimeError("agy_pro failed")

    async def _call_agy_pro_async(
        self,
        ctx: CandidateSetupAuditContext,
        cascade_path: Sequence[str],
    ) -> AIVetoDecision:
        """Run `_call_agy_pro_sync` in a worker thread so the async event loop never blocks."""
        return await asyncio.to_thread(self._call_agy_pro_sync, ctx, cascade_path)

    def _evaluate_pure_math_fallback(
        self,
        ctx: CandidateSetupAuditContext,
        start_perf: float,
        cascade_path: Sequence[str],
        fallback_reason_prefix: str = "",
    ) -> AIVetoDecision:
        """Deterministic 0ms $0 Pure Math pass-through relying on Regime + DOM + TimeMachine + 6 Breakers."""
        latency_ms = (time.perf_counter() - start_perf) * 1000.0
        min_net_r = float(getattr(self, "min_net_avg_R", 0.10))
        # Pure math check: if net_avg_R < min_net_avg_R or spread_pct_of_sl > 0.15, VETO; otherwise APPROVE
        if ctx.net_avg_R < min_net_r:
            dec = "VETO"
            reason = f"Pure Math Gate: net_avg_R ({ctx.net_avg_R:+.3f}R) < {min_net_r:.2f}R"
        elif ctx.spread_pct_of_sl > 0.15 + 1e-9:
            dec = "VETO"
            reason = f"Pure Math Gate: spread is {ctx.spread_pct_of_sl * 100:.1f}% of SL (> 15.0%)"
        else:
            dec = "APPROVE"
            reason = (
                f"Pure Math Deterministic Pass-Through: Regime={ctx.regime}, "
                f"net_avg_R={ctx.net_avg_R:+.3f}R, ProbProfit={ctx.prob_profit * 100:.1f}%"
            )

        if fallback_reason_prefix:
            reason = f"[{fallback_reason_prefix}] {reason}"

        return AIVetoDecision(
            decision=dec,
            confidence=1.0,
            reason=reason,
            latency_ms=latency_ms,
            bypassed=True,
            provider_used=AIProviderName.PURE_MATH.value,
            cascade_path=tuple(cascade_path),
        )

    def evaluate_sync(self, ctx: CandidateSetupAuditContext) -> AIVetoDecision:
        """Hot-reload `.env` & `config/apex_config.json`, then execute the configured 3-tier waterfall."""
        bar_ts = int(ctx.bar_timestamp)
        side_str = ctx.side.value if hasattr(ctx.side, "value") else str(ctx.side).upper()
        cache_key = (ctx.symbol.upper(), side_str, ctx.timeframe.upper(), bar_ts)
        if bar_ts > 0 and cache_key in self._verdict_cache:
            return self._verdict_cache[cache_key]

        decision = self._evaluate_sync_impl(ctx)
        if bar_ts > 0:
            self._verdict_cache[cache_key] = decision
        return decision

    def _evaluate_sync_impl(self, ctx: CandidateSetupAuditContext) -> AIVetoDecision:
        t0 = time.perf_counter()
        status = self.get_router_status()

        if not status.enabled or status.mode == AIVetoMode.FORCE_PURE_MATH.value:
            return self._evaluate_pure_math_fallback(
                ctx=ctx,
                start_perf=t0,
                cascade_path=(AIProviderName.PURE_MATH.value,),
                fallback_reason_prefix="force_pure_math",
            )

        attempted: list[str] = []
        errors: list[str] = []
        ready_chain = [status.active_primary, *status.fallback_chain]

        for provider in ready_chain:
            attempted.append(provider)
            if provider == AIProviderName.STUDIO_API.value:
                try:
                    return self._call_studio_api_sync(ctx, cascade_path=attempted)
                except Exception as exc:
                    err_msg = f"studio_api failed ({exc})"
                    errors.append(err_msg)
                    logger.warning("%s; cascading to next AI tier...", err_msg)
                    continue

            if provider == AIProviderName.AGY_PRO.value:
                try:
                    return self._call_agy_pro_sync(ctx, cascade_path=attempted)
                except Exception as exc:
                    err_msg = f"agy_pro failed ({exc})"
                    errors.append(err_msg)
                    logger.warning("%s; cascading to next AI tier...", err_msg)
                    continue

            if provider == AIProviderName.PURE_MATH.value:
                prefix = " -> ".join(errors) if errors else "auto_pure_math"
                return self._evaluate_pure_math_fallback(
                    ctx=ctx,
                    start_perf=t0,
                    cascade_path=attempted,
                    fallback_reason_prefix=prefix,
                )

        # Ultimate safety net: always return pure_math
        if AIProviderName.PURE_MATH.value not in attempted:
            attempted.append(AIProviderName.PURE_MATH.value)
        return self._evaluate_pure_math_fallback(
            ctx=ctx,
            start_perf=t0,
            cascade_path=attempted,
            fallback_reason_prefix="; ".join(errors),
        )

    async def evaluate_async(self, ctx: CandidateSetupAuditContext) -> AIVetoDecision:
        """Asynchronous 3-tier waterfall evaluation with hot-reload and zero-crash Pure Math fallback."""
        bar_ts = int(ctx.bar_timestamp)
        side_str = ctx.side.value if hasattr(ctx.side, "value") else str(ctx.side).upper()
        cache_key = (ctx.symbol.upper(), side_str, ctx.timeframe.upper(), bar_ts)
        if bar_ts > 0 and cache_key in self._verdict_cache:
            return self._verdict_cache[cache_key]

        decision = await self._evaluate_async_impl(ctx)
        if bar_ts > 0:
            self._verdict_cache[cache_key] = decision
        return decision

    async def _evaluate_async_impl(self, ctx: CandidateSetupAuditContext) -> AIVetoDecision:
        t0 = time.perf_counter()
        status = self.get_router_status()

        if not status.enabled or status.mode == AIVetoMode.FORCE_PURE_MATH.value:
            return self._evaluate_pure_math_fallback(
                ctx=ctx,
                start_perf=t0,
                cascade_path=(AIProviderName.PURE_MATH.value,),
                fallback_reason_prefix="force_pure_math",
            )

        attempted: list[str] = []
        errors: list[str] = []
        ready_chain = [status.active_primary, *status.fallback_chain]

        for provider in ready_chain:
            attempted.append(provider)
            if provider == AIProviderName.STUDIO_API.value:
                try:
                    return await self._call_studio_api_async(ctx, cascade_path=attempted)
                except Exception as exc:
                    err_msg = f"studio_api failed ({exc})"
                    errors.append(err_msg)
                    logger.warning("%s; cascading to next AI tier...", err_msg)
                    continue

            if provider == AIProviderName.AGY_PRO.value:
                try:
                    return await self._call_agy_pro_async(ctx, cascade_path=attempted)
                except Exception as exc:
                    err_msg = f"agy_pro failed ({exc})"
                    errors.append(err_msg)
                    logger.warning("%s; cascading to next AI tier...", err_msg)
                    continue

            if provider == AIProviderName.PURE_MATH.value:
                prefix = " -> ".join(errors) if errors else "auto_pure_math"
                return self._evaluate_pure_math_fallback(
                    ctx=ctx,
                    start_perf=t0,
                    cascade_path=attempted,
                    fallback_reason_prefix=prefix,
                )

        if AIProviderName.PURE_MATH.value not in attempted:
            attempted.append(AIProviderName.PURE_MATH.value)
        return self._evaluate_pure_math_fallback(
            ctx=ctx,
            start_perf=t0,
            cascade_path=attempted,
            fallback_reason_prefix="; ".join(errors),
        )
