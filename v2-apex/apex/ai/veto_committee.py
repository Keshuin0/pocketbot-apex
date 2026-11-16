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
import math
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
from typing import Any, Awaitable, Callable, Sequence

import aiohttp
from dotenv import dotenv_values

from apex.ai.aegis_verifier import AegisVerifier, BlindedTelemetry
from apex.ai.model_selector import AdaptiveModelSelector, compute_setup_stakes
from apex.feeds.models import Side

logger = logging.getLogger(__name__)

_ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


class AIProviderName(str, Enum):
    """Supported tiers in the Hybrid 4-Tier AI Router."""

    STUDIO_API = "studio_api"
    LOCAL_VLLM = "local_vllm"
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
    local_vllm_status: AIProviderStatus | None = None

    @property
    def active_model_label(self) -> str:
        """Return human-readable active provider + top READY model (never UNKNOWN)."""
        if not self.enabled or self.active_primary == AIProviderName.PURE_MATH.value:
            return "PURE_MATH: deterministic"
        sel = str(self.model_quota_summary.get("selected_model", "") or "").strip()
        if sel and sel.lower() not in {"none", "unknown"}:
            return f"{self.active_primary.upper()}: {sel}"
        return f"{self.active_primary.upper()}: READY"

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
    rollover_illiquidity_flag: bool = False
    cross_asset_shock_z: float = 0.0


@dataclass(frozen=True, slots=True)
class AIVetoDecision:
    """Structured JSON decision returned by the Hybrid 3-Tier AI Router + Apex Aegis Firewall."""

    decision: str  # "APPROVE", "VETO", or "ABSTAIN"
    confidence: float  # [0.0, 1.0]
    reason: str
    latency_ms: float
    bypassed: bool = False
    provider_used: str = "pure_math"
    cascade_path: tuple[str, ...] = field(default_factory=lambda: ("pure_math",))
    model_used: str = ""
    risk_multiplier: float = 1.0
    aegis_verified: bool = False
    hallucination_detected: bool = False
    input_sha256: str = ""
    audit_id: int = 0

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
            "risk_multiplier": round(self.risk_multiplier, 4),
            "aegis_verified": self.aegis_verified,
            "hallucination_detected": self.hallucination_detected,
            "input_sha256": self.input_sha256,
            "audit_id": self.audit_id,
        }


class AIVetoCommittee:
    """Hybrid Self-Detecting 3-Tier AI Router (`apex/ai/veto_committee.py`).

    Monitors `.env` and `config/apex_config.json` modification times (`mtime`) before each
    evaluation so changing API keys, logging into `agy.exe`, or switching priority modes
    takes effect immediately in real time without restarting the bot.
    """

    GEMINI_ENDPOINT_TEMPLATE: str = (
        "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    )
    LOCAL_VLLM_ENDPOINT: str = "http://127.0.0.1:8000/v1/chat/completions"
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
        ledger_db: Any = None,
        vllm_endpoint: str | None = None,
        vllm_model: str | None = None,
        vllm_enabled: bool | None = None,
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

        env_vllm_url = os.environ.get("APEX_VLLM_URL", "").strip()
        self._explicit_vllm_endpoint: str | None = vllm_endpoint
        self.vllm_endpoint: str = (vllm_endpoint or env_vllm_url or self.LOCAL_VLLM_ENDPOINT).strip()
        self.vllm_model: str = (
            vllm_model or os.environ.get("APEX_VLLM_MODEL", "") or "Qwen/Qwen2.5-7B-Instruct-AWQ"
        ).strip()
        self._explicit_vllm_enabled: bool | None = vllm_enabled
        self.vllm_enabled: bool = bool(
            vllm_enabled
            if vllm_enabled is not None
            else (
                bool(vllm_endpoint or env_vllm_url)
                or (
                    self._explicit_priority is not None
                    and AIProviderName.LOCAL_VLLM.value in self._explicit_priority
                )
            )
        )

        self._env_mtime: float = -1.0
        self._config_mtime: float = -1.0

        self.enabled: bool = bool(enabled)
        self.mode: str = str(mode)
        self.api_key: str = (api_key or "").strip()
        self._priority_order: tuple[str, ...] = self._resolve_priority_tuple(
            self.mode, priority_order, include_vllm=self.vllm_enabled
        )

        self._cached_agy_path: str | None = None
        self._cached_agy_auth_ok: bool | None = None
        self._verdict_cache: dict[tuple[str, str, str, int], AIVetoDecision] = {}
        self._agy_consecutive_failures: int = 0
        self._agy_cooldown_until_utc: float = 0.0
        self.min_net_avg_R: float = 0.10
        self.ledger_db: Any = ledger_db
        self.model_selector: AdaptiveModelSelector = AdaptiveModelSelector(ledger_db=ledger_db)
        self.aegis: AegisVerifier = AegisVerifier(
            ledger_db=ledger_db,
            model_selector=self.model_selector,
        )
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
        include_vllm: bool = False,
    ) -> tuple[str, ...]:
        norm_mode = str(mode or AIVetoMode.AUTO_STUDIO_FIRST.value).lower().strip()
        if norm_mode == AIVetoMode.FORCE_PURE_MATH.value:
            return (AIProviderName.PURE_MATH.value,)
        if custom_order and (
            norm_mode not in MODE_DEFAULT_PRIORITY
            or AIProviderName.LOCAL_VLLM.value in [str(x).lower().strip() for x in custom_order]
        ):
            chain = [str(x).lower().strip() for x in custom_order if str(x).strip()]
        elif norm_mode in MODE_DEFAULT_PRIORITY:
            chain = list(MODE_DEFAULT_PRIORITY[norm_mode])
        elif custom_order:
            chain = [str(x).lower().strip() for x in custom_order if str(x).strip()]
        else:
            chain = list(MODE_DEFAULT_PRIORITY[AIVetoMode.AUTO_STUDIO_FIRST.value])

        if (
            include_vllm
            and AIProviderName.LOCAL_VLLM.value not in chain
            and len(chain) >= 2
            and norm_mode != AIVetoMode.FORCE_PURE_MATH.value
        ):
            chain.insert(1, AIProviderName.LOCAL_VLLM.value)

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

        # Fast non-blocking check: verify binary exists and user has non-empty authenticated `.gemini` profile
        home = Path(os.environ.get("USERPROFILE") or os.environ.get("HOME") or str(Path.home()))
        gemini_dir = home / ".gemini"
        if not gemini_dir.exists() or not gemini_dir.is_dir():
            return False, f"Google AI Pro not authenticated (missing {gemini_dir})"
        auth_markers = (
            gemini_dir / "oauth_creds.json",
            gemini_dir / "settings.json",
            gemini_dir / "google_accounts.json",
            gemini_dir / "antigravity",
            gemini_dir / "accounts",
        )
        has_marker = any(
            m.exists() and ((m.is_file() and m.stat().st_size > 2) or (m.is_dir() and any(m.iterdir())))
            for m in auth_markers
        )
        if not has_marker:
            try:
                has_marker = any(p.is_file() and p.stat().st_size > 2 for p in gemini_dir.iterdir())
            except OSError:
                has_marker = False
        if not has_marker:
            return False, f"Google AI Pro not authenticated (empty {gemini_dir})"
        return True, f"Ready via Google AI Pro login ({exe_file})"

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
                    if "vllm_endpoint" in ai_cfg and not self._explicit_vllm_endpoint:
                        self.vllm_endpoint = str(ai_cfg["vllm_endpoint"]).strip() or self.LOCAL_VLLM_ENDPOINT
                    if "vllm_model" in ai_cfg:
                        self.vllm_model = str(ai_cfg["vllm_model"]).strip() or self.vllm_model
                    if self._explicit_vllm_enabled is None and "vllm_enabled" in ai_cfg:
                        self.vllm_enabled = bool(ai_cfg["vllm_enabled"])
                    self.mode = cfg_mode
                    effective_order = cfg_order if cfg_order is not None else self._explicit_priority
                    effective_vllm = bool(
                        self.vllm_enabled
                        or (
                            effective_order is not None
                            and AIProviderName.LOCAL_VLLM.value in [str(x).lower().strip() for x in effective_order]
                        )
                    )
                    self._priority_order = self._resolve_priority_tuple(
                        cfg_mode,
                        effective_order,
                        include_vllm=effective_vllm,
                    )
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
                self._priority_order = self._resolve_priority_tuple(
                    self.mode,
                    self._explicit_priority,
                    include_vllm=self.vllm_enabled,
                )

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

        local_vllm_ready = bool(
            self.vllm_endpoint
            and (
                self.vllm_enabled
                or AIProviderName.LOCAL_VLLM.value in self._priority_order
            )
        )
        local_vllm_status = AIProviderStatus(
            provider=AIProviderName.LOCAL_VLLM.value,
            ready=local_vllm_ready,
            detail=f"OpenAI-compatible vLLM ({self.vllm_model})" if local_vllm_ready else "local_vllm disabled",
            executable_or_endpoint=self.vllm_endpoint if local_vllm_ready else "",
        )

        ready_map = {
            AIProviderName.STUDIO_API.value: studio_status.ready,
            AIProviderName.LOCAL_VLLM.value: local_vllm_status.ready,
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
                local_vllm_status=local_vllm_status,
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
            local_vllm_status=local_vllm_status,
        )

    @staticmethod
    def build_adversarial_prompt(ctx: CandidateSetupAuditContext) -> str:
        """Build the Shield 2 & 3 ticker-blinded, closed-world proof-carrying prompt."""
        blinded = AegisVerifier.normalize_ticker_blinded_telemetry(ctx)
        return AegisVerifier.build_aegis_prompt(blinded)

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
        """Parse `{decision, confidence, reason}`; fail closed to ABSTAIN on missing/malformed inputs."""
        path_tuple = tuple(cascade_path) if cascade_path else (provider_used,)
        try:
            data = cls.clean_and_extract_json_payload(raw_text)
            if not isinstance(data, dict):
                raise ValueError("AI payload JSON is not a mapping")
            raw_dec = str(data.get("decision") or "ABSTAIN").upper().strip()
            if raw_dec not in ("APPROVE", "VETO", "ABSTAIN"):
                raise ValueError(f"Invalid AI decision value: {raw_dec!r}")
            conf_raw = float(data.get("confidence", 0.5))
            if math.isnan(conf_raw) or math.isinf(conf_raw):
                raise ValueError("Non-finite confidence")
            conf = float(max(0.0, min(1.0, conf_raw)))
            default_mult = 0.0 if raw_dec in ("VETO", "ABSTAIN") else 1.0
            mult_raw = float(data.get("risk_scale", data.get("risk_multiplier", default_mult)))
            if math.isnan(mult_raw) or math.isinf(mult_raw):
                raise ValueError("Non-finite risk_multiplier")
            mult = float(max(0.0, min(1.0, mult_raw)))
            reason = str(data.get("reason", f"Audited by {provider_used}")).strip()
            return AIVetoDecision(
                decision=raw_dec,
                confidence=conf,
                reason=reason,
                latency_ms=latency_ms,
                bypassed=(raw_dec == "ABSTAIN"),
                provider_used=provider_used,
                cascade_path=path_tuple,
                model_used=model_used,
                risk_multiplier=mult,
            )
        except Exception as exc:
            logger.warning("parse_structured_decision failed closed to ABSTAIN (%s)", exc)
            return AIVetoDecision(
                decision="ABSTAIN",
                confidence=0.0,
                reason=f"Fail-closed ABSTAIN on malformed AI payload: {exc}",
                latency_ms=latency_ms,
                bypassed=True,
                provider_used=provider_used,
                cascade_path=path_tuple,
                model_used=model_used,
                risk_multiplier=0.0,
            )

    def _log_epistemic_and_update_trust(
        self,
        ctx: CandidateSetupAuditContext,
        blinded: BlindedTelemetry,
        provider_used: str,
        model_used: str,
        raw_decision: str,
        final_decision: str,
        confidence: float,
        risk_multiplier: float,
        hallucination_detected: bool,
        violation_codes: str,
        reason: str,
    ) -> int:
        if self.ledger_db is None:
            return 0
        side_str = ctx.side.value if hasattr(ctx.side, "value") else str(ctx.side).upper()
        audit_id = self.ledger_db.log_ai_epistemic_audit(
            symbol=ctx.symbol,
            side=side_str,
            timeframe=ctx.timeframe,
            provider=provider_used,
            model_id=model_used,
            input_sha256=blinded.input_sha256,
            raw_decision=raw_decision,
            final_decision=final_decision,
            confidence=confidence,
            risk_multiplier=risk_multiplier,
            hallucination_detected=hallucination_detected,
            violation_codes=violation_codes,
            reason=reason,
            is_canary=False,
            predicted_net_r=ctx.net_avg_R,
        )
        stats = self.ledger_db.get_model_epistemic_stats(provider=provider_used, model_id=model_used, window=20)
        self.model_selector.update_epistemic_trust(
            provider_used,
            model_used,
            stats.epistemic_trust_score,
            quarantined=stats.should_demote_to_pure_math,
            reason=stats.demote_reason,
        )
        return int(audit_id)

    def _build_swap_abstain_decision(
        self,
        ctx: CandidateSetupAuditContext,
        blinded: BlindedTelemetry,
        provider_used: str,
        model_used: str,
        cascade_path: Sequence[str],
        raw_decision: str,
        swap_decision: str,
        confidence: float,
        latency_ms: float,
        start_perf: float,
    ) -> AIVetoDecision:
        swap_reason = f"Shield 5 SHIELD5_SEMANTIC_ENTROPY ({raw_decision}->{swap_decision}) -> pure_math"
        audit_id = self._log_epistemic_and_update_trust(
            ctx=ctx,
            blinded=blinded,
            provider_used=provider_used,
            model_used=model_used,
            raw_decision=raw_decision,
            final_decision="ABSTAIN",
            confidence=0.0,
            risk_multiplier=1.0,
            hallucination_detected=False,
            violation_codes="SHIELD5_SEMANTIC_ENTROPY",
            reason=swap_reason,
        )
        fb = self._evaluate_pure_math_fallback(
            ctx=ctx,
            start_perf=start_perf,
            cascade_path=list(cascade_path) + [AIProviderName.PURE_MATH.value],
            fallback_reason_prefix=swap_reason,
        )
        return AIVetoDecision(
            decision=fb.decision,
            confidence=fb.confidence,
            reason=fb.reason,
            latency_ms=fb.latency_ms if fb.latency_ms > 0.0 else latency_ms,
            bypassed=True,
            provider_used=fb.provider_used,
            cascade_path=fb.cascade_path,
            model_used=model_used,
            risk_multiplier=0.0 if fb.decision == "VETO" else 1.0,
            aegis_verified=True,
            hallucination_detected=False,
            input_sha256=blinded.input_sha256,
            audit_id=audit_id,
        )

    def _finalize_from_verification(
        self,
        v_res: Any,
        ctx: CandidateSetupAuditContext,
        blinded: BlindedTelemetry,
        latency_ms: float,
        provider_used: str,
        cascade_path: Sequence[str],
        model_used: str,
        start_perf: float,
    ) -> AIVetoDecision:
        audit_id = self._log_epistemic_and_update_trust(
            ctx=ctx,
            blinded=blinded,
            provider_used=provider_used,
            model_used=model_used,
            raw_decision=v_res.raw_decision,
            final_decision=v_res.final_decision,
            confidence=v_res.confidence,
            risk_multiplier=v_res.risk_multiplier,
            hallucination_detected=v_res.hallucination_detected,
            violation_codes=",".join(v_res.violation_codes),
            reason=v_res.reason,
        )

        if v_res.hallucination_detected:
            self.model_selector.record_failure(
                provider_used,
                model_used,
                f"HALLUCINATION_DETECTED: {','.join(v_res.violation_codes)}",
            )
            fb = self._evaluate_pure_math_fallback(
                ctx=ctx,
                start_perf=start_perf,
                cascade_path=list(cascade_path) + [AIProviderName.PURE_MATH.value],
                fallback_reason_prefix=f"Aegis HALLUCINATION_DETECTED ({','.join(v_res.violation_codes)}) -> pure_math",
            )
            return AIVetoDecision(
                decision=fb.decision,
                confidence=fb.confidence,
                reason=fb.reason,
                latency_ms=fb.latency_ms,
                bypassed=True,
                provider_used=fb.provider_used,
                cascade_path=fb.cascade_path,
                model_used=model_used,
                risk_multiplier=0.0 if fb.decision == "VETO" else 1.0,
                aegis_verified=True,
                hallucination_detected=True,
                input_sha256=blinded.input_sha256,
                audit_id=audit_id,
            )

        self.model_selector.record_success(provider_used, model_used, latency_ms)

        if v_res.fallback_to_pure_math:
            fb = self._evaluate_pure_math_fallback(
                ctx=ctx,
                start_perf=start_perf,
                cascade_path=list(cascade_path) + [AIProviderName.PURE_MATH.value],
                fallback_reason_prefix=f"Aegis {v_res.raw_decision} -> pure_math",
            )
            return AIVetoDecision(
                decision=fb.decision,
                confidence=fb.confidence,
                reason=fb.reason,
                latency_ms=fb.latency_ms,
                bypassed=True,
                provider_used=fb.provider_used,
                cascade_path=fb.cascade_path,
                model_used=model_used,
                risk_multiplier=0.0 if fb.decision == "VETO" else 1.0,
                aegis_verified=True,
                hallucination_detected=False,
                input_sha256=blinded.input_sha256,
                audit_id=audit_id,
            )

        return AIVetoDecision(
            decision=v_res.final_decision,
            confidence=v_res.confidence,
            reason=v_res.reason,
            latency_ms=latency_ms,
            bypassed=False,
            provider_used=provider_used,
            cascade_path=tuple(cascade_path),
            model_used=model_used,
            risk_multiplier=v_res.risk_multiplier,
            aegis_verified=True,
            hallucination_detected=False,
            input_sha256=blinded.input_sha256,
            audit_id=audit_id,
        )

    def _verify_and_finalize_ai_response(
        self,
        raw_text: str,
        ctx: CandidateSetupAuditContext,
        blinded: BlindedTelemetry,
        latency_ms: float,
        provider_used: str,
        cascade_path: Sequence[str],
        model_used: str,
        start_perf: float,
        swap_evaluator: Callable[[str], str] | None = None,
    ) -> AIVetoDecision:
        """Run Aegis Shields 1-6 (including Shield 5 order-swap) on a raw sync LLM response."""
        try:
            payload = self.clean_and_extract_json_payload(raw_text)
        except Exception as exc:
            payload = {"decision": "INVALID", "confidence": 0.0, "reason": str(exc)}
        v_res = self.aegis.verify_response(
            payload,
            blinded,
            provider=provider_used,
            model_id=model_used,
        )
        if (
            not v_res.hallucination_detected
            and v_res.final_decision in ("APPROVE", "VETO")
            and self.aegis.is_borderline_high_stakes(blinded.telemetry)
            and swap_evaluator is not None
        ):
            swap_prompt = self.aegis.build_aegis_prompt(blinded, swap_order=True)
            try:
                swap_raw = swap_evaluator(swap_prompt)
                swap_dec = self.parse_structured_decision(
                    swap_raw,
                    latency_ms=latency_ms,
                    provider_used=provider_used,
                    cascade_path=cascade_path,
                    model_used=model_used,
                ).decision
            except Exception:
                swap_dec = "ABSTAIN"
            if swap_dec == "ABSTAIN" or not self.aegis.check_semantic_invariance(v_res.raw_decision, swap_dec):
                return self._build_swap_abstain_decision(
                    ctx=ctx,
                    blinded=blinded,
                    provider_used=provider_used,
                    model_used=model_used,
                    cascade_path=cascade_path,
                    raw_decision=v_res.raw_decision,
                    swap_decision=swap_dec,
                    confidence=v_res.confidence,
                    latency_ms=latency_ms,
                    start_perf=start_perf,
                )
        return self._finalize_from_verification(
            v_res=v_res,
            ctx=ctx,
            blinded=blinded,
            latency_ms=latency_ms,
            provider_used=provider_used,
            cascade_path=cascade_path,
            model_used=model_used,
            start_perf=start_perf,
        )

    async def _verify_and_finalize_ai_response_async(
        self,
        raw_text: str,
        ctx: CandidateSetupAuditContext,
        blinded: BlindedTelemetry,
        latency_ms: float,
        provider_used: str,
        cascade_path: Sequence[str],
        model_used: str,
        start_perf: float,
        swap_evaluator: Callable[[str], Awaitable[str]] | None = None,
    ) -> AIVetoDecision:
        """Run Aegis Shields 1-6 (including async Shield 5 order-swap) on a raw LLM response."""
        try:
            payload = self.clean_and_extract_json_payload(raw_text)
        except Exception as exc:
            payload = {"decision": "INVALID", "confidence": 0.0, "reason": str(exc)}
        v_res = self.aegis.verify_response(
            payload,
            blinded,
            provider=provider_used,
            model_id=model_used,
        )
        if (
            not v_res.hallucination_detected
            and v_res.final_decision in ("APPROVE", "VETO")
            and self.aegis.is_borderline_high_stakes(blinded.telemetry)
            and swap_evaluator is not None
        ):
            swap_prompt = self.aegis.build_aegis_prompt(blinded, swap_order=True)
            try:
                swap_raw = await swap_evaluator(swap_prompt)
                swap_dec = self.parse_structured_decision(
                    swap_raw,
                    latency_ms=latency_ms,
                    provider_used=provider_used,
                    cascade_path=cascade_path,
                    model_used=model_used,
                ).decision
            except Exception:
                swap_dec = "ABSTAIN"
            if swap_dec == "ABSTAIN" or not self.aegis.check_semantic_invariance(v_res.raw_decision, swap_dec):
                return self._build_swap_abstain_decision(
                    ctx=ctx,
                    blinded=blinded,
                    provider_used=provider_used,
                    model_used=model_used,
                    cascade_path=cascade_path,
                    raw_decision=v_res.raw_decision,
                    swap_decision=swap_dec,
                    confidence=v_res.confidence,
                    latency_ms=latency_ms,
                    start_perf=start_perf,
                )
        return self._finalize_from_verification(
            v_res=v_res,
            ctx=ctx,
            blinded=blinded,
            latency_ms=latency_ms,
            provider_used=provider_used,
            cascade_path=cascade_path,
            model_used=model_used,
            start_perf=start_perf,
        )

    def _get_ranked_model_ids(self, provider: str, ctx: CandidateSetupAuditContext, fallback_id: str) -> list[str]:
        """Score & pick the top candidate from each distinct non-quarantined model family."""
        stakes = compute_setup_stakes(
            volatility_percentile=ctx.volatility_percentile,
            net_avg_R=ctx.net_avg_R,
            recommended_risk_pct=ctx.recommended_risk_pct,
            timeframe=ctx.timeframe,
        )
        cross_family = self.model_selector.rank_cross_family_candidates(provider, stakes)
        if cross_family:
            return [r.model.model_id for r in cross_family]
        # Never fall back to bare `--model`-less default if named models exist or a family is quarantined
        if self.model_selector.get_models(provider) or self.model_selector.has_any_quarantined_family(provider):
            return []
        return [fallback_id]

    @staticmethod
    def _build_local_vllm_body(model_id: str, prompt: str) -> dict[str, Any]:
        """Build an OpenAI-compatible ChatCompletions request with strict JSON Schema response_format."""
        claim_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "field": {"type": "string"},
                    "op": {"type": "string", "enum": ["<", "<=", "==", ">=", ">"]},
                    "cited_value": {"type": "number"},
                    "threshold": {"type": "number"},
                },
                "required": ["field", "op", "cited_value", "threshold"],
                "additionalProperties": False,
            },
        }
        return {
            "model": model_id,
            "messages": [
                {
                    "role": "system",
                    "content": "Respond ONLY with valid JSON conforming to the Closed-World Proof-Carrying Schema.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 450,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "aegis_veto_verdict",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "bull_claims": claim_schema,
                            "bear_claims": claim_schema,
                            "decision": {"type": "string", "enum": ["APPROVE", "VETO", "ABSTAIN"]},
                            "confidence": {"type": "number"},
                            "risk_scale": {"type": "number"},
                            "reason": {"type": "string"},
                        },
                        "required": ["bull_claims", "bear_claims", "decision", "confidence", "risk_scale", "reason"],
                        "additionalProperties": False,
                    },
                },
            },
        }

    @staticmethod
    def _extract_local_vllm_text(resp_json: dict[str, Any]) -> str:
        choice = (resp_json.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        if "parsed" in msg and isinstance(msg["parsed"], dict):
            return json.dumps(msg["parsed"])
        return str(msg.get("content") or choice.get("text") or "")

    def _call_local_vllm_sync(
        self,
        ctx: CandidateSetupAuditContext,
        cascade_path: Sequence[str],
    ) -> AIVetoDecision:
        """Call local OpenAI-compatible vLLM / SGLang endpoint synchronously."""
        if not self.vllm_endpoint:
            raise RuntimeError("local_vllm endpoint not configured")
        blinded = self.aegis.normalize_ticker_blinded_telemetry(ctx)
        prompt = self.aegis.build_aegis_prompt(blinded)
        model_id = self.vllm_model

        def _invoke(p_txt: str) -> str:
            body_bytes = json.dumps(self._build_local_vllm_body(model_id, p_txt)).encode("utf-8")
            req = urllib.request.Request(
                self.vllm_endpoint,
                data=body_bytes,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"local_vllm HTTP {resp.status}")
                resp_json = json.loads(resp.read().decode("utf-8"))
                return self._extract_local_vllm_text(resp_json)

        t0 = time.perf_counter()
        raw_text = _invoke(prompt)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return self._verify_and_finalize_ai_response(
            raw_text=raw_text,
            ctx=ctx,
            blinded=blinded,
            latency_ms=latency_ms,
            provider_used=AIProviderName.LOCAL_VLLM.value,
            cascade_path=cascade_path,
            model_used=model_id,
            start_perf=t0,
            swap_evaluator=_invoke,
        )

    async def _call_local_vllm_async(
        self,
        ctx: CandidateSetupAuditContext,
        cascade_path: Sequence[str],
    ) -> AIVetoDecision:
        """Async call to local OpenAI-compatible vLLM / SGLang server with Structured Outputs."""
        if not self.vllm_endpoint:
            raise RuntimeError("local_vllm endpoint not configured")
        blinded = self.aegis.normalize_ticker_blinded_telemetry(ctx)
        prompt = self.aegis.build_aegis_prompt(blinded)
        model_id = self.vllm_model
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async def _invoke(p_txt: str) -> str:
                body = self._build_local_vllm_body(model_id, p_txt)
                async with session.post(
                    self.vllm_endpoint,
                    json=body,
                    headers={"Content-Type": "application/json"},
                ) as resp:
                    if resp.status != 200:
                        err_txt = await resp.text()
                        raise RuntimeError(f"local_vllm HTTP {resp.status}: {err_txt[:120]}")
                    resp_json = await resp.json()
                    return self._extract_local_vllm_text(resp_json)

            t0 = time.perf_counter()
            raw_text = await _invoke(prompt)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return await self._verify_and_finalize_ai_response_async(
                raw_text=raw_text,
                ctx=ctx,
                blinded=blinded,
                latency_ms=latency_ms,
                provider_used=AIProviderName.LOCAL_VLLM.value,
                cascade_path=cascade_path,
                model_used=model_id,
                start_perf=t0,
                swap_evaluator=_invoke,
            )

    def _call_studio_api_sync(
        self,
        ctx: CandidateSetupAuditContext,
        cascade_path: Sequence[str],
    ) -> AIVetoDecision:
        """Call Google AI Studio REST API with adaptive model selection & header auth."""
        if not self.api_key:
            raise RuntimeError("studio_api not ready: GEMINI_API_KEY is empty")

        blinded = self.aegis.normalize_ticker_blinded_telemetry(ctx)
        prompt = self.aegis.build_aegis_prompt(blinded)
        candidate_models = self._get_ranked_model_ids("studio_api", ctx, self.model)
        last_exc: Exception | None = None
        for model_id in candidate_models:
            t0 = time.perf_counter()
            url = self.GEMINI_ENDPOINT_TEMPLATE.format(model=model_id)

            def _invoke_studio(p_txt: str, target_url: str = url) -> str:
                payload_bytes = json.dumps(
                    {
                        "contents": [{"parts": [{"text": p_txt}]}],
                        "generationConfig": {
                            "temperature": 0.1,
                            "maxOutputTokens": 450,
                            "responseMimeType": "application/json",
                        },
                    }
                ).encode("utf-8")
                req = urllib.request.Request(
                    target_url,
                    data=payload_bytes,
                    headers={
                        "Content-Type": "application/json",
                        "x-goog-api-key": self.api_key,
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"studio_api HTTP {resp.status}")
                    resp_json = json.loads(resp.read().decode("utf-8"))
                    return resp_json["candidates"][0]["content"]["parts"][0]["text"]

            try:
                raw_text = _invoke_studio(prompt)
                latency_ms = (time.perf_counter() - t0) * 1000.0
                return self._verify_and_finalize_ai_response(
                    raw_text=raw_text,
                    ctx=ctx,
                    blinded=blinded,
                    latency_ms=latency_ms,
                    provider_used=AIProviderName.STUDIO_API.value,
                    cascade_path=cascade_path,
                    model_used=model_id,
                    start_perf=t0,
                    swap_evaluator=_invoke_studio,
                )
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
        """Async call to Google AI Studio REST API with header auth & Shield 5 order-swap."""
        if not self.api_key:
            raise RuntimeError("studio_api not ready: GEMINI_API_KEY is empty")

        blinded = self.aegis.normalize_ticker_blinded_telemetry(ctx)
        prompt = self.aegis.build_aegis_prompt(blinded)
        candidate_models = self._get_ranked_model_ids("studio_api", ctx, self.model)
        last_exc: Exception | None = None
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for model_id in candidate_models:
                t0 = time.perf_counter()
                url = self.GEMINI_ENDPOINT_TEMPLATE.format(model=model_id)

                async def _invoke_studio_async(p_txt: str, target_url: str = url) -> str:
                    body = {
                        "contents": [{"parts": [{"text": p_txt}]}],
                        "generationConfig": {
                            "temperature": 0.1,
                            "maxOutputTokens": 450,
                            "responseMimeType": "application/json",
                        },
                    }
                    async with session.post(
                        target_url,
                        json=body,
                        headers={
                            "Content-Type": "application/json",
                            "x-goog-api-key": self.api_key,
                        },
                    ) as resp:
                        if resp.status != 200:
                            err_txt = await resp.text()
                            raise RuntimeError(f"studio_api HTTP {resp.status}: {err_txt[:120]}")
                        resp_json = await resp.json()
                        return resp_json["candidates"][0]["content"]["parts"][0]["text"]

                try:
                    raw_text = await _invoke_studio_async(prompt)
                    latency_ms = (time.perf_counter() - t0) * 1000.0
                    return await self._verify_and_finalize_ai_response_async(
                        raw_text=raw_text,
                        ctx=ctx,
                        blinded=blinded,
                        latency_ms=latency_ms,
                        provider_used=AIProviderName.STUDIO_API.value,
                        cascade_path=cascade_path,
                        model_used=model_id,
                        start_perf=t0,
                        swap_evaluator=_invoke_studio_async,
                    )
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

        blinded = self.aegis.normalize_ticker_blinded_telemetry(ctx)
        prompt = self.aegis.build_aegis_prompt(blinded)
        timeout_int = max(5, int(round(self.agy_timeout_seconds)))
        iso_dir = self.get_isolated_temp_dir()
        candidate_models = self._get_ranked_model_ids("agy_pro", ctx, "default")
        if not candidate_models:
            self._agy_cooldown_until_utc = time.time() + 600.0
            raise RuntimeError("All agy_pro model families are quarantined/exhausted; falling back to pure_math + Z3")
        last_exc: Exception | None = None

        for model_id in candidate_models:
            t0 = time.perf_counter()

            def _build_cmd(p_txt: str, mid: str = model_id) -> list[str]:
                cmd = [
                    agy_exe,
                    "-p",
                    p_txt,
                    "--output-format",
                    "json",
                    "--print-timeout",
                    f"{timeout_int}s",
                    "--disable-slash-commands",
                ]
                if mid != "default":
                    cmd.extend(["--model", mid])
                else:
                    cmd.extend(["--effort", "low"])
                return cmd

            def _invoke_agy(p_txt: str, mid: str = model_id) -> str:
                proc = subprocess.run(
                    _build_cmd(p_txt, mid),
                    cwd=iso_dir,
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    timeout=float(timeout_int) + 4.0,
                    encoding="utf-8",
                    errors="replace",
                )
                stdout_txt = (proc.stdout or "").strip()
                stderr_txt = (proc.stderr or "").strip()
                combined_low = (stdout_txt + "\n" + stderr_txt).lower()
                if any(
                    tok in combined_low
                    for tok in ("credits", "balance is too low", "too low to continue", "quota", "resource_exhausted", "429")
                ):
                    raise RuntimeError(f"QUOTA_EXHAUSTED ({mid}): {(stdout_txt or stderr_txt)[:180]}")
                if not stdout_txt:
                    raise RuntimeError(f"agy_pro exited {proc.returncode} with empty stdout: {stderr_txt[:160]}")
                return stdout_txt

            try:
                stdout_txt = _invoke_agy(prompt)
                latency_ms = (time.perf_counter() - t0) * 1000.0
                decision = self._verify_and_finalize_ai_response(
                    raw_text=stdout_txt,
                    ctx=ctx,
                    blinded=blinded,
                    latency_ms=latency_ms,
                    provider_used=AIProviderName.AGY_PRO.value,
                    cascade_path=cascade_path,
                    model_used=model_id,
                    start_perf=t0,
                    swap_evaluator=_invoke_agy,
                )
                self._agy_consecutive_failures = 0
                return decision
            except Exception as exc:
                last_exc = exc
                err_s = str(exc)
                self.model_selector.record_failure(
                    "agy_pro",
                    model_id,
                    err_s,
                    is_quota_or_429=(
                        "quota" in err_s.lower()
                        or "429" in err_s
                        or "exhaust" in err_s.lower()
                        or "credits" in err_s.lower()
                        or "balance" in err_s.lower()
                    ),
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
        """Native non-blocking `asyncio.create_subprocess_exec` invocation of `agy_pro` with clean timeout kill."""
        now_utc = time.time()
        if now_utc < self._agy_cooldown_until_utc:
            rem_s = int(self._agy_cooldown_until_utc - now_utc)
            raise RuntimeError(f"agy_pro in cooldown ({rem_s}s remaining); skipping to fallback")

        agy_exe = self._cached_agy_path
        if not agy_exe or not self._cached_agy_auth_ok:
            raise RuntimeError("agy_pro not ready or not found")

        blinded = self.aegis.normalize_ticker_blinded_telemetry(ctx)
        prompt = self.aegis.build_aegis_prompt(blinded)
        timeout_int = max(5, int(round(self.agy_timeout_seconds)))
        iso_dir = self.get_isolated_temp_dir()
        candidate_models = self._get_ranked_model_ids("agy_pro", ctx, "default")
        if not candidate_models:
            self._agy_cooldown_until_utc = time.time() + 600.0
            raise RuntimeError("All agy_pro model families are quarantined/exhausted; falling back to pure_math + Z3")
        last_exc: Exception | None = None

        for model_id in candidate_models:
            t0 = time.perf_counter()

            async def _invoke_agy_async(p_txt: str, mid: str = model_id) -> str:
                cmd = [
                    agy_exe,
                    "-p",
                    p_txt,
                    "--output-format",
                    "json",
                    "--print-timeout",
                    f"{timeout_int}s",
                    "--disable-slash-commands",
                ]
                if mid != "default":
                    cmd.extend(["--model", mid])
                else:
                    cmd.extend(["--effort", "low"])
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=iso_dir,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    raw_out, raw_err = await asyncio.wait_for(
                        proc.communicate(),
                        timeout=float(timeout_int) + 4.0,
                    )
                except asyncio.TimeoutError as exc:
                    try:
                        proc.kill()
                        await proc.wait()
                    except OSError as kill_exc:
                        logger.debug("agy_pro cleanup after timeout: %s", kill_exc)
                    raise TimeoutError(f"agy_pro async timed out after {timeout_int + 4}s") from exc

                stdout_txt = (raw_out or b"").decode("utf-8", errors="replace").strip()
                stderr_txt = (raw_err or b"").decode("utf-8", errors="replace").strip()
                combined_low = (stdout_txt + "\n" + stderr_txt).lower()
                if any(
                    tok in combined_low
                    for tok in ("credits", "balance is too low", "too low to continue", "quota", "resource_exhausted", "429")
                ):
                    raise RuntimeError(f"QUOTA_EXHAUSTED ({mid}): {(stdout_txt or stderr_txt)[:180]}")
                if not stdout_txt:
                    raise RuntimeError(f"agy_pro exited {proc.returncode} with empty stdout: {stderr_txt[:160]}")
                return stdout_txt

            try:
                stdout_txt = await _invoke_agy_async(prompt)
                latency_ms = (time.perf_counter() - t0) * 1000.0
                decision = await self._verify_and_finalize_ai_response_async(
                    raw_text=stdout_txt,
                    ctx=ctx,
                    blinded=blinded,
                    latency_ms=latency_ms,
                    provider_used=AIProviderName.AGY_PRO.value,
                    cascade_path=cascade_path,
                    model_used=model_id,
                    start_perf=t0,
                    swap_evaluator=_invoke_agy_async,
                )
                self._agy_consecutive_failures = 0
                return decision
            except Exception as exc:
                last_exc = exc
                err_s = str(exc)
                self.model_selector.record_failure(
                    "agy_pro",
                    model_id,
                    err_s,
                    is_quota_or_429=(
                        "quota" in err_s.lower()
                        or "429" in err_s
                        or "exhaust" in err_s.lower()
                        or "credits" in err_s.lower()
                        or "balance" in err_s.lower()
                    ),
                )
                logger.warning("agy_pro async model %s failed (%s); cascading...", model_id, err_s)

        self._agy_consecutive_failures += 1
        if self._agy_consecutive_failures >= 2:
            self._agy_cooldown_until_utc = time.time() + 600.0
            logger.warning(
                "agy_pro failed %d times in a row (%s). Entering 10-minute cooldown to pure_math.",
                self._agy_consecutive_failures,
                last_exc,
            )
        raise last_exc or RuntimeError("agy_pro async failed")

    def _evaluate_pure_math_fallback(
        self,
        ctx: CandidateSetupAuditContext,
        start_perf: float,
        cascade_path: Sequence[str],
        fallback_reason_prefix: str = "",
    ) -> AIVetoDecision:
        """Deterministic 0ms $0 Pure Math + Z3 Axiom Gate pass-through."""
        latency_ms = (time.perf_counter() - start_perf) * 1000.0
        min_net_r = float(getattr(self, "min_net_avg_R", 0.10))
        blinded = self.aegis.normalize_ticker_blinded_telemetry(ctx)
        z3_dec, z3_note = self.aegis.z3_pure_math_axiom_gate(blinded.telemetry)

        if ctx.net_avg_R < min_net_r:
            dec = "VETO"
            reason = f"Pure Math Gate: net_avg_R ({ctx.net_avg_R:+.3f}R) < {min_net_r:.2f}R"
        elif ctx.spread_pct_of_sl > 0.15 + 1e-9:
            dec = "VETO"
            reason = f"Pure Math Gate: spread is {ctx.spread_pct_of_sl * 100:.1f}% of SL (> 15.0%)"
        elif z3_dec == "VETO":
            dec = "VETO"
            reason = z3_note
        else:
            dec = "APPROVE"
            reason = (
                f"Pure Math + Z3 Pass-Through: Regime={ctx.regime}, "
                f"net_avg_R={ctx.net_avg_R:+.3f}R, ProbProfit={ctx.prob_profit * 100:.1f}% ({z3_note})"
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

            if provider == AIProviderName.LOCAL_VLLM.value:
                try:
                    return self._call_local_vllm_sync(ctx, cascade_path=attempted)
                except Exception as exc:
                    err_msg = f"local_vllm failed ({exc})"
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

            if provider == AIProviderName.LOCAL_VLLM.value:
                try:
                    return await self._call_local_vllm_async(ctx, cascade_path=attempted)
                except Exception as exc:
                    err_msg = f"local_vllm failed ({exc})"
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
