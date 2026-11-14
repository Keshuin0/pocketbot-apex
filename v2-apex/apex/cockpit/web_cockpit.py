"""Embedded async aiohttp HTTP + WebSocket server for the PocketBot Apex Chrome App Cockpit."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import secrets
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from aiohttp import WSMsgType, web

logger = logging.getLogger("apex.cockpit.web")

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_HTML_PATH = STATIC_DIR / "index.html"

def _query_windows_app_path(exe_name: str) -> Optional[str]:
    """Query HKLM/HKCU App Paths registry for a browser executable on Windows."""
    try:
        import winreg
    except ImportError:
        return None

    subkey = rf"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{exe_name}"
    for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
        try:
            with winreg.OpenKey(hive, subkey) as key:
                val, _ = winreg.QueryValueEx(key, "")
                if val and os.path.isfile(str(val)):
                    return str(val)
        except OSError:
            continue
    return None


def find_browser_executable() -> Optional[str]:
    """Locate Chrome (preferred) or Edge executable reliably across standard Windows paths & Registry."""
    env_dirs = [
        os.environ.get("ProgramFiles", r"C:\Program Files"),
        os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
        os.environ.get("LocalAppData", ""),
    ]
    chrome_candidates: list[str] = [
        os.path.join(d, "Google", "Chrome", "Application", "chrome.exe")
        for d in env_dirs
        if d
    ]
    for path in chrome_candidates:
        if os.path.isfile(path):
            return path

    reg_chrome = _query_windows_app_path("chrome.exe")
    if reg_chrome:
        return reg_chrome

    edge_candidates: list[str] = [
        os.path.join(d, "Microsoft", "Edge", "Application", "msedge.exe")
        for d in env_dirs
        if d
    ]
    for path in edge_candidates:
        if os.path.isfile(path):
            return path

    reg_edge = _query_windows_app_path("msedge.exe")
    if reg_edge:
        return reg_edge

    return shutil.which("chrome") or shutil.which("msedge")


def launch_chrome_app(url: str, no_gui: bool = False) -> Optional[subprocess.Popen[bytes]]:
    """Launch a standalone frameless Chrome/Edge Desktop App window if GUI is enabled."""
    if no_gui:
        return None
    if os.environ.get("GITHUB_ACTIONS", "").strip().lower() == "true":
        return None
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return None

    exe_path = find_browser_executable()
    if exe_path is not None:
        try:
            creationflags = 0
            if os.name == "nt":
                creationflags = getattr(subprocess, "DETACHED_PROCESS", 0x00000008) | getattr(
                    subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200
                )
            proc = subprocess.Popen(
                [exe_path, f"--app={url}", "--new-window", "--window-size=1480,920"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                creationflags=creationflags,
            )
            logger.info("Launched standalone cockpit app: %s --app=%s", exe_path, url)
            return proc
        except Exception as exc:
            logger.warning("Failed to launch cockpit browser window: %s", exc)

    if hasattr(os, "startfile"):
        try:
            os.startfile(url)  # type: ignore[attr-defined]
            logger.info("Opened cockpit URL via os.startfile: %s", url)
        except Exception as exc:
            logger.warning("os.startfile(%s) failed: %s", url, exc)
    else:
        logger.warning("Chrome/Edge executable not found; open %s manually.", url)
    return None


def _default_state(csrf_token: str) -> dict[str, Any]:
    return {
        "csrf_token": csrf_token,
        "run_mode": "DEMO-AUTO",
        "mind_state_badge": "AWAKE",
        "universe_scanned_count": 0,
        "universe_total_count": 0,
        "account_currency": "CAD",
        "account_equity": 10000.0,
        "daily_pnl": 0.0,
        "open_positions_count": 0,
        "mesh_badge": "ACTIVE",
        "verified_ai_model": "DISCOVERING",
        "ai_brain_badge": "READY",
        "quarantined_families": [],
        "contagion_shock_z": 0.0,
        "aegis_badge": "Z3 ARMED",
        "paused": False,
        "drill_state": {
            "status": "IDLE",
            "symbol": "",
            "ticket": 0,
            "floating_pnl": 0.0,
            "message": "Ready for 0.01-lot FusionMarkets-Demo lifecycle drill.",
        },
        "replay_state": {
            "status": "IDLE",
            "message": "Ready for 100x historical replay.",
            "equity_curve": [],
        },
        "open_positions": [],
        "rows": [],
        "broker_events": [],
        "updated_at": time.time(),
    }


class WebCockpitServer:
    """Thread-hosted aiohttp HTTP + WebSocket server bound strictly to 127.0.0.1."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        max_port: int = 8785,
        control_handlers: Optional[dict[str, Callable[[], Any]]] = None,
    ) -> None:
        self.host = host
        self.requested_port = int(port)
        self.max_port = max(int(max_port), self.requested_port)
        self.port: int = self.requested_port
        self.csrf_token: str = secrets.token_urlsafe(32)
        self._state: dict[str, Any] = _default_state(self.csrf_token)
        self._state_lock = threading.Lock()
        self._control_handlers: dict[str, Callable[[], Any]] = dict(control_handlers or {})

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        self._clients: set[web.WebSocketResponse] = set()
        self._ready_event = threading.Event()
        self._start_error: Optional[BaseException] = None
        self._stopped = False

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def set_control_handler(self, action: str, handler: Callable[[], Any]) -> None:
        self._control_handlers[action] = handler

    def get_state_snapshot(self) -> dict[str, Any]:
        with self._state_lock:
            return json.loads(json.dumps(self._state))

    def start(self, timeout: float = 5.0) -> str:
        if self._thread is not None and self._thread.is_alive():
            return self.url

        self._stopped = False
        self._ready_event.clear()
        self._start_error = None
        self._thread = threading.Thread(
            target=self._run_loop_thread,
            name="ApexWebCockpitThread",
            daemon=True,
        )
        self._thread.start()
        if not self._ready_event.wait(timeout=timeout):
            raise TimeoutError("WebCockpitServer failed to bind within timeout.")
        if self._start_error is not None:
            raise RuntimeError(f"WebCockpitServer failed to start: {self._start_error}") from self._start_error
        return self.url

    def stop(self, timeout: float = 3.0) -> None:
        self._stopped = True
        loop = self._loop
        if loop is not None and loop.is_running():
            fut = asyncio.run_coroutine_threadsafe(self._async_shutdown(), loop)
            try:
                fut.result(timeout=timeout)
            except Exception:
                pass
            loop.call_soon_threadsafe(loop.stop)
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._thread = None
        self._loop = None

    def push_state(self, state_update: dict[str, Any]) -> None:
        with self._state_lock:
            merged = dict(self._state)
            merged.update(state_update)
            merged["csrf_token"] = self.csrf_token
            merged["updated_at"] = time.time()
            self._state = merged
            payload = json.dumps(merged, default=str)
        self._schedule_broadcast(payload)

    def update_drill_state(self, drill_state: dict[str, Any]) -> None:
        with self._state_lock:
            current = dict(self._state.get("drill_state", {}))
            current.update(drill_state)
            self._state["drill_state"] = current
            self._state["updated_at"] = time.time()
            payload = json.dumps(self._state, default=str)
        self._schedule_broadcast(payload)

    def update_replay_state(self, replay_state: dict[str, Any]) -> None:
        with self._state_lock:
            current = dict(self._state.get("replay_state", {}))
            current.update(replay_state)
            self._state["replay_state"] = current
            self._state["updated_at"] = time.time()
            payload = json.dumps(self._state, default=str)
        self._schedule_broadcast(payload)

    def _schedule_broadcast(self, payload: str) -> None:
        loop = self._loop
        if loop is None or not loop.is_running() or self._stopped:
            return
        try:
            asyncio.run_coroutine_threadsafe(self._broadcast(payload), loop)
        except RuntimeError:
            pass

    async def _broadcast(self, payload: str) -> None:
        if not self._clients:
            return
        stale: list[web.WebSocketResponse] = []
        for ws in list(self._clients):
            if ws.closed:
                stale.append(ws)
                continue
            try:
                await ws.send_str(payload)
            except Exception:
                stale.append(ws)
        for ws in stale:
            self._clients.discard(ws)

    def _run_loop_thread(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._async_startup())
            self._ready_event.set()
            loop.run_forever()
        except BaseException as exc:
            self._start_error = exc
            self._ready_event.set()
        finally:
            try:
                loop.run_until_complete(self._async_shutdown())
            except Exception:
                pass
            loop.close()

    async def _async_startup(self) -> None:
        app = web.Application()
        app.router.add_get("/", self._handle_index)
        app.router.add_get("/api/state", self._handle_api_state)
        app.router.add_get("/ws", self._handle_ws)
        app.router.add_post("/api/control/{action}", self._handle_control)

        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        self._runner = runner

        last_err: Optional[OSError] = None
        for candidate_port in range(self.requested_port, self.max_port + 1):
            site = web.TCPSite(runner, self.host, candidate_port)
            try:
                await site.start()
                self._site = site
                self.port = candidate_port
                logger.info("WebCockpitServer listening on %s", self.url)
                return
            except OSError as exc:
                last_err = exc
                continue
        raise RuntimeError(
            f"Unable to bind WebCockpitServer on {self.host}:{self.requested_port}-{self.max_port}: {last_err}"
        )

    async def _async_shutdown(self) -> None:
        for ws in list(self._clients):
            try:
                await ws.close()
            except Exception:
                pass
        self._clients.clear()
        if self._runner is not None:
            try:
                await self._runner.cleanup()
            except Exception:
                pass
            self._runner = None
            self._site = None

    async def _handle_index(self, request: web.Request) -> web.Response:
        if INDEX_HTML_PATH.is_file():
            html = INDEX_HTML_PATH.read_text(encoding="utf-8")
            return web.Response(text=html, content_type="text/html")
        return web.Response(text="<h1>PocketBot Apex Cockpit HTML missing</h1>", content_type="text/html", status=404)

    async def _handle_api_state(self, request: web.Request) -> web.Response:
        return web.json_response(self.get_state_snapshot())

    async def _handle_ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=20.0)
        await ws.prepare(request)
        self._clients.add(ws)
        try:
            await ws.send_str(json.dumps(self.get_state_snapshot(), default=str))
            async for msg in ws:
                if msg.type == WSMsgType.TEXT and msg.data.strip().lower() == "ping":
                    await ws.send_str(json.dumps(self.get_state_snapshot(), default=str))
                elif msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                    break
        finally:
            self._clients.discard(ws)
        return ws

    async def _handle_control(self, request: web.Request) -> web.Response:
        provided_csrf = request.headers.get("X-Apex-CSRF", "")
        if not provided_csrf or not secrets.compare_digest(provided_csrf, self.csrf_token):
            return web.json_response(
                {"ok": False, "error": "Invalid or missing X-Apex-CSRF token"},
                status=403,
            )

        action = request.match_info.get("action", "").strip()
        if action not in {"pause", "drill", "replay", "refresh_ai", "panic_flatten"}:
            return web.json_response({"ok": False, "error": f"Unknown action: {action}"}, status=404)

        if action == "pause" and "pause" not in self._control_handlers:
            with self._state_lock:
                new_paused = not bool(self._state.get("paused", False))
                self._state["paused"] = new_paused
                payload = json.dumps(self._state, default=str)
            self._schedule_broadcast(payload)
            return web.json_response({"ok": True, "paused": new_paused, "message": "Toggled pause state."})

        handler = self._control_handlers.get(action)
        if handler is None:
            return web.json_response(
                {"ok": False, "error": f"No handler registered for '{action}'"},
                status=501,
            )

        try:
            payload: dict[str, Any] = {}
            if request.can_read_body:
                try:
                    raw_body = await request.json()
                    if isinstance(raw_body, dict):
                        payload = raw_body
                except Exception:
                    payload = {}

            sig = inspect.signature(handler)
            if len(sig.parameters) >= 1:
                result = handler(payload)
            else:
                result = handler()
            if asyncio.iscoroutine(result):
                result = await result
            if isinstance(result, dict):
                return web.json_response({"ok": True, **result})
            return web.json_response({"ok": True, "result": result})
        except Exception as exc:
            logger.exception("Control action '%s' failed", action)
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
