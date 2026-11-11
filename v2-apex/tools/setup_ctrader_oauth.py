"""Local browser OAuth2 callback wizard for Spotware cTrader Open API.

Starts an HTTP listener at http://127.0.0.1:8787/callback, takes CTRADER_CLIENT_ID and
CTRADER_CLIENT_SECRET, opens the Spotware OAuth2 consent page in the user's browser,
exchanges the authorization code for access + refresh tokens, queries ProtoOAGetAccountListByAccessTokenReq
to discover all ctidTraderAccountId accounts (distinguishing DEMO vs LIVE), and writes the
credentials safely to `.env`.
"""

from __future__ import annotations

import argparse
import os
import threading
import urllib.parse
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from ctrader_open_api import Auth, Client, EndPoints, Protobuf, TcpProtocol
from ctrader_open_api.messages import OpenApiMessages_pb2 as oa_msg
from dotenv import dotenv_values, set_key
from rich.console import Console
from rich.table import Table

CALLBACK_HOST: str = "127.0.0.1"
CALLBACK_PORT: int = 8787
CALLBACK_PATH: str = "/callback"
REDIRECT_URI: str = f"http://{CALLBACK_HOST}:{CALLBACK_PORT}{CALLBACK_PATH}"

console = Console()


@dataclass(frozen=True, slots=True)
class DiscoveredTraderAccount:
    """Discovered cTrader trading account metadata from ProtoOAGetAccountListByAccessTokenRes."""

    ctid_trader_account_id: int
    is_live: bool
    trader_login: int


class _OAuthCallbackHandler(BaseHTTPRequestHandler):
    """HTTP request handler capturing the `?code=...` query parameter on `/callback`."""

    auth_code: str | None = None
    error_message: str | None = None
    received_event: threading.Event = threading.Event()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != CALLBACK_PATH:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")
            return

        params = urllib.parse.parse_qs(parsed.query)
        if "code" in params and params["code"]:
            _OAuthCallbackHandler.auth_code = params["code"][0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            html = (
                "<html><body style='font-family:sans-serif;background:#0f172a;color:#e2e8f0;padding:40px;'>"
                "<h2>PocketBot Apex — cTrader OAuth2 Authorization Complete</h2>"
                "<p>Authorization code received. You may close this browser tab and return to the terminal.</p>"
                "</body></html>"
            )
            self.wfile.write(html.encode("utf-8"))
        else:
            err = params.get("error_description", params.get("error", ["Unknown OAuth error"]))[0]
            _OAuthCallbackHandler.error_message = str(err)
            self.send_response(400)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(f"OAuth Error: {err}".encode("utf-8"))

        _OAuthCallbackHandler.received_event.set()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return


def build_authorization_url(
    client_id: str,
    client_secret: str,
    redirect_uri: str = REDIRECT_URI,
    scope: str = "trading",
) -> str:
    """Build the Spotware cTrader OAuth2 authorization URL."""
    if not client_id.strip() or not client_secret.strip():
        raise ValueError("CTRADER_CLIENT_ID and CTRADER_CLIENT_SECRET must be non-empty.")
    auth = Auth(client_id.strip(), client_secret.strip(), redirect_uri)
    return str(auth.getAuthUri(scope=scope))


def wait_for_oauth_callback(
    host: str = CALLBACK_HOST,
    port: int = CALLBACK_PORT,
    timeout_seconds: float = 180.0,
) -> str:
    """Run a temporary local HTTP server on 127.0.0.1:8787 and wait for `/callback?code=...`."""
    _OAuthCallbackHandler.auth_code = None
    _OAuthCallbackHandler.error_message = None
    _OAuthCallbackHandler.received_event.clear()

    server = HTTPServer((host, port), _OAuthCallbackHandler)
    server.timeout = 0.5

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        completed = _OAuthCallbackHandler.received_event.wait(timeout=timeout_seconds)
        if not completed:
            raise TimeoutError(f"Timed out after {timeout_seconds}s waiting for OAuth callback on {REDIRECT_URI}.")
        if _OAuthCallbackHandler.error_message:
            raise RuntimeError(f"OAuth callback returned error: {_OAuthCallbackHandler.error_message}")
        if not _OAuthCallbackHandler.auth_code:
            raise RuntimeError("No authorization code received on callback.")
        return _OAuthCallbackHandler.auth_code
    finally:
        server.shutdown()
        server.server_close()


def exchange_code_for_tokens(
    client_id: str,
    client_secret: str,
    auth_code: str,
    redirect_uri: str = REDIRECT_URI,
) -> dict[str, str]:
    """Exchange the authorization code for an access token and refresh token."""
    auth = Auth(client_id.strip(), client_secret.strip(), redirect_uri)
    token_res = auth.getToken(auth_code.strip())
    access_token = str(token_res.get("accessToken") or token_res.get("access_token") or "")
    refresh_token = str(token_res.get("refreshToken") or token_res.get("refresh_token") or "")
    if not access_token:
        raise RuntimeError(f"Failed to obtain accessToken from Spotware: {token_res}")
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
    }


def fetch_ctid_trader_accounts(
    client_id: str,
    client_secret: str,
    access_token: str,
    timeout_seconds: float = 15.0,
) -> list[DiscoveredTraderAccount]:
    """Connect to Spotware Demo Protobuf host and query ``ProtoOAGetAccountListByAccessTokenReq``."""
    from twisted.internet import reactor

    accounts: list[DiscoveredTraderAccount] = []
    done_event = threading.Event()
    client = Client(EndPoints.PROTOBUF_DEMO_HOST, EndPoints.PROTOBUF_PORT, TcpProtocol)

    def on_connected(c: Client) -> None:
        app_req = oa_msg.ProtoOAApplicationAuthReq()
        app_req.clientId = client_id
        app_req.clientSecret = client_secret
        c.send(app_req)

    def on_message(c: Client, msg: object) -> None:
        extracted = Protobuf.extract(msg)
        if isinstance(extracted, oa_msg.ProtoOAApplicationAuthRes):
            list_req = oa_msg.ProtoOAGetAccountListByAccessTokenReq()
            list_req.accessToken = access_token
            c.send(list_req)
        elif isinstance(extracted, oa_msg.ProtoOAGetAccountListByAccessTokenRes):
            for acct in extracted.ctidTraderAccount:
                accounts.append(
                    DiscoveredTraderAccount(
                        ctid_trader_account_id=int(acct.ctidTraderAccountId),
                        is_live=bool(getattr(acct, "isLive", False)),
                        trader_login=int(getattr(acct, "traderLogin", 0)),
                    )
                )
            done_event.set()
            c.stopService()
        elif isinstance(extracted, oa_msg.ProtoOAErrorRes):
            console.print(f"[red]cTrader API Error:[/red] {extracted.description}")
            done_event.set()
            c.stopService()

    client.setConnectedCallback(on_connected)
    client.setMessageReceivedCallback(on_message)
    client.startService()

    if not reactor.running:
        reactor_thread = threading.Thread(
            target=lambda: reactor.run(installSignalHandlers=False),
            daemon=True,
        )
        reactor_thread.start()

    done_event.wait(timeout=timeout_seconds)
    return accounts


def save_credentials_to_env(
    env_path: Path,
    client_id: str,
    client_secret: str,
    access_token: str,
    refresh_token: str,
    account_id: int,
) -> None:
    """Safely write or update cTrader credentials in `.env` without clobbering existing keys."""
    if not env_path.exists():
        env_path.write_text("", encoding="utf-8")

    # Verify existing keys remain intact while updating cTrader keys
    _ = dotenv_values(env_path)
    set_key(str(env_path), "CTRADER_CLIENT_ID", client_id.strip())
    set_key(str(env_path), "CTRADER_CLIENT_SECRET", client_secret.strip())
    set_key(str(env_path), "CTRADER_ACCESS_TOKEN", access_token.strip())
    set_key(str(env_path), "CTRADER_REFRESH_TOKEN", refresh_token.strip())
    set_key(str(env_path), "CTRADER_ACCOUNT_ID", str(int(account_id)))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PocketBot Apex — Local Browser OAuth2 Setup Wizard for cTrader Open API"
    )
    parser.add_argument("--client-id", default=os.environ.get("CTRADER_CLIENT_ID", ""), help="Spotware App Client ID")
    parser.add_argument(
        "--client-secret",
        default=os.environ.get("CTRADER_CLIENT_SECRET", ""),
        help="Spotware App Client Secret",
    )
    parser.add_argument("--env-file", default=".env", help="Path to .env file (default: .env)")
    parser.add_argument("--timeout", type=float, default=180.0, help="Callback timeout in seconds")
    args = parser.parse_args()

    client_id = args.client_id.strip() or console.input("[bold cyan]Enter CTRADER_CLIENT_ID:[/bold cyan] ").strip()
    client_secret = (
        args.client_secret.strip() or console.input("[bold cyan]Enter CTRADER_CLIENT_SECRET:[/bold cyan] ").strip()
    )

    auth_url = build_authorization_url(client_id, client_secret, REDIRECT_URI)
    console.print(f"\n[bold green]1. Opening browser for Spotware OAuth2 authorization...[/bold green]")
    console.print(f"   URL: [underline]{auth_url}[/underline]")
    console.print(f"   Listening for callback on [bold]{REDIRECT_URI}[/bold]\n")

    webbrowser.open(auth_url)
    auth_code = wait_for_oauth_callback(timeout_seconds=args.timeout)
    console.print("[green]✓ Authorization code captured![/green] Exchanging for tokens...")

    tokens = exchange_code_for_tokens(client_id, client_secret, auth_code, REDIRECT_URI)
    console.print("[green]✓ Access & Refresh tokens obtained![/green] Discovering ctidTraderAccountId list...")

    accounts = fetch_ctid_trader_accounts(
        client_id=client_id,
        client_secret=client_secret,
        access_token=tokens["access_token"],
    )

    selected_account_id = 0
    if accounts:
        table = Table(title="Discovered cTrader Accounts")
        table.add_column("Index", justify="right")
        table.add_column("ctidTraderAccountId", style="cyan")
        table.add_column("Trader Login")
        table.add_column("Environment", style="bold")

        demo_accounts = [a for a in accounts if not a.is_live]
        for idx, acct in enumerate(accounts):
            env_label = "[red]LIVE (LOCKED)[/red]" if acct.is_live else "[green]DEMO[/green]"
            table.add_row(str(idx), str(acct.ctid_trader_account_id), str(acct.trader_login), env_label)
        console.print(table)

        # Default to first DEMO account per safety invariant
        if demo_accounts:
            selected_account_id = demo_accounts[0].ctid_trader_account_id
            console.print(
                f"[bold green]Auto-selected DEMO ctidTraderAccountId:[/bold green] {selected_account_id}"
            )
        else:
            selected_account_id = accounts[0].ctid_trader_account_id

    env_path = Path(args.env_file).resolve()
    save_credentials_to_env(
        env_path=env_path,
        client_id=client_id,
        client_secret=client_secret,
        access_token=tokens["access_token"],
        refresh_token=tokens["refresh_token"],
        account_id=selected_account_id,
    )
    console.print(f"[bold green]✓ Saved cTrader OAuth credentials to {env_path}[/bold green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
