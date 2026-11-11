# PocketBot Apex — Diagnostic & Troubleshooting Guide

This guide details resolutions for the most common environment, broker, and network issues.

---

## 1. MetaTrader 5 Issues

### 1.1 "Algo Trading Disabled" / Order Placement Fails
- **Symptom:** Logs report `TRADE_RETCODE_AUTOTRADING_DISABLED` or execution router falls back to Paper-Shadow mode.
- **Cause:** Automated trading is disabled in the MT5 terminal toolbar or terminal options.
- **Fix:**
  1. Look at the top toolbar in MT5. Click the **Algo Trading** button to turn it **GREEN**.
  2. Press `Ctrl + O` $\to$ **Expert Advisors** tab.
  3. Check the box: **"Allow automated trading"**.
  4. Check the box: **"Allow DLL imports"**.
  5. Click **OK**.

### 1.2 "MetaTrader 5 terminal not found" / IPC Initialization Fails
- **Symptom:** `MarketFeedRouter.connect()` reports `mt5_connected: False`.
- **Cause:** MT5 terminal is either not running, or 32-bit Python is being used instead of 64-bit.
- **Fix:**
  1. Make sure your MetaTrader 5 application is open and logged into your account.
  2. Run `0_FIRST_TIME_SETUP.bat` to ensure a 64-bit Python 3.12 virtual environment is created.
  3. If you have multiple MT5 terminals installed, open the specific terminal you wish to trade with before launching PocketBot.

### 1.3 "Symbol Not Found" / Quotes Missing
- **Symptom:** `Failed to fetch ticks for EURUSD`.
- **Cause:** The symbol is hidden from the Market Watch window, or your broker uses a custom suffix (e.g. `EURUSD.raw`, `EURUSDm`, `EURUSD.pro`).
- **Fix:**
  1. In MT5, press `Ctrl + U` (Symbols window).
  2. Locate the symbol and click **Show Symbol**.
  3. If your broker appends a suffix, open `config/apex_config.json` and adjust the symbol name to match your broker's exact format.

---

## 2. cTrader Open API Issues

### 2.1 "Connection Refused / Timeout on Port 5035"
- **Symptom:** cTrader feed cannot connect to `demo.ctraderapi.com:5035` or `live.ctraderapi.com:5035`.
- **Cause:** Network firewall, corporate proxy, or VPN blocking outbound TCP connections on port 5035 or 5036.
- **Fix:**
  1. Open Windows Defender Firewall $\to$ Advanced Settings.
  2. Ensure outbound TCP traffic on ports `5035` and `5036` is allowed.
  3. If using Cloudflare WARP or a third-party VPN, test with VPN temporarily disabled or add an exemption for cTrader API endpoints.

### 2.2 "Invalid Client Credentials or Expired Token"
- **Symptom:** Logs show `ProtoOAErrorRes: CH_ACCESS_TOKEN_INVALID`.
- **Cause:** Access token has expired or credentials were typed incorrectly.
- **Fix:**
  1. Double-click `1_SETUP_CTRADER_KEYS.bat`.
  2. Complete the OAuth authorization in your browser to regenerate fresh access and refresh tokens.

---

## 3. AI Brain & Risk Router Issues

### 3.1 "AI Brain running in PURE_MATH fallback"
- **Symptom:** Dashboard header shows `AI Brain: PURE_MATH [0ms Deterministic Quant Only]`.
- **Cause:** Neither `GEMINI_API_KEY` was provided in `.env` nor was `agy.exe` found on your computer.
- **Note:** This is **NOT an error!** PocketBot Apex is designed to operate 100% autonomously at $0 AI token cost using pure mathematical filters (Regime, Momentum/DOM, 10,000-Path Monte Carlo, and 6 Circuit Breakers).
- **To enable Gemini AI Veto:**
  - Option A: Obtain a free Gemini API key from [Google AI Studio](https://aistudio.google.com/) and paste it into `.env` as `GEMINI_API_KEY=AIzaSy...`.
  - Option B: If you use Antigravity / Google AI Pro (`agy.exe`), double-click `5_CONFIGURE_AI_MODE.bat` and select `[2] Auto-Pro First`.

---

## 4. Emergency Procedures

### 4.1 How to Stop All Trading Immediately
1. Press `Ctrl + C` in the running command window.
2. If you are away from your PC and have Telegram configured, send `/panic` to your bot to instantly close all active trades and cancel all pending orders.
3. You can also manually close all positions inside the MetaTrader 5 or cTrader desktop client at any time.
