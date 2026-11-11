# PocketBot Apex — Broker Connectivity & API Setup Guide

This guide walks you through connecting **MetaTrader 5 (MT5)** and **cTrader Open API** to PocketBot Apex.

---

## 1. MetaTrader 5 Setup (5 Minutes)

PocketBot Apex connects to MT5 through local Windows Inter-Process Communication (IPC). It works with any broker providing MetaTrader 5 (e.g., IC Markets, Pepperstone, FTMO, Eightcap, Fusion Markets, etc.).

### Step 1.1: Install MT5 (64-Bit)
1. Download and install MetaTrader 5 from your chosen broker.
2. Log into your **Demo Account** inside MT5.
3. Keep the MT5 terminal running on your computer.

### Step 1.2: Enable Algo Trading & DLL Imports
1. In MT5, click **Tools** $\to$ **Options** (or press `Ctrl+O`).
2. Navigate to the **Expert Advisors** tab.
3. Check the following boxes:
   - [x] **Allow automated trading**
   - [x] **Allow DLL imports**
   - [x] **Allow WebRequest for listed URL** (optional)
4. Click **OK**.
5. Look at the toolbar at the top of MT5 and ensure the **Algo Trading** button is toggled **ON** (green icon).

```
   +---------------------------------------------+
   |  [Algo Trading: ON]   |   Market Watch      |
   +---------------------------------------------+
```

### Step 1.3: Enable Market Watch Symbols
1. Press `Ctrl+U` in MT5 to open the **Symbols** window.
2. Select your trading symbols (e.g., `EURUSD`, `GBPUSD`, `USDJPY`, `XAUUSD`, `BTCUSD`) and click **Show Symbol**.
3. Press `Ctrl+M` to open the **Market Watch** panel and verify price quotes are updating.

---

## 2. cTrader Open API Setup (1-Click OAuth Wizard)

cTrader Open API gives you free, ultra-low-latency direct access to broker pricing and full **Level 2 Depth of Market (DOM)** quotes with zero platform fees.

### Step 2.1: Register on Spotware Open API Portal
1. Visit the official Spotware Open API Portal: [https://openapi.ctrader.com](https://openapi.ctrader.com)
2. Log in with your cTrader ID (cTID) or create a free account.
3. Click **Create Application**.
4. Fill in the application details:
   - **Application Name:** `PocketBot Apex`
   - **Redirect URI:** `http://localhost:5000/callback`
   - **Application Type:** `Desktop` or `Web`
5. Save the application. Spotware will display your:
   - **Client ID** (e.g. `1234_abc...`)
   - **Client Secret** (e.g. `xyz987...`)

### Step 2.2: Run the 1-Click OAuth Setup Wizard
1. Open the `v2-apex/` folder.
2. Double-click `1_SETUP_CTRADER_KEYS.bat`.
3. Paste your **Client ID** and **Client Secret** when prompted.
4. The wizard automatically opens your default browser to authenticate with Spotware.
5. Click **Allow Access** in your browser.
6. The setup wizard captures the authorization code, fetches your OAuth access token, retrieves your cTrader trading accounts, and writes them securely to your local `.env` file!

```
==============================================================================
  POCKETBOT APEX - CTRADER OAUTH 2.0 SETUP WIZARD
==============================================================================
[*] Client ID and Secret detected.
[*] Starting local callback server on http://localhost:5000/callback ...
[*] Browser opened for Spotware Authorization.
[OK] Authorization code captured!
[OK] Token exchange successful!
[OK] Discovered cTrader Account ID: 12345678 (Demo)
[OK] Credentials saved securely to .env!
```

---

## 3. Selecting Broker Mesh Topology

PocketBot Apex supports 4 dynamic mesh topologies configured via `6_CONFIGURE_BROKER_MESH.bat`:

| Topology Mode | Description | How It Operates |
| :--- | :--- | :--- |
| **`AUTO_HYDRA_MESH`** *(Default)* | Auto-Detects Online Venues | If both MT5 & cTrader are connected $\to$ unlocks **Active-Active SOR**. If only MT5 is open $\to$ runs **MT5 Solo**. If only cTrader is open $\to$ runs **cTrader Solo**. |
| **`CTRADER_EYES_MT5_HANDS`** | Cross-Venue Microstructure | Uses free cTrader Demo Level 2 DOM and Order-Book Imbalance to filter setups, but executes 100% of trades on MT5! |
| **`MT5_EYES_CTRADER_HANDS`** | MT5 Signal with ECN Execution | Uses MT5 indicators to trigger signals, but routes executions to cTrader for lowest raw ECN spreads. |
| **`SPLIT_EXECUTION`** | Parallel Multi-Venue Fill | Splits approved trade risk 50/50 across both MT5 and cTrader simultaneously. |

To change topology at any time without restarting the bot:
1. Double-click `6_CONFIGURE_BROKER_MESH.bat`.
2. Select your desired option (`1` through `4`).
3. PocketBot Apex hot-reloads the configuration immediately!
