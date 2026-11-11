# PocketBot Apex — 1-Page Quick Desk Cheat Sheet

Keep this quick reference desk sheet open or printed next to your trading workstation.

---

## 1. Quick Launch Scripts (Double-Click)

| Script | Purpose | When to Use |
| :--- | :--- | :--- |
| `0_FIRST_TIME_SETUP.bat` | 1-Click Environment Setup | Run once upon initial install or after moving folder |
| `1_SETUP_CTRADER_KEYS.bat`| cTrader OAuth 2.0 Wizard | Connect cTrader API keys and discover accounts |
| `2_START_APEX_AUTO_DEMO.bat` | Fully Autonomous Trading | Runs scans and routes trades automatically |
| `3_START_APEX_CONFIRM.bat` | Interactive Confirmation Mode | Asks for human `[Y/N]` approval on candidate trades |
| `4_APEX_LIVE_DASHBOARD.bat` | Rich Terminal UI (TUI) | Full-screen live telemetry cockpit & feed status |
| `5_CONFIGURE_AI_MODE.bat` | AI Router Mode Switcher | Switch between Auto, Studio API, AI Pro, or Pure Math |
| `6_CONFIGURE_BROKER_MESH.bat` | Broker Topology Switcher | Switch between Hydra SOR, MT5 Solo, or cTrader Solo |
| `7_UPDATE_FROM_GITHUB.bat`| Live GitHub Auto-Updater | Pulls latest updates without touching your `.env` or data |

---

## 2. Interactive Terminal Commands

- **Stop Bot Safely:** Press `Ctrl + C` in the console window. The bot finishes logging and closes feed sockets cleanly.
- **Confirm Trade (`3_START_APEX_CONFIRM.bat`):**
  - Type `Y` and press Enter to approve candidate order.
  - Type `N` and press Enter to reject and continue scanning.

---

## 3. Telegram Remote Cockpit Commands

If you configured `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env`, you can manage PocketBot Apex from your smartphone anywhere in the world:

| Command | Action |
| :--- | :--- |
| `/status` | Returns active broker feeds, current mesh topology, and AI brain mode |
| `/positions` | Lists all currently open positions with real-time floating P/L |
| `/daily` | Displays daily closed P/L, win rate, and drawdown vs. the -2.0% kill-switch |
| `/panic` | **EMERGENCY FLATTEN:** Closes all open positions and cancels all pending orders |
| `/mode <auto\|confirm\|halt>` | Dynamically switches operational execution mode |
| `/help` | Lists available commands and syntax |

---

## 4. Non-Negotiable Risk Rules

```
+--------------------------------------------------------------------------+
|  1. MAX RISK PER TRADE:     0.50% of account equity ($50 on $10k account) |
|  2. TARGET REWARD:          1.00% of account equity ($100 on $10k account)|
|  3. MAX CONCURRENT TRADES:  2 positions across all symbols                |
|  4. DAILY KILL-SWITCH:      -2.00% of equity (halts all trading for day)  |
|  5. COST CEILING:           Spread + Slippage must be <= 15% of SL        |
|  6. DEMO LOCK:              Hard-locked to Demo; Live requires CLI flag   |
+--------------------------------------------------------------------------+
```

---

## 5. Asset Tiers & Timeframe Guidelines

- **Tight-Spread Tier (M5 Timeframe):** `EURUSD`, `GBPUSD`, `USDJPY`, `XAUUSD`, `US500`, `NAS100`
- **Wide-Spread Momentum Tier (M15 Timeframe):** `BTCUSD`, `ETHUSD`, `TSLA`, `NVDA`
- Minimum Impulse Bar Body: $\ge 1.5 \times \text{ATR}(14)$
- Bar Close Location: Extreme $25\%$ of the candle range.
