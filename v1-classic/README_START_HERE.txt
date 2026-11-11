================================================================================
  POCKETBOT MT5 — ZERO-CODING QUICKSTART GUIDE (WINDOWS USB EDITION)
================================================================================

1. COPY THIS "PocketBot" FOLDER ANYWHERE ON YOUR PC
   - Copy the "PocketBot" folder from the USB drive onto your Desktop, C:\, or
     Documents folder.

2. OPEN METATRADER 5 (DEMO ACCOUNT)
   - Open MetaTrader 5 on your PC and make sure you are logged into a DEMO
     account (virtual money).
   - Click the "Algo Trading" button on the top toolbar of MetaTrader 5 so its
     icon turns GREEN.

3. DOUBLE-CLICK THE NUMBERED FILES IN ORDER:
   - [1_FIRST_TIME_SETUP.bat]
     Run this ONCE on the new PC. It checks for 64-bit Python 3.12 (installing
     it automatically if missing), creates the local .venv environment inside
     this folder, installs MetaTrader5 + numpy, and tests your MT5 connection.

   - [2_START_AUTO_DEMO.bat]
     Scans EURUSD, GBPUSD, USDJPY, XAUUSD, SPX, NQ (5-minute candles) and
     BTCUSD, ETHUSD, #TSLA, #NVDA (15-minute Strong Momentum candles) every
     5 minutes and automatically places demo bracket orders when Time Machine
     10,000-replay expected value is positive and all safety checks pass.

   - [3_START_CONFIRM_MODE.bat]
     Same live scan and 10,000-replay simulation, but asks you to type YES or
     NO before sending any order.

   - [4_CHECK_STATS.bat]
     Shows your current MT5 demo balance, today's P/L, open positions, win
     rate, profit factor, and the last 10 actions from trader/bot_log.txt.

================================================================================
  BUILT-IN SAFETY CIRCUIT BREAKERS (ALWAYS ON)
================================================================================
  - Demo-Only Lock      : Blocks real-money accounts automatically.
  - Max Risk Per Trade  : 0.5% of equity ($50 max risk on a $10,000 account).
  - Daily Loss Switch   : Stops trading for the day if daily P/L hits -2.0%.
  - Max Open Positions  : Maximum 2 open positions at the same time.
  - Spread & SL/TP Gate : Requires hard Stop-Loss & Take-Profit and blocks
                          trades if spread exceeds 15% of Stop-Loss distance.
================================================================================
