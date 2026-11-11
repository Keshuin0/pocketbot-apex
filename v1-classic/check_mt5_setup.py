#!/usr/bin/env python3
"""First-Time Setup MetaTrader 5 Connection Verifier for PocketBot USB."""

from pathlib import Path
import struct
import sys

BASE_DIR = Path(__file__).resolve().parent


def main():
    bits = struct.calcsize("P") * 8
    print("=" * 78)
    print("  STEP 4 OF 4: VERIFYING METATRADER 5 CONNECTION")
    print("=" * 78)
    print(f"  Python Runtime : {sys.version.split()[0]} ({bits}-bit)")
    print(f"  PocketBot Folder: {BASE_DIR}")
    print("-" * 78)

    try:
        import MetaTrader5 as mt5
        import numpy as np
        print(f"  [OK] Packages Installed: MetaTrader5 ({mt5.__version__}), numpy ({np.__version__})")
    except Exception as exc:
        print(f"  [!] Package import failed: {exc}")
        return 1

    if not mt5.initialize():
        err = mt5.last_error()
        print("\n  [!] MetaTrader 5 is NOT currently open (or not installed yet).")
        print(f"      Details: {err}")
        print("\n  HOW TO GET READY IN 3 EASY STEPS (NO CODING NEEDED):")
        print("  1. Install & Open MetaTrader 5 (64-bit) on this Windows PC.")
        print("  2. Log into your Demo Account inside MetaTrader 5:")
        print("     - Click 'File' -> 'Open an Account' (or 'Login to Trade Account').")
        print("     - Pick a Demo account (virtual money) so you never risk real cash.")
        print("  3. Turn ON the 'Algo Trading' button on the top toolbar of MetaTrader 5")
        print("     so the little play icon turns GREEN.")
        print("\n  Once MetaTrader 5 is open, double-click:")
        print("     - 2_START_AUTO_DEMO.bat   (for automatic demo trading)")
        print("     - 3_START_CONFIRM_MODE.bat (to approve every trade with YES/NO)")
        print("     - 4_CHECK_STATS.bat        (to see your balance, P/L, and trade stats)")
        print("=" * 78)
        return 0

    try:
        acct = mt5.account_info()
        term = mt5.terminal_info()
        if acct is None:
            print("\n  [!] MetaTrader 5 is open, but no trading account is logged in yet.")
            print("      In MetaTrader 5, click 'File' -> 'Login to Trade Account' (or 'Open an Account').")
            print("=" * 78)
            return 0

        is_demo = acct.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO
        algo_btn = bool(term and term.trade_allowed)
        print("  [OK] Connected to Open MetaTrader 5 Terminal!")
        print(f"  Account Login  : {acct.login} ({acct.server})")
        print(f"  Account Type   : {'DEMO ACCOUNT (Safe Virtual Money)' if is_demo else 'LIVE ACCOUNT (Blocked by Safety Lock)'}")
        print(f"  Balance / Eq   : ${acct.balance:,.2f} / ${acct.equity:,.2f} {acct.currency}")
        print(f"  Algo Trading   : {'GREEN / ON' if algo_btn else 'OFF (Click the Algo Trading button in MT5 top toolbar to turn it GREEN)'}")
        if not getattr(acct, "trade_allowed", True) or not getattr(acct, "trade_expert", True):
            print("\n  [NOTE] Your broker server currently has API order execution disabled on this login")
            print("         (trade_allowed=False / trade_expert=False). PocketBot will automatically")
            print("         record trades in its built-in Paper-Demo ledger, OR you can click")
            print("         'File -> Open an Account -> MetaQuotes Ltd -> Open a demo account'")
            print("         inside MetaTrader 5 to enable instant Trade-tab orders.")
        print("\n  SETUP COMPLETE! You can now double-click:")
        print("     - 2_START_AUTO_DEMO.bat    (Automatic Demo Trading)")
        print("     - 3_START_CONFIRM_MODE.bat  (Ask YES/NO Before Every Trade)")
        print("     - 4_CHECK_STATS.bat         (View Balance, P/L & Trade History)")
        print("=" * 78)
        return 0
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    sys.exit(main())
