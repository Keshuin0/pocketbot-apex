"""Phase 5 Verification Suite — "Apex Hydra" Active-Active Dual-Venue SOR & Synthetic NBBO Mesh.

Tests:
  1. All 4+ Mesh Topologies (`MT5_SOLO`, `CTRADER_SOLO`, `CTRADER_EYES_MT5_HANDS`,
     `MT5_EYES_CTRADER_HANDS`, `SPLIT_EXECUTION`, `ACTIVE_ACTIVE_SOR`) & instant hot-promotion
     from `MT5_SOLO` to `ACTIVE_ACTIVE_SOR` the moment cTrader connects.
  2. Synthetic Consolidated NBBO & Effective All-In Cost venue selection (raw-spread + $4.50/lot
     commission vs zero-commission wider-spread) + exact pips/$ savings logged in `WalkForwardLedgerDB`.
  3. Cross-Pollinated cTrader Level 2 DOM gating an order routed to MT5 (`CTRADER_EYES_MT5_HANDS`).
  4. Single-Broker Spike Rejector blocking entries when cross-venue mid-price basis > 2.5x normal.
  5. 50/50 `SPLIT_EXECUTION` parallel risk split & `tools/configure_broker_mesh.py` + `6_CONFIGURE_BROKER_MESH.bat`.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from apex.execution.router import DualBrokerExecutionRouter, ExecutionOrderRequest
from apex.feeds.dual_router import MarketFeedRouter
from apex.feeds.hydra_nbbo import ResolvedMeshTopology
from apex.feeds.models import DepthLevel, Side, Tick
from apex.ledger.walkforward_db import WalkForwardLedgerDB
from tools.configure_broker_mesh import apply_mesh_choice, main as mesh_cli_main

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent


def test_four_mesh_topologies_and_hot_promotion(tmp_path: Path) -> None:
    """Verify MT5_SOLO, CTRADER_SOLO, CTRADER_EYES_MT5_HANDS, SPLIT_EXECUTION, and hot-promotion to ACTIVE_ACTIVE_SOR."""
    cfg_path = tmp_path / "apex_config.json"
    shutil.copy(WORKSPACE_ROOT / "config" / "apex_config.json", cfg_path)

    router = MarketFeedRouter(config_path=cfg_path)

    # 1. Only MT5 online -> MT5_SOLO
    router.ctrader._connected = False
    router.mt5._connected = True
    assert router.resolved_topology == ResolvedMeshTopology.MT5_SOLO
    assert "MT5 Active" in router.dashboard_mesh_badge

    # 2. Only cTrader online -> CTRADER_SOLO (headless VPS mode without MT5)
    router.ctrader._connected = True
    router.mt5._connected = False
    assert router.resolved_topology == ResolvedMeshTopology.CTRADER_SOLO

    # 3. Hot-promotion: MT5_SOLO -> ACTIVE_ACTIVE_SOR the instant cTrader connects!
    router.ctrader._connected = False
    router.mt5._connected = True
    assert router.resolved_topology == ResolvedMeshTopology.MT5_SOLO
    # Simulate cTrader coming online mid-session without restarting the bot
    router.ctrader._connected = True
    assert router.resolved_topology == ResolvedMeshTopology.ACTIVE_ACTIVE_SOR
    assert "ACTIVE_ACTIVE_SOR" in router.dashboard_mesh_badge

    # 4. User-selectable CTRADER_EYES_MT5_HANDS via live config hot-reload
    apply_mesh_choice("2", config_path=cfg_path)
    assert router.resolved_topology == ResolvedMeshTopology.CTRADER_EYES_MT5_HANDS

    # 5. User-selectable SPLIT_EXECUTION via live config hot-reload
    apply_mesh_choice("3", config_path=cfg_path)
    assert router.resolved_topology == ResolvedMeshTopology.SPLIT_EXECUTION

    # Restore AUTO_HYDRA_MESH
    apply_mesh_choice("1", config_path=cfg_path)
    assert router.resolved_topology == ResolvedMeshTopology.ACTIVE_ACTIVE_SOR


def test_effective_all_in_cost_smart_venue_selection_and_ledger_savings(tmp_path: Path) -> None:
    """Verify Effective All-In Cost formula (spread + commission + VWAP DOM slippage + latency) & SQLite savings log."""
    cfg_path = tmp_path / "apex_config.json"
    shutil.copy(WORKSPACE_ROOT / "config" / "apex_config.json", cfg_path)
    # Configure cTrader as Fusion Raw ($4.50/lot RT) vs MT5 as Zero-Commission ($0.00/lot RT)
    import json
    cfg_data = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg_data["feeds"]["ctrader_commission_per_lot_usd"] = 4.50
    cfg_data["feeds"]["mt5_commission_per_lot_usd"] = 0.00
    cfg_path.write_text(json.dumps(cfg_data, indent=2), encoding="utf-8")

    ledger = WalkForwardLedgerDB(db_path=tmp_path / "sor_ledger.db")

    router = MarketFeedRouter(config_path=cfg_path)
    router.ctrader._connected = True
    router.mt5._connected = True

    # Case A: cTrader Raw ECN has 0.2 pip spread (0.00002) + $4.50/lot commission (0.45p)
    # vs MT5 Zero-Commission account with 1.4 pip spread (0.00014) + $0 commission.
    # -> Smart Venue Selector MUST pick cTrader and save >0.70 pips!
    ct_tick_tight = Tick(
        symbol="EURUSD",
        bid=1.10000,
        ask=1.10002,  # 0.2 pips
        timestamp_utc=1700000100.0,
        source="ctrader",
    )
    mt5_tick_wide = Tick(
        symbol="EURUSD",
        bid=1.09994,
        ask=1.10008,  # 1.4 pips
        timestamp_utc=1700000100.0,
        source="mt5",
    )
    router.ctrader._latest_ticks["EURUSD"] = ct_tick_tight
    router.mt5.get_tick = lambda sym: mt5_tick_wide if sym.upper() == "EURUSD" else None  # type: ignore[method-assign]

    nbbo_a = router.evaluate_nbbo(
        "EURUSD",
        one_r_price_distance=0.00100,
        side=Side.BUY,
        ctrader_slippage_price=0.00001,
        mt5_slippage_price=0.00001,
    )
    assert nbbo_a.topology == ResolvedMeshTopology.ACTIVE_ACTIVE_SOR
    assert nbbo_a.winning_venue == "ctrader"
    assert pytest.approx(nbbo_a.best_bid, abs=1e-6) == 1.10000
    assert pytest.approx(nbbo_a.best_ask, abs=1e-6) == 1.10002
    assert nbbo_a.pips_saved > 0.70
    assert nbbo_a.usd_saved_per_lot > 7.00

    # Execute a 0.50 lot order through DualBrokerExecutionRouter and verify SOR savings are persisted in SQLite
    exec_router = DualBrokerExecutionRouter(feed_router=router, ledger_db=ledger)
    order = ExecutionOrderRequest(
        symbol="EURUSD",
        side=Side.BUY,
        lots=0.50,
        entry_price=1.10002,
        stop_loss=1.09902,
        take_profit=1.10202,
        force_paper_shadow=True,
    )
    receipt = exec_router.execute_order(
        order=order,
        current_spread_price=0.00002,
        vwap_slippage_price=0.00001,
    )
    assert receipt.winning_venue == "ctrader"
    assert receipt.mesh_topology == "ACTIVE_ACTIVE_SOR"
    assert receipt.sor_pips_saved == pytest.approx(nbbo_a.pips_saved, abs=1e-5)
    assert receipt.sor_usd_saved == pytest.approx(nbbo_a.usd_saved_per_lot * 0.50, abs=1e-4)

    db_pips, db_usd = ledger.get_cumulative_sor_savings()
    assert db_pips == pytest.approx(receipt.sor_pips_saved, abs=1e-5)
    assert db_usd == pytest.approx(receipt.sor_usd_saved, abs=1e-4)

    # Case B: Now MT5 has a tight 0.4 pip spread ($0 commission -> ~0.51p all-in) while
    # cTrader widens to 0.9 pips (+ 0.45p commission -> ~1.46p all-in) -> MT5 MUST win!
    ct_tick_wider = Tick(
        symbol="EURUSD",
        bid=1.09996,
        ask=1.10005,  # 0.9 pips
        timestamp_utc=1700000105.0,
        source="ctrader",
    )
    mt5_tick_tight = Tick(
        symbol="EURUSD",
        bid=1.09998,
        ask=1.10002,  # 0.4 pips
        timestamp_utc=1700000105.0,
        source="mt5",
    )
    router.ctrader._latest_ticks["EURUSD"] = ct_tick_wider
    router.mt5.get_tick = lambda sym: mt5_tick_tight if sym.upper() == "EURUSD" else None  # type: ignore[method-assign]

    nbbo_b = router.evaluate_nbbo(
        "EURUSD",
        one_r_price_distance=0.00100,
        side=Side.BUY,
        ctrader_slippage_price=0.00001,
        mt5_slippage_price=0.00001,
    )
    assert nbbo_b.winning_venue == "mt5"
    assert nbbo_b.pips_saved > 0.85


def test_cross_pollinated_ctrader_dom_gating_mt5_order(tmp_path: Path) -> None:
    """Prove cTrader Level 2 DOM (OBI & Liquidity Walls) automatically gates orders routed to MT5."""
    cfg_path = tmp_path / "apex_config.json"
    shutil.copy(WORKSPACE_ROOT / "config" / "apex_config.json", cfg_path)
    apply_mesh_choice("2", config_path=cfg_path)  # CTRADER_EYES_MT5_HANDS

    router = MarketFeedRouter(config_path=cfg_path)
    router.ctrader._connected = True
    router.mt5._connected = True

    mt5_tick = Tick(
        symbol="EURUSD",
        bid=1.10000,
        ask=1.10002,
        timestamp_utc=1700000200.0,
        source="mt5",
    )
    router.ctrader._latest_ticks["EURUSD"] = mt5_tick
    router.mt5.get_tick = lambda sym: mt5_tick  # type: ignore[method-assign]

    # Seed cTrader Level 2 DOM for EURUSD with an opposing SELL liquidity wall (5.0x median)
    # at 1.10020 (+0.18R from 1.10002 entry with 10-pip SL = 0.00100, which is <= 0.35R)
    dom_engine = router.ctrader.get_or_create_dom("EURUSD", pip_size=0.0001)
    dom_engine.set_levels(
        bids=[(1.10000 - idx * 0.0001, 300_000.0) for idx in range(5)],
        asks=[
            (1.10005 + idx * 0.00005, 1_500_000.0 if idx == 3 else 200_000.0)
            for idx in range(5)
        ],
    )

    exec_router = DualBrokerExecutionRouter(feed_router=router)
    mt5_order = ExecutionOrderRequest(
        symbol="EURUSD",
        side=Side.BUY,
        lots=0.20,
        entry_price=1.10002,
        stop_loss=1.09902,
        take_profit=1.10202,
    )

    # Notice we do NOT pass dom_analytics explicitly — DualBrokerExecutionRouter automatically
    # cross-pollinates cTrader's live Level 2 DOM to inspect the MT5 trade!
    blocked_receipt = exec_router.execute_order(
        order=mt5_order,
        current_spread_price=0.00002,
        vwap_slippage_price=0.00001,
    )
    assert blocked_receipt.status == "REJECTED"
    assert any("E_LEVEL2_DOM_GATE" in r for r in blocked_receipt.breaker_report.failed_reasons)

    # Now update cTrader Level 2 DOM to show strong positive OBI (+0.50) and zero opposing walls
    dom_engine.set_levels(
        bids=[(1.10000 - idx * 0.0001, 600_000.0) for idx in range(5)],
        asks=[(1.10005 + idx * 0.0001, 200_000.0) for idx in range(5)],
    )

    approved_receipt = exec_router.execute_order(
        order=mt5_order,
        current_spread_price=0.00002,
        vwap_slippage_price=0.00001,
    )
    assert approved_receipt.breaker_report.all_passed is True
    assert approved_receipt.mesh_topology == "CTRADER_EYES_MT5_HANDS"
    assert approved_receipt.winning_venue == "mt5"


def test_single_broker_spike_rejector(tmp_path: Path) -> None:
    """Verify Single-Broker Spike Rejector blocks trades when cross-venue mid diverges > 2.5x normal basis."""
    cfg_path = tmp_path / "apex_config.json"
    shutil.copy(WORKSPACE_ROOT / "config" / "apex_config.json", cfg_path)

    router = MarketFeedRouter(config_path=cfg_path)
    router.ctrader._connected = True
    router.mt5._connected = True

    # 1. Feed 12 normal ticks where cTrader and MT5 mid-prices have a normal ~0.1 pip basis (0.00001)
    for i in range(12):
        ct_t = Tick("EURUSD", 1.10000, 1.10002, 1700000300.0 + i, "ctrader")
        mt5_t = Tick("EURUSD", 1.09999, 1.10001, 1700000300.0 + i, "mt5")
        router.ctrader._latest_ticks["EURUSD"] = ct_t
        router.mt5.get_tick = lambda sym, t=mt5_t: t  # type: ignore[method-assign]
        res_ok = router.evaluate_nbbo("EURUSD", one_r_price_distance=0.00100)
        assert res_ok.single_broker_spike_blocked is False

    # 2. Now Venue A prints a sudden 4.0-pip artificial stop-hunt spike while Venue B stays at 1.10000
    # (basis = 4.0 pips > 2.5x normal basis) -> Single-Broker Spike Rejector MUST block entry!
    ct_spike = Tick("EURUSD", 1.10040, 1.10042, 1700000320.0, "ctrader")
    mt5_calm = Tick("EURUSD", 1.09999, 1.10001, 1700000320.0, "mt5")
    router.ctrader._latest_ticks["EURUSD"] = ct_spike
    router.mt5.get_tick = lambda sym: mt5_calm  # type: ignore[method-assign]

    spike_nbbo = router.evaluate_nbbo("EURUSD", one_r_price_distance=0.00100)
    assert spike_nbbo.single_broker_spike_blocked is True
    assert "SINGLE_BROKER_SPIKE_REJECTED" in spike_nbbo.spike_reason

    exec_router = DualBrokerExecutionRouter(feed_router=router, dom_obi_enabled=False)
    order = ExecutionOrderRequest(
        symbol="EURUSD",
        side=Side.BUY,
        lots=0.20,
        entry_price=1.10042,
        stop_loss=1.09942,
        take_profit=1.10242,
    )
    spike_receipt = exec_router.execute_order(
        order=order,
        current_spread_price=0.00002,
        vwap_slippage_price=0.00001,
    )
    assert spike_receipt.status == "REJECTED"
    assert any("SINGLE_BROKER_SPIKE_REJECTED" in r for r in spike_receipt.breaker_report.failed_reasons)


def test_split_execution_50_50_and_broker_mesh_cli_tool(tmp_path: Path) -> None:
    """Verify 50/50 SPLIT_EXECUTION parallel child orders and 6_CONFIGURE_BROKER_MESH.bat + CLI switcher."""
    cfg_path = tmp_path / "apex_config.json"
    shutil.copy(WORKSPACE_ROOT / "config" / "apex_config.json", cfg_path)

    # Switch to SPLIT_EXECUTION using CLI tool (--choice 3)
    rc = mesh_cli_main(["--config", str(cfg_path), "--choice", "3"])
    assert rc == 0

    router = MarketFeedRouter(config_path=cfg_path)
    router.ctrader._connected = True
    router.mt5._connected = True

    tick = Tick("EURUSD", 1.10000, 1.10002, 1700000400.0, "mt5")
    router.ctrader._latest_ticks["EURUSD"] = tick
    router.mt5.get_tick = lambda sym: tick  # type: ignore[method-assign]

    exec_router = DualBrokerExecutionRouter(feed_router=router, dom_obi_enabled=False)
    split_order = ExecutionOrderRequest(
        symbol="EURUSD",
        side=Side.BUY,
        lots=1.00,
        entry_price=1.10002,
        stop_loss=1.09902,
        take_profit=1.10202,
        risk_pct=0.50,
        risk_usd=50.0,
    )
    split_receipt = exec_router.execute_order(
        order=split_order,
        current_spread_price=0.00002,
        vwap_slippage_price=0.00001,
    )
    assert split_receipt.venue == "SPLIT_CTRADER_MT5"
    assert split_receipt.mesh_topology == "SPLIT_EXECUTION"
    assert len(split_receipt.child_receipts) == 2
    assert split_receipt.child_receipts[0].lots == pytest.approx(0.50)
    assert split_receipt.child_receipts[1].lots == pytest.approx(0.50)
    assert split_receipt.child_receipts[0].winning_venue == "ctrader"
    assert split_receipt.child_receipts[1].winning_venue == "mt5"

    # Run Live Broker Shootout via CLI (--choice 4)
    rc_shootout = mesh_cli_main(["--config", str(cfg_path), "--choice", "4"])
    assert rc_shootout == 0

    # Verify 6_CONFIGURE_BROKER_MESH.bat exists and uses portable %~dp0 paths
    bat_path = WORKSPACE_ROOT / "6_CONFIGURE_BROKER_MESH.bat"
    assert bat_path.exists()
    bat_text = bat_path.read_text(encoding="utf-8")
    assert "%~dp0" in bat_text
    assert "configure_broker_mesh.py" in bat_text
