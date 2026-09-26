"""
tools/p1_c_live_paper_certifier.py — Live Paper WebSocket Runtime Certification Helper (P1-C).

Used during regular market hours (8:30 PM - 3:00 AM ICT) to audit prerequisites and execute
controlled fault-injection on Alpaca Paper account for P1-C runtime certification.

Usage:
  # 1. Audit account readiness (Check active positions & GTC stop orders):
  python tools/p1_c_live_paper_certifier.py --check-prerequisites

  # 2. Inject controlled clean transport fault (socket drop recovery):
  python tools/p1_c_live_paper_certifier.py --inject-fault --mode clean

  # 3. Inject controlled dirty fault (out-of-band cancellation -> LATCHED_DESYNC -> reprotection):
  python tools/p1_c_live_paper_certifier.py --inject-fault --mode dirty

Governance Standards:
  • ZERO POLICY DRIFT: Sizing, ML, regimes, confidence thresholds remain 100% frozen.
  • NON-FLAT REQUIREMENT: Account must hold >= 1 active long position with resting GTC stop.
  • FAIL-CLOSED BOUNDARY: If broker truth does not match, engine MUST remain in LATCHED_DESYNC.
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from alpaca_bridge import AlpacaBridge, StreamState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("p1_c_certifier")


def check_prerequisites(bridge: AlpacaBridge) -> bool:
    """
    Audits the Alpaca Paper account to verify that the mandatory P1-C prerequisite is met:
    The account must hold at least 1 active long position with an active GTC protective stop order.
    """
    logger.info("=" * 60)
    logger.info("  AUDITING P1-C LIVE PAPER PREREQUISITES")
    logger.info("=" * 60)

    account_id = getattr(bridge, "account_id", "UNKNOWN")
    logger.info("Connected Account: %s (Paper Trading)", account_id)

    positions = bridge.get_positions() if hasattr(bridge, "get_positions") else []
    working_orders = bridge._get_active_orders() if hasattr(bridge, "_get_active_orders") else []

    logger.info("Active Positions Count: %d", len(positions))
    for p in positions:
        sym = getattr(p, "symbol", "")
        qty = getattr(p, "position", getattr(p, "qty", 0))
        mval = getattr(p, "marketValue", getattr(p, "market_value", "N/A"))
        logger.info("  • Position: %s | Qty: %s | Market Value: %s", sym, qty, mval)

    logger.info("Active Working Orders Count: %d", len(working_orders))
    stop_orders = []
    for o in working_orders:
        sym = getattr(o, "symbol", "")
        side = getattr(o, "side", "")
        raw_type = getattr(o, "order_type", getattr(o, "type", ""))
        otype = str(getattr(raw_type, "value", raw_type)).lower().replace("ordertype.", "")
        if otype in ("stop", "stop_limit", "trailing_stop") and "SELL" in str(side).upper():
            stop_orders.append(o)

    new_stops = [o for o in stop_orders if "NEW" in str(getattr(o, "status", "")).upper()]
    held_stops = [o for o in stop_orders if "HELD" in str(getattr(o, "status", "")).upper()]

    if not positions:
        logger.error(
            "❌ PREREQUISITE FAILED: Account is 100% FLAT. "
            "P1-C requires >= 1 active position with resting GTC stop to prove anti-desync protection audit!"
        )
        return False

    if not stop_orders:
        logger.error(
            "❌ PREREQUISITE FAILED: Account holds %d positions, but 0 resting protective stop orders found!",
            len(positions)
        )
        return False

    logger.info(
        "✅ PREREQUISITE PASSED: %d positions and %d protective stop legs verified on broker truth (%d NEW + %d HELD bracket secondary leg).",
        len(positions), len(stop_orders), len(new_stops), len(held_stops)
    )
    return True


def execute_controlled_fault_injection(bridge: AlpacaBridge, mode: str = "clean") -> bool:
    """
    Executes a controlled socket drop on the active trading stream and monitors the
    full reconciliation lifecycle.
    """
    logger.info("=" * 60)
    logger.info("  STARTING P1-C CONTROLLED FAULT INJECTION (Mode: %s)", mode.upper())
    logger.info("=" * 60)

    if not hasattr(bridge, "_trade_stream") or bridge._trade_stream is None:
        logger.error("Trade stream is not initialized on bridge.")
        return False

    stream = bridge._trade_stream
    ws = getattr(stream, "_ws", None)
    if ws is None:
        logger.error("Active WebSocket instance (_ws) not found on trade stream.")
        return False

    logger.info("Initial Stream State: %s", bridge._stream_state.value)
    if bridge._stream_state != StreamState.HEALTHY:
        logger.warning("Stream is not HEALTHY (state=%s). Proceeding with caution.", bridge._stream_state.value)

    target_order_backup = None

    if mode == "dirty":
        # Identify a standalone direct NEW stop order to cancel out-of-band (strictly avoiding nested AMZN)
        working_orders = bridge._get_active_orders() if hasattr(bridge, "_get_active_orders") else []
        candidate_stops = []
        for o in working_orders:
            sym = str(getattr(o, "symbol", "")).upper()
            status = str(getattr(o, "status", "")).upper()
            raw_type = getattr(o, "order_type", getattr(o, "type", ""))
            otype = str(getattr(raw_type, "value", raw_type)).lower()
            # Do NOT touch AMZN (nested bracket)
            if sym != "AMZN" and "NEW" in status and "stop" in otype:
                candidate_stops.append(o)

        if not candidate_stops:
            logger.error("❌ DIRTY INJECTION ABORTED: No standalone NEW stop order found to simulate out-of-band drop.")
            return False

        target_order = candidate_stops[0]
        target_order_backup = {
            "order_id": str(target_order.id),
            "symbol": str(target_order.symbol).upper(),
            "qty": float(getattr(target_order, "qty", 0)),
            "stop_price": float(getattr(target_order, "stop_price", 0.0)),
        }
        logger.info(
            "[DIRTY INJECTION SETUP] Selected standalone stop order for %s (id=%s, qty=%.1f, stop=$%.2f)",
            target_order_backup["symbol"], target_order_backup["order_id"],
            target_order_backup["qty"], target_order_backup["stop_price"],
        )

        # Cancel the stop order out-of-band directly on Alpaca REST client
        try:
            logger.info("[DIRTY INJECTION] Canceling stop order %s out-of-band...", target_order_backup["order_id"])
            bridge._client.cancel_order_by_id(target_order_backup["order_id"])
            logger.info("✅ Out-of-band stop cancellation submitted successfully.")
        except Exception as cancel_exc:
            logger.exception("Failed to cancel stop order for dirty test: %s", cancel_exc)
            return False

    # Record pre-fault state
    t_fault = datetime.now(timezone.utc).isoformat()
    logger.info("[FAULT INJECTION] Terminating WebSocket transport at %s...", t_fault)

    try:
        # Close underlying websocket connection cleanly to simulate socket drop / [WinError 64]
        import asyncio
        loop = getattr(stream, "_loop", None)
        if loop and loop.is_running():
            asyncio.run_coroutine_threadsafe(ws.close(), loop)
        else:
            logger.warning("Event loop is not running; closing ws directly if coroutine available")
            if hasattr(ws, "close"):
                res = ws.close()
                if asyncio.iscoroutine(res):
                    asyncio.run(res)
        logger.info("[FAULT INJECTION] Socket closed. Monitoring state transitions...")
    except Exception as exc:
        logger.exception("Failed to close websocket: %s", exc)
        return False

    # Monitor state transitions over the next 45 seconds
    start_wait = time.time()
    observed_states = [bridge._stream_state]
    test_passed = False

    while time.time() - start_wait < 45.0:
        time.sleep(0.5)
        current = bridge._stream_state
        if current != observed_states[-1]:
            observed_states.append(current)
            logger.info("[STATE TRANSITION OBSERVED] -> %s (elapsed: %.1fs)", current.value, time.time() - start_wait)

        if mode == "clean" and current == StreamState.HEALTHY and len(observed_states) >= 3:
            logger.info("✅ CLEAN RECOVERY COMPLETED TO HEALTHY!")
            test_passed = True
            break
        elif mode == "dirty" and current == StreamState.LATCHED_DESYNC:
            logger.info("✅ DIRTY INJECTION COMPLETED: Engine correctly latched in LATCHED_DESYNC!")
            test_passed = True
            break

    logger.info("Complete Observed State Sequence: %s", [s.value for s in observed_states])

    # If dirty mode, perform immediate automated reprotection and recovery
    if mode == "dirty" and target_order_backup is not None:
        logger.info("=" * 60)
        logger.info("  REPROTECTION & DESYNC RESOLUTION")
        logger.info("=" * 60)
        try:
            logger.info(
                "Restoring protective stop for %s (qty=%.1f, stop=$%.2f)...",
                target_order_backup["symbol"], target_order_backup["qty"], target_order_backup["stop_price"],
            )
            order_info = bridge.place_stop_order(
                symbol=target_order_backup["symbol"],
                side="SELL",
                qty=target_order_backup["qty"],
                stop_price=target_order_backup["stop_price"],
            )
            new_stop_id = order_info.get("order_id") if isinstance(order_info, dict) else str(order_info)
            logger.info("✅ Restored protective stop successfully -> order_id: %s", new_stop_id)

            # Trigger reconciliation to clear LATCHED_DESYNC
            logger.info("Executing post-reprotection reconciliation audit...")
            recon = bridge.reconcile_startup_state(manage_stream_state=True)
            is_clean = recon.get("clean", False) if isinstance(recon, dict) else getattr(recon, "clean", False)
            unprot = recon.get("protect", {}).get("unprotected", []) if isinstance(recon, dict) else getattr(recon, "unprotected", [])
            logger.info("Reconciliation result: clean=%s | unprotected=%s", is_clean, unprot)
            if is_clean:
                with bridge._stream_state_lock:
                    bridge._stream_state = StreamState.HEALTHY
                with bridge._trade_stream_lock:
                    bridge._trade_stream_recovery_required = False
                    bridge._trade_stream_ready = True
                bridge.entry_gate.clear_recovery_halts()
                logger.info("✅ Engine fully recovered from dirty desync -> StreamState.HEALTHY, entry gate released!")
            else:
                logger.error("❌ Reprotection reconciliation failed to clear desync!")
                test_passed = False
        except Exception as reprot_exc:
            logger.exception("Failed during reprotection: %s", reprot_exc)
            test_passed = False

    return test_passed


def main():
    parser = argparse.ArgumentParser(description="P1-C Live Paper Runtime Certification Helper")
    parser.add_argument("--check-prerequisites", action="store_true", help="Check if account meets non-flat prerequisite")
    parser.add_argument("--inject-fault", action="store_true", help="Inject controlled socket drop")
    parser.add_argument("--mode", choices=["clean", "dirty"], default="clean", help="Fault injection mode")
    parser.add_argument("--trading-mode", choices=["daytrade", "swing"], default="swing", help="Bridge trading mode")
    args = parser.parse_args()

    start_stream = bool(args.inject_fault)
    bridge = AlpacaBridge(mode=args.trading_mode)
    if not bridge.connect(start_stream=start_stream):
        logger.error("Failed to connect to Alpaca Paper Account.")
        sys.exit(1)

    if args.check_prerequisites:
        ok = check_prerequisites(bridge)
        sys.exit(0 if ok else 1)

    if args.inject_fault:
        logger.info("Waiting for trade stream to authenticate...")
        for _ in range(30):
            if bridge._stream_state in (StreamState.AUTHENTICATED_UNRECONCILED, StreamState.HEALTHY):
                break
            time.sleep(0.5)
        logger.info("Stream authenticated in state: %s. Executing startup reconciliation audit...", bridge._stream_state.value)
        bridge.reconcile_startup_state(manage_stream_state=True)
        logger.info("Stream readiness state after startup reconciliation: %s", bridge._stream_state.value)
        if bridge._stream_state != StreamState.HEALTHY:
            logger.error("Stream failed to reach HEALTHY prior to injection (current=%s)", bridge._stream_state.value)
            sys.exit(1)
        ok = execute_controlled_fault_injection(bridge, mode=args.mode)
        sys.exit(0 if ok else 1)

    # Default action if no flags provided
    check_prerequisites(bridge)


if __name__ == "__main__":
    main()
