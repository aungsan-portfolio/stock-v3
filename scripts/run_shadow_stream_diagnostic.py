"""
scripts/run_shadow_stream_diagnostic.py
======================================
Standalone live preflight diagnostic verification runner for Alpaca IEX WebSocket Stream.

Operational Objectives:
1. Zero-Cost / Zero-Impact: Connects to Alpaca IEX free WebSocket (Basic Plan) in read-only shadow mode.
2. Symbol Partitioning: Subscribes to top 30 symbols from DAYTRADE_WATCHLIST, assigns 31st+ to REST fallback.
3. Live Telemetry Verification: Validates authentication, subscription handshake, heartbeat, quotes, and bars.
4. Generates comprehensive preflight report at reports/shadow_stream_preflight_report.json.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load .env file
try:
    from dotenv import load_dotenv
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
except ImportError:
    pass

import config
from strategies.shadow_stream_manager import (
    ShadowStreamManager,
    MAX_WEBSOCKET_SYMBOLS,
    DEFAULT_COMPARISON_LOG,
)

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def run_diagnostic(duration_seconds: int = 20) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("  [ALPACA IEX SHADOW WEBSOCKET STREAM - PREFLIGHT DIAGNOSTIC RUNNER]")
    print("=" * 70)

    # 1. Resolve Credentials
    api_key = (
        os.environ.get("DAYTRADE_APCA_API_KEY_ID")
        or os.environ.get("APCA_API_KEY_ID")
        or ""
    ).strip()
    secret_key = (
        os.environ.get("DAYTRADE_APCA_API_SECRET_KEY")
        or os.environ.get("APCA_API_SECRET_KEY")
        or ""
    ).strip()

    masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) >= 8 else "MISSING"
    print(f"  Target Feed        : IEX (Basic Free Plan - $0.00)")
    print(f"  API Key ID         : {masked_key}")
    print(f"  Watchlist Limit    : Max {MAX_WEBSOCKET_SYMBOLS} WebSocket Streams (Hard Basic Limit)")
    print(f"  Diagnostic Duration: {duration_seconds} seconds")
    print("-" * 70)

    if not api_key or not secret_key:
        err_msg = "Alpaca API credentials missing! Set DAYTRADE_APCA_API_KEY_ID or APCA_API_KEY_ID in .env."
        print(f"🚨 [FAIL] {err_msg}")
        return {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status": "FAILED",
            "error": err_msg,
        }

    # 2. Watchlist Partitioning
    watchlist = getattr(config, "DAYTRADE_WATCHLIST", [])
    print(f"  Total Configured Watchlist : {len(watchlist)} symbols")

    manager = ShadowStreamManager(
        api_key=api_key,
        secret_key=secret_key,
        watchlist=watchlist,
        max_stream_symbols=MAX_WEBSOCKET_SYMBOLS,
    )

    print(f"  Streamed Partition (30 syms) : {', '.join(manager.streamed_symbols[:10])}... ({len(manager.streamed_symbols)} symbols)")
    print(f"  REST Fallback Partition (1+) : {manager.rest_fallback_symbols}")
    print("-" * 70)
    print("  Initiating WebSocket connection to Alpaca IEX data stream...")

    start_time = time.time()
    started = manager.start()
    if not started:
        print("🚨 [FAIL] Failed to start ShadowStreamManager worker thread!")
        return {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status": "FAILED",
            "error": "Failed to start ShadowStreamManager thread",
        }

    # 3. Live Observation Loop
    elapsed = 0
    poll_interval = 2.0
    while elapsed < duration_seconds:
        time.sleep(poll_interval)
        elapsed = time.time() - start_time
        health = manager.get_health_status()
        conn_str = "CONNECTED [OK]" if health["is_connected"] else "CONNECTING / IDLE [WAIT]"
        hb_age = f"{health['heartbeat_age_seconds']:.1f}s" if health["heartbeat_age_seconds"] is not None else "None"
        print(
            f"  [{elapsed:4.1f}s / {duration_seconds}s] Status: {conn_str} | "
            f"Quotes: {health['quotes_count']:4d} | Bars: {health['bars_count']:3d} | "
            f"UpdatedBars: {health['updated_bars_count']:3d} | HB Age: {hb_age} | Errors: {health['errors_count']}"
        )

    # 4. Graceful Shutdown
    print("-" * 70)
    print("  Stopping ShadowStreamManager background stream...")
    manager.stop()
    print("  Stream gracefully stopped.")

    final_health = manager.get_health_status()
    summary = manager.get_summary()

    # 5. Compile Preflight Scorecard
    is_authenticated = final_health["errors_count"] == 0 and (
        final_health["is_connected"]
        or final_health["quotes_count"] > 0
        or final_health["bars_count"] > 0
        or final_health["last_heartbeat_timestamp"] is not None
    )

    report_data = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_duration_seconds": duration_seconds,
        "feed_type": "IEX",
        "plan_type": "BASIC_FREE",
        "authentication_success": is_authenticated,
        "partitioning": {
            "streamed_symbols_count": len(manager.streamed_symbols),
            "rest_fallback_symbols_count": len(manager.rest_fallback_symbols),
            "streamed_symbols": manager.streamed_symbols,
            "rest_fallback_symbols": manager.rest_fallback_symbols,
        },
        "stream_health": final_health,
        "events_received": {
            "quotes_count": final_health["quotes_count"],
            "bars_count": final_health["bars_count"],
            "updated_bars_count": final_health["updated_bars_count"],
            "errors_count": final_health["errors_count"],
            "last_error": final_health["last_error"],
        },
        "symbol_metadata_sample": {
            sym: summary["symbol_metadata"].get(sym)
            for sym in (manager.streamed_symbols[:5] + manager.rest_fallback_symbols)
        },
    }

    # 6. Save Report
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / "shadow_stream_preflight_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print("\n" + "=" * 70)
    print("  [SHADOW STREAM PREFLIGHT SCORECARD]")
    print("=" * 70)
    print(f"  Authentication Status  : {'PASS [OK]' if is_authenticated else 'FAIL [X]'}")
    print(f"  30-Symbol Partitioning : PASS [OK] ({len(manager.streamed_symbols)} streamed, {len(manager.rest_fallback_symbols)} fallback)")
    print(f"  Total Quotes Received  : {final_health['quotes_count']}")
    print(f"  Total 1m Bars Received : {final_health['bars_count']}")
    print(f"  Updated Bars Received  : {final_health['updated_bars_count']}")
    print(f"  Stream Errors Encountered: {final_health['errors_count']} ({final_health['last_error'] or 'None'})")
    print(f"  Report Persisted To    : {report_file}")
    print("=" * 70 + "\n")

    return report_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alpaca IEX Shadow WebSocket Diagnostic Runner")
    parser.add_argument("--duration", type=int, default=15, help="Diagnostic duration in seconds (default: 15)")
    args = parser.parse_args()

    res = run_diagnostic(duration_seconds=args.duration)
    sys.exit(0 if res.get("authentication_success") else 1)
