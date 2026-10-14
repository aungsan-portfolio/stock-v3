import os
import time
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv
load_dotenv('.env')

import sys
sys.path.insert(0, os.path.abspath("."))
import config
from alpaca_bridge import AlpacaBridge
from strategies.intraday_data import fetch_intraday_batch, _get_alpaca_client
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

bridge = AlpacaBridge(mode="daytrade")
bridge.connect(start_stream=False)

symbols = list(getattr(config, "DAYTRADE_WATCHLIST", []))
if not symbols:
    symbols = ["SPY", "QQQ", "SOXX", "AAPL", "MSFT", "NVDA", "TSLA", "AMD", "META", "AMZN", "GOOGL"]

print(f"Auditing Intraday Freshness for {len(symbols)} symbols...")

now_utc = datetime.now(timezone.utc)
# Current 5m boundary
# e.g., if now is 15:35:10, target_bar_end is 15:35:00, target_bar_start is 15:30:00
target_bar_minute = (now_utc.minute // 5) * 5
target_bar_end = now_utc.replace(minute=target_bar_minute, second=0, microsecond=0)

t0 = time.perf_counter()
batch_results, telemetry = fetch_intraday_batch(symbols, interval="5m", bridge=bridge, purpose="ENTRY_SIGNAL")
batch_duration_ms = (time.perf_counter() - t0) * 1000.0

received_symbols = list(batch_results.keys())
missing_symbols = [s for s in symbols if s not in batch_results or batch_results[s].empty]

audit_rows = []
for s in symbols:
    df = batch_results.get(s)
    if df is not None and not df.empty:
        last_row = df.iloc[-1]
        last_ts = pd.Timestamp(last_row["date"])
        if last_ts.tz is None:
            last_ts = last_ts.tz_localize("UTC")
        else:
            last_ts = last_ts.tz_convert("UTC")
        
        delay = (now_utc - last_ts).total_seconds()
        # Bar end for a 5m bar that starts at last_ts
        bar_end = last_ts + pd.Timedelta(minutes=5)
        post_close_delay = (now_utc - bar_end).total_seconds()
        audit_rows.append({
            "symbol": s,
            "status": "RECEIVED",
            "bar_start": last_ts.isoformat(),
            "bar_end": bar_end.isoformat(),
            "post_close_delay_s": round(post_close_delay, 1),
            "close_px": float(last_row["close"]),
            "volume": int(last_row["volume"]),
        })
    else:
        audit_rows.append({
            "symbol": s,
            "status": "MISSING_OR_REJECTED",
            "bar_start": None,
            "bar_end": None,
            "post_close_delay_s": None,
            "close_px": None,
            "volume": None,
        })

coverage = len(received_symbols) / len(symbols) if symbols else 0.0

report = {
    "audit_timestamp_utc": now_utc.isoformat(),
    "target_bar_end_utc": target_bar_end.isoformat(),
    "batch_fetch_duration_ms": round(batch_duration_ms, 1),
    "total_symbols": len(symbols),
    "received_count": len(received_symbols),
    "missing_count": len(missing_symbols),
    "coverage_pct": round(coverage * 100.0, 1),
    "min_required_coverage_pct": round(getattr(config, "DAYTRADE_BATCH_FETCH_MIN_COVERAGE", 0.95) * 100.0, 1),
    "missing_symbols": missing_symbols,
    "symbol_details": audit_rows,
}

import json
with open("reports/intraday_freshness_audit.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("\n=== INTRADAY FRESHNESS AUDIT REPORT ===")
print(f"Timestamp: {now_utc.isoformat()}")
print(f"Target Bar End: {target_bar_end.isoformat()}")
print(f"Batch Duration: {batch_duration_ms:.1f}ms")
print(f"Coverage: {coverage*100.0:.1f}% ({len(received_symbols)}/{len(symbols)}) [Required: >= {config.DAYTRADE_BATCH_FETCH_MIN_COVERAGE*100.0:.1f}%]")
print(f"Missing / Rejected Symbols ({len(missing_symbols)}): {missing_symbols}")
for r in audit_rows[:10]:
    print(f"  {r['symbol']:<6}: status={r['status']:<8} bar_end={r['bar_end']} delay={r['post_close_delay_s']}s vol={r['volume']}")
