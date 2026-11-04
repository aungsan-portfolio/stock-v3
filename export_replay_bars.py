"""
export_replay_bars.py - Export market data for offline replay of the day-trading bot
(ADR-001, Phase 1):
  * 1-minute bars for every (symbol, ET trading day) in the bot's decision files, plus
    warm-up days before each one (the bot's 5m indicators use INTRADAY_LOOKBACK_DAYS = 5)
  * optional bid/ask quotes in short windows around quote-sensitive decisions and fills
  * a manifest that states, per input and per output, what is complete and what is not

READ-ONLY: reads the bot's files and calls Alpaca's market-data API. It never places,
changes or cancels orders, and it never prints API keys.

Run from the project root (the folder that holds main.py and .env):

    python export_replay_bars.py --dry-run --quotes   # inputs, coverage plan, no download
    python export_replay_bars.py --quotes             # bars (IEX + SIP) and quotes

Output, one new run id per run (never overwritten):
    data/exports/replay_<run>/bars_1m_iex.csv.gz
    data/exports/replay_<run>/bars_1m_sip.csv.gz
    data/exports/replay_<run>/quotes_iex.csv.gz       (only with --quotes)
    data/exports/replay_<run>/manifest.json

Exit 0 means configured collection checks passed, not that all historical data,
inputs or executions are known. Exit 2 means incomplete; 1 means fatal error.
Dry-run checks the collection plan only and does not validate downloaded data.

What this data can and cannot do (also written into the manifest):
  * signals_*.csv is written AFTER texture admission and portfolio correlation, so
    candidates rejected there carry no entry/stop/target. Replaying them means re-running
    strategy code on bars: a reconstruction, not a record.
  * Validation records carry no fill timestamp; their quote window is centred on the
    decision quote time. Journal FILL records are centred on the fill time.
  * Historical quotes are not the exact snapshot the bot used (feed, timing, latency),
    and say nothing about queue position or the paper fill model: fills stay estimates.
  * The bot's live intraday bars used Adjustment.ALL; this export defaults to RAW (traded
    prices). Historical prices, feed and adjustment can differ from the original run. The
    manifest's "parity" block compares exported bars with the bot's recorded decision prices.
"""
from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
import math
import os
import statistics
import sys
import time
import uuid
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
EXPORT_DIR = ROOT / "data" / "exports"
BENCHMARKS = ("SPY", "QQQ", "SOXX")
SKIP_SYMBOLS = {"", "TEST", "UNIVERSE"}
ET, UTC = "America/New_York", "UTC"
QUOTE_STAGES = {"STAGE_QUOTE_HEALTH", "STAGE_MAX_CHASE", "STAGE_NOISE_FLOOR", "STAGE_ORDER_SUBMIT"}
QUOTE_BEFORE, QUOTE_AFTER = timedelta(seconds=60), timedelta(seconds=120)
# NYSE full-day holidays and 13:00 ET early closes, 2026 (NYSE holiday calendar)
HOLIDAYS = {date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3), date(2026, 5, 25),
            date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7), date(2026, 11, 26), date(2026, 12, 25)}
EARLY_CLOSE = {date(2026, 11, 27), date(2026, 12, 24)}
SIP_MIN_COVERAGE = 0.90          # a liquid name should print nearly every minute on the consolidated tape

INPUTS = {
    "gate_ledger": ROOT / "data" / "trade_gate_decision.jsonl",
    "validation_trades": ROOT / "data" / "paper_validation_trades.jsonl",
    "trade_journal": ROOT / "reports" / "trade_journal.jsonl",
    "signal_exports": EXPORT_DIR / "signals_*.csv",
}


# ---------------------------------------------------------------- helpers
def is_session(d: date) -> bool:
    if d.year != 2026:
        raise ValueError(f"NYSE calendar is only configured for 2026, got {d}")
    return d.weekday() < 5 and d not in HOLIDAYS


def session_bounds(d: date, extended: bool):
    open_t, close_t = ("04:00", "20:00") if extended else ("09:30", "13:00" if d in EARLY_CLOSE else "16:00")
    return pd.Timestamp(f"{d} {open_t}", tz=ET).tz_convert(UTC), pd.Timestamp(f"{d} {close_t}", tz=ET).tz_convert(UTC)


def expected_regular_minutes(d: date) -> int:
    return 210 if d in EARLY_CLOSE else 390


def to_utc(ts, naive_tz: str):
    """ISO string / epoch seconds -> UTC Timestamp. naive_tz = how a timestamp WITHOUT an
    offset is read: UTC for ledger/validation/journal (trade_journal.py reads naive as UTC),
    America/New_York for the signals_*.csv 'timestamp_et' column."""
    if ts is None or ts == "":
        return None
    try:
        if isinstance(ts, (int, float)):
            t = pd.Timestamp(ts, unit="s", tz=UTC)
            return None if pd.isna(t) else t
        t = pd.Timestamp(str(ts))
        if pd.isna(t):
            return None
        return (t.tz_localize(naive_tz) if t.tzinfo is None else t).tz_convert(UTC)
    except Exception:
        return None


def et_date(t):
    return None if t is None else t.tz_convert(ET).date()


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".partial")
    with open(tmp, "x", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


class InputStats(dict):
    def __init__(self, path_label: str):
        super().__init__(path=path_label, exists=False, files=0, lines=0, parsed=0, malformed=0,
                         bad_timestamp=0, out_of_range=0, used=0)


def iter_jsonl(path: Path, st: InputStats):
    paths = sorted(p for p in path.parent.glob(path.name + '*') if p.is_file() and (p == path or p.name.startswith(path.name + '.')) and not p.name.endswith('.partial'))
    st["exists"], st["files"] = bool(paths), len(paths)
    for file_path in paths:
      open_input = gzip.open if file_path.name.endswith('.gz') else open
      with open_input(file_path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            st["lines"] += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                st["malformed"] += 1
                continue
            if not isinstance(rec, dict):
                st["malformed"] += 1
                continue
            st["parsed"] += 1
            yield rec


# ---------------------------------------------------------------- inputs
def collect(since: date, warmup_days: int):
    pairs = defaultdict(set)          # ET day -> symbols needed on that day
    decision_days = set()
    events = []                       # (symbol, utc ts, kind, id) for quote windows
    parity_refs = []                  # validation records with decision price + 5m bar
    stats = {k: InputStats(str(p.relative_to(ROOT))) for k, p in INPUTS.items()}

    def add(sym, t, src) -> bool:
        sym = str(sym or "").upper().strip()
        if t is None:
            stats[src]["bad_timestamp"] += 1
            return False
        d = et_date(t)
        if sym in SKIP_SYMBOLS or d < since or not is_session(d):
            stats[src]["out_of_range"] += 1
            return False
        pairs[d].add(sym)
        decision_days.add(d)
        stats[src]["used"] += 1
        return True

    for r in iter_jsonl(INPUTS["gate_ledger"], stats["gate_ledger"]):
        msg = str(r.get("human_message", ""))
        if "MagicMock" in msg or "ord_a_parent" in msg:
            stats["gate_ledger"]["out_of_range"] += 1      # test fixtures that leaked into the ledger
            continue
        t = to_utc(r.get("timestamp"), UTC)
        if add(r.get("symbol"), t, "gate_ledger") and r.get("stage") in QUOTE_STAGES:
            events.append((str(r["symbol"]).upper().strip(), t, f"ledger:{r.get('stage')}", r.get("evaluation_id", "")))

    for r in iter_jsonl(INPUTS["validation_trades"], stats["validation_trades"]):
        t = to_utc(r.get("quote_timestamp") or r.get("timestamp"), UTC)
        if add(r.get("symbol"), t, "validation_trades"):
            if r.get("entry_order_id"):
                events.append((str(r["symbol"]).upper().strip(), t, "validation_decision_quote", r.get("validation_id", "")))
            bar_t = to_utc(r.get("bar_timestamp"), UTC)
            if bar_t is not None and r.get("decision_price"):
                try:
                    decision_price = float(r["decision_price"])
                    if not math.isfinite(decision_price) or decision_price <= 0:
                        raise ValueError('invalid decision price')
                    parity_refs.append((str(r["symbol"]).upper().strip(), bar_t, decision_price))
                except (TypeError, ValueError):
                    stats["validation_trades"]["malformed"] += 1

    for r in iter_jsonl(INPUTS["trade_journal"], stats["trade_journal"]):
        t = to_utc(r.get("timestamp") or r.get("journal_write_timestamp_utc"), UTC)
        if add(r.get("symbol"), t, "trade_journal") and "FILL" in (str(r.get("type", "")) + str(r.get("event_type", ""))).upper():
            events.append((str(r["symbol"]).upper().strip(), t, "journal_fill", str(r.get("fill_id") or r.get("order_id") or "")))

    st = stats["signal_exports"]
    for f in sorted(glob.glob(str(INPUTS["signal_exports"]))):
        st["exists"], st["files"] = True, st["files"] + 1
        try:
            with open(f, "r", encoding="utf-8", errors="replace", newline="") as fh:
                for row in csv.DictReader(fh):
                    st["lines"] += 1
                    if not row.get("symbol"):
                        st["malformed"] += 1
                        continue
                    st["parsed"] += 1
                    t = to_utc(row["timestamp"], UTC) if row.get("timestamp") else to_utc(row.get("timestamp_et"), ET)
                    add(row.get("symbol"), t, "signal_exports")
        except Exception:
            st["malformed"] += 1

    # warm-up: the bot's 5m indicators look back INTRADAY_LOOKBACK_DAYS calendar days
    for d in sorted(decision_days):
        for k in range(1, warmup_days + 1):
            w = d - timedelta(days=k)
            if is_session(w):
                pairs[w].update(pairs[d])
    for d in list(pairs):
        pairs[d].update(BENCHMARKS)
    return dict(sorted(pairs.items())), decision_days, events, parity_refs, stats


def merge_windows(events):
    by_sym = defaultdict(list)
    for sym, t, kind, eid in events:
        by_sym[sym].append((t - QUOTE_BEFORE, t + QUOTE_AFTER, f"{kind}:{eid}"))
    out = []
    for sym, wins in by_sym.items():
        wins.sort()
        s0, e0, ids = wins[0][0], wins[0][1], [wins[0][2]]
        for s, e, i in wins[1:]:
            if s <= e0:
                e0, ids = max(e0, e), ids + [i]
            else:
                out.append((sym, s0, e0, ids))
                s0, e0, ids = s, e, [i]
        out.append((sym, s0, e0, ids))
    return sorted(out, key=lambda w: w[1])


# ---------------------------------------------------------------- credentials
def load_keys(allow_generic: bool):
    """DAYTRADE_* from .env, then DAYTRADE_* from the environment (the pair
    run_daytrade_paper.bat uses). Generic APCA_* only with --allow-generic-keys."""
    vals = {}
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                vals[k.strip()] = v.strip().strip('"').strip("'")
    options = [
        (".env DAYTRADE_*", vals.get("DAYTRADE_APCA_API_KEY_ID"), vals.get("DAYTRADE_APCA_API_SECRET_KEY")),
        ("environment DAYTRADE_*", os.environ.get("DAYTRADE_APCA_API_KEY_ID"), os.environ.get("DAYTRADE_APCA_API_SECRET_KEY")),
    ]
    if allow_generic:
        options.append(("environment APCA_* (generic)", os.environ.get("APCA_API_KEY_ID"), os.environ.get("APCA_API_SECRET_KEY")))
    for source, key, secret in options:
        if key and secret and "YOUR_" not in key:
            return key, secret, source
    sys.exit("[ERROR] DAYTRADE_APCA_API_KEY_ID / DAYTRADE_APCA_API_SECRET_KEY not found in .env or environment.")


# ---------------------------------------------------------------- download
def fetch_bars(client, symbols, d, feed, adjustment, extended):
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    start, end = session_bounds(d, extended)
    req = StockBarsRequest(symbol_or_symbols=list(symbols), timeframe=TimeFrame.Minute,
                           start=start.to_pydatetime(), end=end.to_pydatetime(),
                           adjustment=Adjustment.ALL if adjustment == "all" else Adjustment.RAW,
                           feed=DataFeed.SIP if feed == "sip" else DataFeed.IEX)
    df = client.get_stock_bars(req).df
    return pd.DataFrame() if df is None or df.empty else df.reset_index()


def fetch_quotes(client, symbol, start, end, feed):
    from alpaca.data.enums import DataFeed
    from alpaca.data.requests import StockQuotesRequest
    req = StockQuotesRequest(symbol_or_symbols=symbol, start=start.floor("ms").to_pydatetime(),
                             end=end.ceil("ms").to_pydatetime(),
                             feed=DataFeed.SIP if feed == "sip" else DataFeed.IEX)
    df = client.get_stock_quotes(req).df
    return pd.DataFrame() if df is None or df.empty else df.reset_index()


def new_run_dir() -> tuple[str, Path]:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    for _ in range(5):
        run = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        path = EXPORT_DIR / f"replay_{run}"
        try:
            path.mkdir()                 # fails if it exists: no two runs share a folder
            return run, path
        except FileExistsError:
            continue
    sys.exit("[ERROR] could not create a unique run folder")


def parity_check(parity_refs, bars_by_feed):
    """Compare the bot's recorded decision price (validation records) with the close of the
    matching 5m bar rebuilt from exported 1m bars. A check, not a proof of parity."""
    out = {}
    for feed, closes in bars_by_feed.items():
        diffs = []
        for sym, bar_t, px in parity_refs:
            prices = [closes.get((sym, bar_t + timedelta(minutes=m))) for m in range(5)]
            if all(value is not None for value in prices) and px > 0:
                diffs.append(abs(prices[-1] - px) / px * 1e4)
        out[feed] = {"refs": len(parity_refs), "matched": len(diffs),
                     "median_abs_bps": round(statistics.median(diffs), 2) if diffs else None,
                     "share_within_5bps": round(sum(x <= 5 for x in diffs) / len(diffs), 3) if diffs else None}
    return out


def assess_bar_coverage(df, symbol, day, extended=False):
    """Count unique valid regular-session minutes, recording duplicate and invalid rows.

    A missing minute can mean no trade rather than missing provider data. Extended-hour
    rows are permitted but never counted as regular-session observations. The
    request's exact end timestamp may be returned by an inclusive API response;
    retain and count that boundary separately without treating it as a regular
    minute. Other out-of-session timestamps remain invalid.
    """
    start, end = session_bounds(day, False)
    extended_start, extended_end = session_bounds(day, True)
    expected = pd.date_range(start, end, freq="min", inclusive="left")
    seen = set()
    boundary_seen = set()
    duplicate = invalid = boundary_rows = out_of_session = 0
    if not df.empty:
        for row in df.itertuples(index=False):
            if str(getattr(row, 'symbol', '')).upper() != symbol.upper():
                continue
            try:
                ts = pd.Timestamp(row.timestamp)
                if ts.tzinfo is None:
                    raise ValueError('naive timestamp')
                ts = ts.tz_convert(UTC)
                low_bound, high_bound = (extended_start, extended_end) if extended else (start, end)
                if not low_bound <= ts <= high_bound:
                    out_of_session += 1
                    invalid += 1
                    continue
                if ts != ts.floor('min'):
                    invalid += 1
                    continue
                o, h, l, c = (float(getattr(row, k)) for k in ('open', 'high', 'low', 'close'))
                volume = float(row.volume)
                if not all(math.isfinite(v) for v in (o, h, l, c, volume)) or min(o, h, l, c) <= 0 or volume < 0 or not (l <= min(o, c) <= max(o, c) <= h):
                    invalid += 1
                    continue
                if ts == high_bound:
                    boundary_rows += 1
                    if ts in boundary_seen:
                        duplicate += 1
                    boundary_seen.add(ts)
                    continue
                if not start <= ts < end:
                    continue
                if ts in seen:
                    duplicate += 1
                seen.add(ts)
            except (AttributeError, TypeError, ValueError, OverflowError):
                invalid += 1
    return {'expected_minutes': len(expected), 'observed_minutes': len(seen),
            'missing_minutes': [t.isoformat() for t in expected if t not in seen],
            'duplicate_rows': duplicate, 'invalid_rows': invalid,
            'out_of_session_rows': out_of_session, 'end_boundary_rows': boundary_rows}


def assess_quote_events(df, symbol, events, max_age_seconds=10):
    """Report each event lacking a usable quote at or before its timestamp."""
    quotes = []
    if not df.empty:
        for row in df.itertuples(index=False):
            try:
                if hasattr(row, 'symbol') and str(row.symbol).upper() != symbol.upper():
                    continue
                ts = pd.Timestamp(row.timestamp)
                if ts.tzinfo is None:
                    continue
                bid, ask = float(row.bid_price), float(row.ask_price)
                if math.isfinite(bid) and math.isfinite(ask) and 0 < bid <= ask:
                    quotes.append(ts.tz_convert(UTC))
            except (AttributeError, TypeError, ValueError, OverflowError):
                continue
    uncovered = []
    for sym, t, kind, event_id in events:
        if sym != symbol:
            continue
        t = pd.Timestamp(t).tz_convert(UTC)
        if not any(pd.Timedelta(0) <= t - q <= pd.Timedelta(seconds=max_age_seconds) for q in quotes):
            uncovered.append({'symbol': sym, 'timestamp': t.isoformat(), 'kind': kind, 'id': str(event_id)})
    return uncovered


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", default="2026-09-29", help="first ET decision day (YYYY-MM-DD)")
    ap.add_argument("--warmup-days", type=int, default=5, help="calendar days of bars before each decision day")
    ap.add_argument("--feed", default="both", choices=["both", "iex", "sip"])
    ap.add_argument("--adjustment", default="raw", choices=["raw", "all"])
    ap.add_argument("--extended-hours", action="store_true", help="04:00-20:00 ET instead of the regular session")
    ap.add_argument("--quotes", action="store_true")
    ap.add_argument("--quote-feed", default="iex", choices=["iex", "sip"], help="iex = what the quote gate saw")
    ap.add_argument("--max-quote-windows", type=int, default=2000, help="windows beyond this make the run incomplete")
    ap.add_argument("--max-quote-age-seconds", type=float, default=10.0, help="maximum age of usable pre-event quote")
    ap.add_argument("--allow-generic-keys", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    since = date.fromisoformat(args.since)
    if args.warmup_days < 0 or args.max_quote_windows < 0 or args.max_quote_age_seconds < 0:
        ap.error("warmup days, quote window limit and quote age must be non-negative")
    if since.year != 2026:
        ap.error("the embedded NYSE calendar only supports 2026")

    pairs, decision_days, events, parity_refs, stats = collect(since, args.warmup_days)
    windows_all = merge_windows(events) if args.quotes else []
    windows = windows_all[: args.max_quote_windows]
    reasons = []
    for k, st in stats.items():
        if not st["exists"]:
            reasons.append(f"input missing: {st['path']}")
        if st["malformed"]:
            reasons.append(f"input {k}: {st['malformed']} malformed lines skipped")
        if st["bad_timestamp"]:
            reasons.append(f"input {k}: {st['bad_timestamp']} rows with unreadable timestamps")
        if st["exists"] and not st["used"]:
            reasons.append(f"input {k}: no usable records in the requested period")
    if not pairs:
        reasons.append("no candidate symbol-days found")
    if args.warmup_days < 5:
        reasons.append("warmup shorter than the bot's five-calendar-day lookback")
    if args.quotes and not windows_all:
        reasons.append("quotes requested but no quote-sensitive decisions or fills were found")
    if len(windows_all) > len(windows):
        reasons.append(f"quote windows truncated: {len(windows)} of {len(windows_all)} (raise --max-quote-windows)")

    print(f"Decision days: {len(decision_days)}  bar days incl. warm-up: {len(pairs)}  "
          f"(symbol, day) pairs: {sum(len(s) for s in pairs.values())}")
    for k, st in stats.items():
        print(f"  input {k:18s} exists={st['exists']!s:5s} files={st['files']} lines={st['lines']} "
              f"used={st['used']} malformed={st['malformed']} bad_ts={st['bad_timestamp']}")
    if args.quotes:
        print(f"  quote windows: {len(windows)} of {len(windows_all)} (from {len(events)} events)")
    for r in reasons:
        print(f"  [ATTENTION] {r}")
    if args.dry_run:
        print("Dry run: plan checked only; no market data was downloaded or verified.")
        return 0 if not reasons else 2
    if not pairs:
        print("[ERROR] nothing to download")
        return 1

    from alpaca.data.historical import StockHistoricalDataClient
    key, secret, key_source = load_keys(args.allow_generic_keys)
    print(f"Using market-data keys from: {key_source}")
    client = StockHistoricalDataClient(key, secret)
    run, out_dir = new_run_dir()
    feeds = ["iex", "sip"] if args.feed == "both" else [args.feed]

    manifest = {
        "run_id": run, "created_utc": datetime.now(timezone.utc).isoformat(), "since": str(since),
        "key_source": key_source, "inputs": stats,
        "bars": {"timeframe": "1Min", "adjustment": args.adjustment.upper(),
                 "session": "extended 04:00-20:00 ET" if args.extended_hours else "regular",
                 "warmup_calendar_days": args.warmup_days, "decision_days": len(decision_days),
                 "bar_days": len(pairs), "feeds": {}},
        "errors": {}, "warnings": [],
        "notes": [
            "signals_*.csv excludes candidates rejected at texture admission and portfolio correlation.",
            "Bot live bars may use IEX and Adjustment.ALL with 5-calendar-day lookback. Feed, adjustment, revisions and availability can differ from historical exports.",
            "Validation quote windows are centred on the decision quote time (no fill timestamp in those records).",
            "Quotes are not the bot's exact snapshot and do not show queue position; fills remain estimates.",
            "Status records configured checks only: a missing bar may mean no trades; a passing threshold does not establish complete feed data.",
            "Calendar is limited to documented 2026 sessions; unexpected closures and late data revisions are not verified.",
            "Exact request-end boundary bars are retained in CSV and counted separately; they do not add regular-session minute coverage. Other out-of-session rows and duplicates fail checks.",
        ],
    }

    closes_by_feed = {}
    for feed in feeds:
        path = out_dir / f"bars_1m_{feed}.csv.gz"
        tmp = path.with_name(path.name + ".partial")
        fs = {"file": path.name, "rows": 0, "missing_pairs": [], "low_coverage_pairs": [], "coverage_median": None,
              "pair_coverage": {}, "end_boundary_rows": 0, "end_boundary_pairs": []}
        closes, covs = {}, []
        with gzip.open(tmp, "xt", newline="", encoding="utf-8") as gz:
            w = csv.writer(gz)
            w.writerow(["symbol", "timestamp", "open", "high", "low", "close", "volume", "trade_count", "vwap", "feed"])
            for d, syms in pairs.items():
                try:
                    df = fetch_bars(client, sorted(syms), d, feed, args.adjustment, args.extended_hours)
                except Exception as exc:
                    manifest["errors"][f"bars:{feed}:{d}"] = str(exc)[:300]
                    print(f"  {d} {feed}: ERROR {str(exc)[:120]}")
                    time.sleep(1.0)
                    continue
                for s in sorted(syms):
                    coverage = assess_bar_coverage(df, s, d, args.extended_hours)
                    fs["pair_coverage"][f"{d}:{s}"] = coverage
                    if coverage['end_boundary_rows']:
                        fs['end_boundary_rows'] += coverage['end_boundary_rows']
                        fs['end_boundary_pairs'].append(f"{d}:{s}")
                    n, exp = coverage['observed_minutes'], coverage['expected_minutes']
                    if coverage['invalid_rows'] or coverage['duplicate_rows']:
                        reasons.append(f"bars {feed} {d}:{s}: invalid or duplicate timestamps")
                    if n == 0:
                        fs["missing_pairs"].append(f"{d}:{s}")
                        continue
                    cov = n / exp
                    covs.append(cov)
                    if cov < SIP_MIN_COVERAGE:
                        fs["low_coverage_pairs"].append(f"{d}:{s}:{n}/{exp}")
                for r in df.itertuples(index=False):
                    try:
                        ts = pd.Timestamp(r.timestamp).tz_convert(UTC)
                    except (AttributeError, TypeError, ValueError):
                        reasons.append(f"bars {feed} {d}: unreadable timestamp")
                        continue
                    start, end = session_bounds(d, False)
                    pair = fs['pair_coverage'].get(f"{d}:{r.symbol}", {})
                    if start <= ts < end and ts == ts.floor('min') and not pair.get('invalid_rows') and not pair.get('duplicate_rows'):
                        closes[(r.symbol, ts)] = float(r.close)
                    w.writerow([r.symbol, ts.isoformat(), r.open, r.high, r.low, r.close, r.volume,
                                getattr(r, "trade_count", ""), getattr(r, "vwap", ""), feed])
                fs["rows"] += len(df)
                print(f"  {d} {feed}: symbols={len(syms):3d} bars={len(df):6d}")
                time.sleep(0.35)
        os.replace(tmp, path)
        fs["coverage_median"] = round(statistics.median(covs), 3) if covs else None
        manifest["bars"]["feeds"][feed] = fs
        closes_by_feed[feed] = closes
        if fs["end_boundary_rows"]:
            manifest["warnings"].append(
                f"bars {feed}: {fs['end_boundary_rows']} valid request-end boundary rows retained and excluded from regular-session coverage"
            )
        if fs["missing_pairs"]:
            reasons.append(f"bars {feed}: {len(fs['missing_pairs'])} (symbol, day) pairs with no bars")
        if fs["low_coverage_pairs"]:
            reasons.append(f"bars {feed}: {len(fs['low_coverage_pairs'])} pairs below {SIP_MIN_COVERAGE:.0%} minute coverage; no-trade minutes are indistinguishable from missing data")

    manifest["parity"] = parity_check(parity_refs, closes_by_feed)

    if windows:
        path = out_dir / f"quotes_{args.quote_feed}.csv.gz"
        tmp = path.with_name(path.name + ".partial")
        qs = {"file": path.name, "feed": args.quote_feed, "windows_total": len(windows_all),
              "windows_requested": len(windows), "windows_with_quotes": 0, "empty_windows": [],
              "rows": 0, "window_seconds": [-QUOTE_BEFORE.seconds, QUOTE_AFTER.seconds],
              "max_quote_age_seconds": args.max_quote_age_seconds, "uncovered_events": []}
        with gzip.open(tmp, "xt", newline="", encoding="utf-8") as gz:
            w = csv.writer(gz)
            w.writerow(["symbol", "timestamp", "bid_price", "bid_size", "ask_price", "ask_size", "feed", "window_events"])
            for i, (sym, s, e, ids) in enumerate(windows, 1):
                try:
                    df = fetch_quotes(client, sym, s, e, args.quote_feed)
                except Exception as exc:
                    manifest["errors"][f"quotes:{sym}:{s.isoformat()}"] = str(exc)[:300]
                    time.sleep(1.0)
                    continue
                if df.empty:
                    qs["empty_windows"].append(f"{sym}:{s.isoformat()}")
                    qs['uncovered_events'].extend(assess_quote_events(df, sym, [ev for ev in events if ev[0] == sym and s <= ev[1] <= e], args.max_quote_age_seconds))
                    continue
                qs["windows_with_quotes"] += 1
                qs['uncovered_events'].extend(assess_quote_events(df, sym, [ev for ev in events if ev[0] == sym and s <= ev[1] <= e], args.max_quote_age_seconds))
                tag = "|".join(ids)[:500]
                for r in df.itertuples(index=False):
                    w.writerow([sym, pd.Timestamp(r.timestamp).tz_convert(UTC).isoformat(), r.bid_price,
                                r.bid_size, r.ask_price, r.ask_size, args.quote_feed, tag])
                qs["rows"] += len(df)
                if i % 50 == 0:
                    print(f"  quotes: {i}/{len(windows)} windows, {qs['rows']:,} quotes")
                time.sleep(0.35)
        os.replace(tmp, path)
        manifest["quotes"] = qs
        if qs["empty_windows"]:
            reasons.append(f"quotes: {len(qs['empty_windows'])} windows returned no quotes")
        if qs['uncovered_events']:
            reasons.append(f"quotes: {len(qs['uncovered_events'])} events have no valid recent quote at or before the event")

    if manifest["errors"]:
        reasons.append(f"{len(manifest['errors'])} request errors")
    manifest["status"] = "checks_passed" if not reasons else "incomplete"
    manifest["incomplete_reasons"] = reasons
    atomic_write_text(out_dir / "manifest.json", json.dumps(manifest, indent=2))

    print(f"\nOutput folder: {out_dir}")
    print(f"Status: {manifest['status']}")
    for warning in manifest['warnings']:
        print(f"  [WARNING] {warning}")
    for r in reasons:
        print(f"  - {r}")
    return 0 if not reasons else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
