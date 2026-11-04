"""Offline regression tests for replay-export coverage and input provenance.

All data here are synthetic; no broker or market-data service is contacted.
"""
from datetime import date
import json
import sys

import pandas as pd
import pytest

import export_replay_bars as exporter


@pytest.fixture
def sources(tmp_path, monkeypatch):
    root = tmp_path / "project"
    data = root / "input_fixtures"
    reports = root / "journal_fixtures"
    exports = root / "synthetic_exports"
    exports.mkdir(parents=True)
    data.mkdir()
    reports.mkdir()
    monkeypatch.setattr(exporter, "ROOT", root)
    monkeypatch.setattr(exporter, "EXPORT_DIR", exports)
    monkeypatch.setattr(exporter, "INPUTS", {
        "gate_ledger": data / "trade_gate_decision.jsonl",
        "validation_trades": data / "paper_validation_trades.jsonl",
        "trade_journal": reports / "trade_journal.jsonl",
        "signal_exports": exports / "signals_*.csv",
    })
    return root


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_dry_run_missing_inputs_is_incomplete_without_network(sources, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["export_replay_bars.py", "--dry-run", "--quotes"])
    assert exporter.main() != 0
    output = capsys.readouterr().out.lower()
    assert "missing" in output or "no quote" in output
    assert not list((sources / "synthetic_exports").glob("replay_*"))


def test_rotation_records_are_included_with_own_day_and_symbol(sources):
    data = sources / "input_fixtures"
    _write_jsonl(data / "trade_gate_decision.jsonl", [
        {"timestamp": "2026-11-03T15:00:00Z", "symbol": "AAPL", "stage": "STAGE_CONFIDENCE"},
    ])
    _write_jsonl(data / "trade_gate_decision.jsonl.1", [
        {"timestamp": "2026-11-02T15:00:00Z", "symbol": "MSFT", "stage": "STAGE_CONFIDENCE"},
    ])
    pairs, _, _, _, _ = exporter.collect(date(2026, 10, 1), 0)
    assert "AAPL" in pairs[date(2026, 11, 3)]
    assert "MSFT" in pairs[date(2026, 11, 2)]


def _bar(timestamp, symbol="AAPL", **changes):
    row = {"symbol": symbol, "timestamp": pd.Timestamp(timestamp),
           "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
           "volume": 100}
    row.update(changes)
    return row


def test_bar_coverage_does_not_count_duplicate_or_off_session_minutes():
    # 351 duplicates previously satisfied the 90% SIP threshold despite only
    # one usable minute. A different out-of-hours row must not improve it.
    frame = pd.DataFrame([_bar("2026-11-03T14:30:00Z")] * 351 + [
        _bar("2026-11-03T22:00:00Z"),
    ])
    result = exporter.assess_bar_coverage(frame, "AAPL", date(2026, 11, 3))
    assert result["expected_minutes"] == 390
    assert result["observed_minutes"] == 1
    assert result["duplicate_rows"] >= 350
    assert len(result["missing_minutes"]) == 389


def test_invalid_ohlc_does_not_count_as_observed_bar():
    frame = pd.DataFrame([
        _bar("2026-11-03T14:30:00Z", low=102.0),
        _bar("2026-11-03T14:31:00Z"),
    ])
    result = exporter.assess_bar_coverage(frame, "AAPL", date(2026, 11, 3))
    assert result["invalid_rows"] >= 1
    assert result["observed_minutes"] == 1


def test_extended_hours_bars_do_not_inflate_regular_session_coverage():
    frame = pd.DataFrame([
        _bar("2026-11-03T13:00:00Z"),
        _bar("2026-11-03T14:30:00Z"),
    ])
    result = exporter.assess_bar_coverage(frame, "AAPL", date(2026, 11, 3), extended=True)
    assert result["invalid_rows"] == 0
    assert result["observed_minutes"] == 1


def test_quote_after_event_cannot_cover_decision():
    ts = pd.Timestamp("2026-11-03T14:30:00Z")
    events = [("AAPL", ts, "order_submit", "decision-1")]
    quotes = pd.DataFrame([{
        "symbol": "AAPL", "timestamp": ts + pd.Timedelta(seconds=119),
        "bid_price": 99.9, "ask_price": 100.1, "bid_size": 100, "ask_size": 100,
    }])
    assert len(exporter.assess_quote_events(quotes, "AAPL", events)) == 1


def test_each_event_needs_a_fresh_prior_quote_not_just_a_nonempty_merged_window():
    ts = pd.Timestamp("2026-11-03T14:30:00Z")
    events = [("AAPL", ts, "quote_gate", "first"),
              ("AAPL", ts + pd.Timedelta(seconds=30), "order_submit", "second")]
    quotes = pd.DataFrame([{
        "symbol": "AAPL", "timestamp": ts - pd.Timedelta(seconds=1),
        "bid_price": 99.9, "ask_price": 100.1, "bid_size": 100, "ask_size": 100,
    }])
    assert len(exporter.assess_quote_events(quotes, "AAPL", events, max_age_seconds=10)) == 1


def test_crossed_or_invalid_quotes_do_not_make_an_event_covered():
    ts = pd.Timestamp("2026-11-03T14:30:00Z")
    events = [("AAPL", ts, "quote_gate", "crossed")]
    quotes = pd.DataFrame([{
        "symbol": "AAPL", "timestamp": ts, "bid_price": 101.0,
        "ask_price": 100.0, "bid_size": 100, "ask_size": 100,
    }])
    assert len(exporter.assess_quote_events(quotes, "AAPL", events)) == 1


def test_parity_requires_a_complete_five_minute_input_window():
    ts = pd.Timestamp("2026-11-03T14:30:00Z")
    reference = [("AAPL", ts, 100.0)]
    incomplete = {"iex": {("AAPL", ts): 100.0}}
    assert exporter.parity_check(reference, incomplete)["iex"]["matched"] == 0
    complete = {"iex": {("AAPL", ts + pd.Timedelta(minutes=m)): 100.0 for m in range(5)}}
    assert exporter.parity_check(reference, complete)["iex"]["matched"] == 1


def test_unsupported_calendar_year_is_not_silently_treated_as_2026():
    with pytest.raises((ValueError, NotImplementedError)):
        exporter.is_session(date(2027, 1, 1))


def _valid_sources(root):
    event = {"timestamp": "2026-11-03T14:30:10Z", "symbol": "AAPL"}
    _write_jsonl(root / "input_fixtures" / "trade_gate_decision.jsonl", [
        {**event, "stage": "STAGE_QUOTE_HEALTH", "evaluation_id": "gate-a"},
    ])
    _write_jsonl(root / "input_fixtures" / "paper_validation_trades.jsonl", [
        {**event, "entry_order_id": "fake-entry", "validation_id": "val-a"},
    ])
    _write_jsonl(root / "journal_fixtures" / "trade_journal.jsonl", [
        {**event, "type": "FILL", "fill_id": "fill-a"},
    ])
    (root / "synthetic_exports" / "signals_20261103.csv").write_text(
        "timestamp,symbol\n2026-11-03T14:30:10Z,AAPL\n", encoding="utf-8"
    )


def _run_fake_export(sources, monkeypatch, bars, quotes, capsys):
    """Exercise manifest status through main, with no external I/O."""
    _valid_sources(sources)
    import alpaca.data.historical

    monkeypatch.setattr(sys, "argv", ["export_replay_bars.py", "--since", "2026-11-03",
                                      "--warmup-days", "5", "--feed", "sip", "--quotes"])
    monkeypatch.setattr(exporter, "load_keys", lambda *_: ("FAKE_KEY_111", "FAKE_SECRET_222", "fixture"))
    monkeypatch.setattr(alpaca.data.historical, "StockHistoricalDataClient", lambda *a, **k: object())
    monkeypatch.setattr(exporter, "fetch_bars", bars)
    monkeypatch.setattr(exporter, "fetch_quotes", quotes)
    monkeypatch.setattr(exporter.time, "sleep", lambda *_: None)
    exit_code = exporter.main()
    outputs = list((sources / "synthetic_exports").glob("replay_*/manifest.json"))
    assert len(outputs) == 1
    manifest = json.loads(outputs[0].read_text(encoding="utf-8"))
    captured = capsys.readouterr()
    assert "FAKE_KEY_111" not in captured.out + captured.err
    assert "FAKE_SECRET_222" not in captured.out + captured.err
    return exit_code, manifest


def test_end_to_end_duplicate_minutes_cannot_pass_sip_coverage(sources, monkeypatch, capsys):
    def duplicates(_client, symbols, day, feed, adjustment, extended):
        session_start = exporter.session_bounds(day, False)[0]
        return pd.DataFrame([_bar(session_start, symbol=s) for s in symbols for _ in range(351)])

    def prior_quote(_client, symbol, start, end, feed):
        return pd.DataFrame([{"symbol": symbol, "timestamp": pd.Timestamp("2026-11-03T14:30:09Z"),
                              "bid_price": 99.9, "ask_price": 100.1, "bid_size": 100, "ask_size": 100}])

    code, manifest = _run_fake_export(sources, monkeypatch, duplicates, prior_quote, capsys)
    assert code != 0
    assert manifest["status"] == "incomplete"


def test_end_to_end_quotes_only_after_event_cannot_report_complete(sources, monkeypatch, capsys):
    def full_bars(_client, symbols, day, feed, adjustment, extended):
        times = pd.date_range(exporter.session_bounds(day, False)[0], periods=exporter.expected_regular_minutes(day), freq="min")
        return pd.DataFrame([_bar(t, symbol=s) for s in symbols for t in times])

    def future_quote(_client, symbol, start, end, feed):
        return pd.DataFrame([{"symbol": symbol, "timestamp": pd.Timestamp("2026-11-03T14:32:00Z"),
                              "bid_price": 99.9, "ask_price": 100.1, "bid_size": 100, "ask_size": 100}])

    code, manifest = _run_fake_export(sources, monkeypatch, full_bars, future_quote, capsys)
    assert code != 0
    assert manifest["status"] == "incomplete"


def test_end_to_end_full_sip_bars_and_prior_quotes_pass_configured_checks(sources, monkeypatch, capsys):
    def full_bars(_client, symbols, day, feed, adjustment, extended):
        times = pd.date_range(exporter.session_bounds(day, False)[0], periods=exporter.expected_regular_minutes(day), freq="min")
        return pd.DataFrame([_bar(t, symbol=s) for s in symbols for t in times])

    def prior_quote(_client, symbol, start, end, feed):
        return pd.DataFrame([{"symbol": symbol, "timestamp": pd.Timestamp("2026-11-03T14:30:09Z"),
                              "bid_price": 99.9, "ask_price": 100.1, "bid_size": 100, "ask_size": 100}])

    code, manifest = _run_fake_export(sources, monkeypatch, full_bars, prior_quote, capsys)
    assert code == 0
    assert manifest["status"] == "checks_passed"
