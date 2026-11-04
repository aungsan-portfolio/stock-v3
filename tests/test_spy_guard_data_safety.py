"""An enabled benchmark guard must have usable evidence before sizing a BUY."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

import config
from strategies.base import EntryReadiness, TradeSignal
import strategies.order_manager as om


def _bars(open_price=500.0, close_price=501.0):
    return pd.DataFrame(
        {"open": [500.0, open_price], "close": [500.0, close_price],
         "high": [502.0, max(open_price, close_price) + 1.0],
         "low": [499.0, min(open_price, close_price) - 1.0]},
        index=pd.date_range("2026-11-02T18:00:00Z", periods=2, freq="5min"),
    )


@pytest.fixture
def guard_context(monkeypatch):
    import strategies.intraday_data as data
    import strategies.session as session

    monkeypatch.setattr(config, "SPY_TREND_GUARD_ENABLED", True)
    monkeypatch.setattr(config, "SPY_TREND_GUARD_SYMBOL", "SPY")
    monkeypatch.setattr(config, "INVERSE_ETF_SYMBOLS", {"SOXS"})
    monkeypatch.setattr(config, "INVERSE_ETF_RISK_OFF_EXEMPT", True)
    monkeypatch.setattr(config, "INVERSE_ETF_BENCHMARK_MAP", {"SOXS": "SOXX"})
    monkeypatch.setattr(config, "INVERSE_ETF_MIN_CONFIDENCE", 0.65)
    monkeypatch.setattr(config, "ALLOW_SHORT", True)
    monkeypatch.setattr(session, "now_eastern", lambda: pd.Timestamp("2026-11-02T18:20:00Z").to_pydatetime())
    monkeypatch.setattr(om, "check_session_suppression", lambda *a, **kw: (False, ""))
    monkeypatch.setattr(om, "validate_quote_health", lambda *a, **kw: (True, "", {"age_seconds": 0.0}))
    # Stop immediately after this guard. A zero-size outcome proves the guard
    # approved without allowing unrelated downstream gates or broker mutation.
    sizing = MagicMock(return_value=0)
    monkeypatch.setattr(om, "calculate_shares", sizing)
    reservation = MagicMock(return_value=(True, ""))
    monkeypatch.setattr(om.idempotency_manager, "reserve", reservation)
    fetch = MagicMock(return_value=_bars())
    monkeypatch.setattr(data, "fetch_intraday", fetch)
    bridge = MagicMock()
    bridge.is_connected = True
    bridge.is_paper = True
    bridge.entry_gate = SimpleNamespace(halted=False)
    bridge.get_quote.return_value = {"bid": 99.99, "ask": 100.01}
    readiness = EntryReadiness(
        entry_allowed=True, suppression_reason=None, cycle_id="guard-test",
        bar_ts="2026-11-02T18:05:00Z", created_at=0.0, expires_at=float("inf"),
    )

    def execute(symbol="AAPL", side="BUY", dry_run=False):
        signal = TradeSignal(
            symbol=symbol, strategy="CANDLESTICK", side=side,
            entry_price=100.0, stop_price=98.0 if side == "BUY" else 102.0,
            target_price=104.0 if side == "BUY" else 96.0,
            risk_per_share=2.0, atr=1.0, confidence=0.9, reason="guard test",
            bar_timestamp=readiness.bar_ts,
        )
        return om.execute_signal(
            signal, bridge, equity=100000.0, current_pnl=0.0,
            dry_run=dry_run, entry_readiness=readiness,
            regime_result=SimpleNamespace(
                allow_inverse_etf_longs=True, market_texture="TRENDING",
                confidence_boost=0.0, mode="NEUTRAL",
            ),
        )

    return SimpleNamespace(execute=execute, fetch=fetch, sizing=sizing, reservation=reservation, bridge=bridge)


@pytest.mark.parametrize("symbol", ["AAPL", "SOXS"])
@pytest.mark.parametrize("case", ["exception", "none", "empty", "one_closed", "forming"])
def test_missing_benchmark_evidence_blocks_before_sizing(guard_context, symbol, case):
    ctx = guard_context
    if case == "exception":
        ctx.fetch.side_effect = RuntimeError("benchmark unavailable")
    elif case == "none":
        ctx.fetch.return_value = None
    elif case == "empty":
        ctx.fetch.return_value = pd.DataFrame()
    elif case == "one_closed":
        ctx.fetch.return_value = _bars().iloc[:1]
    else:
        bars = _bars()
        bars.index = pd.to_datetime(["2026-11-02T18:10:00Z", "2026-11-02T18:20:00Z"])
        ctx.fetch.return_value = bars

    result = ctx.execute(symbol)
    assert result["status"] == "REJECTED"
    assert "Trend Guard" in result["reason"] and "safety" in result["reason"]
    ctx.sizing.assert_not_called()
    ctx.reservation.assert_not_called()
    ctx.bridge.place_bracket_order.assert_not_called()
    ctx.bridge.submit_order.assert_not_called()


@pytest.mark.parametrize("symbol", ["AAPL", "SOXS"])
@pytest.mark.parametrize("field,value", [
    ("open", float("nan")), ("close", float("nan")),
    ("open", float("inf")), ("close", float("inf")),
    ("open", 0.0), ("close", -1.0),
    ("high", float("nan")), ("low", float("inf")),
    ("high", 499.0), ("low", 502.0),
])
def test_invalid_benchmark_candle_cannot_allow_long(guard_context, field, value, symbol):
    ctx = guard_context
    bars = _bars()
    bars.loc[bars.index[-1], field] = value
    ctx.fetch.return_value = bars
    result = ctx.execute(symbol)
    assert result["status"] == "REJECTED"
    assert "Trend Guard" in result["reason"] and "safety" in result["reason"]
    ctx.sizing.assert_not_called()
    ctx.reservation.assert_not_called()
    ctx.bridge.place_bracket_order.assert_not_called()
    ctx.bridge.submit_order.assert_not_called()


@pytest.mark.parametrize("symbol", ["AAPL", "SOXS"])
@pytest.mark.parametrize("missing", ["open", "close"])
def test_missing_required_candle_column_blocks_before_sizing(guard_context, symbol, missing):
    ctx = guard_context
    ctx.fetch.return_value = _bars().drop(columns=[missing])
    result = ctx.execute(symbol)
    assert result["status"] == "REJECTED"
    assert "Trend Guard" in result["reason"] and "safety" in result["reason"]
    ctx.sizing.assert_not_called()
    ctx.reservation.assert_not_called()
    ctx.bridge.place_bracket_order.assert_not_called()
    ctx.bridge.submit_order.assert_not_called()


@pytest.mark.parametrize("symbol,bullish,allowed,benchmark", [
    ("AAPL", True, True, "SPY"), ("AAPL", False, False, "SPY"),
    ("SOXS", True, False, "SOXX"), ("SOXS", False, True, "SOXX"),
])
def test_valid_candles_keep_normal_and_inverse_direction_rules(guard_context, symbol, bullish, allowed, benchmark):
    ctx = guard_context
    ctx.fetch.return_value = _bars(500.0, 501.0 if bullish else 499.0)
    result = ctx.execute(symbol)
    assert ctx.fetch.call_args.args[0] == benchmark
    if allowed:
        assert result["reason"] == "Position size = 0"
        ctx.sizing.assert_called_once()
    else:
        assert "Blocking" in result["reason"]
        ctx.sizing.assert_not_called()
    ctx.bridge.place_bracket_order.assert_not_called()
    ctx.bridge.submit_order.assert_not_called()


@pytest.mark.parametrize("mode", ["disabled", "dry_run", "short"])
def test_existing_guard_scope_is_preserved(guard_context, monkeypatch, mode):
    ctx = guard_context
    if mode == "disabled":
        monkeypatch.setattr(config, "SPY_TREND_GUARD_ENABLED", False)
    result = ctx.execute(side="SELL" if mode == "short" else "BUY", dry_run=mode == "dry_run")
    assert result["reason"] == "Position size = 0"
    ctx.fetch.assert_not_called()
    ctx.sizing.assert_called_once()
