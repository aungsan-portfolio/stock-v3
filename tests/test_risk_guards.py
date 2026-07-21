import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from strategies.orchestrator import evaluate_and_execute
from strategies.intraday_risk import (
    get_consecutive_losses,
    get_recent_symbol_loss_time,
    pre_trade_check,
)
from strategies.trade_journal import backfill_closed_trades, get_today_closed_trades


def test_daily_loss_flatten_opt_in():
    bridge = MagicMock()
    bridge.is_connected = True
    bridge.ib.positions.return_value = []
    bridge.get_net_liquidation.return_value = 25000.0
    bridge.ib.openTrades.return_value = []
    bridge.account_daily_pnl.return_value = -3000.0

    with (
        patch("strategies.orchestrator.config.FLATTEN_ON_DAILY_LOSS", True),
        patch("strategies.orchestrator.evaluate_symbols_parallel") as evaluate_symbols,
    ):
        evaluate_and_execute(["AAPL"], bridge, live_paper=True)

    bridge.flatten_all.assert_called_once()
    evaluate_symbols.assert_not_called()


def test_daily_loss_flatten_opt_out():
    bridge = MagicMock()
    bridge.is_connected = True
    bridge.ib.positions.return_value = []
    bridge.get_net_liquidation.return_value = 25000.0
    bridge.ib.openTrades.return_value = []
    bridge.account_daily_pnl.return_value = -3000.0

    with (
        patch("strategies.orchestrator.config.FLATTEN_ON_DAILY_LOSS", False),
        patch("strategies.orchestrator.evaluate_symbols_parallel", return_value=[]),
    ):
        evaluate_and_execute(["AAPL"], bridge, live_paper=True)

    bridge.flatten_all.assert_not_called()


def test_consecutive_losses_limit():
    with (
        patch("strategies.intraday_risk.get_consecutive_losses", return_value=3),
        patch("strategies.intraday_risk.config.MAX_CONSECUTIVE_LOSSES", 3),
    ):
        reason = pre_trade_check(
            equity=30000.0,
            current_pnl=-50.0,
            trades_today=3,
            open_positions=0,
            day_trades_last_5_days=0,
            risk_dollars=50.0,
        )

    assert "Consecutive loss limit reached" in reason


def test_same_symbol_cooldown_active():
    last_loss = datetime.now(timezone.utc) - timedelta(minutes=2)
    with (
        patch("strategies.intraday_risk.get_recent_symbol_loss_time", return_value=last_loss),
        patch("strategies.intraday_risk.get_consecutive_losses", return_value=0),
        patch("strategies.intraday_risk.config.REENTRY_COOLDOWN_MINUTES", 5),
        patch("strategies.intraday_risk.check_flatten_zone", return_value=True),
    ):
        reason = pre_trade_check(
            equity=30000.0,
            current_pnl=-50.0,
            trades_today=1,
            open_positions=0,
            day_trades_last_5_days=0,
            risk_dollars=50.0,
            symbol="AAPL",
        )

    assert "Re-entry cooldown active for AAPL" in reason


def test_same_symbol_cooldown_expired():
    last_loss = datetime.now(timezone.utc) - timedelta(minutes=6)
    with (
        patch("strategies.intraday_risk.get_recent_symbol_loss_time", return_value=last_loss),
        patch("strategies.intraday_risk.get_consecutive_losses", return_value=0),
        patch("strategies.intraday_risk.config.REENTRY_COOLDOWN_MINUTES", 5),
        patch("strategies.intraday_risk.check_flatten_zone", return_value=True),
    ):
        reason = pre_trade_check(
            equity=30000.0,
            current_pnl=-50.0,
            trades_today=1,
            open_positions=0,
            day_trades_last_5_days=0,
            risk_dollars=50.0,
            symbol="AAPL",
        )

    assert reason == ""


def test_consecutive_loss_counter_stops_at_latest_win():
    closed_trades = [
        {"symbol": "AAPL", "realized_pnl": 10.0},
        {"symbol": "MSFT", "realized_pnl": -20.0},
        {"symbol": "TSLA", "realized_pnl": -30.0},
    ]
    with patch("strategies.intraday_risk.get_today_closed_trades", return_value=closed_trades):
        assert get_consecutive_losses() == 2


def test_profit_exit_clears_symbol_cooldown():
    closed_trades = [
        {
            "symbol": "AAPL",
            "closed_at": datetime.now(timezone.utc) - timedelta(minutes=6),
            "realized_pnl": -10.0,
        },
        {
            "symbol": "AAPL",
            "closed_at": datetime.now(timezone.utc) - timedelta(minutes=1),
            "realized_pnl": 20.0,
        },
    ]
    with patch("strategies.intraday_risk.get_today_closed_trades", return_value=closed_trades):
        assert get_recent_symbol_loss_time("AAPL") is None


def test_closed_fills_reconstruct_loss(tmp_path):
    journal = tmp_path / "journal.jsonl"
    journal.write_text(
        "\n".join(
            [
                '{"timestamp":"2026-07-13T13:30:00+00:00","event_type":"FILL","type":"FILL","execution_id":"entry","symbol":"AAPL","side":"BUY","qty":10,"fill_price":100.0}',
                '{"timestamp":"2026-07-13T13:35:00+00:00","event_type":"FILL","type":"FILL","execution_id":"exit","symbol":"AAPL","side":"SELL","qty":10,"fill_price":95.0}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fixed_now = datetime(2026, 7, 13, 12, tzinfo=timezone.utc)

    with patch("strategies.trade_journal.now_eastern", return_value=fixed_now):
        closed_trades = get_today_closed_trades(journal)

    assert len(closed_trades) == 1
    assert closed_trades[0]["symbol"] == "AAPL"
    assert closed_trades[0]["realized_pnl"] == -50.0
    assert closed_trades[0]["is_win"] is False


def test_backfill_closed_trades(tmp_path):
    journal = tmp_path / "journal.jsonl"
    journal.write_text(
        "\n".join(
            [
                '{"timestamp":"2026-07-13T13:30:00+00:00","event_type":"FILL","type":"FILL","execution_id":"entry","symbol":"AAPL","side":"BUY","qty":10,"fill_price":100.0,"signal_id":"sig-test-1"}',
                '{"timestamp":"2026-07-13T13:35:00+00:00","event_type":"FILL","type":"FILL","execution_id":"exit","symbol":"AAPL","side":"SELL","qty":10,"fill_price":105.0,"signal_id":null}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    new_count = backfill_closed_trades(journal)
    assert new_count == 1

    # Check that TRADE_CLOSED record was appended
    with open(journal, encoding="utf-8") as f:
        lines = f.readlines()

    assert len(lines) == 3
    last_record = json.loads(lines[-1])
    assert last_record["event_type"] == "TRADE_CLOSED"
    assert last_record["symbol"] == "AAPL"
    assert last_record["signal_id"] == "sig-test-1"
    assert last_record["realized_pnl"] == 50.0
    assert last_record["is_win"] is True
    assert last_record["exit_reason"] == "PROFIT_EXIT"


def test_emergency_flatten_cooldown_integration(tmp_path):
    from unittest.mock import patch
    from strategies.trade_journal import log_trade
    from strategies.intraday_risk import pre_trade_check
    from datetime import datetime, timezone
    import json

    journal = tmp_path / "journal.jsonl"
    
    # Log the exit record (representing an emergency flatten loss of -$50.0)
    log_trade(
        symbol="AAPL",
        side="SELL",
        strategy="ORB",
        qty=10,
        entry_price=100.0,
        stop_price=95.0,
        target_price=120.0,
        exit_price=95.0,
        pnl=-50.0,
        event_type="TRADE_CLOSED",
        journal_file=journal
    )
    
    with (
        patch("strategies.trade_journal.config.DAYTRADE_TRADE_JOURNAL_FILE", journal),
        patch("strategies.intraday_risk.config.DAYTRADE_TRADE_JOURNAL_FILE", journal),
        patch("strategies.intraday_risk.config.REENTRY_COOLDOWN_MINUTES", 5),
        patch("strategies.intraday_risk.get_consecutive_losses", return_value=0),
        patch("strategies.intraday_risk.check_flatten_zone", return_value=True),
        patch("strategies.trade_journal.now_eastern", return_value=datetime.now(timezone.utc).astimezone())
    ):
        # The cooldown should block AAPL since it was closed with a loss less than 5 mins ago
        reason = pre_trade_check(
            equity=100000.0,
            current_pnl=-50.0,
            trades_today=1,
            open_positions=0,
            day_trades_last_5_days=0,
            risk_dollars=50.0,
            symbol="AAPL"
        )
        assert reason is not None
        assert "Re-entry cooldown active for AAPL" in reason
