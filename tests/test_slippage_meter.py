import sys
import os
import pathlib
sys.path.insert(0, os.path.abspath("."))
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock
from strategies.trade_journal import read_journal
from strategies.order_registry import register_order_expected_price
from alpaca_bridge import AlpacaBridge

def test_patch_1b_acceptance_cases(tmp_path=None):
    if tmp_path is None:
        tmp_path = pathlib.Path("./temp_test_dir")
        tmp_path.mkdir(exist_ok=True)
    journal_file = tmp_path / "test_journal.jsonl"
    if journal_file.exists():
        journal_file.unlink()

    bridge = AlpacaBridge.__new__(AlpacaBridge)
    bridge._connected = True
    bridge._client = MagicMock()

    now = datetime.now(timezone.utc)

    # 1. Entry Market Fill (signal 319.72, fill 319.77 BUY)
    now_1 = now
    sub_1 = now_1 - timedelta(milliseconds=250)
    mock_order_1 = MagicMock()
    mock_order_1.id = "mock_entry_googl_1"
    mock_order_1.symbol = "GOOGL"
    mock_order_1.side.value = "buy"
    mock_order_1.filled_qty = "15"
    mock_order_1.filled_avg_price = "319.77"
    mock_order_1.stop_price = None
    mock_order_1.limit_price = None
    mock_order_1.submitted_at = sub_1
    mock_order_1.filled_at = now_1
    mock_order_1.client_order_id = "dt_VWAP_BOUNCE_GOOGL"

    register_order_expected_price(
        order_id="mock_entry_googl_1",
        expected_price=319.72,
        side="BUY",
        order_type="ENTRY",
        submit_ts=sub_1.timestamp(),
        symbol="GOOGL"
    )

    # 2. Stop Exit Adverse (stop 604.88, fill 603.87 SELL)
    now_2 = now + timedelta(minutes=2)
    sub_2 = now_2 - timedelta(minutes=2) # 2 minutes ago
    mock_order_2 = MagicMock()
    mock_order_2.id = "mock_stop_meta_adverse"
    mock_order_2.symbol = "META"
    mock_order_2.side.value = "sell"
    mock_order_2.filled_qty = "8"
    mock_order_2.filled_avg_price = "603.87"
    mock_order_2.stop_price = "604.88"
    mock_order_2.limit_price = None
    mock_order_2.submitted_at = sub_2
    mock_order_2.filled_at = now_2
    mock_order_2.client_order_id = "dt_VWAP_BOUNCE_META"

    register_order_expected_price(
        order_id="mock_stop_meta_adverse",
        expected_price=604.88,
        side="SELL",
        order_type="STOP_LOSS",
        submit_ts=sub_2.timestamp(),
        symbol="META"
    )

    # 3. Stop Exit Favorable (stop 604.88, fill 605.10 SELL)
    now_3 = now + timedelta(minutes=5)
    sub_3 = now_3 - timedelta(minutes=3)
    mock_order_3 = MagicMock()
    mock_order_3.id = "mock_stop_meta_favorable"
    mock_order_3.symbol = "META"
    mock_order_3.side.value = "sell"
    mock_order_3.filled_qty = "8"
    mock_order_3.filled_avg_price = "605.10"
    mock_order_3.stop_price = "604.88"
    mock_order_3.limit_price = None
    mock_order_3.submitted_at = sub_3
    mock_order_3.filled_at = now_3
    mock_order_3.client_order_id = "dt_VWAP_BOUNCE_META"

    register_order_expected_price(
        order_id="mock_stop_meta_favorable",
        expected_price=604.88,
        side="SELL",
        order_type="STOP_LOSS",
        submit_ts=sub_3.timestamp(),
        symbol="META"
    )

    bridge._client.get_orders.return_value = [mock_order_1, mock_order_2, mock_order_3]

    import config
    orig_path = config.DAYTRADE_TRADE_JOURNAL_FILE
    config.DAYTRADE_TRADE_JOURNAL_FILE = journal_file
    try:
        synced = bridge.sync_today_trades_to_journal()
        assert synced == 3, f"Expected 3 synced orders, got {synced}"

        records = read_journal(journal_file=journal_file)
        assert len(records) == 3

        # Case 1: Entry market fill (GOOGL)
        r1 = records[0]
        exp_slip_1 = (319.77 - 319.72) / 319.72 # +0.00015638...
        assert r1["symbol"] == "GOOGL"
        assert abs(r1["slippage"] - exp_slip_1) < 1e-6
        assert r1["slippage"] > 0, "Entry market fill slippage must be non-zero and positive adverse"
        assert r1["fill_latency_ms"] is not None and abs(r1["fill_latency_ms"] - 250.0) < 1.0
        print(f"[PASS Case 1] Entry Market Fill: expected={r1['expected_price']} fill={r1['fill_price']} slippage={r1['slippage']:.6f} latency={r1['fill_latency_ms']:.1f}ms")

        # Case 2: Stop Exit Adverse (META)
        r2 = records[1]
        exp_slip_2 = (604.88 - 603.87) / 604.88 # +0.0016698... (+0.17%)
        assert r2["symbol"] == "META"
        assert abs(r2["slippage"] - exp_slip_2) < 1e-6
        assert r2["slippage"] > 0, "Adverse stop exit slippage must be positive"
        assert r2["fill_latency_ms"] is None, f"Stop leg holding duration must not be logged as fill_latency_ms, got {r2['fill_latency_ms']}"
        print(f"[PASS Case 2] Stop Exit Adverse: expected={r2['expected_price']} fill={r2['fill_price']} slippage=+{r2['slippage']:.4f} latency={r2['fill_latency_ms']}")

        # Case 3: Stop Exit Favorable (META)
        r3 = records[2]
        exp_slip_3 = (604.88 - 605.10) / 604.88 # -0.0003637... (-0.036%)
        assert r3["symbol"] == "META"
        assert abs(r3["slippage"] - exp_slip_3) < 1e-6
        assert r3["slippage"] < 0, "Favorable stop exit slippage must be negative"
        assert r3["fill_latency_ms"] is None
        print(f"[PASS Case 3] Stop Exit Favorable: expected={r3['expected_price']} fill={r3['fill_price']} slippage={r3['slippage']:.6f} latency={r3['fill_latency_ms']}")

        print("\n[ALL 4 PATCH 1B ACCEPTANCE TESTS PASSED SUCCESSFULLY!]")
    finally:
        config.DAYTRADE_TRADE_JOURNAL_FILE = orig_path

if __name__ == "__main__":
    test_patch_1b_acceptance_cases()
