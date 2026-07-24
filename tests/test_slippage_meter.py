import sys
import os
import pathlib
sys.path.insert(0, os.path.abspath("."))
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock
from strategies.trade_journal import read_journal
from alpaca_bridge import AlpacaBridge

def test_slippage_meter_calculation(tmp_path=None):
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
    sub_time = now - timedelta(milliseconds=250)
    
    mock_order = MagicMock()
    mock_order.id = "mock_meta_stop_123"
    mock_order.symbol = "META"
    mock_order.side.value = "sell"
    mock_order.filled_qty = "8"
    mock_order.filled_avg_price = "603.87"
    mock_order.stop_price = "604.88"
    mock_order.limit_price = None
    mock_order.submitted_at = sub_time
    mock_order.filled_at = now
    mock_order.client_order_id = "dt_VWAP_BOUNCE_META"
    
    bridge._client.get_orders.return_value = [mock_order]
    
    import config
    orig_path = config.DAYTRADE_TRADE_JOURNAL_FILE
    config.DAYTRADE_TRADE_JOURNAL_FILE = journal_file
    try:
        synced = bridge.sync_today_trades_to_journal()
        assert synced == 1, f"Expected 1 synced order, got {synced}"
        
        records = read_journal(journal_file=journal_file)
        assert len(records) == 1
        r = records[0]
        
        assert r["symbol"] == "META"
        assert r["side"] == "SELL"
        assert r["fill_price"] == 603.87
        assert r["expected_price"] == 604.88
        
        # Slippage: |603.87 - 604.88| / 604.88 = 1.01 / 604.88 = 0.0016698... (~0.0017)
        expected_slip = 1.01 / 604.88
        assert abs(r["slippage"] - expected_slip) < 1e-6, f"Expected slippage {expected_slip}, got {r['slippage']}"
        assert r["slippage"] > 0.0015
        assert r["fill_latency_ms"] >= 200.0
        print(f"[PASS] Slippage Meter Test Verified: symbol={r['symbol']} side={r['side']} fill={r['fill_price']} expected={r['expected_price']} slippage={r['slippage']:.4f} latency={r['fill_latency_ms']:.1f}ms")
    finally:
        config.DAYTRADE_TRADE_JOURNAL_FILE = orig_path

if __name__ == "__main__":
    test_slippage_meter_calculation()
