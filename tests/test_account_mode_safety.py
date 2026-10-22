"""
tests/test_account_mode_safety.py — Dedicated Test Suite for Account-Mode Safety Guard

Enforces fail-closed invariants:
1. TradingClient is constructed with paper=True.
2. Live base URL fails closed before order submission.
3. Account number mismatch blocks submission and aborts.
"""

import os
import json
import unittest
from unittest import mock
from decimal import Decimal

import alpaca_bridge
from alpaca_bridge import AlpacaBridge, AccountModeViolation


class MockAccount:
    def __init__(self, account_number="PA3UODIC5JTA", status="ACTIVE", equity=100000.0):
        self.account_number = account_number
        self.id = account_number
        self.status = status
        self.account_blocked = False
        self.trading_blocked = False
        self.equity = equity


class TestAccountModeSafety(unittest.TestCase):
    def setUp(self):
        self.env_backup = dict(os.environ)
        os.environ["APCA_API_KEY_ID"] = "test_paper_key"
        os.environ["APCA_API_SECRET_KEY"] = "test_paper_secret"
        os.environ["DAYTRADE_APCA_API_KEY_ID"] = "test_paper_key"
        os.environ["DAYTRADE_APCA_API_SECRET_KEY"] = "test_paper_secret"
        os.environ["APCA_API_BASE_URL"] = "https://paper-api.alpaca.markets"
        os.environ["EXPECTED_PAPER_ACCOUNT_ID"] = "PA3UODIC5JTA"

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.env_backup)

    @mock.patch("alpaca_bridge.StockHistoricalDataClient")
    @mock.patch("alpaca_bridge.TradingClient")
    def test_client_is_constructed_with_paper_true(self, mock_tc, mock_data):
        """TradingClient is constructed strictly with paper=True in alpaca_bridge."""
        mock_client_instance = mock.MagicMock()
        mock_client_instance._paper = True
        mock_client_instance._base_url = "https://paper-api.alpaca.markets"
        mock_client_instance.get_account.return_value = MockAccount(account_number="PA3UODIC5JTA")
        mock_tc.return_value = mock_client_instance

        bridge = AlpacaBridge(mode="daytrade")
        connected = bridge.connect()

        self.assertTrue(connected)
        mock_tc.assert_called_once_with("test_paper_key", "test_paper_secret", paper=True)
        self.assertTrue(bridge.is_paper)
        self.assertEqual(bridge.account_mode, "PAPER")

    @mock.patch("alpaca_bridge.StockHistoricalDataClient")
    @mock.patch("alpaca_bridge.TradingClient")
    def test_live_base_url_fails_closed_before_order_submission(self, mock_tc, mock_data):
        """Live API base URL fails closed with AccountModeViolation before any order submission."""
        mock_client_instance = mock.MagicMock()
        mock_client_instance._paper = False
        mock_client_instance._base_url = "https://api.alpaca.markets"
        mock_client_instance.get_account.return_value = MockAccount(account_number="PA3UODIC5JTA")
        mock_tc.return_value = mock_client_instance

        bridge = AlpacaBridge(mode="daytrade")
        bridge._client = mock_client_instance
        bridge._connected = True

        # Directly testing assert_account_mode_safety()
        with self.assertRaises(AccountModeViolation) as ctx:
            bridge.assert_account_mode_safety()
        self.assertIn("ACCOUNT_MODE_VIOLATION", str(ctx.exception))

        # Testing order submission path: must fail closed before submit_order
        with self.assertRaises(AccountModeViolation):
            bridge.place_bracket_order(
                symbol="AAPL",
                side="BUY",
                qty=10,
                entry_price=150.0,
                stop_price=145.0,
                target_price=160.0,
            )
        mock_client_instance.submit_order.assert_not_called()

        # Stop order path must also fail closed
        with self.assertRaises(AccountModeViolation):
            bridge.place_stop_order(
                symbol="AAPL",
                side="SELL",
                qty=10,
                stop_price=145.0,
            )
        mock_client_instance.submit_order.assert_not_called()

    @mock.patch("alpaca_bridge.StockHistoricalDataClient")
    @mock.patch("alpaca_bridge.TradingClient")
    def test_account_number_mismatch_blocks_submission(self, mock_tc, mock_data):
        """Account number mismatch fails closed with AccountModeViolation and blocks order submission."""
        mock_client_instance = mock.MagicMock()
        mock_client_instance._paper = True
        mock_client_instance._base_url = "https://paper-api.alpaca.markets"
        # Return wrong account number
        mock_client_instance.get_account.return_value = MockAccount(account_number="PA_WRONG_999999")
        mock_tc.return_value = mock_client_instance

        bridge = AlpacaBridge(mode="daytrade")
        bridge._client = mock_client_instance
        bridge._connected = True
        bridge.account_id = "PA_WRONG_999999"

        with self.assertRaises(AccountModeViolation) as ctx:
            bridge.assert_account_mode_safety(expected_account_id="PA3UODIC5JTA")
        self.assertIn("does not match expected", str(ctx.exception))

        # Verify order submission is blocked
        with self.assertRaises(AccountModeViolation):
            bridge.place_bracket_order(
                symbol="AAPL",
                side="BUY",
                qty=10,
                entry_price=150.0,
                stop_price=145.0,
                target_price=160.0,
            )
        mock_client_instance.submit_order.assert_not_called()

    @mock.patch("alpaca_bridge.StockHistoricalDataClient")
    @mock.patch("alpaca_bridge.TradingClient")
    def test_correct_paper_account_passes_safety_assertion(self, mock_tc, mock_data):
        """Matching paper credentials, base URL, and account number pass safety checks cleanly."""
        mock_client_instance = mock.MagicMock()
        mock_client_instance._paper = True
        mock_client_instance._base_url = "https://paper-api.alpaca.markets"
        mock_client_instance.get_account.return_value = MockAccount(account_number="PA3UODIC5JTA")
        mock_tc.return_value = mock_client_instance

        bridge = AlpacaBridge(mode="daytrade")
        bridge._client = mock_client_instance
        bridge._connected = True
        bridge.account_id = "PA3UODIC5JTA"

        self.assertTrue(bridge.assert_account_mode_safety(expected_account_id="PA3UODIC5JTA"))

    def test_url_normalization_rejects_lookalike_and_insecure_urls(self):
        """Rejects non-https, lookalike domains, non-443 ports, and embedded credentials."""
        from alpaca_bridge import _validate_paper_url

        # Insecure schemes
        with self.assertRaises(AccountModeViolation):
            _validate_paper_url("http://paper-api.alpaca.markets")
        with self.assertRaises(AccountModeViolation):
            _validate_paper_url("ftp://paper-api.alpaca.markets")

        # Lookalike domains
        with self.assertRaises(AccountModeViolation):
            _validate_paper_url("https://paper-api.alpaca.markets.attacker.com")
        with self.assertRaises(AccountModeViolation):
            _validate_paper_url("https://attacker.paper-api.alpaca.markets.evil.org")
        with self.assertRaises(AccountModeViolation):
            _validate_paper_url("https://api.alpaca.markets")

        # Non-standard ports
        with self.assertRaises(AccountModeViolation):
            _validate_paper_url("https://paper-api.alpaca.markets:8080")
        with self.assertRaises(AccountModeViolation):
            _validate_paper_url("https://paper-api.alpaca.markets:8443")

        # Embedded credentials
        with self.assertRaises(AccountModeViolation):
            _validate_paper_url("https://user:pass@paper-api.alpaca.markets")

        # Valid paper URLs should not raise
        _validate_paper_url("https://paper-api.alpaca.markets")
        _validate_paper_url("https://paper-api.alpaca.markets/")
        _validate_paper_url("https://paper-api.alpaca.markets:443")
        _validate_paper_url("  https://paper-api.alpaca.markets  ")

    @mock.patch("alpaca_bridge.StockHistoricalDataClient")
    @mock.patch("alpaca_bridge.TradingClient")
    def test_unreadable_or_erroring_account_fails_closed(self, mock_tc, mock_data):
        """Unreadable, missing, erroring, or sentinel account IDs fail closed with AccountModeViolation."""
        mock_client = mock.MagicMock()
        mock_client._paper = True
        mock_client._base_url = "https://paper-api.alpaca.markets"
        mock_tc.return_value = mock_client

        bridge = AlpacaBridge(mode="daytrade")
        bridge._client = mock_client
        bridge._connected = True

        # 1. get_account raises Exception / Timeout
        mock_client.get_account.side_effect = TimeoutError("Gateway timeout connecting to broker")
        with self.assertRaises(AccountModeViolation) as ctx:
            bridge.assert_account_mode_safety()
        self.assertIn("failed", str(ctx.exception).lower())

        # 2. get_account returns None
        mock_client.get_account.side_effect = None
        mock_client.get_account.return_value = None
        with self.assertRaises(AccountModeViolation) as ctx:
            bridge.assert_account_mode_safety()
        self.assertIn("returned None", str(ctx.exception))

        # 3. get_account returns empty or sentinel account_number
        for bad_id in ["", "   ", "UNKNOWN", "DEFAULT", "NONE"]:
            mock_client.get_account.return_value = MockAccount(account_number=bad_id)
            with self.assertRaises(AccountModeViolation) as ctx:
                bridge.assert_account_mode_safety()
            self.assertIn("unreadable or unknown", str(ctx.exception).lower())

    @mock.patch("alpaca_bridge.StockHistoricalDataClient")
    @mock.patch("alpaca_bridge.TradingClient")
    def test_all_10_mutation_methods_invoke_safety_assertion(self, mock_tc, mock_data):
        """All 10 broker-mutating methods must invoke assert_account_mode_safety and fail closed."""
        mock_client = mock.MagicMock()
        mock_client._paper = False  # LIVE mode violation
        mock_client._base_url = "https://api.alpaca.markets"
        mock_client.get_account.return_value = MockAccount(account_number="PA3UODIC5JTA")
        mock_tc.return_value = mock_client

        bridge = AlpacaBridge(mode="daytrade")
        bridge._client = mock_client
        bridge._connected = True

        # 1. place_bracket_order
        with self.assertRaises(AccountModeViolation):
            bridge.place_bracket_order(symbol="AAPL", side="BUY", qty=1, entry_price=150.0, stop_price=145.0, target_price=160.0)

        # 2. place_stop_order
        with self.assertRaises(AccountModeViolation):
            bridge.place_stop_order(symbol="AAPL", side="SELL", qty=1, stop_price=145.0)

        # 3. modify_stop_order
        with self.assertRaises(AccountModeViolation):
            bridge.modify_stop_order(order_id="fake_order_123", new_stop=146.0)

        # 4. cancel_order
        with self.assertRaises(AccountModeViolation):
            bridge.cancel_order(order_id="fake_order_123")

        # 5. _close_position
        with self.assertRaises(AccountModeViolation):
            bridge._close_position(symbol="AAPL", action="SELL", qty=1, price=150.0, note="test_close")

        # 6. _cancel_symbol_working_orders
        with self.assertRaises(AccountModeViolation):
            bridge._cancel_symbol_working_orders(symbol="AAPL")

        # 7. flatten_all
        with self.assertRaises(AccountModeViolation):
            bridge.flatten_all()

        # 8. _cancel_all_orders_for_symbol
        with self.assertRaises(AccountModeViolation):
            bridge._cancel_all_orders_for_symbol(symbol="AAPL")

        # 9. close_position
        with self.assertRaises(AccountModeViolation):
            bridge.close_position(symbol="AAPL")

        # 10. _cancel_working_entry_orders
        with self.assertRaises(AccountModeViolation):
            bridge._cancel_working_entry_orders(symbol="AAPL")

        # Verify no broker call was actually submitted
        mock_client.submit_order.assert_not_called()
        mock_client.cancel_order_by_id.assert_not_called()
        mock_client.close_position.assert_not_called()

    def test_live_cli_flag_rejected_with_exit_code_6(self):
        """CLI execution with --live is intercepted and returns Exit Code 6 (ACCOUNT_MODE_VIOLATION)."""
        import sys
        from main import main as cli_main

        # Test standalone --live
        with mock.patch.object(sys, "argv", ["main.py", "--live"]):
            code = cli_main()
            self.assertEqual(code, 6)

        # Test subcommand with --live
        with mock.patch.object(sys, "argv", ["main.py", "daytrade-bot", "--live"]):
            code = cli_main()
            self.assertEqual(code, 6)

    def test_diagnostic_snapshot_and_anomaly_logging(self):
        """Diagnostic snapshot and reconciliation anomaly logging write valid durable records."""
        import tempfile
        from alpaca_bridge import record_account_mode_violation_snapshot, log_reconciliation_anomaly

        # Test snapshot in isolated temp directory to prevent production log pollution
        with tempfile.TemporaryDirectory() as tmp_dir:
            snapshot_path = record_account_mode_violation_snapshot(
                expected_account_id="PA3UODIC5JTA",
                observed_account_id="LIVE_ACCOUNT_123",
                endpoint_url="https://api.alpaca.markets",
                failure_reason="Live base URL detected",
                output_dir=tmp_dir,
            )
            self.assertTrue(os.path.exists(snapshot_path))
            self.assertTrue(snapshot_path.startswith(tmp_dir))
            with open(snapshot_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["exit_code"], 6)
            self.assertEqual(data["failure_type"], "ACCOUNT_MODE_VIOLATION")
            self.assertFalse(data["order_submission_attempted"])

        # Test anomaly logging
        anomaly_file = os.path.join("reports", "reconciliation_anomalies.jsonl")
        test_event = {
            "symbol": "BAC",
            "reason": "TEST_ANOMALY",
            "delta_qty": 6,
            "calculated_price": 44.50,
        }
        log_reconciliation_anomaly(test_event)
        self.assertTrue(os.path.exists(anomaly_file))
        with open(anomaly_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        last_record = json.loads(lines[-1])
        self.assertEqual(last_record["symbol"], "BAC")
        self.assertEqual(last_record["reason"], "TEST_ANOMALY")
        self.assertIn("timestamp_utc", last_record)


if __name__ == "__main__":
    unittest.main()
