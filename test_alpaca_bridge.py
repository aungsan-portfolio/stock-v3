import os
import unittest
from unittest import mock
import pytest

import alpaca_bridge
import config

# Create simple fake account object to mock get_account() response
class FakeAccount:
    def __init__(self, status="ACTIVE", account_blocked=False, trading_blocked=False, equity=100000.0):
        self.status = status
        self.account_blocked = account_blocked
        self.trading_blocked = trading_blocked
        self.equity = equity


class TestAlpacaBridgeConnection(unittest.TestCase):
    def setUp(self):
        # Backup environment variables
        self.env_backup = dict(os.environ)
        # Setup valid default paper credentials for tests that check post-credentials path
        os.environ["APCA_API_KEY_ID"] = "fake_key"
        os.environ["APCA_API_SECRET_KEY"] = "fake_secret"
        os.environ["APCA_API_BASE_URL"] = "https://paper-api.alpaca.markets"

    def tearDown(self):
        # Restore environment variables
        os.environ.clear()
        os.environ.update(self.env_backup)

    def test_connect_missing_credentials(self):
        # If API key or secret key is missing, connect() must fail closed (return False)
        os.environ.pop("APCA_API_KEY_ID", None)
        br = alpaca_bridge.AlpacaBridge()
        self.assertFalse(br.connect())
        self.assertFalse(br._conn_health.is_healthy)

        os.environ["APCA_API_KEY_ID"] = "fake_key"
        os.environ.pop("APCA_API_SECRET_KEY", None)
        br = alpaca_bridge.AlpacaBridge()
        self.assertFalse(br.connect())

    @mock.patch("alpaca_bridge.TradingClient")
    @mock.patch("alpaca_bridge.StockHistoricalDataClient")
    def test_connect_live_url_safety_block(self, mock_data, mock_trade):
        # Mock get_account to avoid live broker connection
        mock_trade.return_value.get_account.return_value = FakeAccount()

        # URLs that represent the live Alpaca API or misconfigured URLs
        unsafe_urls = [
            "https://api.alpaca.markets",
            "https://api.alpaca.markets/",
            "  https://api.alpaca.markets  ",
            "HTTPS://API.ALPACA.MARKETS",
            "https://api.alpaca.markets/v2",
        ]

        for url in unsafe_urls:
            with self.subTest(url=url):
                os.environ["APCA_API_BASE_URL"] = url
                br = alpaca_bridge.AlpacaBridge()
                with self.assertRaises(RuntimeError) as ctx:
                    br.connect()
                self.assertIn("Live trading is disabled", str(ctx.exception))
                self.assertFalse(br._conn_health.is_healthy)

    @mock.patch("alpaca_bridge.TradingClient")
    @mock.patch("alpaca_bridge.StockHistoricalDataClient")
    def test_connect_allowlist_fails_for_typos(self, mock_data, mock_trade):
        # Verify allowlist logic: any URL that is not explicitly the paper URL must fail.
        mock_trade.return_value.get_account.return_value = FakeAccount()

        typo_urls = [
            "https://api.alpca.markets", # typo in alpaca
            "https://paper-api.alpca.markets",
            "https://google.com",
            "https://paper-api.alpaca.market" # missing trailing 's'
        ]

        for url in typo_urls:
            with self.subTest(url=url):
                os.environ["APCA_API_BASE_URL"] = url
                br = alpaca_bridge.AlpacaBridge()
                with self.assertRaises(RuntimeError):
                    br.connect()
                self.assertFalse(br._conn_health.is_healthy)

    @mock.patch("alpaca_bridge.TradingClient")
    @mock.patch("alpaca_bridge.StockHistoricalDataClient")
    def test_connect_missing_url_uses_paper_default(self, mock_data, mock_trade):
        # If base URL is not specified, it should default to the paper URL and succeed if credentials are ok
        mock_trade.return_value.get_account.return_value = FakeAccount()
        os.environ.pop("APCA_API_BASE_URL", None)

        br = alpaca_bridge.AlpacaBridge()
        self.assertTrue(br.connect())
        self.assertTrue(br._conn_health.is_healthy)

    @mock.patch("alpaca_bridge.TradingClient")
    @mock.patch("alpaca_bridge.StockHistoricalDataClient")
    def test_connect_blocked_or_inactive_account_fails(self, mock_data, mock_trade):
        # 1. Inactive account status
        mock_trade.return_value.get_account.return_value = FakeAccount(status="SUSPENDED")
        br = alpaca_bridge.AlpacaBridge()
        self.assertFalse(br.connect())
        self.assertFalse(br._conn_health.is_healthy)

        # 2. Account blocked
        mock_trade.return_value.get_account.return_value = FakeAccount(account_blocked=True)
        br = alpaca_bridge.AlpacaBridge()
        self.assertFalse(br.connect())

        # 3. Trading blocked
        mock_trade.return_value.get_account.return_value = FakeAccount(trading_blocked=True)
        br = alpaca_bridge.AlpacaBridge()
        self.assertFalse(br.connect())

    @mock.patch("alpaca_bridge.TradingClient")
    @mock.patch("alpaca_bridge.StockHistoricalDataClient")
    def test_disconnect_resets_state(self, mock_data, mock_trade):
        mock_trade.return_value.get_account.return_value = FakeAccount()
        br = alpaca_bridge.AlpacaBridge()
        
        # Connect first
        self.assertTrue(br.connect())
        self.assertTrue(br._connected)
        self.assertTrue(br._conn_health.is_healthy)

        # Disconnect
        br.disconnect()
        self.assertFalse(br._connected)
        self.assertIsNone(br._client)
        self.assertIsNone(br._data_client)
        self.assertTrue(br._conn_health.is_healthy)

    @mock.patch("alpaca_bridge.TradingClient")
    @mock.patch("alpaca_bridge.StockHistoricalDataClient")
    def test_reconnect_forces_clean_verification(self, mock_data, mock_trade):
        mock_trade.return_value.get_account.return_value = FakeAccount()
        br = alpaca_bridge.AlpacaBridge()

        # Connect attempt 1
        self.assertTrue(br.connect())
        self.assertEqual(mock_trade.return_value.get_account.call_count, 1)

        # Disconnect
        br.disconnect()

        # Connect attempt 2 - must verify clean again, calling get_account again
        self.assertTrue(br.connect())
        self.assertEqual(mock_trade.return_value.get_account.call_count, 2)


if __name__ == "__main__":
    unittest.main()
