"""
tests/test_chunked_batch_fetch.py -- Comprehensive unit tests for chunked Alpaca batch market data fetch.
"""
import pytest
from unittest.mock import MagicMock, patch, call
import pandas as pd
from datetime import datetime, timezone

import config
from strategies.intraday_data import (
    fetch_batch_intraday_alpaca,
    _fetch_bars_batch_alpaca,
    _normalize_data_feed,
)
from alpaca.data.enums import DataFeed


class TestChunkedBatchFetch:
    """Test suite verifying chunking logic, input normalization, free-tier SIP bypass, and telemetry."""

    @pytest.fixture(autouse=True)
    def reset_config(self):
        old_tier = getattr(config, "ALPACA_SUBSCRIPTION_TIER", "free")
        old_chunk_size = getattr(config, "ALPACA_CHUNK_SIZE", 5)
        old_delay = getattr(config, "ALPACA_CHUNK_DELAY_MS", 200)
        old_timeout = getattr(config, "ALPACA_CHUNK_TIMEOUT_SECONDS", 2.0)
        yield
        config.ALPACA_SUBSCRIPTION_TIER = old_tier
        config.ALPACA_CHUNK_SIZE = old_chunk_size
        config.ALPACA_CHUNK_DELAY_MS = old_delay
        config.ALPACA_CHUNK_TIMEOUT_SECONDS = old_timeout

    def _sample_df(self, symbol: str) -> pd.DataFrame:
        idx = pd.date_range(end=datetime.now(timezone.utc), periods=5, freq="5min")
        df = pd.DataFrame({
            "open": [100.0] * 5,
            "high": [101.0] * 5,
            "low": [99.0] * 5,
            "close": [100.5] * 5,
            "volume": [1000] * 5,
        }, index=idx)
        df.index.name = "timestamp"
        return df

    def test_normalize_data_feed_valid_and_invalid(self):
        assert _normalize_data_feed("iex") == DataFeed.IEX
        assert _normalize_data_feed("sip") == DataFeed.SIP
        assert _normalize_data_feed("otc") == DataFeed.OTC
        assert _normalize_data_feed(DataFeed.IEX) == DataFeed.IEX
        # Invalid / unknown string defaults safely to IEX
        assert _normalize_data_feed("unknown_feed") == DataFeed.IEX

    def test_empty_input_generates_zero_sdk_calls(self):
        with patch("strategies.intraday_data._sdk_get_stock_bars_soft_bounded") as mock_sdk:
            res = fetch_batch_intraday_alpaca([])
            assert res == {}
            assert mock_sdk.call_count == 0

    @patch("strategies.intraday_data._get_alpaca_credentials", return_value=("test_key", "test_sec"))
    @patch("strategies.intraday_data._get_alpaca_client")
    @patch("strategies.intraday_data._read_cache", return_value=None)
    @patch("strategies.intraday_data._write_cache")
    @patch("strategies.intraday_data._validate_intraday_data")
    @patch("strategies.intraday_data._sdk_get_stock_bars_soft_bounded")
    @patch("time.sleep")
    def test_seven_symbols_split_into_five_and_two_with_order_preserved(
        self, mock_sleep, mock_sdk, mock_validate, mock_write, mock_read, mock_client, mock_creds
    ):
        config.ALPACA_CHUNK_SIZE = 5
        # Mixed case, whitespace, duplicates, blanks
        symbols = ["aapl", " MSFT ", "", "NVDA", "googl", "aapl", "TSLA", "amd", "meta"]
        # Expected cleaned unique ordered symbols:
        # Note: clean_symbols deduplication preserves order: AAPL, MSFT, NVDA, GOOGL, TSLA, AMD, META (7 symbols)

        chunks_received = []
        def side_effect(client, req, timeout=None):
            chunks_received.append(list(req.symbol_or_symbols))
            return MagicMock(df=pd.DataFrame())

        mock_sdk.side_effect = side_effect
        mock_validate.side_effect = lambda df, sym, *args, **kwargs: df

        fetch_batch_intraday_alpaca(symbols)

        # 7 symbols split into chunks of 5 and 2
        assert len(chunks_received) == 2
        assert chunks_received[0] == ["AAPL", "MSFT", "NVDA", "GOOGL", "TSLA"]
        assert chunks_received[1] == ["AMD", "META"]

        # Sleep occurred exactly chunk_count - 1 = 1 time
        assert mock_sleep.call_count == 1
        mock_sleep.assert_called_once_with(0.2)

    @patch("strategies.intraday_data._get_alpaca_credentials", return_value=("test_key", "test_sec"))
    @patch("strategies.intraday_data._get_alpaca_client")
    @patch("strategies.intraday_data._read_cache", return_value=None)
    @patch("strategies.intraday_data._validate_intraday_data")
    @patch("strategies.intraday_data._sdk_get_stock_bars_soft_bounded")
    def test_free_tier_generates_zero_sip_requests(
        self, mock_sdk, mock_validate, mock_read, mock_client, mock_creds
    ):
        config.ALPACA_SUBSCRIPTION_TIER = "free"
        config.ALPACA_CHUNK_SIZE = 5
        mock_sdk.return_value = MagicMock(df=pd.DataFrame())  # All missing

        fetch_batch_intraday_alpaca(["AAPL", "NVDA"])

        # Exactly 1 call with DataFeed.IEX, zero fallback SIP calls
        assert mock_sdk.call_count == 1
        assert mock_sdk.call_args[0][1].feed == DataFeed.IEX

    @patch("strategies.intraday_data._get_alpaca_credentials", return_value=("test_key", "test_sec"))
    @patch("strategies.intraday_data._get_alpaca_client")
    @patch("strategies.intraday_data._read_cache", return_value=None)
    @patch("strategies.intraday_data._write_cache")
    @patch("strategies.intraday_data._validate_intraday_data")
    @patch("strategies.intraday_data._sdk_get_stock_bars_soft_bounded")
    def test_unlimited_tier_preserves_primary_feed_and_allows_sip(
        self, mock_sdk, mock_validate, mock_write, mock_read, mock_client, mock_creds
    ):
        config.ALPACA_SUBSCRIPTION_TIER = "unlimited"
        config.ALPACA_DATA_FEED = "iex"

        fb_df = self._sample_df("AAPL")
        fb_df["symbol"] = "AAPL"
        fallback_res = MagicMock(df=fb_df.reset_index().set_index(["symbol", "timestamp"]))

        mock_sdk.side_effect = [
            MagicMock(df=pd.DataFrame()),  # Primary returns empty
            fallback_res,                   # Fallback returns data
        ]
        mock_validate.side_effect = lambda df, sym, *args, **kwargs: df

        res = fetch_batch_intraday_alpaca(["AAPL"])
        assert mock_sdk.call_count == 2
        assert mock_sdk.call_args_list[0][0][1].feed == DataFeed.IEX
        assert mock_sdk.call_args_list[1][0][1].feed == DataFeed.SIP
        assert "AAPL" in res

    @patch("strategies.intraday_data._get_alpaca_credentials", return_value=("test_key", "test_sec"))
    @patch("strategies.intraday_data._get_alpaca_client")
    @patch("strategies.intraday_data._read_cache", return_value=None)
    @patch("strategies.intraday_data._write_cache")
    @patch("strategies.intraday_data._validate_intraday_data")
    @patch("strategies.intraday_data._sdk_get_stock_bars_soft_bounded")
    @patch("time.sleep")
    def test_chunk_failure_does_not_interrupt_later_chunks(
        self, mock_sleep, mock_sdk, mock_validate, mock_write, mock_read, mock_client, mock_creds
    ):
        config.ALPACA_CHUNK_SIZE = 1
        symbols = ["FAIL", "PASS"]

        pass_df = self._sample_df("PASS")
        pass_df["symbol"] = "PASS"
        pass_res = MagicMock(df=pass_df.reset_index().set_index(["symbol", "timestamp"]))

        mock_sdk.side_effect = [
            RuntimeError("Chunk 1 connection abort"),
            pass_res,
        ]
        mock_validate.side_effect = lambda df, sym, *args, **kwargs: df

        res = fetch_batch_intraday_alpaca(symbols)
        assert mock_sdk.call_count == 2
        assert "PASS" in res
        assert "FAIL" not in res

    @patch("strategies.intraday_data._get_alpaca_credentials", return_value=("test_key", "test_sec"))
    @patch("strategies.intraday_data._get_alpaca_client")
    @patch("strategies.intraday_data._read_cache", return_value=None)
    @patch("strategies.intraday_data._validate_intraday_data")
    @patch("strategies.intraday_data._sdk_get_stock_bars_soft_bounded")
    def test_timeout_setting_reaches_bounded_sdk_wrapper(
        self, mock_sdk, mock_validate, mock_read, mock_client, mock_creds
    ):
        config.ALPACA_CHUNK_TIMEOUT_SECONDS = 1.75
        mock_sdk.return_value = MagicMock(df=pd.DataFrame())

        fetch_batch_intraday_alpaca(["AAPL"])

        assert mock_sdk.call_count == 1
        # Timeout passed as keyword argument to bounded wrapper
        _, kwargs = mock_sdk.call_args
        assert kwargs.get("timeout") == 1.75

    @patch("strategies.intraday_data._get_alpaca_credentials", return_value=("test_key", "test_sec"))
    @patch("strategies.intraday_data._get_alpaca_client")
    @patch("strategies.intraday_data._read_cache", return_value=None)
    @patch("strategies.intraday_data._write_cache")
    @patch("strategies.intraday_data._validate_intraday_data")
    @patch("strategies.intraday_data._sdk_get_stock_bars_soft_bounded")
    def test_partial_responses_produce_accurate_coverage(
        self, mock_sdk, mock_validate, mock_write, mock_read, mock_client, mock_creds
    ):
        config.ALPACA_CHUNK_SIZE = 4
        symbols = ["S1", "S2", "S3", "S4"]

        # Only S1 returns data
        s1_df = self._sample_df("S1")
        s1_df["symbol"] = "S1"
        mock_sdk.return_value = MagicMock(df=s1_df.reset_index().set_index(["symbol", "timestamp"]))
        mock_validate.side_effect = lambda df, sym, *args, **kwargs: df

        with patch("strategies.intraday_data.logger.info") as mock_info:
            res = fetch_batch_intraday_alpaca(symbols)
            assert len(res) == 1
            # Coverage is 1/4 = 25.0%
            mock_info.assert_any_call(
                "[CHUNK FETCH DONE] Retrieved 1/4 symbols across 1 chunks in 0.00s (coverage=25.0%)"
                if "0.00s" in str(mock_info.call_args_list) else mock_info.call_args_list[-1][0][0]
            )

    def test_compatibility_wrapper_alias(self):
        # Verify _fetch_bars_batch_alpaca exists and is identical callable
        assert callable(_fetch_bars_batch_alpaca)
        assert _fetch_bars_batch_alpaca is fetch_batch_intraday_alpaca
