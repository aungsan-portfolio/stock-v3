import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import pytz

import config
from strategies.intraday_data import _validate_intraday_data


def _make_sample_df(staleness_seconds: float) -> pd.DataFrame:
    """Helper to create a valid dataframe with a specific timestamp staleness."""
    tz = pytz.timezone("US/Eastern")
    now = datetime.now(tz)
    last_bar_time = now - timedelta(seconds=staleness_seconds)
    
    # 15 rows to satisfy length requirements
    times = [last_bar_time - timedelta(minutes=15 - i) for i in range(15)]
    times[-1] = last_bar_time
    
    data = {
        "open": np.linspace(100.0, 102.0, 15),
        "high": np.linspace(101.0, 103.0, 15),
        "low": np.linspace(99.0, 101.0, 15),
        "close": np.linspace(100.5, 102.5, 15),
        "volume": [1000] * 15,
        "atr": [0.5] * 15,
    }
    df = pd.DataFrame(data, index=pd.DatetimeIndex(times, tz=tz))
    return df


def test_tradable_symbol_stale_logs_warning():
    """Tradable symbol with stale quote (e.g. 400s) must emit a loud logger.warning."""
    df = _make_sample_df(staleness_seconds=400)
    
    with patch("strategies.session.is_market_open", return_value=True), \
         patch("strategies.intraday_data._in_market_open_transition", return_value=False), \
         patch("strategies.intraday_data.logger") as mock_logger:
        
        result = _validate_intraday_data(df, symbol="SPY", interval="1m", validate_staleness=True)
        
        # Must return empty DataFrame to prevent trading on stale price
        assert result.empty
        # Must log WARNING on tradable symbols
        mock_logger.warning.assert_called_once()
        args = mock_logger.warning.call_args[0]
        formatted_msg = args[0] % args[1:] if len(args) > 1 else args[0]
        assert "Data validation failed for SPY: Stale quote" in formatted_msg


def test_blacklisted_symbol_stale_logs_debug_only():
    """Blacklisted symbol (e.g. HOOX) with stale quote must log to DEBUG, avoiding log noise."""
    df = _make_sample_df(staleness_seconds=400)
    
    with patch("strategies.session.is_market_open", return_value=True), \
         patch("strategies.intraday_data._in_market_open_transition", return_value=False), \
         patch("config.SYMBOL_STALE_QUOTE_BLACKLIST", ["HOOX", "ELOX"]), \
         patch("strategies.intraday_data.logger") as mock_logger:
        
        result = _validate_intraday_data(df, symbol="HOOX", interval="1m", validate_staleness=True)
        
        # Must return empty DataFrame
        assert result.empty
        # Must NOT log warning
        mock_logger.warning.assert_not_called()
        # Must log DEBUG with [STALE-QUOTE]
        mock_logger.debug.assert_called_once()
        debug_format = mock_logger.debug.call_args[0][0]
        assert "[STALE-QUOTE]" in debug_format


def test_severe_data_outage_on_tradable_symbol_still_logs_warning():
    """Severe outage (>1800s, e.g. 35 mins) on a tradable symbol must NEVER be silenced to debug."""
    df = _make_sample_df(staleness_seconds=2100)
    
    with patch("strategies.session.is_market_open", return_value=True), \
         patch("strategies.intraday_data._in_market_open_transition", return_value=False), \
         patch("strategies.intraday_data.logger") as mock_logger:
        
        result = _validate_intraday_data(df, symbol="MARA", interval="1m", validate_staleness=True)
        
        assert result.empty
        # Must loudly warn of the severe pipeline outage
        mock_logger.warning.assert_called_once()
        args = mock_logger.warning.call_args[0]
        formatted_msg = args[0] % args[1:] if len(args) > 1 else args[0]
        assert "Data validation failed for MARA: Stale quote" in formatted_msg
