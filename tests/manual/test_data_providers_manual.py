"""
Manual test runner for the multi-source data provider.
Run this script to verify that the new data_providers module works correctly.
"""
import sys
import os
import tempfile
import logging

# Add the parent directory to sys.path so we can import the modules from the project root
sys.path.insert(0, str(os.path.abspath(os.path.join(__file__, ".."))))

import data_manager
import data_providers

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def test_basic_fetch():
    """Test basic fetch from yfinance (legacy path)."""
    print("Test 1: Basic yfinance fetch...")
    df = data_manager.fetch_ohlcv('SPY', force_refresh=True)
    print(f"   - Got {len(df)} rows")
    print(f"   - Close range: ${df.Close.min():.2f} - ${df.Close.max():.2f}")
    print("   PASSED\n")

def test_data_providers():
    """Test MultiSourceDataProvider."""
    print("Test 2: MultiSourceDataProvider (yfinance fallback)...")
    provider = data_providers.MultiSourceDataProvider()
    df = provider.fetch('QQQ', force_refresh=True)
    print(f"   - Got {len(df)} rows")
    print(f"   - Close range: ${df.Close.min():.2f} - ${df.Close.max():.2f}")
    print("   PASSED\n")

def test_cache_functionality():
    """Test CSV caching works."""
    print("Test 3: CSV cache functionality...")
    from data_providers import CACHE_DIR
    # Ensure cache directory exists
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Use temporary cache dir to avoid interfering with existing data
    from pathlib import Path
    original_cache = data_providers.CACHE_DIR
    test_cache_dir = Path(tempfile.mkdtemp())
    data_providers.CACHE_DIR = test_cache_dir

    try:
        provider = data_providers.MultiSourceDataProvider()
        # First fetch - should save to cache
        df1 = provider.fetch('MSFT', force_refresh=True)
        print(f"   - First fetch: {len(df1)} rows")

        # Second fetch - should read from cache
        df2 = provider.fetch('MSFT', force_refresh=False)
        print(f"   - Cached fetch: {len(df2)} rows")

        # Verify data matches
        assert len(df1) == len(df2), "Cache size mismatch"
        assert abs(df1.Close.iloc[-1] - df2.Close.iloc[-1]) < 0.01, "Cache data mismatch"
        print("   - Cache check: PASSED\n")
    finally:
        # Restore original cache directory
        data_providers.CACHE_DIR = original_cache

    print("   PASSED\n")

def test_data_provider_setter():
    """Test setting data provider globally."""
    print("Test 4: Global data provider setter...")

    # Create a custom provider that logs calls
    call_log = []

    class MockProvider:
        def __init__(self):
            self.calls = []

        def fetch(self, symbol, period='5y', interval='1d', force_refresh=False):
            self.calls.append((symbol, period, interval, force_refresh))
            # Return fake data
            return data_providers.pd.DataFrame({
                'Open': [100, 101, 102],
                'High': [101, 102, 103],
                'Low': [99, 100, 101],
                'Close': [100, 101, 102],
                'Volume': [1000, 1100, 1200]
            }, index=pd.date_range('2023-01-01', periods=3, freq='D'))

    mock_provider = MockProvider()

    # Set it as the global provider
    data_manager.set_data_provider(mock_provider)

    # Fetch with data_manager - should use mock
    df = data_manager.fetch_ohlcv('TEST', force_refresh=True)
    print(f"   - Fetched {len(df)} rows via mock provider")
    print(f"   - Calls: {len(mock_provider.calls)}")

    # Reset to None
    data_manager.set_data_provider(None)

    # Back to legacy yfinance
    df2 = data_manager.fetch_ohlcv('SPY', force_refresh=True)
    print(f"   - Back to legacy yfinance: {len(df2)} rows")

    print("   PASSED\n")

def run_all_tests():
    """Run all tests."""
    import pandas as pd
    data_providers.pd = pd

    print("=" * 60)
    print("Running data_providers tests...")
    print("=" * 60)

    try:
        test_basic_fetch()
        test_data_providers()
        test_cache_functionality()
        test_data_provider_setter()

        print("=" * 60)
        print("All tests PASSED! ✓")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == '__main__':
    import pandas as pd
    # Patch data_providers with pd
    data_providers.pd = pd
    sys.exit(run_all_tests())