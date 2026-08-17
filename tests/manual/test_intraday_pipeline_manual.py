"""
Manual verification script for the 1m/5m Intraday Data Pipeline.
"""
import sys
import os

# Add root directory to sys.path
sys.path.insert(0, str(os.path.abspath(os.path.join(__file__, ".."))))

import pandas as pd
import numpy as np
import test_intraday_pipeline

def run_tests():
    print("=" * 60)
    print("Running Intraday Pipeline manual validation...")
    print("=" * 60)

    try:
        t = test_intraday_pipeline.TestIntradayPipeline()

        print("Test 1: VWAP daily resetting...")
        t.test_vwap_resets_daily()
        print("   PASSED\n")

        print("Test 2: ORB High/Low projection...")
        t.test_orb_high_low_levels()
        print("   PASSED\n")

        print("Test 3: Aggregation to 5m...")
        t.test_aggregation_to_5m()
        print("   PASSED\n")

        print("Test 4: M5 Trend following indicators...")
        t.test_m5_indicators()
        print("   PASSED\n")

        print("=" * 60)
        print("All Intraday Pipeline tests PASSED! ✓")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    return 0

if __name__ == '__main__':
    sys.exit(run_tests())
