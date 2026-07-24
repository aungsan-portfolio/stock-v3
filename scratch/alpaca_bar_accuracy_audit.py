import sys
import os
os.environ["APCA_API_KEY_ID"] = "PKMEKPN5QXHM5QRGKWMIQGDQDD"
os.environ["APCA_API_SECRET_KEY"] = "Fnh7DDi3AV2spLCACm8CJh1ksjaworq7ig84oPReP4hn"
os.environ["APCA_API_BASE_URL"] = "https://paper-api.alpaca.markets"
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import pandas as pd
import yfinance as yf
from strategies.intraday_data import fetch_intraday_yfinance, fetch_intraday_alpaca

def run_bar_accuracy_comparison():
    print("======================================================================")
    print("  PHASE A: BAR-LEVEL DATA ACCURACY AUDIT (yfinance vs Alpaca API)     ")
    print("======================================================================")

    symbols = ["NVDA", "AAPL", "MSFT", "AMZN", "TSLA"]
    results = []

    for sym in symbols:
        df_yf = fetch_intraday_yfinance(sym, interval="5m", lookback_days=7)
        df_alp = fetch_intraday_alpaca(sym, interval="5m", lookback_days=7)

        if df_yf.empty or df_alp.empty:
            print(f"[{sym}] Skipped due to empty data (yf_len={len(df_yf)}, alp_len={len(df_alp)})")
            continue

        # Align timezone
        if df_yf.index.tz is None:
            df_yf.index = df_yf.index.tz_localize("UTC")
        else:
            df_yf.index = df_yf.index.tz_convert("UTC")

        if "datetime" in df_alp.columns:
            df_alp["datetime"] = pd.to_datetime(df_alp["datetime"])
            df_alp.set_index("datetime", inplace=True)

        if df_alp.index.tz is None:
            df_alp.index = df_alp.index.tz_localize("UTC")
        else:
            df_alp.index = df_alp.index.tz_convert("UTC")

        # Find common timestamps
        common_idx = df_yf.index.intersection(df_alp.index)
        missing_in_alp = len(df_yf.index.difference(df_alp.index))
        missing_in_yf = len(df_alp.index.difference(df_yf.index))

        if len(common_idx) == 0:
            print(f"[{sym}] No common timestamps found between yf ({len(df_yf)}) and alpaca ({len(df_alp)})")
            continue

        yf_common = df_yf.loc[common_idx]
        alp_common = df_alp.loc[common_idx]

        close_diff_pct = (np.abs(yf_common["close"] - alp_common["close"]) / alp_common["close"]) * 100.0
        open_diff_pct = (np.abs(yf_common["open"] - alp_common["open"]) / alp_common["open"]) * 100.0

        results.append({
            "Symbol": sym,
            "Common_Bars": len(common_idx),
            "Missing_in_Alpaca": missing_in_alp,
            "Missing_in_YF": missing_in_yf,
            "Mean_Close_Dev_%": round(close_diff_pct.mean(), 4),
            "Max_Close_Dev_%": round(close_diff_pct.max(), 4),
            "Mean_Open_Dev_%": round(open_diff_pct.mean(), 4),
        })

    df_res = pd.DataFrame(results)
    print("\n--- BAR-LEVEL ACCURACY COMPARISON MATRIX ---")
    print(df_res.to_string())

if __name__ == "__main__":
    run_bar_accuracy_comparison()
