import sys
import os
os.environ["APCA_API_KEY_ID"] = "PKMEKPN5QXHM5QRGKWMIQGDQDD"
os.environ["APCA_API_SECRET_KEY"] = "Fnh7DDi3AV2spLCACm8CJh1ksjaworq7ig84oPReP4hn"
os.environ["APCA_API_BASE_URL"] = "https://paper-api.alpaca.markets"
sys.path.insert(0, os.path.abspath("."))

from datetime import datetime, timedelta, timezone
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

SYMBOLS = ["NVDA", "AAPL", "MSFT", "AMZN", "TSLA", "META", "AMD", "GOOGL", "QQQ", "SPY", "PLTR", "AVGO", "JPM", "BAC", "UBER", "MU", "NFLX"]

def build_60day_dataset():
    print("======================================================================")
    print("  PHASE A: ALPACA 60-90 DAY CANONICAL DATASET BUILDER                 ")
    print("======================================================================")

    api_key = os.environ["APCA_API_KEY_ID"]
    secret_key = os.environ["APCA_API_SECRET_KEY"]
    client = StockHistoricalDataClient(api_key, secret_key)

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=60)

    print(f"Fetching 60-day 5m bars from {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')} for {len(SYMBOLS)} symbols...")

    data_map = {}
    for sym in SYMBOLS:
        try:
            req = StockBarsRequest(
                symbol_or_symbols=sym,
                timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                start=start_dt,
                end=end_dt,
                feed="iex"
            )
            bars = client.get_stock_bars(req)
            if hasattr(bars, "df") and not bars.df.empty:
                df = bars.df
                if isinstance(df.index, pd.MultiIndex):
                    df = df.xs(sym, level="symbol")
                df = df.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"})
                df.index.name = "datetime"
                data_map[sym] = df
                print(f"  [{sym}] Successfully loaded {len(df)} 5m bars via Alpaca IEX.")
            else:
                print(f"  [{sym}] Warning: Empty bars returned from Alpaca IEX.")
        except Exception as e:
            print(f"  [{sym}] Error fetching Alpaca IEX bars: {e}")

    out_dir = os.path.join(".", "data", "canonical")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "alpaca_60d_5m_canonical.pkl")
    pd.to_pickle(data_map, out_file)
    print(f"\n[SUCCESS] Saved 60-day canonical dataset with {len(data_map)} symbols to: {out_file}")

if __name__ == "__main__":
    build_60day_dataset()
