import sys
import os
sys.path.insert(0, os.path.abspath("."))
import pandas as pd
from strategies.backtester import run_portfolio_backtest

def main():
    data_file = os.path.join(".", "data", "canonical", "alpaca_60d_insample_days1_45.pkl")
    data_dict = pd.read_pickle(data_file)

    res = run_portfolio_backtest(
        symbols=list(data_dict.keys()),
        strategy_name="VWAP_BOUNCE",
        lookback_days=45,
        override_min_confidence=0.40,
        plot=False,
        data_dict=data_dict
    )

    trades = res.get("trades", [])
    print(f"Total In-Sample Trades: {len(trades)}")
    print(f"In-Sample WR:  {res.get('win_rate', 0.0):.2f}%")
    print(f"In-Sample PF:  {res.get('profit_factor', 0.0):.2f}")
    print(f"In-Sample DD:  {res.get('max_drawdown', 0.0):.2f}%")

    if trades:
        df_t = pd.DataFrame(trades)
        df_t["stop_dist"] = abs(df_t["entry"] - df_t["stop"])
        
        def classify_exit(r):
            r_str = str(r).upper()
            if "STOP" in r_str or "SL" in r_str: return "STOP_LOSS"
            if "TARGET" in r_str or "TP" in r_str: return "TAKE_PROFIT"
            return "EOD_FLATTEN"

        df_t["exit_bucket"] = df_t["reason"].apply(classify_exit)
        print("\n--- EXIT DECOMPOSITION ---")
        print(df_t.groupby("exit_bucket")["pnl"].agg(["count", "sum", "mean"]))
        print("\n--- STOP DISTANCE STATS ---")
        print(f"Mean Stop:   ${df_t['stop_dist'].mean():.4f}")
        print(f"Median Stop: ${df_t['stop_dist'].median():.4f}")
        print(f"75th Pct:    ${df_t['stop_dist'].quantile(0.75):.4f}")

if __name__ == "__main__":
    main()
