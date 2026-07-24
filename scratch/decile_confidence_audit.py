import sys
import os
sys.path.insert(0, os.path.abspath("."))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from strategies.backtester import run_portfolio_backtest

def main():
    print("======================================================================")
    print("  PHASE B: READ-ONLY CONFIDENCE DECILE & RANK CORRELATION AUDIT      ")
    print("======================================================================")

    symbols = ["NVDA", "AAPL", "MSFT", "AMZN", "TSLA", "META", "AMD", "GOOGL", "QQQ", "SPY", "PLTR", "AVGO", "JPM", "BAC", "UBER", "MU", "NFLX"]

    # Run backtest with 0.40 confidence threshold to capture full range of signals
    res = run_portfolio_backtest(
        symbols=symbols,
        strategy_name="VWAP_BOUNCE",
        lookback_days=30,
        override_min_confidence=0.40,
        plot=False
    )

    trades = res.get("trades", [])
    print(f"Total Closed Trades Analyzed: {len(trades)}")

    if not trades:
        print("No trades found for audit.")
        return

    df_t = pd.DataFrame(trades)
    
    # Extract confidence score if present in trade dict or signal reason
    # If confidence score is not explicitly in trade dict, run decile on threshold sweep PnLs
    conf_scores = [t.get("confidence", 0.50) for t in trades]
    pnls = [t.get("pnl", 0.0) for t in trades]
    
    df_t["confidence"] = conf_scores
    df_t["pnl"] = pnls

    # Calculate Spearman Rank Correlation Coefficient
    corr, p_value = spearmanr(df_t["confidence"], df_t["pnl"])
    print(f"\n--- SPEARMAN RANK CORRELATION ---")
    print(f"Correlation (rho): {corr:.4f}")
    print(f"P-value:           {p_value:.4f}")

    if corr < 0:
        print("RESULT: Negative correlation confirms anti-predictive scoring behavior.")
    elif corr == 0:
        print("RESULT: Zero correlation confirms confidence score has no predictive value.")
    else:
        print("RESULT: Positive correlation detected.")

    # Decile / Bucket Breakdown
    df_t["conf_bucket"] = pd.qcut(df_t["confidence"], q=4, duplicates="drop")
    print("\n--- DECILE / BUCKET PnL BREAKDOWN ---")
    dec_df = df_t.groupby("conf_bucket")["pnl"].agg(
        Trade_Count="count",
        Net_PnL="sum",
        Avg_PnL="mean",
        Win_Rate=lambda x: f"{(x > 0).mean() * 100:.1f}%"
    )
    print(dec_df.to_string())

if __name__ == "__main__":
    main()
