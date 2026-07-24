import sys
import os
sys.path.insert(0, os.path.abspath("."))
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from strategies.backtester import run_portfolio_backtest

def main():
    print("======================================================================")
    print("  PHASE B: COMPONENT-LEVEL FEATURE CORRELATION & DECISION POINT B AUDIT ")
    print("======================================================================")

    data_file = os.path.join(".", "data", "canonical", "alpaca_60d_5m_canonical.pkl")
    if not os.path.exists(data_file):
        print(f"Error: Canonical dataset file {data_file} not found.")
        return

    data_dict = pd.read_pickle(data_file)
    print(f"Loaded 60-day canonical dataset with {len(data_dict)} symbols.")

    symbols = list(data_dict.keys())
    res = run_portfolio_backtest(
        symbols=symbols,
        strategy_name="VWAP_BOUNCE",
        lookback_days=60,
        override_min_confidence=0.40,
        plot=False,
        data_dict=data_dict
    )

    trades = res.get("trades", [])
    print(f"Total Closed Trades in 60-Day In-Sample Dataset: {len(trades)}")

    if not trades:
        print("No trades generated for component audit.")
        return

    df_t = pd.DataFrame(trades)
    
    # Analyze component features
    features = ["confidence", "stop", "target", "entry"]
    df_t["stop_dist"] = abs(df_t["entry"] - df_t["stop"])
    df_t["target_dist"] = abs(df_t["target"] - df_t["entry"])
    df_t["rr_ratio"] = df_t["target_dist"] / df_t["stop_dist"].replace(0, np.nan)

    audit_features = ["confidence", "stop_dist", "target_dist", "rr_ratio"]

    print("\n--- COMPONENT-LEVEL SPEARMAN RANK CORRELATION MATRIX vs REALIZED PnL ---")
    corr_results = []
    for feat in audit_features:
        if feat in df_t.columns and df_t[feat].nunique() > 1:
            rho, pval = spearmanr(df_t[feat], df_t["pnl"])
            corr_results.append({
                "Feature": feat,
                "Spearman_Rho": round(rho, 4),
                "P_Value": round(pval, 4),
                "Predictive_Status": "VALID" if (pval < 0.05 and rho > 0.10) else ("ANTI-PREDICTIVE" if (pval < 0.05 and rho < -0.10) else "ZERO (NO SIGNAL)")
            })

    df_corr = pd.DataFrame(corr_results)
    print(df_corr.to_string())

    # Decision Point B Check
    overall_conf_rho = df_corr.loc[df_corr["Feature"] == "confidence", "Spearman_Rho"].values
    conf_status = df_corr.loc[df_corr["Feature"] == "confidence", "Predictive_Status"].values

    print("\n======================================================================")
    print("  DECISION POINT B EVALUATION                                         ")
    print("======================================================================")
    if len(overall_conf_rho) > 0 and (abs(overall_conf_rho[0]) < 0.05 or conf_status[0] == "ZERO (NO SIGNAL)"):
        print("[DECISION POINT B TRIGGERED]")
        print("Confidence scoring core has statistically ZERO predictive power over trade PnL.")
        print("Cosmetic threshold adjustments will NOT improve strategy performance.")
        print("Action Required: Pivot to Signal Core Replacement & Structural Entry Edge Reform.")
    else:
        print("Confidence scoring core shows valid predictive power. Proceed with threshold/filter tuning.")

if __name__ == "__main__":
    main()
