import sys
import os
sys.path.insert(0, os.path.abspath("."))
import numpy as np
import pandas as pd
from strategies.backtester import run_portfolio_backtest
import config

def run_empirical_bootstrap_audit():
    print("======================================================================")
    print("  EMPIRICAL TRADE-LEVEL BOOTSTRAP AUDIT (WITH SANITY ASSERTION)      ")
    print("======================================================================")

    data_file = os.path.join(".", "data", "canonical", "alpaca_60d_insample_days1_45.pkl")
    if not os.path.exists(data_file):
        print(f"Error: Dataset {data_file} not found.")
        return

    data_dict = pd.read_pickle(data_file)
    symbols = list(data_dict.keys())

    # Run real backtester engine for 2.5x ATR14
    config.ATR_STOP_MULTIPLE = 2.5
    res = run_portfolio_backtest(
        symbols=symbols,
        strategy_name="VWAP_BOUNCE",
        lookback_days=45,
        override_min_confidence=0.40,
        plot=False,
        data_dict=data_dict
    )

    trades = res.get("trades", [])
    if not trades:
        print("No trades generated.")
        return

    df_t = pd.DataFrame(trades)
    pnls = df_t["pnl"].values
    n = len(pnls)

    point_pf = res.get("profit_factor", 0.0)
    point_exp = np.mean(pnls)

    print(f"\n--- EMPIRICAL BACKTEST OUTPUT (Days 1-45 In-Sample) ---")
    print(f"Total Trade Count N:    {n}")
    print(f"Win Rate:               {res.get('win_rate', 0.0):.2f}%")
    print(f"Point Est Profit Factor: {point_pf:.4f}")
    print(f"Point Est Expectancy ($): ${point_exp:.4f}")

    # Bootstrap 1,000 Iterations on ACTUAL Empirical Trade PnLs
    n_iterations = 1000
    boot_pfs = []
    boot_exps = []

    np.random.seed(42)
    for _ in range(n_iterations):
        sample = np.random.choice(pnls, size=n, replace=True)
        wins = sample[sample > 0]
        losses = abs(sample[sample < 0])

        sum_w = np.sum(wins) if len(wins) > 0 else 0.0
        sum_l = np.sum(losses) if len(losses) > 0 else 0.0

        pf = sum_w / sum_l if sum_l > 0 else (sum_w if sum_w > 0 else 0.0)
        exp = np.mean(sample)

        boot_pfs.append(pf)
        boot_exps.append(exp)

    mean_boot_exp = np.mean(boot_exps)
    mean_boot_pf = np.mean(boot_pfs)

    # Sanity Assertion check (bootstrap mean vs point estimate within 5%)
    exp_diff_pct = abs(mean_boot_exp - point_exp) / abs(point_exp) * 100.0 if point_exp != 0 else 0.0
    print(f"\n--- SANITY ASSERTION CHECK ---")
    print(f"Point Est Expectancy:    ${point_exp:.4f}")
    print(f"Bootstrap Mean Exp:     ${mean_boot_exp:.4f}")
    print(f"Relative Deviation:     {exp_diff_pct:.2f}%")

    assert exp_diff_pct <= 5.0, f"SANITY FAILURE: Bootstrap mean ${mean_boot_exp:.4f} deviates >5% from point est ${point_exp:.4f}!"
    print("SANITY ASSERTION: PASSED (Bootstrap mean matches point estimate perfectly within 5%!).")

    # Calculate 95% Confidence Interval
    pf_ci = (np.percentile(boot_pfs, 2.5), np.percentile(boot_pfs, 97.5))
    exp_ci = (np.percentile(boot_exps, 2.5), np.percentile(boot_exps, 97.5))

    print(f"\n--- EMPIRICAL 95% BOOTSTRAP CONFIDENCE INTERVAL (1,000 Iterations) ---")
    print(f"Profit Factor (PF) Point Est: {point_pf:.2f}")
    print(f"Profit Factor 95% CI:        [{pf_ci[0]:.2f}, {pf_ci[1]:.2f}]")
    print(f"Expectancy ($) Point Est:    ${point_exp:.2f}")
    print(f"Expectancy 95% CI:           [${exp_ci[0]:.2f}, ${exp_ci[1]:.2f}]")

    if pf_ci[0] >= 1.0:
        print("\n[VERDICT] Statistically Robust Edge: 95% Lower Bound of PF >= 1.0.")
    else:
        print(f"\n[VERDICT] Statistical Fragility Detected: 95% Lower Bound of PF [{pf_ci[0]:.2f}] < 1.0.")

if __name__ == "__main__":
    run_empirical_bootstrap_audit()
