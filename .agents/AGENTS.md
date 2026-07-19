# Workspace Customization Rules

This file defines the project-scoped operational constraints and rules for stock engine trading activities.

## 1. Trader Location & Timezone
* **Location**: Chiang Mai, Thailand (ICT, UTC+7 timezone).
* **Daylight Saving Time (DST / EDT)** (Summer/Fall):
  - Pre-market scanner: **8:00 PM - 8:15 PM** Thailand Time.
  - Regular Market trading: **8:30 PM - 3:00 AM (next day)** Thailand Time.
* **Standard Time (EST)** (Winter/Spring):
  - Pre-market scanner: **9:00 PM - 9:15 PM** Thailand Time.
  - Regular Market trading: **9:30 PM - 4:00 AM (next day)** Thailand Time.

## 2. Multi-Account Execution Safety (CRITICAL)
> [!IMPORTANT]
> **DO NOT** run Day Trading (`daytrade-bot`) and Swing Trading (`paper`/`predict`) on the same single Alpaca account.
> 
> **Why**: Day Trading contains automated end-of-day flatten rules (`flatten_all()`) and risk circuit breakers. When triggered, these API calls will wipe out all resting orders and close *every* position on the account, accidentally liquidating long-term Swing trading positions.
> 
> **Rule**: Always use separate Alpaca Sub-accounts (with separate API keys) to keep Day Trading and Swing Trading assets completely isolated.

## 3. Quantitative Promotion Gates (Paper -> Real Money)
Before migrating the Day Trading strategy from paper trading to real money trading, the following thresholds must be met (audited via `evaluate_paper_results.py`):
1. **Statistical Validity**: $N \ge 30$ closed trades.
2. **Profit Factor**: $PF \ge 1.25$.
3. **Win Rate**: $WR \ge 40\%$.
4. **Drawdown Tolerance**: $\text{Max Drawdown} \le 5\%$ (calculated on a trade-by-trade intraday basis).
5. **Slippage Audit**: $\text{Avg Slippage} \le 0.05\%$.
