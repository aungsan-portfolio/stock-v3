# Dangerous / Ad-Hoc Scripts

> [!CAUTION]
> **HIGH RISK NOTICE**: Scripts in this folder may directly modify `config.py` in-place, alter account thresholds, or submit orders directly to broker endpoints (Paper or Live).

## Rules Before Execution:
1. **Never run these scripts without inspecting the source code first.**
2. **Verify target account**: Confirm API keys and paper/live mode in `.env`.
3. **Verify configuration state**: Check that `config.py` has not been corrupted or altered with unexpected hardcoded values.
4. **Trading risk acknowledgment**: Ensure you accept the market, slippage, and execution risk before executing.

## Script Catalog:
* `buy_amzn.bat`: Ad-hoc helper that modifies `MAX_OPEN_POSITIONS` in `config.py` and triggers order submission.
* `run_buy_check.bat`: Standalone full-market prediction scanner and print helper.
