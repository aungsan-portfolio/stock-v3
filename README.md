# Stock Prediction Engine — Production Ready Logic Patch

RF + LSTM + Technical ensemble → BUY/HOLD/SELL → IBKR Paper Trading.

This package includes the production-safety fixes requested after the v2 review.

## Critical Fixes Included

| Area | Fix |
|---|---|
| `lstm_engine.py` | Fixed atomic `.npz` scaler save bug. No more accidental `.tmp.npz` filename mismatch. |
| `predictor.py` | If RF and LSTM are both missing, signal is forced to `HOLD`; technical-only trading is disabled by default. |
| `backtest.py` | Rewritten as a broker-like position-state simulator instead of isolated 5-day trade events. |
| `backtest.py` | Uses same technical-score helper and BUY/SELL thresholds as live predictor. |
| `backtest.py` | Includes transaction cost + slippage assumptions from `config.py`. |
| `backtest.py` | Optional slow full-ensemble backtest via `--include-lstm`. Default uses walk-forward RF + Technical for speed. |
| `ibkr_bridge.py` | Fixed `marketPrice()` handling; it is called as a method. |
| `ibkr_bridge.py` | Orders are counted as placed only when status is accepted, not rejected/inactive/cancelled. |
| `ai_engine.py` | RF model is evaluated on chronological holdout, then final production model is refit on all valid history. |
| `main.py` | Training returns non-zero exit codes if RF or LSTM produces no models. |

## Install

```bash
pip install -r requirements.txt
```

## Recommended Usage

```bash
# 1) Train models
python main.py train

# 2) Show today's signals
python main.py predict

# 3) Fast walk-forward backtest: RF + Technical, broker-like position rules
python main.py backtest --train-min 252 --step 21

# 4) Slow full-ensemble backtest: RF + LSTM + Technical
python main.py backtest --train-min 252 --step 21 --include-lstm

# 5) Preview IBKR paper orders only
python main.py paper --dry-run

# 6) Execute paper orders after reviewing dry-run output
python main.py paper
```

## Production Safety Defaults

- `ALLOW_SHORT = False`
  - SELL closes existing long positions.
  - SELL does **not** open short positions unless explicitly enabled.
- `MIN_ML_MODELS_FOR_SIGNAL = 1`
  - At least one trained ML model must be available.
  - If both RF and LSTM are missing, action is forced to `HOLD`.
- `BACKTEST_TRANSACTION_COST_PCT = 0.0005`
- `BACKTEST_SLIPPAGE_PCT = 0.0005`
- `MAX_OPEN_POSITIONS = 5`
- Opening trades use bracket orders; closing trades use plain limit orders.

## Important Notes

The default backtest is `RF_TECHNICAL_WALK_FORWARD` because fold-local LSTM training can be slow. To test RF + LSTM + Technical without leakage, use `--include-lstm`; it trains a small LSTM inside each walk-forward fold using only past rows.

Paper trading only. Past performance does not guarantee future results.
