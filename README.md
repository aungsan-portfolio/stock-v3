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
- Opening trades use bracket orders; closing trades use plain limit orders.

## Safe Paper Config

Conservative defaults validated end-to-end against IBKR paper (tag `paper-e2e-v1`).

| Setting | Value | Meaning |
|---|---|---|
| `MAX_TRADE_VALUE` | `1000.0` | Hard dollar cap per trade. Final size = `min(cash * MAX_POSITION_PCT, MAX_TRADE_VALUE)`, so a large account never sizes a single trade above this. |
| `MAX_POSITION_PCT` | `0.01` | Percent-of-cash cap, applied before the dollar cap. |
| `MAX_OPEN_POSITIONS` | `1` | Only one open position at a time. |
| `ALLOW_SHORT` | `False` | Long-only. SELL closes longs; it never opens shorts. |
| `IBKR_PORT` | `7497` | TWS/Gateway must be in **Paper** mode on this port. |
| `IBKR_MARKET_DATA_TYPE` | `3` | Delayed (15-min) data, so paper accounts without a real-time subscription still get prices. Falls back to last daily close when snapshots fail. |

**TWS prerequisites:** API enabled, **Read-Only API unchecked**, socket clients enabled on port 7497.

**Windows:** always run with `python -X utf8 ...` so the emoji/box-drawing output does not crash on the `cp1252` console.

```bash
python -X utf8 main.py paper --dry-run   # preview, no orders
python -X utf8 main.py paper             # place paper orders
```

Helper scripts: `test_ibkr_connect.py` (connection check), `check_positions.py` (positions + open orders), `cancel_open_orders.py` (cancel all working orders), `flatten_vti.py` (market-sell a position).

## Important Notes

The default backtest is `RF_TECHNICAL_WALK_FORWARD` because fold-local LSTM training can be slow. To test RF + LSTM + Technical without leakage, use `--include-lstm`; it trains a small LSTM inside each walk-forward fold using only past rows.

Paper trading only. Past performance does not guarantee future results.
