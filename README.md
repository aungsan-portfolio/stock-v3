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

## Full Market Hot Scanner

A **safe, opt-in** way to look beyond the small static `WATCHLIST` and discover
momentum/volume candidates across the broad US market. It is built as a
discovery layer that sits *in front of* the existing prediction + risk pipeline
— it never replaces it.

### What the words mean

- **Full-market scan** = broad **symbol discovery**, not auto-buy. It downloads
  the official Nasdaq Trader symbol directory, filters out junk instruments, and
  produces a candidate list. Producing a symbol is **not** a buy.
- **"Hot"** = a momentum/volume/trend **candidate**, not a guaranteed profit.
- **Prediction still decides** BUY/HOLD/SELL. The RF + LSTM + Technical ensemble
  (and `MIN_ML_MODELS_FOR_SIGNAL`) runs unchanged on the candidates.
- **`paper-hot` defaults to dry-run.** No orders are placed unless you add
  `--execute`.
- **`--execute` is required** for any paper order placement, and even then orders
  only go through the existing `IBKRBridge` with all existing risk controls.
- **Learning / paper trading only.** Past performance does not guarantee future
  results.

### How it works

1. `market_universe.py` downloads `nasdaqlisted.txt` + `otherlisted.txt`, caches
   them to `data/symbol_universe.csv` (falls back to the cache if offline),
   drops test issues / warrants / rights / units / preferred / notes / bonds, and
   normalizes symbols for yfinance (e.g. `BRK.B` → `BRK-B`).
2. `hot_scanner.py` fetches OHLCV via the existing `data_manager.fetch_ohlcv`,
   computes price / 20-day average volume / 1-5-20-day returns / volume ratio /
   ATR% / SMA position, rejects anything outside the configured price, liquidity
   and volatility bands, ranks the survivors by a `hot_score`, keeps the top N,
   and writes `reports/hot_candidates.csv`.
3. The existing `Predictor` and `IBKRBridge` take it from there.

One bad ticker is logged and skipped — it never aborts the scan.

### Commands

```bash
# Discover + rank only (no prediction, no orders)
python -X utf8 main.py scan-hot
python -X utf8 main.py scan-hot --full-market

# Scan, then run the model ensemble (no orders)
python -X utf8 main.py predict-hot
python -X utf8 main.py predict-hot --full-market

# Scan, then train RF + LSTM only for the hot candidates
python -X utf8 main.py train-hot
python -X utf8 main.py train-hot --full-market

# Scan + predict, DRY-RUN by default (never places orders)
python -X utf8 main.py paper-hot
python -X utf8 main.py paper-hot --full-market

# Scan + predict, then place PAPER orders through the existing IBKRBridge
python -X utf8 main.py paper-hot --full-market --execute

# Beginner safety check (read-only; loads universe, scans 50, predicts, no orders)
python -X utf8 verify_full_market_scanner.py
```

Optional per-run overrides: `--max-symbols N`, `--top-n N`, and
`--selection-mode {hybrid,random,rotation,alphabetical}`.

### Which symbols a full-market scan picks (`--selection-mode`)

The universe is stored alphabetically, so naively scanning the first
`--max-symbols` would only ever look at A/AB tickers. Selection decides *which*
slice of the broad market each run scans (discovery only — it never places
orders or changes risk rules):

| Mode | Behavior |
|---|---|
| `hybrid` *(default)* | Always scan `FULL_MARKET_CORE_SYMBOLS` first, then fill the rest with a shuffled, rotating sample of the universe. |
| `random` | Seeded random sample from the whole universe. |
| `rotation` | A different sequential slice each run — covers the whole market over time. |
| `alphabetical` | Original first-N behavior; for debugging only. |

```bash
python -X utf8 main.py scan-hot --full-market --max-symbols 500 --top-n 30 --selection-mode hybrid
python -X utf8 main.py scan-hot --full-market --max-symbols 500 --top-n 30 --selection-mode random
python -X utf8 main.py scan-hot --full-market --max-symbols 500 --top-n 30 --selection-mode rotation
```

The same flag applies to `predict-hot`, `paper-hot`, and `threshold-report`.
Every full-market run prints the universe size, mode, selected count, first 20
symbols, and how many core symbols were included, and writes the full selection
(with a `source` + `selection_reason` per symbol) to
`reports/selected_scan_symbols.csv`.

### Tunable config (in `config.py`)

| Setting | Default | Meaning |
|---|---|---|
| `FULL_MARKET_SCAN_ENABLED` | `True` | Master switch for broad discovery. |
| `FULL_MARKET_CACHE_HOURS` | `24` | How long the symbol directory cache stays fresh. |
| `FULL_MARKET_MAX_SYMBOLS_TO_CHECK` | `500` | Hard cap on symbols fetched per scan. Beginner-safe; raise deliberately. |
| `FULL_MARKET_SELECTION_MODE` | `"hybrid"` | Default symbol-selection strategy (see above). |
| `FULL_MARKET_CORE_SYMBOLS` | liquid majors | Anchors always scanned in `hybrid` mode. |
| `FULL_MARKET_RANDOM_SEED` | `42` | Seed for `random`/​hybrid-fill sampling (reproducible). |
| `FULL_MARKET_ROTATION_STATE_FILE` | `data/scan_rotation_state.json` | Persisted rotation cursor so runs advance through the market. |
| `HOT_SCAN_TOP_N` | `30` | How many ranked candidates to keep. |
| `HOT_SCAN_MIN_PRICE` / `HOT_SCAN_MAX_PRICE` | `5.0` / `1000.0` | Price band. |
| `HOT_SCAN_MIN_AVG_VOLUME` | `1_000_000` | Liquidity floor (20-day avg volume). |
| `HOT_SCAN_EXCLUDE_ETFS` | `False` | Keep normal ETFs as candidates. |
| `HOT_SCAN_MAX_ATR_PCT` | `0.12` | Reject names more volatile than this. |
| `HOT_SCAN_CHUNK_SIZE` / `HOT_SCAN_SLEEP_SECONDS` | `50` / `1.0` | Batch size + polite throttle. |

> The scanner can scale later (raise `FULL_MARKET_MAX_SYMBOLS_TO_CHECK`), but the
> beginner default deliberately caps work so a single run stays safe and not too
> slow.

### Safety notes for the scanner

- `scan-hot`, `predict-hot`, `train-hot`, and `paper-hot` **without `--execute`**
  never connect to IBKR and never place an order.
- `--execute` uses the **IBKR paper account only**: `REQUIRE_PAPER_PORT=True`,
  paper port `7497`, `ALLOW_SHORT=False`, and the existing `MAX_POSITION_PCT`,
  `MAX_TRADE_VALUE`, `MAX_OPEN_POSITIONS`, `MAX_DAILY_TRADES`, and duplicate-order
  guard all still apply. None of these are bypassed.
- If no BUY/SELL signal exists, no orders are placed.
- Existing positions / working orders are skipped, with the reason logged.

## Important Notes

The default backtest is `RF_TECHNICAL_WALK_FORWARD` because fold-local LSTM training can be slow. To test RF + LSTM + Technical without leakage, use `--include-lstm`; it trains a small LSTM inside each walk-forward fold using only past rows.

Paper trading only. Past performance does not guarantee future results.
