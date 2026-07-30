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
| `data_manager.py` | Fixed `volume_shock` lookahead leakage, `rolling_efficiency` unit mismatch, and upgraded default RSI to Wilder's exponential smoothing. |
| `trade_coach.py` | Aligned primary preview `quantity` with risk-based sizing (`quantity_by_risk`), retaining `quantity_by_cap` as diagnostic field. |
| `backtest.py` | Evaluates candle Low for intraday hard stop triggers and records initial stop/risk fields (`initial_stop_price`, `risk_per_share`, `risk_source`, `exit_price`, `exit_reason`) in ledger CSV output. |
| `expectancy.py` | Automatically computes true trade-level R-multiples when initial stop prices are present. |

> [!IMPORTANT]
> **Model Retraining Required**: Default RSI now uses Wilder's exponential smoothing (`RSI_SMOOTHING="wilder"`). Run `python main.py train` to retrain models on updated feature distributions. (Set `RSI_SMOOTHING="simple"` in `config.py` if legacy SMA RSI feature alignment is needed).

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

## Guided Paper Trading Coach

A **beginner-friendly, preview-first** flow that turns the same RF + LSTM +
Technical signals into a plain-English trade *lesson* and a safe **paper-trade
preview**. It is designed to help you *learn* while paper trading — it **never
auto-buys**. A paper order is placed only when you explicitly pass **both**
`--confirm` **and** `--chart-checked`.

### Recommended workflow

```bash
# 1) Learn: signals + lessons for your WATCHLIST (no IBKR, no orders)
python -X utf8 main.py coach

# 2) Preview: connect to paper, see ONE proposed trade with full risk math (no orders)
python -X utf8 main.py paper-coach

# 3) Check the chart manually — support/resistance, recent news, the proposed stop

# 4) Place at most ONE paper order, only after the chart check
python -X utf8 main.py paper-coach --confirm --chart-checked
```

If you run `paper-coach --confirm` *without* `--chart-checked`, the coach
refuses with **"Chart check required before paper execution."** and places
nothing.

### What the coach explains

For each signal it teaches: what the **ticker** is, what **BUY / HOLD / SELL**
mean, what **confidence** means and whether the signal is **strong or
borderline**, what the **RF / LSTM / Technical** sub-scores mean, **why** a name
is or is not a trade candidate, the **estimated quantity and cost**, the
**average cost** if adding, the **stop / trailing stop** and the **estimated max
loss** if the stop triggers, and **why a manual chart check is required** before
any execution.

### Commands

```bash
# Lessons on WATCHLIST (no IBKR connection, no orders)
python -X utf8 main.py coach

# Lessons on the latest reports/hot_candidates.csv (run `scan-hot` first);
# prints only the top 1-3 candidates. No IBKR connection, no orders.
python -X utf8 main.py coach-hot

# Connect to the paper account (read cash / positions / open orders) and PREVIEW
# the single best BUY candidate. No order is placed.
python -X utf8 main.py paper-coach

# Refuses — chart check is required before any paper execution.
python -X utf8 main.py paper-coach --confirm

# Place at most ONE paper order through the existing IBKRBridge risk controls.
python -X utf8 main.py paper-coach --confirm --chart-checked
```

A markdown audit trail is appended to `reports/trade_coach_report.md` on every
run (date/time, candidate, signal, confidence, RF/LSTM/Tech, price, estimated
quantity/cost, stop and trailing-stop explanation, max estimated loss, chart
check reminder, and whether the run was *preview only* or *paper order placed*).

### Coach safety

- **Default is no orders.** `coach` and `coach-hot` never connect to IBKR;
  `paper-coach` connects read-only unless you confirm.
- **One trade max per run** (`COACH_MAX_NEW_TRADES_PER_RUN = 1`).
- **Chart check required** before any execution (`--chart-checked`).
- **Paper account only** — `REQUIRE_PAPER_PORT = True`, paper port `7497`.
- **All existing risk controls preserved** — `ALLOW_SHORT = False`,
  `MAX_TRADE_VALUE`, `MAX_POSITION_PCT`, `MAX_OPEN_POSITIONS`, `MAX_DAILY_TRADES`,
  and the duplicate-order guard all still apply. The coach decides *whether* to
  call `IBKRBridge.execute_signal`; it never bypasses it.
- **No `--execute` and no historical-close pricing** in the coach flow. Orders
  go through the live snapshot pricing path (`ALLOW_HISTORICAL_PRICE_FOR_ORDERS`
  stays `False`).

### Coach config (in `config.py`)

| Setting | Default | Meaning |
|---|---|---|
| `COACH_MIN_CONFIDENCE_FOR_CANDIDATE` | `0.65` | Confidence floor for a BUY to be a coach candidate. |
| `COACH_REQUIRE_CHART_CHECK` | `True` | Require a manual chart check before execution. |
| `COACH_REQUIRE_USER_CONFIRM` | `True` | Require explicit `--confirm` before execution. |
| `COACH_MAX_NEW_TRADES_PER_RUN` | `1` | Hard cap on new trades for the single-symbol `paper-coach` flow. |
| `COACH_REPORT_FILE` | `reports/trade_coach_report.md` | Where the lesson/preview audit trail is written. |

## Guided Daily Trading Coach

`daily-coach` is the **multi-trade PAPER practice** flow. It scans the **full
market** (the hybrid scanner), previews the strongest BUY candidates, and — only
when you pass **both `--confirm` and `--chart-checked`** — may place **more than
one** paper order in a single run (up to `COACH_MAX_PAPER_TRADES_PER_RUN`, default
**3**) so you can **learn faster while paper trading**. It is still **paper-only**
and every existing risk control is preserved.

> **Live trading remains disabled. This bot is paper-trading only.** There is no
> live-trading mode and no live account port. Do **not** move to live trading
> until you have reviewed your paper results.

### Commands

```bash
# Default: full-market scan, preview the top 3 candidates. No orders.
python -X utf8 main.py daily-coach

# Preview up to 3 best candidates. No orders.
python -X utf8 main.py daily-coach --max-trades 3

# Refuses — chart check is required before any paper execution.
python -X utf8 main.py daily-coach --confirm --max-trades 3

# May place up to 3 PAPER trades (only valid candidates, only if they exist).
python -X utf8 main.py daily-coach --confirm --chart-checked --max-trades 3
```

A beginner can safely run the first two to **preview the top 3 candidates** with
full risk math. **Paper execution** (up to 3 trades) requires
`--confirm --chart-checked --max-trades 3`.

### What it prints

- **Scanned market count** and the top candidates.
- For **each candidate**: symbol, bot action, confidence, chart status, *why
  selected*, *why skipped or accepted*, estimated quantity, estimated cost,
  stop / trailing stop, and estimated possible loss.
- An **execution summary**: requested max trades, valid candidates, orders
  placed, every skipped candidate with its reason, and the reminder
  *"This is paper trading practice."*

### Execution gates (all must hold to place a paper order)

`--confirm` **and** `--chart-checked` are both required (without `--chart-checked`
the run refuses with *"Chart check required before paper execution."*). Then each
candidate must independently satisfy:

- a **BUY** signal with **confidence ≥ `BUY_THRESHOLD`**;
- **no existing position** and **no working order** in the same symbol;
- a chart status that is **not** `TOO_EXTENDED`, `BELOW_TREND`, `LOW_VOLUME`, or
  `MODEL_MISSING`;
- the run-level caps: `--max-trades` (clamped to `COACH_MAX_PAPER_TRADES_PER_RUN`,
  it can only lower the cap), `MAX_OPEN_POSITIONS`, and `MAX_DAILY_TRADES`.

If only 1 candidate is valid, exactly 1 order is placed; if 0 are valid, no orders
are placed. Every candidate that is not traded prints a skip reason.

### Daily-coach safety

- **Paper only — live trading is hard-blocked.** `daily-coach` calls an explicit
  guard before doing anything: if any config or command attempts live trading it
  refuses with *"Live trading is disabled. This bot is paper-trading only."* The
  guard requires `COACH_LIVE_TRADING_ENABLED = False`, `REQUIRE_PAPER_PORT = True`,
  `IBKR_PORT == PAPER_IBKR_PORT`, `ALLOW_SHORT = False`, and
  `ALLOW_HISTORICAL_PRICE_FOR_ORDERS = False`.
- **No live account port** is ever added or used.
- **All existing risk controls preserved** — `MAX_TRADE_VALUE`,
  `MAX_POSITION_PCT`, `MAX_OPEN_POSITIONS`, `MAX_DAILY_TRADES`, the
  duplicate-position / working-order guards, and live-snapshot-only pricing all
  still apply. Orders go only through the existing `IBKRBridge`; nothing bypasses
  it or its caps.

### Daily-coach config (in `config.py`)

| Setting | Default | Meaning |
|---|---|---|
| `COACH_MAX_PAPER_TRADES_PER_RUN` | `3` | Hard upper bound on PAPER trades one `daily-coach` run may place. `--max-trades` can only lower it. |
| `COACH_DEFAULT_PREVIEW_CANDIDATES` | `3` | How many top candidates the default (preview-only) run shows. |
| `COACH_LIVE_TRADING_ENABLED` | `False` | Master live-trading kill switch. Must stay `False` — this bot is paper-only. |

## Important Notes

The default backtest is `RF_TECHNICAL_WALK_FORWARD` because fold-local LSTM training can be slow. To test RF + LSTM + Technical without leakage, use `--include-lstm`; it trains a small LSTM inside each walk-forward fold using only past rows.

Paper trading only. Past performance does not guarantee future results.
