# Testing & Verification

> ⚠️ **Paper-trading only.** This engine targets IBKR **paper** accounts. It is a
> research/education tool. Past performance does not guarantee future results.
> Do not point it at a live, real-money account.

## Dependencies

The test suite and verification script use only Python's standard-library
`unittest` — no pytest, no lint/type tooling. Installing the runtime
dependencies is enough:

```bash
pip install -r requirements-dev.txt   # pulls in -r requirements.txt
```

## Running the unit tests

```bash
python -m unittest test_logic -v      # verbose
python test_logic.py                  # equivalent direct run
```

The tests are **deterministic and offline**: they feed synthetic OHLCV data and
stub the per-symbol model predictions, so no network access or trained model
files are required. Backtest tests redirect `REPORTS_DIR` to a temp directory,
so they never overwrite real reports.

### What the tests cover

| Area | Coverage |
|---|---|
| `apply_position_rule_with_hold` | long-only open/hold/close, short-enabled open/cover, no single-bar flip |
| `min_hold` guard | opposite signals ignored until `MIN_HOLD_BARS`; exact hold-period |
| Ensemble safety | both ML models missing → forced `HOLD`; one model present → real ensemble |
| Config | `MIN_HOLD_BARS == ML_HORIZON` |
| Backtest I/O | empty-data → empty CSV + metrics JSON; populated → CSV rows + schema |
| Imports | `main`, `backtest`, `predictor`, `ibkr_bridge`, engines import cleanly |

## Continuous integration

`.github/workflows/ci.yml` runs on push / pull request against a Python
**3.11** and **3.12** matrix:

1. `python -m py_compile` over every module.
2. `python -m unittest test_logic -v`.

No build artifacts are produced.

## Production verification (multi-symbol)

`verify_watchlist.py` exercises the live prediction + backtest paths over the
whole `WATCHLIST` **without placing any orders**, then writes a summary JSON to
`reports/verification_summary.json`.

```bash
python verify_watchlist.py                  # predict + full backtest
python verify_watchlist.py --skip-backtest  # faster, predict-only
```

It asserts the safety invariants below and exits `0` on success, `1` on any
violation. (Unlike the unit tests, this script fetches **live** data via
`yfinance`, so it needs network access.)

## Current production-safety behavior

These behaviors are guaranteed by the current logic and verified by the tests
and the verification script:

- **Model missing → forced `HOLD`.**
  If a symbol has no trained ML model (neither RF nor LSTM), the engine forces
  `action = HOLD`. A missing per-symbol model raises `ModelNotAvailableError`
  rather than returning a neutral `0.5`, so it can never silently count as an
  available model. Technical-only BUY/SELL trading stays disabled
  (`MIN_ML_MODELS_FOR_SIGNAL = 1`).

- **`MIN_HOLD_BARS = ML_HORIZON`.**
  A position is held for at least `MIN_HOLD_BARS` bars before an opposite signal
  may close it. This is set equal to `ML_HORIZON` so the held period matches the
  forward-direction horizon the models are trained on.

- **Long-only by default (`ALLOW_SHORT = False`).**
  `SELL` closes an existing long; it does not open a short unless `ALLOW_SHORT`
  is explicitly enabled. BUY/SELL never flip a position in a single bar — they
  close to flat first.
