# API Reference — Stock Engine Pro V3

## Entry Point

### `main.py`

Command-line interface with subcommands. All commands follow:
```bash
python -X utf8 main.py <command> [options]
```

| Command | Description | Example |
|---------|-------------|---------|
| `train` | Train RF + LSTM + XGB + Transformer models | `main.py train` |
| `predict` | Show today's BUY/HOLD/SELL signals for WATCHLIST | `main.py predict` |
| `backtest` | Walk-forward backtest (RF + Technical) | `main.py backtest --train-min 252` |
| `backtest --include-lstm` | Full-ensemble backtest (slow) | `main.py backtest --include-lstm` |
| `paper` | Run signals on IBKR paper account | `main.py paper --dry-run` |
| `scan-hot` | Full-market momentum/volume scanner | `main.py scan-hot --full-market` |
| `predict-hot` | Scan + predict hot candidates | `main.py predict-hot` |
| `paper-hot` | Scan + predict + paper orders | `main.py paper-hot --full-market --execute` |
| `coach` | Beginner trade lessons (no IBKR) | `main.py coach` |
| `paper-coach` | Preview + place ONE paper trade | `main.py paper-coach --confirm --chart-checked` |
| `daily-coach` | Full-market scan + multi-trade paper | `main.py daily-coach --max-trades 1` |
| `model-doctor` | Check RF/LSTM coverage + freshness | `main.py model-doctor` |
| `model-refresh` | Train models for hot candidates | `main.py model-refresh --from-report --top-n 30` |
| `threshold-report` | Sweep BUY thresholds (read-only) | `main.py threshold-report` |
| `expectancy-report` | R-multiple expectancy from backtest | `main.py expectancy-report` |
| `forward-test-report` | Paper fill journal stats | `main.py forward-test-report` |
| `live-readiness` | Go-live scorecard | `main.py live-readiness --connect` |
| `panic-flatten` | Emergency cancel + close all | `main.py panic-flatten --confirm` |
| `signal-parity` | Check backtest/live signal match | `main.py signal-parity` |
| `permutation-test` | Shuffled-label null test | `main.py permutation-test` |
| `run-scheduled` | One-shot market-hours scheduler | `main.py run-scheduled` |
| `dashboard` | Compile HTML dashboard | `main.py dashboard` |

---

## Data Layer

### `data_manager.py`

| Function | Returns | Description |
|----------|---------|-------------|
| `fetch_ohlcv(symbol, period, interval, force_refresh)` | `pd.DataFrame` | Multi-source OHLCV fetch with cache |
| `build_features(df, market_df)` | `pd.DataFrame` | 14+ engineered features (RSI, MACD, ATR, BB%, candlestick) |
| `make_labels(df, horizon, mode)` | `pd.Series` | Binary (endpoint) or triple_barrier labels |
| `get_feature_columns()` | `list[str]` | Active feature column names |
| `fetch_market_benchmark(period, interval)` | `pd.DataFrame` | SPY benchmark feed |
| `set_data_provider(provider)` | — | Inject custom MultiSourceDataProvider |

### `data_providers.py`

| Class | Description |
|-------|-------------|
| `MultiSourceDataProvider` | yfinance → IBKR → Polygon.io fallback chain |
| `.fetch(symbol, period, interval, force_refresh)` | OHLCV with disk cache + split detection |
| `DataProviderError` | Raised when all sources fail |

### `intraday_pipeline.py`

| Function | Description |
|----------|-------------|
| `calculate_vwap(df)` | Daily-resetting Volume Weighted Avg Price |
| `calculate_orb(df, minutes)` | Opening Range Breakout High/Low |
| `aggregate_to_5m(df_1m)` | 1m → 5m OHLCV resampling |
| `calculate_m5_indicators(df_5m)` | 8/21 EMA crossover, MACD, typical price |

---

## ML Engines

### `ai_engine.py` — Random Forest

| Function / Method | Description |
|-------------------|-------------|
| `build_rf(oob_score, cfg)` | Canonical RF factory (n=100, depth=6, balanced) |
| `StockRFEngine(settings)` | Per-symbol RF with atomic save, walk-forward CV, holdout |
| `.train(symbols, verbose)` | Returns dict of {symbol: metrics} |
| `.load()` | Load from joblib file |
| `.predict(symbol, df)` | Returns class-1 probability [0, 1] |
| `cv_pooled_auc(X, y, horizon, n_splits, cfg)` | Walk-forward OOS AUC |

### `lstm_engine.py` — Attention LSTM

| Class / Method | Description |
|----------------|-------------|
| `AttentionLSTM(input_size, hidden, layers, dropout, bidirectional)` | Self-attention LSTM with scaled dot-product |
| `StockLSTMEngine(settings)` | Per-symbol LSTM with AMP, early stopping, cyclic LR |
| `.train(symbols, verbose)` | Returns dict of symbol → {val_loss, val_auc, ...} |
| `.load()` | Load checkpoint with weights_only=True |
| `.predict(symbol, df)` | Sigmoid output [0, 1] |
| Meta-labeling | Secondary model predicts primary correctness |

### `alternative_models.py` — XGBoost / Transformer

| Class | Description |
|-------|-------------|
| `StockXGBEngine(settings)` | XGBoost → sklearn GradientBoosting fallback |
| `TransformerTSModel(input_size)` | Transformer Encoder with positional encoding |
| `StockTransformerEngine(settings)` | Per-symbol Transformer with save/load/predict |

---

## Ensemble

### `predictor.py`

| Function | Description |
|----------|-------------|
| `Signal` | Dataclass: symbol, action, confidence, rf/lstm/xgb/trans/tech scores |
| `technical_score_from_feature_row(row)` | Deterministic [0,1] score from RSI/MACD/BB |
| `weighted_blend(...)` | Renormalized weighted average over available models |
| `ml_model_count(...)` | Count models with positive weights |
| `enough_ml_models(...)` | Gate: at least `MIN_ML_MODELS_FOR_SIGNAL` available |
| `action_from_confidence(conf, cfg)` | `BUY` ≥ threshold, `SELL` ≤ threshold, else `HOLD` |
| `Predictor(settings)` | Loads RF, LSTM, XGB, Transformer; predicts all symbols |
| `.predict_all(symbols)` | Returns sorted `List[Signal]` |

**Ensemble Weights:**

| Weight | Default | Description |
|--------|---------|-------------|
| `WEIGHT_RF` | 0.40 | Random Forest |
| `WEIGHT_LSTM` | 0.00 | LSTM (disabled — sub-chance AUC) |
| `WEIGHT_XGB` | 0.20 | XGBoost / GBDT |
| `WEIGHT_TRANSFORMER` | 0.15 | Transformer Encoder |
| `WEIGHT_TECHNICAL` | 0.25 | Rule-based technical score |

---

## Execution Layer

### `ibkr_bridge.py`

| Method | Description |
|--------|-------------|
| `connect()` | Paper-port lock + account-type assertion + bounded retry |
| `disconnect()` | Clean shutdown with watchdog |
| `execute_signal(signal, coach)` | BUY/SELL through pricing, entry gates, bracket placement, protection |
| `execute_all(signals)` | Batch execute respecting MAX_OPEN_POSITIONS |
| `flatten_all(confirm)` | Cancel orders + market-close all positions |
| `ensure_protective_stops()` | Scan + repair GTC protective stops |
| `reconcile_startup_state()` | Broker-truth reconciliation (H18) |
| `graceful_shutdown(repair)` | Preserve stops, repair unprotected longs, disconnect |

### `pricing_service.py`

| Method | Description |
|--------|-------------|
| `contract(symbol)` | Cache-qualify Stock contract |
| `get_order_quote(symbol, timeout)` | Phase-4 validated quote (crossed/wide/stale blocked) |
| `get_price(symbol, timeout, allow_historical)` | Order price with historical fallback |
| `calc_quantity(price, cash)` | MIN(MAX_POSITION_PCT * cash, MAX_TRADE_VALUE) / price |
| `limit_price(action, price)` | BUY: +0.1%, SELL: -0.1% |

### `entry_gate_service.py`

| Method | Description |
|--------|-------------|
| `entry_blocked(symbol, value, signal, price)` | Returns (blocked, reason) — market hours, daily loss, drawdown, exposure, stage2 filter |
| `risk_sized_qty(symbol, signal, entry, notional)` | Minervini 1R shrink; fail-open |
| `market_hours_ok()` | US RTH gate (09:30-16:00 ET, holidays) |

### `account_service.py`

| Method | Description |
|--------|-------------|
| `get_cash()` | AvailableFunds (USD) |
| `get_position(symbol)` | Current position qty |
| `get_net_liquidation()` | Account equity for drawdown/daily-loss |
| `working_order_symbols()` | Symbols with active orders |
| `has_working_order(symbol, action)` | Duplicate-entry guard |

### `order_service.py`

| Method | Description |
|--------|-------------|
| `close_position(symbol, action, qty, price, note)` | Marketable-limit → market escalation (H8) |
| `place_open_bracket(symbol, action, qty, price, confidence)` | Limit entry + fill-confirmed protection |
| `place_protection(symbol, filled, basis)` | GTC/OCA hard stop + trailing/fixed stop |
| `verify_or_protect(symbol, parent, result, basis)` | Emergency flatten if protection fails |
| `market_close_symbol(symbol)` | Cancel orders + market-close position |

---

## Safety & Risk

| Module | Key Features |
|--------|--------------|
| `risk_engine.py` | `drawdown_halt_breached()`, `symbol_exposure_exceeded()` |
| `risk_state.py` | `daily_loss_blocked()`, `can_open_more()`, `snapshot_start_of_day_equity()` |
| `account_guard.py` | `assert_account()` — DU/U prefix check |
| `reconnect_watchdog.py` | Bounded exponential backoff (3 attempts max) |
| `shutdown_guard.py` | `build_shutdown_plan()` — preserve stops, detect unprotected longs |

---

## Config

### `config.py`

Dual-mode:
- **Module-level constants** (backward-compatible): `config.WEIGHT_RF`, `config.BUY_THRESHOLD`
- **Settings dataclass** (dependency injection): `Settings` → 200+ fields
- API: `get_settings()` → `Settings`, `set_settings(cfg)` (for tests/Optuna)

---

## Reports & Files

| File | Format | Contents |
|------|--------|----------|
| `reports/backtest_metrics.json` | JSON | Walk-forward metrics (avg return, sharpe, drawdown, per-symbol) |
| `reports/backtest_trades.csv` | CSV | Per-bar position-state ledger |
| `reports/forward_test_metrics.json` | JSON | Paper P&L, win/loss, drawdown, open positions |
| `reports/expectancy_report.md` | MD | R-multiple expectancy analysis |
| `reports/hot_candidates.csv` | CSV | Scanner ranked candidates |
| `reports/dashboard.html` | HTML | Static interactive dashboard |
| `models/rf_models.joblib` | joblib | RF classifiers per symbol |
| `models/lstm_checkpoint.pt` | PyTorch | LSTM state dicts + scalers |
| `models/transformer_checkpoint.pt` | PyTorch | Transformer state dicts + scalers |
| `models/xgb_models.joblib` | joblib | GBDT classifiers per symbol |
| `models/model_metrics.json` | JSON | Per-symbol RF/LSTM metrics + train timestamps |
