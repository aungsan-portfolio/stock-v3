# Module Reference — Quick Summary

| Module | Lines | Key Classes / Functions | Dependencies |
|--------|-------|------------------------|--------------|
| `main.py` | ~2210 | 20+ CLI commands via argparse | All modules |
| `config.py` | ~655 | `Settings` dataclass, 200+ constants | `pathlib`, `dataclasses` |
| `data_manager.py` | ~565 | `fetch_ohlcv()`, `build_features()`, `make_labels()`, `get_feature_columns()` | `yfinance`, `pandas`, `numpy` |
| `data_providers.py` | ~245 | `MultiSourceDataProvider` | `yfinance`, `pandas`, `requests` |
| `intraday_pipeline.py` | ~195 | `calculate_vwap()`, `calculate_orb()`, `M5 indicators` | `pandas`, `numpy` |
| `predictor.py` | ~320 | `Signal`, `weighted_blend()`, `Predictor` | `numpy`, `pandas` |
| `ai_engine.py` | ~340 | `StockRFEngine`, `build_rf()`, `cv_pooled_auc()` | `sklearn`, `joblib` |
| `lstm_engine.py` | ~640 | `AttentionLSTM`, `StockLSTMEngine`, meta-labeling | `torch`, `numpy` |
| `alternative_models.py` | ~340 | `StockXGBEngine`, `TransformerTSModel`, `StockTransformerEngine` | `xgboost`/sklearn, `torch` |
| `backtest.py` | ~490 | `run_backtest()`, `check_signal_parity()` | `numpy`, `pandas`, `sklearn` |
| `ibkr_bridge.py` | ~310 | `IBKRBridge` (orchestrator, delegates to 4 services) | `ib_insync` |
| `pricing_service.py` | ~165 | `PricingService` — contracts, quotes, sizing | `ib_insync` |
| `account_service.py` | ~125 | `AccountService` — cash, positions, orders | `ib_insync` |
| `entry_gate_service.py` | ~165 | `EntryGateService` — gates + Minervini sizing | — |
| `order_service.py` | ~410 | `OrderService` — bracket, close, protection | `ib_insync` |
| `trade_coach.py` | ~1250 | Lessons, previews, daily-coach, chart checks | — |
| `hot_scanner.py` | ~330 | Full-market scan, hybrid selection, ranking | `yfinance` |
| `risk_engine.py` | ~80 | Pure drawdown/exposure math | — |
| `risk_state.py` | ~100 | Daily loss kill-switch, trade counting | — |
| `account_guard.py` | ~270 | Account-type assertion (DU/U), fail-closed | — |
| `reconnect_watchdog.py` | ~275 | Bounded retry, disconnect handler | — |
| `shutdown_guard.py` | ~145 | Shutdown plan builder | — |
| `order_exec.py` | ~390 | Fill/classification, close-followup, GTC stop checks | — |
| `order_audit.py` | ~110 | Audit log event constants | — |
| `paper_ledger.py` | ~625 | Append-only entry/journal, exit sweep | — |
| `reconciliation.py` | ~175 | Broker-snapshot builder | — |
| `forward_test.py` | ~370 | Paper journal report generator | — |
| `expectancy.py` | ~525 | Closed-trade reconstruction, R-multiple | — |
| `eval_metrics.py` | ~130 | Pooled AUC, precision-at-BUY-threshold | — |
| `model_metrics.py` | ~240 | Per-symbol metric persistence + gate eval | — |
| `model_doctor.py` | ~430 | Coverage/freshness report + refresh | — |
| `generate_dashboard.py` | ~485 | Static HTML dashboard from JSON reports | `pandas`, `json` |
| `lstm_hyperopt.py` | ~100 | Optuna hyperparameter sweep | `optuna` |
| `minervini.py` | ~345 | Stage-2, VCP-like, pocket pivot, 1R sizing | — |
| `permutation_test.py` | ~260 | Shuffled-label null test | `sklearn` |
| `scheduler_runner.py` | ~315 | Market-hours gate + dispatch | — |
| `alerts.py` | ~270 | Log-only alerting (inert by default) | — |
| `live_invariants.py` | ~140 | Position/order adapters for pure checks | — |
| `coach_i18n.py` | ~260 | English / Burmese display layer | — |

## Dependency Graph

```
main.py ──── config.py ──── data_manager.py ──── data_providers.py ──── yfinance
                              │                           └─── intraday_pipeline.py
                              └─── ai_engine.py ──── predictor.py ──── ibkr_bridge.py
                              └─── lstm_engine.py         │               ├── pricing_service.py
                              └─── alternative_models.py  │               ├── account_service.py
                                                          │               ├── entry_gate_service.py
                                                          │               └── order_service.py
                                                          └─── backtest.py
                                                          └─── trade_coach.py
```
