# Stock Engine Pro V3 — Architecture

```mermaid
graph TB
    subgraph Data["📡 Data Layer"]
        DP[data_providers.py<br/>MultiSourceProvider] --> DM[data_manager.py<br/>fetch_ohlcv / build_features]
        IP[intraday_pipeline.py<br/>VWAP / ORB / M5] --> DM
        DM --> FEAT[Feature Engineering<br/>RSI / MACD / ATR / Candlestick]
    end

    subgraph ML["🧠 ML Engines"]
        RF[ai_engine.py<br/>Random Forest] --> PRED[predictor.py<br/>Ensemble Weighted Blend]
        LSTM[lstm_engine.py<br/>Attention LSTM] --> PRED
        XGB[alternative_models.py<br/>GBDT / XGBoost] --> PRED
        TRANS[alternative_models.py<br/>Transformer Encoder] --> PRED
    end

    subgraph EXEC["⚡ Execution"]
        PRED --> IB[ibkr_bridge.py<br/>IBKRBridge Orchestrator]
        PS[pricing_service.py<br/>Contract / Quote / Sizing] --> IB
        EGS[entry_gate_service.py<br/>Market Hours / Daily Loss] --> IB
        AS[account_service.py<br/>Cash / Positions] --> IB
        OS[order_service.py<br/>Bracket / Fill / Protection] --> IB
    end

    subgraph ANALYZE["📊 Analysis"]
        BT[backtest.py<br/>Walk-Forward Sim] --> REP[reports/]
        FT[forward_test.py<br/>Paper Journal] --> REP
        DASH[generate_dashboard.py<br/>HTML Dashboard] --> REP
        HYP[lstm_hyperopt.py<br/>Optuna Sweep] --> LSTM
    end

    subgraph SAFETY["🛡️ Safety"]
        RG[risk_engine.py<br/>Drawdown / Exposure]
        RS[risk_state.py<br/>Daily Kill-Switch]
        AG[account_guard.py<br/>Account Assertion]
        RW[reconnect_watchdog.py<br/>Bounded Retry]
        SG[shutdown_guard.py<br/>Graceful Teardown]
    end

    subgraph CI["🔄 CI/CD Pipeline"]
        MY[mypy<br/>Static Types]
        PT[pytest-cov<br/>Coverage ≥ 60%]
        GH[GitHub Actions<br/>3.11 / 3.12 Matrix]
    end

    IB --> SAFETY
    EXEC --> ANALYZE
    CI -.-> ML
    CI -.-> EXEC
```

## Layer Overview

| Layer | Key Files | Responsibility |
|-------|-----------|----------------|
| **Data** | `data_providers.py`, `data_manager.py`, `intraday_pipeline.py` | Multi-source fetch (yfinance/IBKR/Polygon), caching, feature engineering, VWAP/ORB/M5 |
| **ML Engines** | `ai_engine.py`, `lstm_engine.py`, `alternative_models.py` | RF, Attention LSTM, XGBoost/GBDT, Transformer — each with atomic save, per-symbol fallback, walk-forward CV |
| **Ensemble** | `predictor.py` | Weighted blend with missing-model detection, model-gate safety, confidence→BUY/HOLD/SELL |
| **Execution** | `ibkr_bridge.py`, `pricing_service.py`, `entry_gate_service.py`, `account_service.py`, `order_service.py` | Broker orchestration, quote validation, entry gates, bracket orders, fill verification, GTC/OCA protection |
| **Analysis** | `backtest.py`, `forward_test.py`, `expectancy.py`, `generate_dashboard.py` | Walk-forward backtest, paper journal, R-multiple expectancy, HTML dashboard |
| **Safety** | `risk_engine.py`, `risk_state.py`, `account_guard.py`, `reconnect_watchdog.py` | Drawdown halt, daily loss kill-switch, account-type assertion, bounded reconnect |
| **CI/CD** | `.github/workflows/ci.yml`, `pyproject.toml`, `requirements-dev.txt` | mypy type checking, pytest coverage gate ≥ 60%, matrix build 3.11/3.12 |
