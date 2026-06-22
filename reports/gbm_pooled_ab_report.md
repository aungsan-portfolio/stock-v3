# Pooled cross-symbol GBM A/B (offline edge test)

_Generated: 2026-06-22 12:45:24_

Read-only / offline ML evaluation. No IBKR, no orders, no model saved (rf_models.joblib / lstm_checkpoint.pt / model_metrics.json untouched).
Leakage-safe panel walk-forward CV: split on the global DATE axis with a purge gap of 5 bars (ML_HORIZON); pooled GBM = sklearn HistGradientBoostingClassifier.

## Setup

- Universe : 108 symbols requested (model_metrics.json rf keys)
- Label modes : ['binary'] | arms : ['A', 'B']
- CV splits : 5 | purge : 5 | null draws : 20 (scope global)
- Candlestick features : off | seed : 42

### Per-symbol RF baseline (model_metrics.json (binary-trained RF))

- Mean AUC **0.5068** | median 0.5072 | symbols with AUC: 8/108

## binary — arm A (14 features, 108 symbols, 125521 rows)

- **Pooled CV-AUC : 0.4988** (null mean 0.5003, p95 0.5027, p-value **0.8095**, 20 draws)
- Precision@BUY(0.65) : 0.4987 on 1119 signals | positive rate 0.4944
- Per-symbol AUC : mean 0.5032 median 0.5009 [0.4346..0.704], 5/108 > 0.55

**Verdict — NO EDGE: pooled AUC 0.4988 (p=0.8095) vs RF baseline 0.5068. Pooling does not recover edge from this feature/label set.**

## binary — arm B (18 features, 108 symbols, 124009 rows)

- **Pooled CV-AUC : 0.5156** (null mean 0.5, p95 0.5027, p-value **0.0476**, 20 draws)
- Precision@BUY(0.65) : 0.5424 on 7048 signals | positive rate 0.4964
- Per-symbol AUC : mean 0.5162 median 0.5174 [0.4541..0.5682], 6/108 > 0.55

**Verdict — NO EDGE: pooled AUC 0.5156 (p=0.0476) vs RF baseline 0.5068. Pooling does not recover edge from this feature/label set.**
